# model/trainer.py

import torch
import torch.nn as nn
# from torch_geometric import nn
from tqdm import tqdm
from utils import Utils
from loss_functions import get_loss_function
from logger import *
import json
import os
from typing import Tuple
from evaluator import Evaluator
import math
import time


class Trainer:
    """
    Trainer class for hybrid classical-quantum models.

    Handles training loop with separate optimizers for classical and quantum components,
    validation, and experiment tracking. Supports different loss functions and
    learning rate schedules.

    Attributes:
        model: The UniversalHybridSlotModel to train.
        device: Device for training (cuda/cpu).
        config: Training configuration dictionary.
        classic_optimizer: Optimizer for classical parameters.
        quantum_optimizer: Optimizer for quantum parameters (if quantum branch enabled).

    Example:
        >>> trainer = HybridTrainer(model, config, device='cuda')
        >>> trainer.train(train_loader, val_loader)
    """

    def __init__(self, model: nn.Module, evaluator: Evaluator, config: dict, device: str = 'cuda') -> None:
        """
        Initialize the hybrid trainer.

        Args:
            model: Model instance to train.
            config: Configuration dictionary with training parameters.
            device: Device for training ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.train_cfg = config['training']
        self.evaluator = evaluator or Evaluator(model, device)

        log_debug("--- FULL MODEL PARAMETERS SCAN ---", stage="DEBUG")
        all_params = list(model.named_parameters())
        for name, p in all_params:
            log_debug(f"Name: {name} | Shape: {list(p.shape)} | RequiresGrad: {p.requires_grad}", stage="OPTIMIZER")

        q_keys = ["qlayer", "final_layer"]
        quantum_params = [p for n, p in model.named_parameters() if any(k in n for k in q_keys)]
        classic_params = [p for n, p in model.named_parameters() if not any(k in n for k in q_keys)]

        # 2. Initialize optimizers from config
        c_opt_cfg = self.train_cfg['optimizers']['classic']
        self.opt_classic = getattr(torch.optim, c_opt_cfg['type'])(classic_params, **c_opt_cfg['params'])
        log_info(f"Classic: {c_opt_cfg['type']} with {sum(p.numel() for p in classic_params)}"
                 f" parameters in {len(classic_params)} tensors", stage="OPTIMIZER")

        q_opt_cfg = self.train_cfg['optimizers']['quantum']
        if len(quantum_params) > 0:
            self.opt_quantum = getattr(torch.optim, q_opt_cfg['type'])(quantum_params, **q_opt_cfg['params'])
            log_info(f"Quantum: {q_opt_cfg['type']} with {sum(p.numel() for p in quantum_params)}"
                     f" parameters in {len(quantum_params)} tensors", stage="OPTIMIZER")
        else:
            self.opt_quantum = None
            log_warn(f"Quantum: No quantum parameters found, optimizer disabled", stage="OPTIMIZER")

        self.sched_classic = self._build_scheduler(self.opt_classic, c_opt_cfg.get('scheduler'))
        self.sched_quantum = self._build_scheduler(self.opt_quantum, q_opt_cfg.get('scheduler')) if self.opt_quantum else None

        # 3. Loss function
        self.criterion = get_loss_function(config['training'])

        es_cfg = config['training']['early_stopping']
        self.es_enabled = es_cfg['enabled']
        self.monitors = es_cfg['monitors']
        self.primary_metric = es_cfg['primary_monitor']
        self.primary_metric_mode = es_cfg['monitors'][self.primary_metric]
        if self.primary_metric_mode == 'ignore':
            log_error(f"Primary metric {self.primary_metric_mode} can't be ignored", stage="METRICS")
        log_info(f"Primary metric: {self.primary_metric} with mode {self.primary_metric_mode}", stage="METRICS")
        self.best_scores = {self.primary_metric: float('-inf') if self.primary_metric_mode == 'max' else float('inf')}
        if self.es_enabled:
            self.es_patience = es_cfg['patience']
            self.early_stop = False

            for k, v in self.monitors.items():
                if v == 'max':
                    self.best_scores[k] = float('-inf')
                elif v == 'min':
                    self.best_scores[k] = float('inf')
                elif v == 'ignore':
                    pass
                else:
                    log_error(f"Unknown mode {v} on metrics {k} monitoring", stage="METRICS")
            log_info(f"Early stopping enabled with patience {self.es_patience}", stage="METRICS")
            log_info(f"Monitors: {self.monitors}", stage="METRICS")
        else:
            log_info("Early stopping disabled", stage="TRAINER")

    def _build_scheduler(self, optimizer, sched_cfg):
        if not sched_cfg:
            return None
        # Например: ReduceLROnPlateau или CosineAnnealingLR
        sched_cls = getattr(torch.optim.lr_scheduler, sched_cfg['type'])
        params = sched_cfg.get('params', {}) or {}
        valid_params = Utils.filter_kwargs(sched_cls.__init__, params)
        invalid_params = [key for key in params if key not in valid_params]
        if invalid_params:
            log_warn(
                f"Ignoring unsupported params for {sched_cfg['type']}: {invalid_params}",
                stage="SCHEDULER"
            )
        return sched_cls(optimizer, **valid_params)

    @staticmethod
    def _collect_tensor_stats(params):
        total_num = 0
        total_sum = 0.0
        total_sum_sq = 0.0
        min_val = float('inf')
        max_val = float('-inf')

        for p in params:
            if p is None:
                continue
            tensor = p.detach()
            if tensor.numel() == 0:
                continue
            flat = tensor.view(-1)
            total_num += flat.numel()
            total_sum += float(flat.sum().item())
            total_sum_sq += float(flat.pow(2).sum().item())
            min_val = min(min_val, float(flat.min().item()))
            max_val = max(max_val, float(flat.max().item()))

        if total_num == 0:
            return None
        mean = total_sum / total_num
        variance = max(total_sum_sq / total_num - mean ** 2, 0.0)
        std = variance ** 0.5
        return {
            'count': total_num,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
        }

    @staticmethod
    def _optimizer_lrs(optimizer):
        if optimizer is None:
            return None
        return [float(param_group.get('lr', 0.0)) for param_group in optimizer.param_groups]

    @staticmethod
    def _tensor_min_max_repr(tensor) -> str:
        if tensor is None or not hasattr(tensor, "numel") or tensor.numel() == 0:
            return "na"
        tensor = tensor.detach()
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() == 0:
            return "no_finite_values"
        return f"{float(finite.min().item()):.6f}..{float(finite.max().item()):.6f}"

    @staticmethod
    def _batch_summary(batch) -> str:
        try:
            _, _, _, complex_data, targets = batch
            z = complex_data.x[:, 0] if hasattr(complex_data, 'x') and complex_data.x is not None else None
            pos = complex_data.pos if hasattr(complex_data, 'pos') and complex_data.pos is not None else None
            pdb_id = getattr(complex_data, 'pdb_id', None)
            if isinstance(pdb_id, (list, tuple)):
                pdb_repr = ",".join(str(x) for x in pdb_id[:4])
                if len(pdb_id) > 4:
                    pdb_repr += ",..."
            else:
                pdb_repr = str(pdb_id) if pdb_id is not None else 'na'

            min_pair_dist = 'na'
            if pos is not None and pos.numel() and pos.size(0) > 1:
                pos_cpu = pos.detach().float().cpu()
                dist = torch.cdist(pos_cpu, pos_cpu, p=2)
                dist.fill_diagonal_(float('inf'))
                min_pair_dist = float(dist.min().item())

            return (
                f"pdb_id={pdb_repr} "
                f"targets_shape={tuple(targets.shape)} "
                f"targets_finite={bool(torch.isfinite(targets).all().item())} "
                f"num_nodes={int(complex_data.num_nodes) if hasattr(complex_data, 'num_nodes') else 'na'} "
                f"z_min={int(z.min().item()) if z is not None and z.numel() else 'na'} "
                f"z_max={int(z.max().item()) if z is not None and z.numel() else 'na'} "
                f"pos_finite={bool(torch.isfinite(pos).all().item()) if pos is not None and pos.numel() else 'na'} "
                f"min_pair_dist={min_pair_dist}"
            )
        except Exception as exc:
            return f"batch_summary_error={exc}"

    def _maybe_log_large_loss(self, loss_value: float, preds, targets, batch, batch_idx: int) -> None:
        threshold = float(self.train_cfg.get("large_loss_threshold", 1e3))
        if not math.isfinite(loss_value) or loss_value <= threshold:
            return

        self.large_loss_events_in_epoch += 1
        log_warn(
            "Large finite loss encountered in "
            f"train batch {batch_idx}: loss={loss_value:.6f} "
            f"pred_range={self._tensor_min_max_repr(preds)} "
            f"target_range={self._tensor_min_max_repr(targets)} "
            f"{self._batch_summary(batch)}",
            stage="TRAINER"
        )

    def _log_epoch_start(self, epoch_id: int, total_epochs: int, progress: float) -> None:
        classic_lr = self._optimizer_lrs(self.opt_classic)
        quantum_lr = self._optimizer_lrs(self.opt_quantum)

        q_keys = ["qlayer", "final_layer"]
        q_params = [p for n, p in self.model.named_parameters() if any(k in n for k in q_keys)]
        q_param_count = len(q_params)
        q_stats = self._collect_tensor_stats(q_params)

        log_info(
            f"[EPOCH {epoch_id}/{total_epochs}] progress={progress:.3f} classic_lr={classic_lr}"
            f" quantum_lr={quantum_lr} ",
            stage="TRAINER"
        )
        if q_stats is not None:
            log_debug(
                f"[EPOCH {epoch_id}/{total_epochs}] quantum_params={sum(p.numel() for p in q_params)}"
                f" in {q_param_count} tensors mean={q_stats['mean']:.6f} "
                f"std={q_stats['std']:.6f} min={q_stats['min']:.6f} max={q_stats['max']:.6f}",
                stage="TRAINER"
            )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, rem = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes)}m {rem:.1f}s"
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours)}h {int(minutes)}m {rem:.1f}s"

    @staticmethod
    def _format_bytes(num_bytes) -> str:
        if num_bytes is None:
            return "na"
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(num_bytes)
        unit_idx = 0
        while value >= 1024.0 and unit_idx < len(units) - 1:
            value /= 1024.0
            unit_idx += 1
        return f"{value:.1f}{units[unit_idx]}"

    @staticmethod
    def _read_proc_status_value_bytes(key: str):
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith(key + ":"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except (FileNotFoundError, OSError, ValueError):
            return None
        return None

    @staticmethod
    def _read_meminfo_value_bytes(key: str):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith(key + ":"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except (FileNotFoundError, OSError, ValueError):
            return None
        return None

    def _memory_log_every_n_batches(self) -> int:
        return int(self.train_cfg.get("memory_log_every_n_batches", 250))

    def _log_memory_snapshot(self, label: str, batch_idx: int | None = None, total_batches: int | None = None) -> None:
        process_rss = self._read_proc_status_value_bytes("VmRSS")
        process_hwm = self._read_proc_status_value_bytes("VmHWM")
        system_available = self._read_meminfo_value_bytes("MemAvailable")
        system_total = self._read_meminfo_value_bytes("MemTotal")

        parts = [label]
        if batch_idx is not None and total_batches is not None:
            parts.append(f"batch={batch_idx}/{total_batches}")
        parts.extend([
            f"rss={self._format_bytes(process_rss)}",
            f"rss_peak={self._format_bytes(process_hwm)}",
            f"sys_avail={self._format_bytes(system_available)}",
            f"sys_total={self._format_bytes(system_total)}",
        ])

        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_allocated = torch.cuda.max_memory_allocated()
                max_reserved = torch.cuda.max_memory_reserved()
                parts.extend([
                    f"cuda_alloc={self._format_bytes(allocated)}",
                    f"cuda_reserved={self._format_bytes(reserved)}",
                    f"cuda_max_alloc={self._format_bytes(max_allocated)}",
                    f"cuda_max_reserved={self._format_bytes(max_reserved)}",
                ])
            except RuntimeError as exc:
                parts.append(f"cuda_mem_error={exc}")

        log_debug(" | ".join(parts), stage="MEMORY")

    def _build_checkpoint_payload(self):
        if hasattr(self.model, "build_checkpoint_payload"):
            return self.model.build_checkpoint_payload()
        return self.model.state_dict()

    def _load_checkpoint_payload(self, payload) -> None:
        if hasattr(self.model, "load_checkpoint_payload"):
            self.model.load_checkpoint_payload(payload)
            return
        self.model.load_state_dict(payload)

    def step_schedulers(self, metrics):
        """Обновление шага обучения"""
        if self.sched_classic:
            # ReduceLROnPlateau требует метрику (loss)
            if isinstance(self.sched_classic, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.sched_classic.step(metrics)
            else:
                self.sched_classic.step()
        
        if self.sched_quantum and self.opt_quantum:
            self.sched_quantum.step()

    def train_epoch(self, loader, progress: float = 0.0) -> float:
        """
        Train for one epoch.

        Args:
            loader: DataLoader for training data.
            progress: Schedule progress in [0, 1] for the quantum encoder.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0
        self.large_loss_events_in_epoch = 0
        memory_log_every_n_batches = self._memory_log_every_n_batches()
        total_batches = len(loader) if hasattr(loader, "__len__") else None
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._log_memory_snapshot("train_epoch_start", batch_idx=0, total_batches=total_batches)

        pbar = tqdm(loader, desc="Training", unit="batch", leave=True)
        
        for i, batch in enumerate(pbar):
            # Transfer data to device (remember tuple structure)
            batch = [b.to(self.device) if hasattr(b, 'to') else b for b in batch]
            _, _, _, _, targets = batch
            
            self.opt_classic.zero_grad()
            if self.opt_quantum:
                self.opt_quantum.zero_grad()
            
            # Forward pass
            preds = self.model(batch, progress=progress).view(-1)
            if not torch.isfinite(preds).all():
                raise RuntimeError(
                    f"Non-finite predictions encountered in train batch {i}. "
                    f"{self._batch_summary(batch)}"
                )
            if not torch.isfinite(targets).all():
                raise RuntimeError(
                    f"Non-finite targets encountered in train batch {i}. "
                    f"{self._batch_summary(batch)}"
                )
            loss = self.criterion(preds, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss encountered in train batch {i}. "
                    f"{self._batch_summary(batch)}"
                )
            self._maybe_log_large_loss(float(loss.item()), preds, targets, batch, i)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Step both optimizers
            self.opt_classic.step()
            if self.opt_quantum:
                self.opt_quantum.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            avg_loss = epoch_loss / (i + 1)
            
            # Рендерим полоску и обновляем прогресс-бар
            l_bar = Utils.get_loss_bar(current_loss)
            pbar.set_postfix_str(f"Loss: {current_loss:.4f} {l_bar} Avg: {avg_loss:.4f}")

            if memory_log_every_n_batches > 0 and ((i + 1) % memory_log_every_n_batches == 0):
                self._log_memory_snapshot(
                    "train_epoch_progress",
                    batch_idx=i + 1,
                    total_batches=total_batches,
                )
            
        self._log_memory_snapshot("train_epoch_end", batch_idx=total_batches, total_batches=total_batches)
        return avg_loss

    def validate(self, loader, progress: float = 1.0):
        """
        Validate the model on validation set.

        Args:
            loader: DataLoader for validation data.
            progress: Schedule progress in [0, 1] for the quantum encoder.

        Returns:
            Tuple of validation statistics:
            (avg_loss, rmse, pearson_r, ci, predictions, targets).
        """
        return self.evaluator.evaluate_with_loss(loader, self.criterion, progress=progress)

    def train(self, train_loader, val_loader, exp_dir, save_only_best_epoch=True) -> Tuple[int, float]:
        """
        Run the complete training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
        """
        plot_every_n_epochs = self.train_cfg.get('plot_every_n_epochs', 10)
        # best_val_r = -1.0
        best_epoch = 0
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_pearson': [], 
            'val_ci': [], 'best_y_true': None, 'best_y_pred': None
            }
        if self.es_enabled:
            self.es_counter = 0
        
        total_number_of_epochs = self.train_cfg['epochs']
        for epoch in range(total_number_of_epochs):
            epoch_started_at = time.perf_counter()
            epoch_id = epoch + 1
            progress = epoch / max(1, total_number_of_epochs - 1)
            self._log_epoch_start(epoch_id, total_number_of_epochs, progress)
            self._log_memory_snapshot(f"epoch_{epoch_id}_start")
            log_info(f"[EPOCH {epoch_id}/{total_number_of_epochs}] Preparing training epoch.", stage="TRAINER")
            train_started_at = time.perf_counter()
            train_loss = self.train_epoch(train_loader, progress=progress)
            train_duration = time.perf_counter() - train_started_at
            self._log_memory_snapshot(f"epoch_{epoch_id}_after_train")
            log_debug(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] Training pass completed in "
                f"{self._format_duration(train_duration)}",
                stage="TRAINER"
            )
            log_info(f"[EPOCH {epoch_id}/{total_number_of_epochs}] Validation.", stage="TRAINER")
            val_started_at = time.perf_counter()
            val_loss, _, r_val, ci_val, preds, targets = self.validate(val_loader, progress=progress)
            val_duration = time.perf_counter() - val_started_at
            epoch_duration = time.perf_counter() - epoch_started_at
            self._log_memory_snapshot(f"epoch_{epoch_id}_after_val")

            log_info(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}",
                stage="TRAINER"
            )
            log_debug(f"[EPOCH {epoch_id}/{total_number_of_epochs}] "
                      f"Train_time={self._format_duration(train_duration)} "
                      f"| Val_time={self._format_duration(val_duration)} "
                      f"| epoch_time={self._format_duration(epoch_duration)}",
                      stage="TRAINER")
            if self.large_loss_events_in_epoch:
                log_warn(
                    f"[EPOCH {epoch_id}/{total_number_of_epochs}] large finite loss batches: "
                    f"{self.large_loss_events_in_epoch}",
                    stage="TRAINER"
                )

            if self.config['dataset']['stats'] is not None:
                stats = self.config['dataset']['stats']
            else:
                raise ValueError("Data are denormalized.")
            log_debug(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] Preparing denormalized validation metrics.",
                stage="TRAINER"
            )
            metric_post_started_at = time.perf_counter()
            preds_denorm = Utils.denormalize(preds, stats)
            targets_denorm = Utils.denormalize(targets, stats)
            rmse_denorm = Utils.calculate_rmse(targets_denorm, preds_denorm)
            log_debug(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] Denormalized validation metrics prepared in "
                f"{self._format_duration(time.perf_counter() - metric_post_started_at)}",
                stage="TRAINER"
            )
            log_debug(f"{type(train_loss)}, {type(rmse_denorm)}, {type(r_val)}, {type(ci_val)}", stage="TRAINER")
            log_debug(f"{type(preds_denorm)}, {type(targets_denorm)}, {type(preds_denorm[0])}, {type(targets_denorm[0])}", stage="TRAINER")
            log_debug(f"{train_loss}, {rmse_denorm}, {r_val}, {ci_val}", stage="TRAINER")
            log_debug(f"{preds_denorm}, {targets_denorm}, {preds_denorm[0]}, {targets_denorm[0]}", stage="TRAINER")
            self.history['train_loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_rmse'].append(float(rmse_denorm))
            self.history['val_pearson'].append(float(r_val))
            self.history['val_ci'].append(float(ci_val))
            log_info(f"[EPOCH {epoch_id}/{total_number_of_epochs}] Valid: RMSE {rmse_denorm:.4f} | R {r_val:.4f} | CI {ci_val:.4f}", stage="TRAINER")

            current_metrics = {
                "val_pearson": r_val,
                "val_rmse": rmse_denorm,
                "train_loss": train_loss,
                "val_ci": ci_val
            }

            improved_any = False
            improved_primary = False

            log_debug(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] Checking monitored metrics for improvements.",
                stage="TRAINER"
            )
            monitor_started_at = time.perf_counter()
            for metric, mode in self.monitors.items():
                if mode == 'ignore': continue
                val = current_metrics.get(metric)
                if val is None: continue
                if isinstance(val, float) and not math.isfinite(val):
                    continue
                if (mode == 'max' and val > self.best_scores[metric]) or \
                    (mode == 'min' and val < self.best_scores[metric]):

                    self.best_scores[metric] = val
                    improved_any = True
                    log_info(f"Improved: "
                             f"{'Primary' if self.primary_metric == metric else 'Secondary'} metric"
                    f" '{metric}': {val:.4f}", stage="TRAINER")

                    if metric == self.primary_metric:
                        improved_primary = True
            log_debug(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] Metric improvement check completed in "
                f"{self._format_duration(time.perf_counter() - monitor_started_at)}",
                stage="TRAINER"
            )

            if improved_primary:
                best_epoch = epoch_id
                log_debug(
                    f"[EPOCH {epoch_id}/{total_number_of_epochs}] Saving best model.",
                    stage="TRAINER"
                )
                best_save_started_at = time.perf_counter()
                torch.save(self._build_checkpoint_payload(), f"{exp_dir}/best_model.pt")
                self.history['best_y_true'] = targets_denorm.tolist()
                self.history['best_y_pred'] = preds_denorm.tolist()
                log_debug(
                    f"[EPOCH {epoch_id}/{total_number_of_epochs}] Best model checkpoint saved in "
                    f"{self._format_duration(time.perf_counter() - best_save_started_at)}",
                    stage="TRAINER"
                )
                log_debug(f"New best for primary metric {self.primary_metric}:"
                f" {self.best_scores[self.primary_metric]:.4f} (Saved to best_model.pt)",
                stage="TRAINER")
            if self.es_enabled:
                log_debug(
                    f"[EPOCH {epoch_id}/{total_number_of_epochs}] Updating early stopping state.",
                    stage="TRAINER"
                )
                if improved_any:
                    self.es_counter = 0
                else:
                    self.es_counter += 1
                    log_info(f"EarlyStopping counter: {self.es_counter}/{self.es_patience}",
                                stage="TRAINER")
                    if self.es_counter >= self.es_patience:
                        log_info(f"[EPOCH {epoch_id}/{total_number_of_epochs}]"
                        f" Early stopping triggered", stage="TRAINER")
                        self.early_stop = True

            for k in ['train_loss', 'val_loss', 'val_rmse', 'val_pearson', 'val_ci']:
                data = self.history[k]
                log_debug(f"Key: {k}, Length: {len(data)}, Types: {[type(x) for x in data]}", stage="DEBUG_PLOT")

            log_debug(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] Saving history.json.",
                stage="TRAINER"
            )
            history_save_started_at = time.perf_counter()
            with open(f"{exp_dir}/history.json", 'w') as f:
                json.dump(self.history, f, indent=4)
            log_debug(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] history.json saved in "
                f"{self._format_duration(time.perf_counter() - history_save_started_at)}",
                stage="TRAINER"
            )

            if not save_only_best_epoch:
                log_debug(
                    f"[EPOCH {epoch_id}/{total_number_of_epochs}] Saving per-epoch checkpoint.",
                    stage="TRAINER"
                )
                epoch_save_started_at = time.perf_counter()
                torch.save(self._build_checkpoint_payload(), f"{exp_dir}/model_epoch_{epoch_id}.pt")
                log_debug(
                    f"[EPOCH {epoch_id}/{total_number_of_epochs}] Per-epoch checkpoint saved in "
                    f"{self._format_duration(time.perf_counter() - epoch_save_started_at)}",
                    stage="TRAINER"
                )

            if self.early_stop:
                break

            log_debug(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] Updating schedulers.",
                stage="TRAINER"
            )
            scheduler_started_at = time.perf_counter()
            self.step_schedulers(val_loss)
            log_debug(
                f"[EPOCH {epoch_id}/{total_number_of_epochs}] Schedulers updated in "
                f"{self._format_duration(time.perf_counter() - scheduler_started_at)}",
                stage="TRAINER"
            )

            # if it is the last epoch, the runner anyway will draw the results
            if (epoch_id % plot_every_n_epochs == 0) and (epoch_id != total_number_of_epochs):
                log_debug(
                    f"[EPOCH {epoch_id}/{total_number_of_epochs}] Rendering ASCII dashboard.",
                    stage="TRAINER"
                )
                plot_started_at = time.perf_counter()
                console_plots(self.history, side_by_side=True, stage="TRAINER")
                log_debug(
                    f"[EPOCH {epoch_id}/{total_number_of_epochs}] ASCII dashboard rendered in "
                    f"{self._format_duration(time.perf_counter() - plot_started_at)}",
                    stage="TRAINER"
                )
            log_info(get_divider("-"), stage="TRAINER")

        log_info(
            f"Training completed. Best {self.primary_metric}: "
            f"{self.best_scores[self.primary_metric]:.4f} at epoch {best_epoch}",
            stage="TRAINER"
        )
        return best_epoch, self.best_scores[self.primary_metric]

    def test(self, test_loader, exp_dir, best_epoch, show_plots=False, save_plots=True):
        if hasattr(self, 'history'):
            log_debug("Preparing final performance report plot.", stage="TEST")
            plot_started_at = time.perf_counter()
            self.evaluator.plot_history(exp_dir, self.history, show=show_plots, save=save_plots)
            log_debug(
                f"Final performance report plot completed in "
                f"{self._format_duration(time.perf_counter() - plot_started_at)}",
                stage="TEST"
            )
        else:
            log_debug("No training history available for plotting.", stage="TEST")

        self._log_memory_snapshot("test_start")
        # Подгружаем веса лучшей эпохи (в идеале нужно написать логику загрузки лучшего .pt,
        # но пока протестируем на весах последней эпохи)

        best_model_path = f"{exp_dir}/best_model.pt"
        if os.path.exists(best_model_path):
            log_debug("Loading best checkpoint for final test.", stage="TEST")
            load_started_at = time.perf_counter()
            checkpoint_payload = torch.load(best_model_path)
            self._load_checkpoint_payload(checkpoint_payload)
            log_debug(
                f"Weights for the best model (epoch {best_epoch}) loaded from {best_model_path} "
                f"in {self._format_duration(time.perf_counter() - load_started_at)}",
                stage="TEST"
            )

        log_debug("Preparing final test metric evaluation.", stage="TEST")
        eval_started_at = time.perf_counter()
        # test_rmse, test_r_val, test_ci_val, test_preds, test_targets
        _, test_r, test_ci, test_preds, test_targets = self.evaluator.evaluate(test_loader, progress=1.0)

        if self.config['dataset']['stats'] is not None:
            stats = self.config['dataset']['stats']
        else:
            raise ValueError("Data are denormalized.")

        test_preds_denorm = Utils.denormalize(test_preds, stats)
        test_targets_denorm = Utils.denormalize(test_targets, stats)
        test_rmse_denorm = Utils.calculate_rmse(test_targets_denorm, test_preds_denorm)
        self._log_memory_snapshot("test_after_eval")
        log_info(
            f"FINAL TEST -> RMSE: {test_rmse_denorm:.4f} | Pearson R: {test_r:.4f} | CI: {test_ci:.4f} "
            f"| eval_time={self._format_duration(time.perf_counter() - eval_started_at)}",
            stage="TEST"
        )
        
        # Сохраняем результаты теста
        log_debug("Saving final test results.", stage="TEST")
        test_results_started_at = time.perf_counter()
        with open(f"{exp_dir}/test_results.json", 'w') as f:
            json.dump({
                "RMSE": test_rmse_denorm,
                "Pearson_R": test_r,
                "CI": test_ci
            }, f, indent=4)
        log_debug(
            f"Final test results saved in "
            f"{self._format_duration(time.perf_counter() - test_results_started_at)}",
            stage="TEST"
        )
