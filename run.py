# run.py

import json
import os
import csv
import socket
import subprocess
import platform
import pandas as pd
import gc
import argparse
import tempfile
import shutil
import time
import random
import torch
import numpy as np
from datetime import datetime
from importlib import metadata as importlib_metadata
from torch_geometric.loader import DataLoader

from logger import *
from extractor import PDBBindOrchestrator
from trainer import Trainer
from evaluator import Evaluator
from models.a1 import A1DimeNet
from models.a2 import A2DimeNet
from models.a3 import A3DimeNet
from models.graph_components import (
    get_a3_mixer_bias,
    get_global_encoder_config,
    get_global_graph_config,
    get_global_graph_mode,
    get_head_config,
    get_head_mode,
    get_local_encoder_config,
    get_local_encoder_mode,
    get_local_graph_config,
    get_local_graph_mode,
    get_model_family,
)
from parsers.interaction_graph_parser import InteractionGraphParser
from tokenizer import UniversalPDBBindDataset
from splitter import PDBBindSplitter
from utils import Utils
from models.protein_context import FrozenESMEncoder, ProteinContextConfig, get_protein_context_mode
from models.ligand_context import FrozenLigandDescriptorEncoder, LigandContextConfig, get_ligand_context_mode


DATASETS_DIR = "datasets"
PROTEIN_CONTEXT_FEATURES_DIR = "protein_context_features"
LIGAND_CONTEXT_FEATURES_DIR = "ligand_context_features"


class ExperimentRunner:
    """
    Orchestrate dataset preparation, context precompute, training, and testing.

    `ExperimentRunner` is the project-level entry point. It binds together the
    parser/orchestrator layer, split logic, context caches, model construction,
    trainer lifecycle, and experiment-folder management. The intent is to keep
    the notebook and shell entrypoints extremely thin while preserving one
    explicit place where the end-to-end experiment contract is documented.
    """

    def __init__(self, config_path: str, extract: bool = False,
                 train_dataset_path: None | str = None,
                 test_dataset_path: None |str = None,
                 val_dataset_path: None | str = None,
                 temp_run: bool = False,
                 keep_temp: bool = False,
                 exp_dir: None | str = None,
                 core_as_test: None | bool = None,
                 a3_mixer_bias: None | bool = None,
                 splitter_seed: None | int = None,
                 auto_summary: bool = True):
        self.config_path = os.path.abspath(config_path or 'config.json')
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log_info(f"Starting experiment: {self.config['experiment_name']}", stage="EXPERIMENT")
        self.extract = extract
        self.save_train_test_val_datasets = self.config['dataset']['save_train_test_val_datasets']
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.val_dataset_path = val_dataset_path
        self.temp_run = temp_run
        self.keep_temp = keep_temp
        self.exp_dir = exp_dir
        self.run_started_at = datetime.now()
        self.source_subset = self.config['dataset'].get('source_subset', 'refined')
        self.model_family = get_model_family(self.config)
        if self.source_subset not in {'refined', 'general'}:
            raise ValueError("source_subset must be either 'refined' or 'general'")

        configured_core_as_test = self.config['dataset'].get('core_as_test', True)
        self.core_as_test = configured_core_as_test if core_as_test is None else core_as_test
        self.a3_mixer_bias = a3_mixer_bias
        self.splitter_seed = splitter_seed
        self.auto_summary = auto_summary
        if self.a3_mixer_bias is not None and self.model_family != "A3":
            raise ValueError(
                "--a3-mixer-bias / --no-a3-mixer-bias can only be used when "
                "model.selected == 'A3'."
            )
        self.test_frac = self.config['dataset'].get('test_frac', 0.15)
        if not self.core_as_test and not 0.0 < float(self.test_frac) < 1.0:
            raise ValueError("test_frac must be between 0 and 1")
        if self.splitter_seed is not None:
            strategy = self.config['splitter']['selected']
            self.config['splitter']['available'][strategy]['seed'] = int(self.splitter_seed)
        self.config['dataset']['source_subset'] = self.source_subset
        self.config['dataset']['core_as_test'] = self.core_as_test
        self.config['dataset']['test_frac'] = self.test_frac
        assert all([self.train_dataset_path, self.test_dataset_path, self.val_dataset_path]) or not any([self.train_dataset_path, self.test_dataset_path, self.val_dataset_path])
        self.experiment_signature = None
        self.best_epoch = None
        self.epochs_completed = None
        self.git_commit = self._resolve_git_commit()
        self.execution_env = self._detect_execution_env()
        self.hostname = socket.gethostname()
        self.runtime_fingerprint = self._collect_runtime_fingerprint()

    def prepare_folders(self):
        """
        Create runtime folders for datasets, caches, and run artifacts.

        This method is called before any expensive work begins so that logs and
        generated outputs have a stable destination even if the run fails
        midway through parsing or training.
        """
        log_info(get_stage_banner("INITIALIZING"), stage="EXPERIMENT")
        log_info(get_divider("="), stage="EXPERIMENT")
        if self.exp_dir is not None:
            self.exp_run_dir = self.exp_dir
        elif self.temp_run:
            os.makedirs('runs', exist_ok=True)
            self.exp_run_dir = tempfile.mkdtemp(prefix=f"{self.config['experiment_name']}_tmp_", dir="runs")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            exp_name = f"{self.config['experiment_name']}_{timestamp}"
            self.experiment_signature = exp_name
            log_info(f"Experiment signature: {exp_name}", stage="EXPERIMENT")
            self.exp_run_dir = f"runs/{exp_name}"
        if self.experiment_signature is None:
            self.experiment_signature = os.path.basename(os.path.normpath(self.exp_run_dir))

        os.makedirs(DATASETS_DIR, exist_ok=True)
        log_info(f"Base Datasets folder: {DATASETS_DIR}", stage="EXPERIMENT")

        os.makedirs(PROTEIN_CONTEXT_FEATURES_DIR, exist_ok=True)
        log_info(
            f"Base protein context features folder: {PROTEIN_CONTEXT_FEATURES_DIR}",
            stage="EXPERIMENT"
        )
        os.makedirs(LIGAND_CONTEXT_FEATURES_DIR, exist_ok=True)
        log_info(
            f"Base ligand context features folder: {LIGAND_CONTEXT_FEATURES_DIR}",
            stage="EXPERIMENT"
        )

        os.makedirs(self.exp_run_dir, exist_ok=True)
        log_info(f"Run results folder: {self.exp_run_dir}", stage="EXPERIMENT")

        if self.save_train_test_val_datasets:
            self.exp_run_datasets_dir = f"{self.exp_run_dir}/datasets"
            os.makedirs(self.exp_run_datasets_dir, exist_ok=True)
            log_info(f"Experiment datasets path: {self.exp_run_datasets_dir}", stage="EXPERIMENT")

        log_path = os.path.join(self.exp_run_dir, "run.log")
        setup_file_logging(log_path)
        log_info(f"Log file: {log_path}", stage="EXPERIMENT")
        log_info(
            f"Run started at: {self.run_started_at.isoformat(timespec='seconds')}",
            stage="EXPERIMENT"
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
    def _describe_model_family(model_family: str) -> str:
        descriptions = {
            "A1": "coarse global model only; one fused interaction graph plus optional context branches",
            "A2": "global coarse branch plus an explicit local geometric correction branch",
            "A3": "A2 branch encoders with an explicit linear coarse-plus-local readout",
        }
        return descriptions.get(model_family, "unknown model family")

    def _log_model_overview(
        self,
        model_family: str,
        global_graph_mode: str,
        local_graph_mode: str,
        protein_context_mode: str,
        ligand_context_mode: str,
    ) -> None:
        log_info(
            f"Model selected -> {model_family}: {self._describe_model_family(model_family)}",
            stage="MODEL"
        )

        local_active = model_family in {"A2", "A3"} and local_graph_mode != "none"
        log_info(
            f"Model branch summary -> global_graph={global_graph_mode}, "
            f"local_branch={'enabled' if local_active else 'disabled'}, "
            f"protein_context={protein_context_mode}, ligand_context={ligand_context_mode}",
            stage="MODEL"
        )

        if model_family == "A3":
            mixer_bias = get_a3_mixer_bias(self.config, self.a3_mixer_bias)
            log_info(
                f"A3 readout summary -> mixer_bias={'enabled' if mixer_bias else 'disabled'} "
                f"({'gamma is trainable' if mixer_bias else 'gamma absent'})",
                stage="MODEL"
            )

    def _set_global_random_state(self) -> None:
        """
        Seed Python, NumPy, and PyTorch from the effective splitter seed.

        This keeps the user-facing `--rseed` override meaningful beyond the
        dataframe split itself while preserving the existing config-driven
        default when no override is passed.
        """
        strategy = self.config['splitter']['selected']
        seed = int(self.config['splitter']['available'][strategy].get('seed', 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_git_commit() -> str:
        """
        Return the current Git commit hash when available.

        The experiment registry uses this as a compact code-version fingerprint.
        If Git is unavailable or the repo state cannot be resolved, an empty
        string is stored instead of failing the run.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    @staticmethod
    def _detect_execution_env() -> str:
        """
        Return a compact label for the current runtime.

        The value is intentionally coarse; it is meant for experiment
        bookkeeping rather than environment introspection.
        """
        forced = os.environ.get("DUMPLING_EXECUTION_ENV", "").strip()
        if forced:
            return forced
        if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
            return "colab"
        if os.environ.get("SLURM_JOB_ID"):
            return "cluster"
        return "local"

    def _build_registry_row(
        self,
        *,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        test_metrics: dict | None = None,
    ) -> dict[str, object]:
        strategy = self.config['splitter']['selected']
        splitter_seed = int(self.config['splitter']['available'][strategy].get('seed', 42))
        batch_run_index = os.environ.get("DUMPLING_BATCH_RUN_INDEX", "")
        batch_n_times = os.environ.get("DUMPLING_BATCH_N_TIMES", "")
        a3_mixer_bias = ""
        if self.model_family == "A3":
            a3_mixer_bias = get_a3_mixer_bias(self.config, self.a3_mixer_bias)

        row = {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_sec": round((finished_at - started_at).total_seconds(), 1),
            "status": status,
            "experiment_name": self.config["experiment_name"],
            "experiment_signature": self.experiment_signature,
            "exp_dir": self.exp_run_dir,
            "config_path": self.config_path,
            "git_commit": self.git_commit,
            "execution_env": self.execution_env,
            "hostname": self.hostname,
            "artifact_root": os.path.abspath(os.path.dirname(self.exp_run_dir)),
            "model_family": self.model_family,
            "source_subset": self.source_subset,
            "splitter": strategy,
            "splitter_seed": splitter_seed,
            "core_as_test": self.core_as_test,
            "a3_mixer_bias": a3_mixer_bias,
            "batch_run_index": batch_run_index,
            "batch_n_times": batch_n_times,
            "device": self.device,
            "primary_metric": self.config["training"]["early_stopping"]["primary_monitor"],
            "best_epoch": self.best_epoch if self.best_epoch is not None else "",
            "epochs_completed": self.epochs_completed if self.epochs_completed is not None else "",
            "test_rmse": "",
            "test_pearson": "",
            "test_ci": "",
        }
        if test_metrics:
            row["test_rmse"] = test_metrics.get("RMSE", "")
            row["test_pearson"] = test_metrics.get("Pearson_R", "")
            row["test_ci"] = test_metrics.get("CI", "")
        return row

    def _write_run_manifest(
        self,
        *,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        test_metrics: dict | None = None,
    ) -> None:
        payload = {
            "registry_row": self._build_registry_row(
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                test_metrics=test_metrics,
            ),
            "history_metrics": self._load_history(),
            "test_metrics": test_metrics,
            "runtime_fingerprint": self.runtime_fingerprint,
            "artifacts": {
                "exp_dir": os.path.abspath(self.exp_run_dir),
                "config_path": self.config_path,
                "assistant_summary_path": os.path.join(self.exp_run_dir, "assistant_summary.md"),
                "journal_path": os.path.abspath(os.path.join("runs", "experiment_journal.md")),
                "registry_path": os.path.abspath(os.path.join("runs", "experiment_registry.csv")),
            },
        }
        manifest_path = os.path.join(self.exp_run_dir, "run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_test_metrics(exp_dir: str) -> dict | None:
        test_results_path = os.path.join(exp_dir, "test_results.json")
        if not os.path.exists(test_results_path):
            return None
        with open(test_results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_history(self) -> dict | None:
        history_path = os.path.join(self.exp_run_dir, "history.json")
        if not os.path.exists(history_path):
            return None
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _safe_package_version(package_name: str) -> str:
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            return ""
        except Exception:
            return ""

    def _collect_runtime_fingerprint(self) -> dict[str, object]:
        gpu: dict[str, object] = {
            "available": bool(torch.cuda.is_available()),
            "count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                gpu.update(
                    {
                        "name": torch.cuda.get_device_name(0),
                        "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                        "cuda_runtime": str(torch.version.cuda or ""),
                    }
                )
            except Exception:
                pass

        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count() or "",
            "libraries": {
                "torch": getattr(torch, "__version__", ""),
                "torch_geometric": self._safe_package_version("torch-geometric"),
                "rdkit": self._safe_package_version("rdkit"),
                "fair_esm": self._safe_package_version("fair-esm"),
                "pandas": self._safe_package_version("pandas"),
                "numpy": getattr(np, "__version__", ""),
            },
            "hardware": {
                "device": self.device,
                "hostname": self.hostname,
                "gpu": gpu,
            },
        }

    @staticmethod
    def _last_metric(history: dict | None, key: str) -> float | None:
        if not history:
            return None
        values = history.get(key, [])
        if not values:
            return None
        try:
            return float(values[-1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_factual_note_parts(
        self,
        *,
        history: dict | None,
        test_metrics: dict | None,
    ) -> list[str]:
        strategy = self.config['splitter']['selected']
        splitter_seed = int(self.config['splitter']['available'][strategy].get('seed', 42))
        note_parts: list[str] = [
            f"model=`{self.model_family}`",
            f"execution_env=`{self.execution_env}`",
            f"splitter=`{strategy}`",
            f"splitter_seed=`{splitter_seed}`",
        ]

        protein_context = str(self.config.get("model", {}).get("protein_context", {}).get("selected", ""))
        ligand_context = str(self.config.get("model", {}).get("ligand_context", {}).get("selected", ""))
        if protein_context:
            note_parts.append(f"protein_context=`{protein_context}`")
        if ligand_context:
            note_parts.append(f"ligand_context=`{ligand_context}`")

        if self.best_epoch is not None:
            note_parts.append(f"best_epoch=`{self.best_epoch}`")
        if self.epochs_completed is not None:
            note_parts.append(f"epochs_completed=`{self.epochs_completed}`")

        if test_metrics:
            if test_metrics.get("RMSE", None) != "":
                note_parts.append(f"test_RMSE=`{test_metrics.get('RMSE', 'n/a')}`")
            if test_metrics.get("Pearson_R", None) != "":
                note_parts.append(f"test_Pearson_R=`{test_metrics.get('Pearson_R', 'n/a')}`")
            if test_metrics.get("CI", None) != "":
                note_parts.append(f"test_CI=`{test_metrics.get('CI', 'n/a')}`")

        readout = (test_metrics or {}).get("readout_diagnostics", {})
        if self.model_family == "A3" and readout:
            note_parts.append(f"mixer_has_bias=`{readout.get('mixer_has_bias', 'n/a')}`")
            for key in ("alpha", "beta", "gamma", "local_to_global_abs_contribution_ratio"):
                value = readout.get(key, "n/a")
                note_parts.append(f"{key}=`{value}`")

        return note_parts

    def _build_assistant_summary_lines(
        self,
        *,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        test_metrics: dict | None,
        history: dict | None,
    ) -> list[str]:
        duration_sec = round((finished_at - started_at).total_seconds(), 1)
        strategy = self.config['splitter']['selected']
        splitter_seed = int(self.config['splitter']['available'][strategy].get('seed', 42))
        primary_metric = self.config["training"]["early_stopping"]["primary_monitor"]

        lines = [
            f"# Assistant Summary: {self.experiment_signature}",
            "",
            "## Run Snapshot",
            f"- status: `{status}`",
            f"- experiment_name: `{self.config['experiment_name']}`",
            f"- experiment_signature: `{self.experiment_signature}`",
            f"- started_at: `{started_at.isoformat(timespec='seconds')}`",
            f"- finished_at: `{finished_at.isoformat(timespec='seconds')}`",
            f"- duration_sec: `{duration_sec}`",
            f"- model_family: `{self.model_family}`",
            f"- execution_env: `{self.execution_env}`",
            f"- hostname: `{self.hostname}`",
            f"- artifact_root: `{os.path.abspath(os.path.dirname(self.exp_run_dir))}`",
            f"- exp_dir: `{os.path.abspath(self.exp_run_dir)}`",
            f"- config_path: `{self.config_path}`",
            f"- git_commit: `{self.git_commit or 'unknown'}`",
            f"- splitter: `{strategy}`",
            f"- splitter_seed: `{splitter_seed}`",
            f"- primary_metric: `{primary_metric}`",
            f"- best_epoch: `{self.best_epoch if self.best_epoch is not None else 'n/a'}`",
            f"- epochs_completed: `{self.epochs_completed if self.epochs_completed is not None else 'n/a'}`",
            "",
        ]

        libs = self.runtime_fingerprint.get("libraries", {})
        hardware = self.runtime_fingerprint.get("hardware", {})
        gpu = hardware.get("gpu", {}) if isinstance(hardware, dict) else {}
        lines.extend([
            "## Runtime Fingerprint",
            f"- python_version: `{self.runtime_fingerprint.get('python_version', 'n/a')}`",
            f"- platform: `{self.runtime_fingerprint.get('platform', 'n/a')}`",
            f"- cpu_count: `{self.runtime_fingerprint.get('cpu_count', 'n/a')}`",
            f"- torch: `{libs.get('torch', 'n/a')}` | torch_geometric: `{libs.get('torch_geometric', 'n/a')}` | "
            f"rdkit: `{libs.get('rdkit', 'n/a')}` | fair_esm: `{libs.get('fair_esm', 'n/a')}` | "
            f"pandas: `{libs.get('pandas', 'n/a')}` | numpy: `{libs.get('numpy', 'n/a')}`",
            f"- device: `{hardware.get('device', 'n/a')}` | gpu_available: `{gpu.get('available', 'n/a')}` | "
            f"gpu_count: `{gpu.get('count', 'n/a')}` | gpu_name: `{gpu.get('name', 'n/a')}` | "
            f"gpu_total_memory_gb: `{gpu.get('total_memory_gb', 'n/a')}` | cuda_runtime: `{gpu.get('cuda_runtime', 'n/a')}`",
            "",
        ])

        if test_metrics:
            lines.extend([
                "## Final Metrics",
                f"- RMSE: `{test_metrics.get('RMSE', 'n/a')}`",
                f"- Pearson_R: `{test_metrics.get('Pearson_R', 'n/a')}`",
                f"- CI: `{test_metrics.get('CI', 'n/a')}`",
                "",
            ])

        readout = (test_metrics or {}).get("readout_diagnostics", {})
        if self.model_family == "A3" and readout:
            ratio = self._safe_float(readout.get("local_to_global_abs_contribution_ratio"))
            alpha = self._safe_float(readout.get("alpha"))
            beta = self._safe_float(readout.get("beta"))
            gamma = self._safe_float(readout.get("gamma"))
            global_mean = self._safe_float((readout.get("global_contribution") or {}).get("mean"))
            local_mean = self._safe_float((readout.get("local_contribution") or {}).get("mean"))
            lines.extend([
                "## A3 Readout Snapshot",
                f"- mixer_has_bias: `{readout.get('mixer_has_bias', 'n/a')}`",
                f"- alpha: `{alpha if alpha is not None else 'n/a'}`",
                f"- beta: `{beta if beta is not None else 'n/a'}`",
                f"- gamma: `{gamma if gamma is not None else 'n/a'}`",
                f"- local_to_global_abs_contribution_ratio: `{ratio if ratio is not None else 'n/a'}`",
                f"- global_contribution_mean: `{global_mean if global_mean is not None else 'n/a'}`",
                f"- local_contribution_mean: `{local_mean if local_mean is not None else 'n/a'}`",
                "",
            ])

        lines.append("## Heuristic Notes")
        note_parts = self._build_factual_note_parts(history=history, test_metrics=test_metrics)
        if status != "success":
            note_parts.append(f"status=`{status}`")
            note_parts.append(f"run_err_log=`{os.path.join(self.exp_run_dir, 'run_err.log')}`")
        if note_parts:
            lines.extend([f"- {note}" for note in note_parts])
        else:
            lines.append("- No compact factual note was generated for this run.")

        lines.extend([
            "",
            "## Caveat",
            "- This file is an automatic assistant-style summary. It is meant to speed up review, not replace manual judgment.",
            "",
        ])
        return lines

    def _append_experiment_journal_entry(
        self,
        *,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        test_metrics: dict | None = None,
    ) -> None:
        if not self.auto_summary:
            return

        os.makedirs("runs", exist_ok=True)
        journal_path = os.path.join("runs", "experiment_journal.md")
        history = self._load_history()
        strategy = self.config['splitter']['selected']
        splitter_seed = int(self.config['splitter']['available'][strategy].get('seed', 42))
        duration_sec = round((finished_at - started_at).total_seconds(), 1)

        headline = (
            f"## {finished_at.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{self.experiment_signature}"
        )
        lines = [headline, ""]
        lines.append(
            f"- status: `{status}` | model: `{self.model_family}` | env: `{self.execution_env}` | "
            f"seed: `{splitter_seed}` | duration_sec: `{duration_sec}`"
        )
        lines.append(
            f"- location: `{os.path.abspath(self.exp_run_dir)}` on `{self.hostname}`"
        )

        if test_metrics:
            lines.append(
                f"- final metrics: RMSE=`{test_metrics.get('RMSE', 'n/a')}`, "
                f"Pearson_R=`{test_metrics.get('Pearson_R', 'n/a')}`, "
                f"CI=`{test_metrics.get('CI', 'n/a')}`"
            )

        note_parts = self._build_factual_note_parts(history=history, test_metrics=test_metrics)
        if status != "success":
            note_parts.append(f"status=`{status}`")

        if note_parts:
            lines.append(f"- assistant note: {' '.join(note_parts)}")
        else:
            lines.append("- assistant note: no compact factual snapshot.")

        lines.extend([
            "",
            "> Auto-generated assistant journal note. Handy for scanning history; not a substitute for manual interpretation.",
            "",
        ])

        file_exists = os.path.exists(journal_path)
        with open(journal_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("# Experiment Journal\n\n")
                f.write(
                    "This file is an automatically accumulated run log. "
                    "Treat it as a convenience layer over the raw artifacts and the terse CSV registry.\n\n"
                )
            f.write("\n".join(lines))

    def _write_assistant_summary(
        self,
        *,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        test_metrics: dict | None = None,
    ) -> None:
        if not self.auto_summary:
            return
        history = self._load_history()
        lines = self._build_assistant_summary_lines(
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            test_metrics=test_metrics,
            history=history,
        )
        summary_path = os.path.join(self.exp_run_dir, "assistant_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _append_experiment_registry_row(
        self,
        *,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        test_metrics: dict | None = None,
    ) -> None:
        os.makedirs("runs", exist_ok=True)
        registry_path = os.path.join("runs", "experiment_registry.csv")
        row = self._build_registry_row(
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            test_metrics=test_metrics,
        )
        fieldnames = list(row.keys())
        existing_rows: list[dict[str, object]] = []
        file_exists = os.path.exists(registry_path)
        if file_exists:
            with open(registry_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                existing_rows = list(reader)
            if existing_fieldnames != fieldnames:
                with open(registry_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for old_row in existing_rows:
                        normalized = {name: old_row.get(name, "") for name in fieldnames}
                        writer.writerow(normalized)
                existing_rows = []

        with open(registry_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _get_dataset_from_path(self, path: str):
        """
        Load a dataframe from one of the supported on-disk formats.

        Args:
            path: Path to a `.csv`, `.pickle`/`.pkl`, or `.parquet` file.

        Returns:
            A pandas dataframe loaded from the requested file.
        """
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith(('.pkl', '.pickle')):
            return pd.read_pickle(path)
        elif path.endswith('.parquet'):
            return pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file format. Use .csv, .pkl, or .parquet")

    def prepare_datasets(self):
        """
        Build or load the train/validation/test dataframes for a run.

        This method supports two modes:

        - use externally provided split dataframes from disk,
        - rebuild the source dataset from the PDBBind archive and derive splits
          from the configured strategy.

        Returns:
            Tuple of `(train_df, val_df, test_df)` with normalization stats
            already attached to the configuration.
        """
        log_info(get_stage_banner("DATASET"), stage="EXPERIMENT")
        log_info(get_divider("="), stage="EXPERIMENT")
        prepare_started_at = time.perf_counter()
        if self.train_dataset_path is not None:
            log_info("Preparing custom train/test/val datasets.", stage="EXPERIMENT")

            self.config["dataset"].update({
                "train_path": self.train_dataset_path,
                "test_path": self.test_dataset_path,
                "val_path": self.val_dataset_path,
                })
            train_df = self._get_dataset_from_path(self.train_dataset_path)
            test_df = self._get_dataset_from_path(self.test_dataset_path)
            val_df = self._get_dataset_from_path(self.val_dataset_path)
            log_debug(
                f"Custom train/test/val datasets loaded in "
                f"{self._format_duration(time.perf_counter() - prepare_started_at)}",
                stage="EXPERIMENT"
            )

        else:
            global_graph_cfg = get_global_graph_config(self.config)
            parsers = [
                None,
                InteractionGraphParser(
                    dist_threshold=float(global_graph_cfg.get('dist_threshold', 5.0)),
                    ca_only=bool(global_graph_cfg.get('ca_only', False)),
                    local_chemical_features=self.config.get("model", {}).get("local_chemical_features", {}),
                ),
            ]

            orchestrator = PDBBindOrchestrator(parsers, self.config)
            if self.extract:
                log_info(f"Preparing extraction for subset '{self.source_subset}'.", stage="EXPERIMENT")
                extract_started_at = time.perf_counter()
                orchestrator.extract_subset(self.source_subset)
                log_debug(
                    f"Extraction for subset '{self.source_subset}' completed in "
                    f"{self._format_duration(time.perf_counter() - extract_started_at)}",
                    stage="EXPERIMENT"
                )
            log_info(f"Preparing dataset build for subset '{self.source_subset}'.", stage="EXPERIMENT")
            build_started_at = time.perf_counter()
            df_source = orchestrator.build_dataset(subset=self.source_subset, fmt="pickle", save_dir=DATASETS_DIR)
            log_debug(
                f"Dataset build for subset '{self.source_subset}' completed in "
                f"{self._format_duration(time.perf_counter() - build_started_at)}",
                stage="EXPERIMENT"
            )
            core_ids = set(orchestrator.get_complex_ids("core").keys()) if self.core_as_test else None
            log_info("Preparing train/val/test split.", stage="SPLIT")
            split_started_at = time.perf_counter()
            train_df, val_df, test_df, test_file_name = PDBBindSplitter.split_with_test(
                df_source,
                self.config["splitter"],
                core_as_test=self.core_as_test,
                test_frac=self.test_frac,
                core_ids=core_ids,
                source_subset=self.source_subset
            )
            log_debug(
                f"Train/val/test split completed in "
                f"{self._format_duration(time.perf_counter() - split_started_at)}",
                stage="SPLIT"
            )

            if self.save_train_test_val_datasets:
                log_info("Saving per-run train/val/test dataset snapshots.", stage="SAVE")
                save_started_at = time.perf_counter()
                train_path = f"{self.exp_run_datasets_dir}/train.pickle"
                val_path   = f"{self.exp_run_datasets_dir}/val.pickle"
                test_path  = f"{self.exp_run_datasets_dir}/{test_file_name}"

                train_df.to_pickle(train_path)
                test_df.to_pickle(test_path)
                val_df.to_pickle(val_path)

                self.config["dataset"].update({
                    "train_path": train_path,
                    "test_path": test_path,
                    "val_path": val_path,
                    })
                log_info(
                    f"Per-run dataset snapshots saved in "
                    f"{self._format_duration(time.perf_counter() - save_started_at)}",
                    stage="SAVE"
                )
        log_info(
            f"Prepared splits | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}",
            stage="EXPERIMENT"
        )
        # Normalization
        log_info("Preparing data normalization.", stage="NORMALIZATION")
        norm_started_at = time.perf_counter()
        train_target = train_df['pkd'].values
        stats = {'mean': float(train_target.mean()), 'std': float(train_target.std())}
        self.config['dataset']['stats'] = stats
        for df in [train_df, val_df, test_df]:
            df['pkd'] = Utils.normalize(df['pkd'].values, stats)
        log_debug(
            f"Data normalization completed in {self._format_duration(time.perf_counter() - norm_started_at)}",
            stage="NORMALIZATION"
        )
        log_info(f"Data Stats: {stats}", stage="NORMALIZATION")
        log_info(f"min value: {train_df['pkd'].min()}, max value: {train_df['pkd'].max()}", stage="NORMALIZATION")

        # Run a lightweight sanity check after dataset orchestration completes.
        sample = train_df.iloc[0]
        complex_dict = sample['complex_graph']

        log_debug("Performing quick data checks on the first training sample.", stage="CHECK")
        log_debug(f"Complex ID: {sample['pdb_id']}", stage="CHECK")
        log_debug(f"Graph atoms: {len(complex_dict['x'])}", stage="CHECK")
        log_debug(f"Feature dimension: {len(complex_dict['x'][0])}", stage="CHECK")
        log_debug(f"Coordinates present (pos): {'pos' in complex_dict}", stage="CHECK")
        if 'orchestrator' in locals() and orchestrator.bad_complexes_registry:
            log_info(
                f"Data quality registry active: {orchestrator.bad_complexes_path} "
                f"({len(orchestrator.bad_complexes_registry)} entries)",
                stage="DATA_QUALITY"
            )
        log_info(
            f"Dataset preparation finished in "
            f"{self._format_duration(time.perf_counter() - prepare_started_at)}",
            stage="EXPERIMENT"
        )

        return train_df, val_df, test_df

    def prewarm_protein_context(self) -> None:
        """
        Warm up the frozen ESM encoder weights before the main training phase.

        The goal is not to compute any real embeddings here, but to pull model
        weights into the runtime once so later context precompute and forward
        passes do not pay the first-load penalty at an awkward moment.
        """
        protein_context_cfg = ProteinContextConfig.from_config(self.config)
        if protein_context_cfg.mode == "none":
            return

        log_info(
            f"Preparing protein context prewarm: mode={protein_context_cfg.mode}, "
            f"model={protein_context_cfg.model_name}, repr_layer={protein_context_cfg.repr_layer}, "
            f"pooling={protein_context_cfg.pooling}, cache_path={protein_context_cfg.cache_path}",
            stage="PROTEIN_CONTEXT"
        )
        started_at = time.perf_counter()

        encoder = FrozenESMEncoder(
            model_name=protein_context_cfg.model_name,
            repr_layer=protein_context_cfg.repr_layer,
            pooling=protein_context_cfg.pooling,
            device=self.device,
            cache_path=protein_context_cfg.cache_path,
            max_length=protein_context_cfg.max_length,
        )
        del encoder
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        log_debug(
            f"Protein context encoder weights are warmed up in "
            f"{self._format_duration(time.perf_counter() - started_at)}.",
            stage="PROTEIN_CONTEXT"
        )

    @staticmethod
    def _extract_split_sequences(df: pd.DataFrame) -> list[str]:
        if 'protein' not in df.columns:
            return []
        sequences: list[str] = []
        for value in df['protein'].tolist():
            if value is None or pd.isna(value):
                continue
            sequences.append(str(value))
        return sequences

    @staticmethod
    def _extract_split_ligand_smiles(df: pd.DataFrame) -> list[str]:
        source_col = 'ligand_smiles' if 'ligand_smiles' in df.columns else 'ligand'
        if source_col not in df.columns:
            return []
        smiles_list: list[str] = []
        for value in df[source_col].tolist():
            if value is None or pd.isna(value) or not isinstance(value, str):
                continue
            smiles_list.append(str(value))
        return smiles_list

    def precompute_protein_context_embeddings(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """
        Precompute and cache protein-context vectors for all active splits.

        Args:
            train_df: Training dataframe.
            val_df: Validation dataframe.
            test_df: Test dataframe.
        """
        protein_context_cfg = ProteinContextConfig.from_config(self.config)
        if protein_context_cfg.mode == "none":
            return

        started_at = time.perf_counter()
        batch_size = int(protein_context_cfg.precompute_batch_size)
        log_info(
            f"Preparing protein context embedding precompute: mode={protein_context_cfg.mode}, "
            f"model={protein_context_cfg.model_name}, batch_size={batch_size}, "
            f"cache_path={protein_context_cfg.cache_path}",
            stage="PROTEIN_CONTEXT"
        )

        encoder = FrozenESMEncoder(
            model_name=protein_context_cfg.model_name,
            repr_layer=protein_context_cfg.repr_layer,
            pooling=protein_context_cfg.pooling,
            device=self.device,
            cache_path=protein_context_cfg.cache_path,
            max_length=protein_context_cfg.max_length,
        )

        split_frames = [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]

        total_unique = 0
        total_cached_before = 0
        total_computed_now = 0

        for split_name, split_df in split_frames:
            split_sequences = self._extract_split_sequences(split_df)
            log_debug(
                f"Preparing protein context precompute for split '{split_name}' "
                f"with {len(split_sequences)} sequences.",
                stage="PROTEIN_CONTEXT"
            )
            split_started_at = time.perf_counter()
            stats = encoder.precompute_sequences(
                split_sequences,
                batch_size=batch_size,
                progress_desc=f"ESM precompute ({split_name})",
            )
            total_unique += stats["total_unique"]
            total_cached_before += stats["cached_before"]
            total_computed_now += stats["computed_now"]
            log_debug(
                f"Protein context precompute for split '{split_name}' completed in "
                f"{self._format_duration(time.perf_counter() - split_started_at)} "
                f"(unique={stats['total_unique']}, cached_before={stats['cached_before']}, "
                f"computed_now={stats['computed_now']})",
                stage="PROTEIN_CONTEXT"
            )

        del encoder
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        log_debug(
            f"Protein context embedding precompute completed in "
            f"{self._format_duration(time.perf_counter() - started_at)} "
            f"(split_unique_total={total_unique}, cached_before_total={total_cached_before}, "
            f"computed_now_total={total_computed_now})",
            stage="PROTEIN_CONTEXT"
        )

    def precompute_ligand_context_embeddings(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """
        Precompute and cache ligand-context descriptor vectors for all splits.

        Args:
            train_df: Training dataframe.
            val_df: Validation dataframe.
            test_df: Test dataframe.
        """
        ligand_context_cfg = LigandContextConfig.from_config(self.config)
        if ligand_context_cfg.mode == "none":
            return

        started_at = time.perf_counter()
        batch_size = int(ligand_context_cfg.precompute_batch_size)
        log_info(
            f"Preparing ligand context embedding precompute: mode={ligand_context_cfg.mode}, "
            f"descriptor_set={ligand_context_cfg.descriptor_set}, batch_size={batch_size}, "
            f"cache_path={ligand_context_cfg.cache_path}",
            stage="LIGAND_CONTEXT"
        )

        encoder = FrozenLigandDescriptorEncoder(
            cache_path=ligand_context_cfg.cache_path,
            descriptor_set=ligand_context_cfg.descriptor_set,
            device=self.device,
        )

        split_frames = [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]

        total_unique = 0
        total_cached_before = 0
        total_computed_now = 0

        for split_name, split_df in split_frames:
            split_smiles = self._extract_split_ligand_smiles(split_df)
            log_debug(
                f"Preparing ligand context precompute for split '{split_name}' "
                f"with {len(split_smiles)} ligands.",
                stage="LIGAND_CONTEXT"
            )
            split_started_at = time.perf_counter()
            stats = encoder.precompute_smiles(
                split_smiles,
                batch_size=batch_size,
                progress_desc=f"Ligand context precompute ({split_name})",
            )
            total_unique += stats["total_unique"]
            total_cached_before += stats["cached_before"]
            total_computed_now += stats["computed_now"]
            log_debug(
                f"Ligand context precompute for split '{split_name}' completed in "
                f"{self._format_duration(time.perf_counter() - split_started_at)} "
                f"(unique={stats['total_unique']}, cached_before={stats['cached_before']}, "
                f"computed_now={stats['computed_now']})",
                stage="LIGAND_CONTEXT"
            )

        del encoder
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        log_debug(
            f"Ligand context embedding precompute completed in "
            f"{self._format_duration(time.perf_counter() - started_at)} "
            f"(split_unique_total={total_unique}, cached_before_total={total_cached_before}, "
            f"computed_now_total={total_computed_now})",
            stage="LIGAND_CONTEXT"
        )

    def run(self, train_df, val_df, test_df):
        """
        Execute the full experiment after dataframes have been prepared.

        Args:
            train_df: Training dataframe.
            val_df: Validation dataframe.
            test_df: Test dataframe.
        """
        self._set_global_random_state()
        run_started_at = time.perf_counter()
        log_info(get_stage_banner("ENRICHMENT"), stage="EXPERIMENT")
        log_info(get_divider("="), stage="EXPERIMENT")
        log_info("Preparing training datasets and loaders.", stage="EXPERIMENT")
        self.precompute_protein_context_embeddings(train_df, val_df, test_df)
        self.precompute_ligand_context_embeddings(train_df, val_df, test_df)
        train_ds = UniversalPDBBindDataset(train_df, self.config)
        test_ds = UniversalPDBBindDataset(test_df, self.config)
        val_ds   = UniversalPDBBindDataset(val_df, self.config)
        gc.collect()

        batch_size = self.config['dataset']['batch_size']
        num_workers = int(self.config['dataset'].get('num_workers', 0))
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": self.device == 'cuda',
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

        self.config['dataset']['actual_sizes'] = {
            'train': len(train_ds),
            'val': len(val_ds),
            'test': len(test_ds)
        }

        with open(f"{self.exp_run_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

        model_family = self.model_family
        if model_family not in {"A1", "A2", "A3"}:
            raise ValueError(f"Unsupported model.selected={model_family!r}. Expected 'A1', 'A2', or 'A3'.")
        global_graph_mode = get_global_graph_mode(self.config)
        global_graph_cfg = get_global_graph_config(self.config)
        global_encoder_cfg = get_global_encoder_config(self.config)
        local_encoder_cfg = get_local_encoder_config(self.config)
        local_graph_mode = get_local_graph_mode(self.config)
        local_graph_cfg = get_local_graph_config(self.config)
        local_chemical_cfg = self.config.get("model", {}).get("local_chemical_features", {})
        hidden_dim = global_encoder_cfg.hidden_channels
        protein_context_mode = get_protein_context_mode(self.config)
        ligand_context_mode = get_ligand_context_mode(self.config)
        protein_context_cfg = ProteinContextConfig.from_config(self.config)
        ligand_context_cfg = LigandContextConfig.from_config(self.config)
        self.prewarm_protein_context()
        log_info(
            f"Protein context settings -> mode={protein_context_cfg.mode}, model={protein_context_cfg.model_name}, "
            f"repr_layer={protein_context_cfg.repr_layer}, pooling={protein_context_cfg.pooling}, "
            f"cache_path={protein_context_cfg.cache_path}, max_length={protein_context_cfg.max_length}",
            stage="PROTEIN_CONTEXT"
        )
        log_info(
            f"Ligand context settings -> mode={ligand_context_cfg.mode}, descriptor_set={ligand_context_cfg.descriptor_set}, "
            f"cache_path={ligand_context_cfg.cache_path}, embedding_dim={ligand_context_cfg.embedding_dim}",
            stage="LIGAND_CONTEXT"
        )
        splitter_seed = self.config['splitter']['available'][self.config['splitter']['selected']].get('seed', 'na')
        log_info(get_stage_banner("RUN SETTINGS"), stage="EXPERIMENT")
        log_info(get_divider("="), stage="EXPERIMENT")
        log_info(
            f"Run settings -> source_subset={self.source_subset}, core_as_test={self.core_as_test}, "
            f"splitter={self.config['splitter']['selected']}, batch_size={self.config['dataset']['batch_size']}, "
            f"num_workers={num_workers}, splitter_seed={splitter_seed}, model={model_family}, "
            f"epochs={self.config['training']['epochs']}",
            stage="EXPERIMENT"
        )
        short_description = str(self.config.get("short_description", "")).strip()
        if short_description:
            log_info(
                f"Experiment description -> {short_description}",
                stage="EXPERIMENT"
            )
        self._log_model_overview(
            model_family=model_family,
            global_graph_mode=global_graph_mode,
            local_graph_mode=local_graph_mode,
            protein_context_mode=protein_context_mode,
            ligand_context_mode=ligand_context_mode,
        )
        log_info(
            f"Global graph settings -> mode={global_graph_mode}, "
            f"dist_threshold={global_graph_cfg.get('dist_threshold', 'na')}, "
            f"ca_only={global_graph_cfg.get('ca_only', 'na')}",
            stage="MODEL"
        )
        log_info(
            f"Global encoder settings -> hidden_channels={hidden_dim}, cutoff={global_encoder_cfg.cutoff}, "
            f"max_num_neighbors={global_encoder_cfg.max_num_neighbors}, "
            f"num_blocks={global_encoder_cfg.num_blocks}, "
            f"protein_context={protein_context_mode}, ligand_context={ligand_context_mode}",
            stage="MODEL"
        )
        head_mode = get_head_mode(self.config)
        head_cfg = get_head_config(self.config)
        if model_family == "A2":
            if head_mode == "vqc":
                adapter_hidden_layers = head_cfg.get("adapter_hidden_layers")
                if adapter_hidden_layers is None and "pre_hidden_dim" in head_cfg:
                    adapter_hidden_layers = [int(head_cfg["pre_hidden_dim"])]
                log_info(
                    "A2 readout settings -> "
                    f"head=vqc, adapter_hidden_layers={adapter_hidden_layers or [128, 64]}, "
                    f"adapter_activation={head_cfg.get('adapter_activation', 'Tanh')}, "
                    f"n_qubits={head_cfg.get('n_qubits', 6)}, n_layers={head_cfg.get('n_layers', 2)}, "
                    f"backend={head_cfg.get('backend', 'default.qubit')}, "
                    f"rotation={head_cfg.get('rotation', 'X')}, "
                    f"initial_rotation={head_cfg.get('initial_rotation', 'Y')}, "
                    f"input_scale={head_cfg.get('input_scale', 0.01)}, "
                    f"angle_schedule=[{head_cfg.get('start_scale', torch.pi / 6):.4f}, "
                    f"{head_cfg.get('end_scale', torch.pi):.4f}], "
                    f"readout_hidden_dim={head_cfg.get('readout_hidden_dim', 16)}",
                    stage="MODEL"
                )
            else:
                log_info(
                    f"A2 readout settings -> head=mlp, hidden_dim={head_cfg.get('hidden_dim', max(hidden_dim // 2, 1))}",
                    stage="MODEL"
                )
        elif head_mode != "mlp":
            log_info(
                f"Additional head configuration present but currently unused by model={model_family}: "
                f"head={head_mode}",
                stage="MODEL"
            )
        if model_family in {"A2", "A3"}:
            local_encoder_mode = get_local_encoder_mode(self.config)
            if local_encoder_cfg is None or local_graph_mode == "none":
                log_info(
                    f"Local branch settings -> local_graph={local_graph_mode}, local_encoder={local_encoder_mode}",
                    stage="MODEL"
                )
            else:
                log_info(
                    f"Local branch settings -> local_graph={local_graph_mode} "
                    f"(dist_threshold={local_graph_cfg.get('dist_threshold', 'na')}), "
                    f"local_encoder={local_encoder_mode}, hidden_channels={local_encoder_cfg.hidden_channels}, "
                    f"pooling_mode={getattr(local_encoder_cfg, 'pooling_mode', 'na')}, "
                    f"cutoff={local_encoder_cfg.cutoff}, max_num_neighbors={local_encoder_cfg.max_num_neighbors}, "
                    f"num_blocks={local_encoder_cfg.num_blocks}",
                    stage="MODEL"
                )
                if bool(local_chemical_cfg.get("enabled", False)):
                    raw_features = local_chemical_cfg.get("features", {})
                    if isinstance(raw_features, dict):
                        enabled_feature_names = sorted(
                            key for key, enabled in raw_features.items() if bool(enabled)
                        )
                    else:
                        enabled_feature_names = []
                    log_info(
                        "Local chemical enrichment -> enabled=True, "
                        f"features={enabled_feature_names if enabled_feature_names else '<none>'}",
                        stage="MODEL"
                    )
                else:
                    log_info(
                        "Local chemical enrichment -> enabled=False",
                        stage="MODEL"
                    )
        if model_family == "A3":
            log_info(
                f"A3 readout settings -> mixer_bias={get_a3_mixer_bias(self.config, self.a3_mixer_bias)}",
                stage="MODEL"
            )
        elif model_family not in {"A2", "A3"} and (
            local_graph_mode != "none" or get_local_encoder_mode(self.config) != "none"
        ):
            log_info(
                f"Local branch configuration present but ignored because model={model_family}: "
                f"local_graph={local_graph_mode}, local_encoder={get_local_encoder_mode(self.config)}",
                stage="MODEL"
            )
        classic_opt_cfg = self.config['training']['optimizers']['classic']
        log_info(
            f"Optimizer settings -> type={classic_opt_cfg['type']}, "
            f"lr={classic_opt_cfg['params'].get('lr')}, "
            f"weight_decay={classic_opt_cfg['params'].get('weight_decay', 0.0)}",
            stage="OPTIMIZER"
        )
        if model_family == "A2" and head_mode == "vqc":
            quantum_opt_cfg = self.config['training']['optimizers']['quantum']
            quantum_scheduler = quantum_opt_cfg.get('scheduler')
            log_info(
                f"Quantum optimizer settings -> type={quantum_opt_cfg['type']}, "
                f"lr={quantum_opt_cfg['params'].get('lr')}, "
                f"weight_decay={quantum_opt_cfg['params'].get('weight_decay', 0.0)}, "
                f"scheduler={quantum_scheduler['type'] if quantum_scheduler else 'none'}",
                stage="OPTIMIZER"
            )
        log_info("Preparing model initialization.", stage="MODEL")
        model_init_started_at = time.perf_counter()
        if model_family == "A3":
            model = A3DimeNet(
                config=self.config,
                device=self.device,
                out_channels=1,
                mixer_bias=self.a3_mixer_bias,
            )
        elif model_family == "A2":
            model = A2DimeNet(
                config=self.config,
                device=self.device,
                out_channels=1,
            )
        else:
            model = A1DimeNet(
                config=self.config,
                device=self.device,
                out_channels=1,
            )
        log_debug(
            f"Model initialization completed in "
            f"{self._format_duration(time.perf_counter() - model_init_started_at)}",
            stage="MODEL"
        )
        evaluator = Evaluator(model, self.device)

        log_info(f"Launch on: {self.device}", stage="EXPERIMENT")

        self.trainer = Trainer(model, evaluator, self.config, self.device)
        train_started_at = time.perf_counter()
        log_info(get_stage_banner("TRAINING"), stage="EXPERIMENT")
        log_info(get_divider("="), stage="EXPERIMENT")
        best_epoch, _ = self.trainer.train(train_loader, val_loader, self.exp_run_dir, self.config['training']['save_only_best_epoch'])
        self.best_epoch = int(best_epoch)
        if hasattr(self.trainer, "history"):
            self.epochs_completed = len(self.trainer.history.get("train_loss", []))
        log_debug(
            f"Training loop completed in {self._format_duration(time.perf_counter() - train_started_at)}",
            stage="EXPERIMENT"
        )
        test_started_at = time.perf_counter()
        log_info(get_stage_banner("TESTING"), stage="EXPERIMENT")
        log_info(get_divider("="), stage="EXPERIMENT")
        self.trainer.test(test_loader, self.exp_run_dir, best_epoch)
        log_debug(
            f"Final test evaluation completed in "
            f"{self._format_duration(time.perf_counter() - test_started_at)}",
            stage="TEST"
        )
        if hasattr(model, "log_local_guard_summary"):
            model.log_local_guard_summary()

        log_info("Generating ASCII performance summary.", stage="SUMMARY")
        console_plots(self.trainer.history, side_by_side=False, stage="SUMMARY")
        console_plots(self.trainer.history, side_by_side=True, stage="SUMMARY")
        log_info(
            f"Experiment completed successfully in "
            f"{self._format_duration(time.perf_counter() - run_started_at)}.",
            stage="EXPERIMENT"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDBBind Experiment")

    parser.add_argument('--config', type=str, default='config.json', 
                        help='Path to the configuration JSON file')
    
    # Extraction flag: if present on the CLI, extraction is forced before build.
    parser.add_argument('--extract', action='store_true', 
                        help='Extract subset before building dataset')
    
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--temp-run', action='store_true', help='Use a temporary experiment directory and remove it after successful completion')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary directory even after successful temp-run')
    parser.add_argument('--exp-dir', type=str, default=None, help='Use a fixed experiment directory instead of timestamped runs/<name>_<ts>')
    parser.add_argument('--core-as-test', action=argparse.BooleanOptionalAction, default=None,
                        help='Use PDBBind core as test set. Use --no-core-as-test to split the configured source subset into train/val/test.')
    parser.add_argument('--a3-mixer-bias', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable the explicit gamma term in the A3 readout mixer. Use --no-a3-mixer-bias for bias-free A3 ablations.')
    parser.add_argument('--rseed', '--splitter-seed', dest='splitter_seed', type=int, default=None,
                        help='Override the configured splitter seed from the CLI without editing config.json.')
    parser.add_argument('--auto-summary', action=argparse.BooleanOptionalAction, default=True,
                        help='Write assistant_summary.md after each run. Use --no-auto-summary to disable the per-run automatic note.')

    args = parser.parse_args()
    runner = ExperimentRunner(
        config_path=args.config,
        extract=args.extract,
        train_dataset_path=args.train_path,
        test_dataset_path=args.test_path,
        val_dataset_path=args.val_path,
        temp_run=args.temp_run,
        keep_temp=args.keep_temp,
        exp_dir=args.exp_dir,
        core_as_test=args.core_as_test,
        a3_mixer_bias=args.a3_mixer_bias,
        splitter_seed=args.splitter_seed,
        auto_summary=args.auto_summary,
    )
    runner.prepare_folders()
    train_df, val_df, test_df = runner.prepare_datasets()

    run_started_at = datetime.now()
    try:
        runner.run(train_df, val_df, test_df)
        run_finished_at = datetime.now()
        test_metrics = runner._load_test_metrics(runner.exp_run_dir)
        runner._append_experiment_registry_row(
            status="success",
            started_at=run_started_at,
            finished_at=run_finished_at,
            test_metrics=test_metrics,
        )
        runner._write_assistant_summary(
            status="success",
            started_at=run_started_at,
            finished_at=run_finished_at,
            test_metrics=test_metrics,
        )
        runner._write_run_manifest(
            status="success",
            started_at=run_started_at,
            finished_at=run_finished_at,
            test_metrics=test_metrics,
        )
        runner._append_experiment_journal_entry(
            status="success",
            started_at=run_started_at,
            finished_at=run_finished_at,
            test_metrics=test_metrics,
        )
        if runner.temp_run and not runner.keep_temp:
            shutil.rmtree(runner.exp_run_dir, ignore_errors=True)
    except Exception as e:
        import traceback
        err_path = os.path.join(runner.exp_run_dir, "run_err.log")
        error_msg = traceback.format_exc()
        log_info(f"ERROR message saved to: {err_path}", stage="CRASH")

        with open(err_path, "w", encoding="utf-8") as f:
            f.write(error_msg)

        failed_at = datetime.now()
        test_metrics = runner._load_test_metrics(runner.exp_run_dir)
        runner._append_experiment_registry_row(
            status="failed",
            started_at=run_started_at,
            finished_at=failed_at,
            test_metrics=test_metrics,
        )
        runner._write_assistant_summary(
            status="failed",
            started_at=run_started_at,
            finished_at=failed_at,
            test_metrics=test_metrics,
        )
        runner._write_run_manifest(
            status="failed",
            started_at=run_started_at,
            finished_at=failed_at,
            test_metrics=test_metrics,
        )
        runner._append_experiment_journal_entry(
            status="failed",
            started_at=run_started_at,
            finished_at=failed_at,
            test_metrics=test_metrics,
        )

        raise e

"""
python run.py --config configs/gnn_test.json    # path to the custom config
python run.py --config config.json --extract    # then extract == True

"""
