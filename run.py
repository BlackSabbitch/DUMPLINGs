# run.py

import json
import os
import pandas as pd
import gc
import argparse
import tempfile
import shutil
import time
import torch
from datetime import datetime
from torch_geometric.loader import DataLoader

from logger import *
from extractor import PDBBindOrchestrator
from trainer import Trainer
from evaluator import Evaluator
from models.a1 import A1DimeNet
from parsers.interaction_graph_parser import InteractionGraphParser
from tokenizer import UniversalPDBBindDataset
from splitter import PDBBindSplitter
from parsers.cnn_parser import CNNParser
from parsers.gnn_parser import GNNParser
from utils import Utils
from models.protein_context import FrozenESMEncoder, ProteinContextConfig, get_protein_context_mode


DATASETS_DIR = "datasets"
ESM_CACHE_DIR = "esm_cache"


class ExperimentRunner:
    """
    Orchestrates the full pipeline: data extraction, parsing, splitting,
    training, and evaluation based on a configuration file.
    """

    def __init__(self, config_path: str, extract: bool = False,
                 train_dataset_path: None | str = None,
                 test_dataset_path: None |str = None,
                 val_dataset_path: None | str = None,
                 temp_run: bool = False,
                 keep_temp: bool = False,
                 exp_dir: None | str = None,
                 core_as_test: None | bool = None):
        with open(config_path or 'config.json', 'r') as f:
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
        self.source_subset = self.config['dataset'].get('source_subset', 'refined')
        if self.source_subset not in {'refined', 'general'}:
            raise ValueError("source_subset must be either 'refined' or 'general'")

        configured_core_as_test = self.config['dataset'].get('core_as_test', True)
        self.core_as_test = configured_core_as_test if core_as_test is None else core_as_test
        self.test_frac = self.config['dataset'].get('test_frac', 0.15)
        if not self.core_as_test and not 0.0 < float(self.test_frac) < 1.0:
            raise ValueError("test_frac must be between 0 and 1")
        self.config['dataset']['source_subset'] = self.source_subset
        self.config['dataset']['core_as_test'] = self.core_as_test
        self.config['dataset']['test_frac'] = self.test_frac
        assert all([self.train_dataset_path, self.test_dataset_path, self.val_dataset_path]) or not any([self.train_dataset_path, self.test_dataset_path, self.val_dataset_path])

    def prepare_folders(self):
        if self.exp_dir is not None:
            self.exp_run_dir = self.exp_dir
        elif self.temp_run:
            os.makedirs('runs', exist_ok=True)
            self.exp_run_dir = tempfile.mkdtemp(prefix=f"{self.config['experiment_name']}_tmp_", dir="runs")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{self.config['experiment_name']}_{timestamp}"
            log_info(f"Experiment signature: {exp_name}", stage="EXPERIMENT")
            self.exp_run_dir = f"runs/{exp_name}"

        os.makedirs(DATASETS_DIR, exist_ok=True)
        log_info(f"Base Datasets folder: {DATASETS_DIR}", stage="EXPERIMENT")

        os.makedirs(ESM_CACHE_DIR, exist_ok=True)
        log_info(f"Base ESM cache folder: {ESM_CACHE_DIR}", stage="EXPERIMENT")

        os.makedirs(self.exp_run_dir, exist_ok=True)
        log_info(f"Run results folder: {self.exp_run_dir}", stage="EXPERIMENT")

        if self.save_train_test_val_datasets:
            self.exp_run_datasets_dir = f"{self.exp_run_dir}/datasets"
            os.makedirs(self.exp_run_datasets_dir, exist_ok=True)
            log_info(f"Experiment datasets path: {self.exp_run_datasets_dir}", stage="EXPERIMENT")

        log_path = os.path.join(self.exp_run_dir, "log.txt")
        setup_file_logging(log_path)
        log_info(f"Log file: {log_path}", stage="EXPERIMENT")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, rem = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes)}m {rem:.1f}s"
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours)}h {int(minutes)}m {rem:.1f}s"

    def _get_dataset_from_path(self, path: str):
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith(('.pkl', '.pickle')):
            return pd.read_pickle(path)
        elif path.endswith('.parquet'):
            return pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file format. Use .csv, .pkl, or .parquet")

    def prepare_datasets(self):
        prepare_started_at = time.perf_counter()
        if self.train_dataset_path is not None:
            log_info("Preparing custom train/test/val datasets...", stage="EXPERIMENT")

            self.config["dataset"].update({
                "train_path": self.train_dataset_path,
                "test_path": self.test_dataset_path,
                "val_path": self.val_dataset_path,
                })
            train_df = self._get_dataset_from_path(self.train_dataset_path)
            test_df = self._get_dataset_from_path(self.test_dataset_path)
            val_df = self._get_dataset_from_path(self.val_dataset_path)
            log_info(
                f"Custom train/test/val datasets loaded in "
                f"{self._format_duration(time.perf_counter() - prepare_started_at)}",
                stage="EXPERIMENT"
            )

        else:
            mode = self.config['model']['graph_encoder']['selected']
            available_cfg = self.config['model']['graph_encoder']['available'][mode]
            mode_str = available_cfg['protein_ligand_pocket_encoders']
            parsers = []
            for i, char in enumerate(mode_str):
                is_lig = (i == 1)
                if char == 'C': parsers.append(CNNParser(is_ligand=is_lig))
                elif char == 'G': parsers.append(GNNParser(is_ligand=is_lig))
                elif char == 'E':
                    egnn_params = available_cfg.get('egnn_params', {})
                    parsers.append(InteractionGraphParser(
                        dist_threshold=egnn_params.get('dist_threshold', 5.0),
                        ca_only=egnn_params.get('ca_only', False)
                    ))
                elif char == 'N': parsers.append(None)
                else: log_info(f"Unknown symbol {char} in the architecture description.", stage="EXPERIMENT")

            orchestrator = PDBBindOrchestrator(parsers, self.config)
            if self.extract:
                log_info(f"Preparing extraction for subset '{self.source_subset}'...", stage="EXPERIMENT")
                extract_started_at = time.perf_counter()
                orchestrator.extract_subset(self.source_subset)
                log_info(
                    f"Extraction for subset '{self.source_subset}' completed in "
                    f"{self._format_duration(time.perf_counter() - extract_started_at)}",
                    stage="EXPERIMENT"
                )
            log_info(f"Preparing dataset build for subset '{self.source_subset}'...", stage="EXPERIMENT")
            build_started_at = time.perf_counter()
            df_source = orchestrator.build_dataset(subset=self.source_subset, fmt="pickle", save_dir=DATASETS_DIR)
            log_info(
                f"Dataset build for subset '{self.source_subset}' completed in "
                f"{self._format_duration(time.perf_counter() - build_started_at)}",
                stage="EXPERIMENT"
            )
            core_ids = set(orchestrator.get_complex_ids("core").keys()) if self.core_as_test else None
            log_info("Preparing train/val/test split...", stage="SPLIT")
            split_started_at = time.perf_counter()
            train_df, val_df, test_df, test_file_name = PDBBindSplitter.split_with_test(
                df_source,
                self.config["splitter"],
                core_as_test=self.core_as_test,
                test_frac=self.test_frac,
                core_ids=core_ids,
                source_subset=self.source_subset
            )
            log_info(
                f"Train/val/test split completed in "
                f"{self._format_duration(time.perf_counter() - split_started_at)}",
                stage="SPLIT"
            )

            if self.save_train_test_val_datasets:
                log_info("Saving per-run train/val/test dataset snapshots...", stage="SAVE")
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

        # Normalization
        log_info("Preparing data normalization...", stage="EXPERIMENT")
        norm_started_at = time.perf_counter()
        train_target = train_df['pkd'].values
        stats = {'mean': float(train_target.mean()), 'std': float(train_target.std())}
        self.config['dataset']['stats'] = stats
        for df in [train_df, val_df, test_df]:
            df['pkd'] = Utils.normalize(df['pkd'].values, stats)
        log_info(
            f"Data normalization completed in {self._format_duration(time.perf_counter() - norm_started_at)}",
            stage="EXPERIMENT"
        )
        log_info(f"Data Stats: {stats}", stage="EXPERIMENT")
        log_info(f"min value: {train_df['pkd'].min()}, max value: {train_df['pkd'].max()}", stage="EXPERIMENT")

        # Запуск минимальной проверки после работы оркестратора
        sample = train_df.iloc[0]
        complex_dict = sample['complex_graph']

        log_info(f"ID комплекса: {sample['pdb_id']}", stage="CHECK")
        log_info(f"Атомов в графе: {len(complex_dict['x'])}", stage="CHECK")
        log_info(f"Размерность признаков: {len(complex_dict['x'][0])}", stage="CHECK")
        log_info(f"Наличие координат (pos): {'pos' in complex_dict}", stage="CHECK")
        if 'orchestrator' in locals() and orchestrator.bad_complexes_registry:
            log_info(
                f"Data quality registry active: {orchestrator.bad_complexes_path} "
                f"({len(orchestrator.bad_complexes_registry)} entries)",
                stage="DATA_QUALITY"
            )
        log_info(
            f"Prepared splits -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}",
            stage="EXPERIMENT"
        )
        log_info(
            f"Dataset preparation finished in "
            f"{self._format_duration(time.perf_counter() - prepare_started_at)}",
            stage="EXPERIMENT"
        )

        return train_df, val_df, test_df

    def prewarm_protein_context(self) -> None:
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

        log_info(
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

    def precompute_protein_context_embeddings(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
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
            log_info(
                f"Preparing protein context precompute for split '{split_name}' "
                f"with {len(split_sequences)} sequences...",
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
            log_info(
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

        log_info(
            f"Protein context embedding precompute completed in "
            f"{self._format_duration(time.perf_counter() - started_at)} "
            f"(split_unique_total={total_unique}, cached_before_total={total_cached_before}, "
            f"computed_now_total={total_computed_now})",
            stage="PROTEIN_CONTEXT"
        )

    def run(self, train_df, val_df, test_df):
        run_started_at = time.perf_counter()
        log_info("Preparing training datasets and loaders...", stage="EXPERIMENT")
        self.precompute_protein_context_embeddings(train_df, val_df, test_df)
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
        log_info(
            f"Dataset sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}",
            stage="EXPERIMENT"
        )

        with open(f"{self.exp_run_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

        dimenet_cfg = self.config['model']['graph_encoder']['available']['duo']['egnn_params']
        hidden_dim = dimenet_cfg.get('hidden_channels', 128)
        protein_context_mode = get_protein_context_mode(self.config)
        protein_context_cfg = ProteinContextConfig.from_config(self.config)
        splitter_seed = self.config['splitter']['available'][self.config['splitter']['selected']].get('seed', 'na')
        log_info(
            f"Run settings -> source_subset={self.source_subset}, core_as_test={self.core_as_test}, "
            f"splitter={self.config['splitter']['selected']}, batch_size={self.config['dataset']['batch_size']}, "
            f"num_workers={num_workers}, splitter_seed={splitter_seed}, "
            f"epochs={self.config['training']['epochs']}",
            stage="EXPERIMENT"
        )
        log_info(
            f"DimeNet settings -> hidden_channels={hidden_dim}, cutoff={dimenet_cfg.get('dist_threshold', 5.0)}, "
            f"max_num_neighbors={dimenet_cfg.get('max_num_neighbors', 32)}, "
            f"num_blocks={dimenet_cfg.get('num_blocks', 3)}, ca_only={dimenet_cfg.get('ca_only', False)}, "
            f"protein_context={protein_context_mode}",
            stage="MODEL"
        )
        log_info(
            f"Protein context settings -> mode={protein_context_cfg.mode}, model={protein_context_cfg.model_name}, "
            f"repr_layer={protein_context_cfg.repr_layer}, pooling={protein_context_cfg.pooling}, "
            f"cache_path={protein_context_cfg.cache_path}, max_length={protein_context_cfg.max_length}",
            stage="PROTEIN_CONTEXT"
        )
        classic_opt_cfg = self.config['training']['optimizers']['classic']
        log_info(
            f"Optimizer settings -> type={classic_opt_cfg['type']}, "
            f"lr={classic_opt_cfg['params'].get('lr')}, "
            f"weight_decay={classic_opt_cfg['params'].get('weight_decay', 0.0)}",
            stage="OPTIMIZER"
        )
        self.prewarm_protein_context()
        log_info("Preparing model initialization...", stage="MODEL")
        model_init_started_at = time.perf_counter()
        model = A1DimeNet(
            config=self.config,
            device=self.device,
            hidden_channels=hidden_dim,
            out_channels=1,
            cutoff=dimenet_cfg.get('dist_threshold', 5.0),
            max_num_neighbors=dimenet_cfg.get('max_num_neighbors', 32),
            num_blocks=dimenet_cfg.get('num_blocks', 3),
        )
        log_info(
            f"Model initialization completed in "
            f"{self._format_duration(time.perf_counter() - model_init_started_at)}",
            stage="MODEL"
        )
        evaluator = Evaluator(model, self.device)

        log_info(f"Launch on: {self.device}", stage="EXPERIMENT")

        self.trainer = Trainer(model, evaluator, self.config, self.device)
        log_info("Preparing training loop...", stage="TRAINER")
        train_started_at = time.perf_counter()
        best_epoch, _ = self.trainer.train(train_loader, val_loader, self.exp_run_dir, self.config['training']['save_only_best_epoch'])
        log_info(
            f"Training loop completed in {self._format_duration(time.perf_counter() - train_started_at)}",
            stage="TRAINER"
        )
        log_info("Preparing final test evaluation...", stage="TEST")
        test_started_at = time.perf_counter()
        self.trainer.test(test_loader, self.exp_run_dir, best_epoch)
        log_info(
            f"Final test evaluation completed in "
            f"{self._format_duration(time.perf_counter() - test_started_at)}",
            stage="TEST"
        )

        log_info("Generating ASCII performance summary...", stage="SUMMARY")
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
    
    # Флаг экстракции (если указан в bash — станет True)
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
        core_as_test=args.core_as_test
    )
    runner.prepare_folders()
    train_df, val_df, test_df = runner.prepare_datasets()

    try:
        runner.run(train_df, val_df, test_df)
        if runner.temp_run and not runner.keep_temp:
            shutil.rmtree(runner.exp_run_dir, ignore_errors=True)
    except Exception as e:
        import traceback
        err_path = os.path.join(runner.exp_run_dir, "err_log.txt")
        error_msg = traceback.format_exc()
        log_info(f"ERROR message saved to: {err_path}", stage="CRASH")

        with open(err_path, "w", encoding="utf-8") as f:
            f.write(error_msg)

        raise e

"""
python run.py --config configs/gnn_test.json    # path to the custom config
python run.py --config config.json --extract    # then extract == True

"""
