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
from models.a2 import A2DimeNet
from models.a3 import A3DimeNet
from models.graph_components import (
    get_a3_mixer_bias,
    get_global_encoder_config,
    get_global_graph_config,
    get_global_graph_mode,
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
                 a3_mixer_bias: None | bool = None):
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
        self.model_family = get_model_family(self.config)
        if self.source_subset not in {'refined', 'general'}:
            raise ValueError("source_subset must be either 'refined' or 'general'")

        configured_core_as_test = self.config['dataset'].get('core_as_test', True)
        self.core_as_test = configured_core_as_test if core_as_test is None else core_as_test
        self.a3_mixer_bias = a3_mixer_bias
        if self.a3_mixer_bias is not None and self.model_family != "A3":
            raise ValueError(
                "--a3-mixer-bias / --no-a3-mixer-bias can only be used when "
                "model.selected == 'A3'."
            )
        self.test_frac = self.config['dataset'].get('test_frac', 0.15)
        if not self.core_as_test and not 0.0 < float(self.test_frac) < 1.0:
            raise ValueError("test_frac must be between 0 and 1")
        self.config['dataset']['source_subset'] = self.source_subset
        self.config['dataset']['core_as_test'] = self.core_as_test
        self.config['dataset']['test_frac'] = self.test_frac
        assert all([self.train_dataset_path, self.test_dataset_path, self.val_dataset_path]) or not any([self.train_dataset_path, self.test_dataset_path, self.val_dataset_path])

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{self.config['experiment_name']}_{timestamp}"
            log_info(f"Experiment signature: {exp_name}", stage="EXPERIMENT")
            self.exp_run_dir = f"runs/{exp_name}"

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
                    f"cutoff={local_encoder_cfg.cutoff}, max_num_neighbors={local_encoder_cfg.max_num_neighbors}, "
                    f"num_blocks={local_encoder_cfg.num_blocks}",
                    stage="MODEL"
                )
        if model_family == "A3":
            log_info(
                f"A3 readout settings -> mixer_bias={get_a3_mixer_bias(self.config, self.a3_mixer_bias)}",
                stage="MODEL"
            )
        elif local_graph_mode != "none" or get_local_encoder_mode(self.config) != "none":
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
