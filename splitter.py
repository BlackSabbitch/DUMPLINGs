# splitter.py

import numpy as np
import random
import hashlib
from collections import defaultdict
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Tuple, Optional, Set
from logger import *


class PDBBindSplitter:
    """
    Collection of splitting strategies for PDBBind refined dataset.

    Provides various methods for splitting molecular datasets into train/validation sets,
    including random, scaffold-based, and scaffold-balanced splits. All methods
    return (train_df, val_df) tuples.

    Example:
        >>> train_df, val_df = PDBBindSplitter.random_split(df, val_frac=0.15, seed=42)
        >>> print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    """

    # ======================
    # RANDOM SPLIT
    # ======================
    @staticmethod
    def random_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform random split of the dataset.

        Args:
            df: Input DataFrame to split.
            val_frac: Fraction of data for validation (0.0 to 1.0).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_df, val_df).
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(df))

        val_size = int(len(df) * val_frac)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        return df.iloc[train_idx], df.iloc[val_idx]

    @staticmethod
    def random_test_split(df: pd.DataFrame, test_frac: float = 0.15, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a dataframe into a train/validation pool and a held-out test set.
        """
        if not 0.0 < float(test_frac) < 1.0:
            raise ValueError("test_frac must be between 0 and 1")

        indices = np.random.default_rng(seed).permutation(len(df))
        test_size = int(len(df) * float(test_frac))
        if test_size == 0 or test_size == len(df):
            raise ValueError("test_frac creates an empty train/validation pool or test set")

        test_idx = indices[:test_size]
        train_val_idx = indices[test_size:]
        return df.iloc[train_val_idx].copy(), df.iloc[test_idx].copy()

    @staticmethod
    def core_test_split(df: pd.DataFrame, core_ids: Set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split refined into train/validation pool and PDBBind core test set.
        """
        test_df = df[df['pdb_id'].isin(core_ids)].copy()
        missing_core_ids = core_ids - set(test_df['pdb_id'])
        if missing_core_ids:
            log_warn(
                f"{len(missing_core_ids)} core complexes are missing from refined dataset",
                stage="DATASET"
            )

        train_val_df = df[~df['pdb_id'].isin(test_df['pdb_id'])].copy()
        return train_val_df, test_df

    # ======================
    # SCAFFOLD SPLIT
    # ======================
    @staticmethod
    def _get_scaffold(smiles: str) -> Optional[str]:
        """
        Get Murcko scaffold from SMILES string.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Scaffold SMILES string or None if failed to generate.
        """
        if not isinstance(smiles, str) or len(smiles) == 0:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None:
                return None

            # Канонизация без стереохимии (стандартная практика)
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)

        except Exception:
            return None

    @staticmethod
    def scaffold_split_strict(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform strict scaffold-based split.

        Ensures that molecules with the same scaffold are never split between
        train and validation sets. Assigns entire scaffold groups to either
        train or validation.

        Args:
            df: DataFrame with 'smiles' column.
            val_frac: Target fraction for validation.
            seed: Random seed.

        Returns:
            Tuple of (train_df, val_df).
        """
        random.seed(seed)

        scaffold_to_indices = defaultdict(list)

        for i, smi in enumerate(df['smiles']):
            scaf = PDBBindSplitter._get_scaffold(smi)

            if scaf is None:
                scaf = f"NO_SCAF_{i}"

            scaffold_to_indices[scaf].append(i)

        # Sort by size (largest first)
        scaffolds = sorted(scaffold_to_indices.values(), key=len, reverse=True)

        train_idx, val_idx = [], []
        val_target = int(len(df) * val_frac)

        for group in scaffolds:
            if len(val_idx) + len(group) <= val_target:
                val_idx.extend(group)
            else:
                train_idx.extend(group)

        return df.iloc[train_idx], df.iloc[val_idx]

    @staticmethod
    def scaffold_split_balanced(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform balanced scaffold-based split.

        Attempts to balance scaffold distribution while maintaining the target
        validation fraction. May split some scaffold groups if necessary.

        Args:
            df: DataFrame with 'smiles' column.
            val_frac: Target fraction for validation.
            seed: Random seed.

        Returns:
            Tuple of (train_df, val_df).
        """
        random.seed(seed)

        scaffold_to_indices = defaultdict(list)

        for i, smi in enumerate(df['smiles']):
            scaf = PDBBindSplitter._get_scaffold(smi)

            if scaf is None:
                scaf = f"NO_SCAF_{i}"

            scaffold_to_indices[scaf].append(i)

        scaffolds = sorted(scaffold_to_indices.values(), key=len, reverse=True)

        train_idx, val_idx = [], []
        train_target = int(len(df) * (1 - val_frac))

        for group in scaffolds:
            if len(train_idx) + len(group) <= train_target:
                train_idx.extend(group)
            else:
                val_idx.extend(group)

        return df.iloc[train_idx], df.iloc[val_idx]

    @staticmethod
    def cold_protein_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform cold protein split.

        Ensures that proteins not seen in training appear in validation.
        Groups complexes by protein sequence and assigns entire groups.

        Args:
            df: DataFrame with 'seq' column containing protein sequences.
            val_frac: Target fraction for validation.
            seed: Random seed.

        Returns:
            Tuple of (train_df, val_df).
        """
        random.seed(seed)

        protein_to_indices = defaultdict(list)

        for i, seq in enumerate(df['seq']):
            protein_to_indices[seq].append(i)

        groups = list(protein_to_indices.values())
        random.shuffle(groups)

        train_idx, val_idx = [], []
        val_target = int(len(df) * val_frac)

        for group in groups:
            if len(val_idx) + len(group) <= val_target:
                val_idx.extend(group)
            else:
                train_idx.extend(group)

        return df.iloc[train_idx], df.iloc[val_idx]

    # ======================
    # UNIFIED INTERFACE
    # ======================
    @staticmethod
    def split(df: pd.DataFrame, conf_splitter: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Unified interface for all splitting strategies.

        Args:
            df: DataFrame to split.
            strategy: Splitting strategy ('random', 'scaffold', 'scaffold_balanced', 'cold_protein').
            val_frac: Validation fraction.
            seed: Random seed.

        Returns:
            Tuple of (train_df, val_df).

        Raises:
            ValueError: If strategy is not supported.
        """
        strategy = conf_splitter['selected']
        params = conf_splitter['available'][strategy]

        log_info(f"Strategy: {strategy}", stage="SPLIT")

        if strategy == "random":
            train_df, val_df = PDBBindSplitter.random_split(df, **params)

        elif strategy == "scaffold":
            train_df, val_df = PDBBindSplitter.scaffold_split_strict(df, **params)

        elif strategy == "scaffold_balanced":
            train_df, val_df = PDBBindSplitter.scaffold_split_balanced(df, **params)

        elif strategy == "cold_protein":
            train_df, val_df = PDBBindSplitter.cold_protein_split(df, **params)

        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
        
        # Log split results
        log_info(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}", stage="SPLIT")
        
        return train_df, val_df

    @staticmethod
    def split_with_test(
        df: pd.DataFrame,
        conf_splitter: dict,
        core_as_test: bool = True,
        test_frac: float = 0.15,
        core_ids: Optional[Set[str]] = None,
        source_subset: str = "refined",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        """
        Split a source dataframe into train, validation, and test sets.
        """
        strategy = conf_splitter['selected']
        split_params = conf_splitter['available'][strategy]
        seed = split_params.get("seed", 42)

        log_info(f"Core as test: {core_as_test}", stage="SPLIT")
        if core_as_test:
            if core_ids is None:
                raise ValueError("core_ids are required when core_as_test=True")
            train_val_df, test_df = PDBBindSplitter.core_test_split(df, core_ids)
            test_file_name = "test_core.pickle"
        else:
            train_val_df, test_df = PDBBindSplitter.random_test_split(df, test_frac=test_frac, seed=seed)
            test_file_name = f"test_{source_subset}.pickle"
            log_debug(
                f"{source_subset} random test split: test_frac={test_frac} "
                f"| Train/Val pool: {len(train_val_df)} | Test: {len(test_df)}",
                stage="SPLIT"
            )

        train_df, val_df = PDBBindSplitter.split(train_val_df, conf_splitter)
        return train_df, val_df, test_df, test_file_name
