from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from torch import nn
from tqdm import tqdm


SUPPORTED_LIGAND_CONTEXT_MODES = {
    "none",
    "basic_rdkit",
}


@dataclass(frozen=True)
class LigandContextConfig:
    mode: str = "none"
    cache_path: str = "ligand_context_features"
    descriptor_set: str = "basic_physchem_v1"
    embedding_dim: int = 7
    precompute_batch_size: int = 256

    @classmethod
    def from_config(cls, config: dict) -> "LigandContextConfig":
        entry = config.get("model", {}).get("ligand_context", {})
        selected = entry.get("selected", "none")
        available = entry.get("available", {})
        selected_cfg = available.get(selected, {}) if selected != "none" else {}
        cfg = cls(
            mode=selected,
            cache_path=selected_cfg.get("cache_path", cls.cache_path),
            descriptor_set=selected_cfg.get("descriptor_set", cls.descriptor_set),
            embedding_dim=selected_cfg.get("embedding_dim", cls.embedding_dim),
            precompute_batch_size=selected_cfg.get("precompute_batch_size", cls.precompute_batch_size),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in SUPPORTED_LIGAND_CONTEXT_MODES:
            raise ValueError(
                f"Unsupported ligand_context mode: {self.mode}. "
                f"Supported: {sorted(SUPPORTED_LIGAND_CONTEXT_MODES)}"
            )
        if int(self.precompute_batch_size) <= 0:
            raise ValueError("ligand_context precompute_batch_size must be > 0")


class FrozenLigandDescriptorEncoder(nn.Module):
    """
    Deterministic RDKit-based ligand descriptor encoder with on-disk caching.

    The first iteration intentionally stays compact and interpretable:
    MW, logP, TPSA, HBD, HBA, Lipinski violations, Wiener index.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        descriptor_set: str = "basic_physchem_v1",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.cache_path = Path(cache_path) if cache_path else None
        self.descriptor_set = descriptor_set
        self.device_name = device
        self.output_dim = 7
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        if self.cache_path is not None:
            self.cache_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _canonicalize_smiles(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid ligand SMILES: {smiles}")
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    def _cache_file_for_smiles(self, smiles: str) -> Optional[Path]:
        if self.cache_path is None:
            return None
        key_payload = "|".join([self.descriptor_set, smiles])
        digest = sha1(key_payload.encode("utf-8")).hexdigest()
        return self.cache_path / f"{digest}.pt"

    def _load_cached_embedding(self, smiles: str) -> Optional[torch.Tensor]:
        cache_file = self._cache_file_for_smiles(smiles)
        if cache_file is None or not cache_file.exists():
            return None
        return torch.load(cache_file, map_location="cpu")

    def _save_cached_embedding(self, smiles: str, embedding: torch.Tensor) -> None:
        cache_file = self._cache_file_for_smiles(smiles)
        if cache_file is None:
            return
        torch.save(embedding.detach().cpu(), cache_file)

    @staticmethod
    def _wiener_index(mol: Chem.Mol) -> float:
        dist = Chem.GetDistanceMatrix(mol)
        if dist.size == 0:
            return 0.0
        upper = np.triu(dist, k=1)
        return float(upper.sum())

    @classmethod
    def _compute_descriptor_vector(cls, smiles: str) -> torch.Tensor:
        canonical = cls._canonicalize_smiles(smiles)
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            raise ValueError(f"Failed to rebuild canonical ligand SMILES: {smiles}")

        hbd = float(rdMolDescriptors.CalcNumHBD(mol))
        hba = float(rdMolDescriptors.CalcNumHBA(mol))
        vector = torch.tensor(
            [
                float(Descriptors.MolWt(mol)),
                float(Crippen.MolLogP(mol)),
                float(rdMolDescriptors.CalcTPSA(mol)),
                hbd,
                hba,
                float(Lipinski.NumLipinskiHBA(mol) > 10)
                + float(Lipinski.NumLipinskiHBD(mol) > 5)
                + float(Descriptors.MolWt(mol) > 500.0)
                + float(Crippen.MolLogP(mol) > 5.0),
                cls._wiener_index(mol),
            ],
            dtype=torch.float32,
        )
        return vector

    def encode_smiles_batch(self, smiles_batch: Sequence[str]) -> torch.Tensor:
        canonical_batch = [self._canonicalize_smiles(smiles) for smiles in smiles_batch]
        for smiles in canonical_batch:
            if smiles in self.embedding_cache:
                continue
            cached = self._load_cached_embedding(smiles)
            if cached is not None:
                self.embedding_cache[smiles] = cached

        missing = [smiles for smiles in canonical_batch if smiles not in self.embedding_cache]
        for smiles in dict.fromkeys(missing):
            embedding = self._compute_descriptor_vector(smiles)
            self.embedding_cache[smiles] = embedding
            self._save_cached_embedding(smiles, embedding)

        return torch.stack([self.embedding_cache[smiles] for smiles in canonical_batch], dim=0).to(self.device_name)

    def encode_smiles(self, smiles: str) -> torch.Tensor:
        return self.encode_smiles_batch([smiles])[0]

    def precompute_smiles(
        self,
        smiles_list: Sequence[str],
        batch_size: int = 256,
        progress_desc: str = "Ligand context precompute",
    ) -> dict:
        canonical_smiles = [self._canonicalize_smiles(smiles) for smiles in smiles_list if smiles is not None]
        unique_smiles = list(dict.fromkeys(canonical_smiles))

        cached_before = 0
        missing_smiles: List[str] = []
        for smiles in unique_smiles:
            if smiles in self.embedding_cache:
                cached_before += 1
                continue
            cached = self._load_cached_embedding(smiles)
            if cached is not None:
                self.embedding_cache[smiles] = cached
                cached_before += 1
            else:
                missing_smiles.append(smiles)

        if missing_smiles:
            total_batches = (len(missing_smiles) + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(total_batches), desc=progress_desc, unit="batch", leave=True):
                start = batch_idx * batch_size
                end = start + batch_size
                batch = missing_smiles[start:end]
                _ = self.encode_smiles_batch(batch)

        return {
            "total_unique": len(unique_smiles),
            "cached_before": cached_before,
            "computed_now": len(missing_smiles),
        }


def build_ligand_context_encoder(config: dict, device: str) -> Optional[FrozenLigandDescriptorEncoder]:
    ctx_cfg = LigandContextConfig.from_config(config)
    if ctx_cfg.mode == "none":
        return None
    if ctx_cfg.mode == "basic_rdkit":
        return FrozenLigandDescriptorEncoder(
            cache_path=ctx_cfg.cache_path,
            descriptor_set=ctx_cfg.descriptor_set,
            device=device,
        )
    raise ValueError(f"Unhandled ligand_context mode: {ctx_cfg.mode}")


def get_ligand_context_mode(config: dict) -> str:
    return LigandContextConfig.from_config(config).mode
