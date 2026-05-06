from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from logger import log_info
from tqdm import tqdm


SUPPORTED_PROTEIN_CONTEXT_MODES = {
    "none",
    "esm_frozen_whole",
    "esm_only",
    "esm_frozen_pocket",
}


@dataclass(frozen=True)
class ProteinContextConfig:
    mode: str = "none"
    model_name: str = "esm2_t33_650M_UR50D"
    repr_layer: int = 33
    pooling: str = "mean"
    cache_path: str = "esm_cache"
    embedding_dim: int = 1280
    max_length: Optional[int] = None
    precompute_batch_size: int = 8

    @classmethod
    def from_config(cls, config: dict) -> "ProteinContextConfig":
        if "protein_context" in config:
            entry = config.get("protein_context", {})
        else:
            entry = config.get("model", {}).get("protein_context", {})
        selected = entry.get("selected", "none")
        available = entry.get("available", {})
        selected_cfg = available.get(selected, {}) if selected != "none" else {}
        cfg = cls(
            mode=selected,
            model_name=selected_cfg.get("model_name", cls.model_name),
            repr_layer=selected_cfg.get("repr_layer", cls.repr_layer),
            pooling=selected_cfg.get("pooling", cls.pooling),
            cache_path=selected_cfg.get("cache_path", cls.cache_path),
            embedding_dim=selected_cfg.get("embedding_dim", cls.embedding_dim),
            max_length=selected_cfg.get("max_length", cls.max_length),
            precompute_batch_size=selected_cfg.get("precompute_batch_size", cls.precompute_batch_size),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in SUPPORTED_PROTEIN_CONTEXT_MODES:
            raise ValueError(
                f"Unsupported protein_context mode: {self.mode}. "
                f"Supported: {sorted(SUPPORTED_PROTEIN_CONTEXT_MODES)}"
            )
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("protein_context pooling must be 'mean' or 'cls'")
        if int(self.precompute_batch_size) <= 0:
            raise ValueError("protein_context precompute_batch_size must be > 0")


class FrozenESMEncoder(nn.Module):
    """
    Thin wrapper around a pre-trained ESM model.

    Design goals:
    - lazy import of the `esm` package so A1 runs do not require it,
    - explicit frozen inference mode,
    - sequence-to-vector encoding for late fusion experiments,
    - future extension point for pocket-only residue pooling.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        repr_layer: int = 33,
        pooling: str = "mean",
        device: str = "cpu",
        cache_path: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.repr_layer = repr_layer
        self.pooling = pooling
        self.device_name = device
        self.max_length = max_length
        self.cache_path = Path(cache_path) if cache_path else None

        try:
            import esm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "FrozenESMEncoder requires the `esm` package. "
                "Install it only for A1.5+ runs where protein context is enabled."
            ) from exc

        self._esm = esm
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.output_dim = self.model.embed_dim
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        if self.cache_path is not None:
            self.cache_path.mkdir(parents=True, exist_ok=True)

    def _cache_file_for_sequence(self, sequence: str) -> Optional[Path]:
        if self.cache_path is None:
            return None
        key_payload = "|".join([
            self.model_name,
            str(self.repr_layer),
            self.pooling,
            sequence,
        ])
        digest = sha1(key_payload.encode("utf-8")).hexdigest()
        return self.cache_path / f"{digest}.pt"

    def _load_cached_embedding(self, sequence: str) -> Optional[torch.Tensor]:
        cache_file = self._cache_file_for_sequence(sequence)
        if cache_file is None or not cache_file.exists():
            return None
        return torch.load(cache_file, map_location="cpu")

    def _save_cached_embedding(self, sequence: str, embedding: torch.Tensor) -> None:
        cache_file = self._cache_file_for_sequence(sequence)
        if cache_file is None:
            return
        torch.save(embedding.detach().cpu(), cache_file)

    def _truncate(self, sequence: str) -> str:
        if self.max_length is None:
            return sequence
        return sequence[: self.max_length]

    def _sanitize_sequence(self, sequence: str) -> str:
        return self._truncate(sequence)

    def _pool(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return token_embeddings[0]
        return token_embeddings.mean(dim=0)

    @torch.no_grad()
    def encode_sequences(self, sequences: Sequence[str]) -> torch.Tensor:
        sanitized = [self._sanitize_sequence(seq) for seq in sequences]
        for seq in sanitized:
            if seq in self.embedding_cache:
                continue
            cached = self._load_cached_embedding(seq)
            if cached is not None:
                self.embedding_cache[seq] = cached

        missing = [seq for seq in sanitized if seq not in self.embedding_cache]
        if missing:
            unique_missing = list(dict.fromkeys(missing))
            labels_and_sequences = [
                (f"seq_{idx}", seq if seq else "X") for idx, seq in enumerate(unique_missing)
            ]
            _, _, tokens = self.batch_converter(labels_and_sequences)
            tokens = tokens.to(self.device_name)

            results = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
            representations = results["representations"][self.repr_layer]

            for idx, seq in enumerate(unique_missing):
                seq_len = len(seq if seq else "X")
                token_slice = representations[idx, 1 : seq_len + 1]
                pooled = self._pool(token_slice).detach().cpu()
                self.embedding_cache[seq] = pooled
                self._save_cached_embedding(seq, pooled)

        return torch.stack([self.embedding_cache[seq] for seq in sanitized], dim=0).to(self.device_name)

    @torch.no_grad()
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        return self.encode_sequences([sequence])[0]

    def precompute_sequences(
        self,
        sequences: Sequence[str],
        batch_size: int = 8,
        progress_desc: str = "ESM precompute",
    ) -> dict:
        sanitized = [self._sanitize_sequence(seq) for seq in sequences if seq is not None]
        unique_sequences = list(dict.fromkeys(sanitized))

        cached_before = 0
        missing_sequences: List[str] = []
        for seq in unique_sequences:
            if seq in self.embedding_cache:
                cached_before += 1
                continue
            cached = self._load_cached_embedding(seq)
            if cached is not None:
                self.embedding_cache[seq] = cached
                cached_before += 1
            else:
                missing_sequences.append(seq)

        if missing_sequences:
            total_batches = (len(missing_sequences) + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(total_batches), desc=progress_desc, unit="batch", leave=True):
                start = batch_idx * batch_size
                end = start + batch_size
                batch = missing_sequences[start:end]
                _ = self.encode_sequences(batch)

        return {
            "total_unique": len(unique_sequences),
            "cached_before": cached_before,
            "computed_now": len(missing_sequences),
        }


class ProteinContextProjector(nn.Module):
    """
    Small projection block to align a protein-context vector with the
    geometric latent space before late fusion.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_protein_context_encoder(config: dict, device: str) -> Optional[FrozenESMEncoder]:
    """
    Build the protein-context encoder from config.

    Modes:
    - none: disabled
    - esm_frozen_whole: full-sequence frozen ESM embedding
    - esm_only: same encoder as above, but intended for ablation where geometry is bypassed
    - esm_frozen_pocket: reserved for future residue-aware pooling implementation
    """
    ctx_cfg = ProteinContextConfig.from_config(config)
    if ctx_cfg.mode == "none":
        return None
    if ctx_cfg.mode == "esm_frozen_pocket":
        raise NotImplementedError(
            "protein_context mode 'esm_frozen_pocket' is reserved for A1.5b and "
            "requires residue-level pocket-to-sequence mapping that is not wired yet."
        )
    if ctx_cfg.mode in {"esm_frozen_whole", "esm_only"}:
        encoder = FrozenESMEncoder(
            model_name=ctx_cfg.model_name,
            repr_layer=ctx_cfg.repr_layer,
            pooling=ctx_cfg.pooling,
            device=device,
            cache_path=ctx_cfg.cache_path,
            max_length=ctx_cfg.max_length,
        )
        log_info(
            f"Protein context encoder ready: mode={ctx_cfg.mode}, model={ctx_cfg.model_name}, "
            f"repr_layer={ctx_cfg.repr_layer}, pooling={ctx_cfg.pooling}, cache_path={ctx_cfg.cache_path}",
            stage="PROTEIN_CONTEXT"
        )
        return encoder
    raise ValueError(f"Unhandled protein_context mode: {ctx_cfg.mode}")


def get_protein_context_mode(config: dict) -> str:
    return ProteinContextConfig.from_config(config).mode
