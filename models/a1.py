from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch_geometric.nn import DimeNetPlusPlus

from models.protein_context import (
    ProteinContextProjector,
    build_protein_context_encoder,
    get_protein_context_mode,
)


class A1DimeNet(nn.Module):
    def __init__(
        self,
        config: dict,
        device: str,
        hidden_channels: int = 128,
        out_channels: int = 1,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.device_name = device
        self.hidden_channels = hidden_channels
        self.protein_context_mode = get_protein_context_mode(config)

        self.gnn = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=128,
            num_spherical=7,
            num_radial=6,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=5,
        )

        self.protein_context_encoder = build_protein_context_encoder(config, device)
        self.protein_context_projector = None

        if self.protein_context_mode == "none":
            head_in_dim = hidden_channels
        elif self.protein_context_mode == "esm_only":
            if self.protein_context_encoder is None:
                raise ValueError("esm_only mode requires a protein context encoder")
            head_in_dim = self.protein_context_encoder.output_dim
        else:
            if self.protein_context_encoder is None:
                raise ValueError(f"{self.protein_context_mode} requires a protein context encoder")
            self.protein_context_projector = ProteinContextProjector(
                in_dim=self.protein_context_encoder.output_dim,
                out_dim=hidden_channels,
            )
            head_in_dim = hidden_channels * 2

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def checkpoint_exclude_prefixes(self) -> tuple[str, ...]:
        prefixes: list[str] = []
        if self.protein_context_encoder is not None:
            prefixes.append("protein_context_encoder.model.")
        return tuple(prefixes)

    def build_checkpoint_payload(self) -> dict:
        state = self.state_dict()
        excluded_prefixes = self.checkpoint_exclude_prefixes()
        filtered_state = {
            key: value
            for key, value in state.items()
            if not any(key.startswith(prefix) for prefix in excluded_prefixes)
        }
        return {
            "checkpoint_format": "lightweight_no_frozen_esm_v1",
            "state_dict": filtered_state,
            "excluded_prefixes": list(excluded_prefixes),
            "protein_context_mode": self.protein_context_mode,
        }

    def load_checkpoint_payload(self, payload) -> None:
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            excluded_prefixes = tuple(payload.get("excluded_prefixes", []))
            incompatible = self.load_state_dict(state_dict, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            disallowed_missing = [
                key for key in incompatible.missing_keys
                if not any(key.startswith(prefix) for prefix in excluded_prefixes)
            ]
            if unexpected or disallowed_missing:
                raise RuntimeError(
                    "Checkpoint load mismatch: "
                    f"unexpected_keys={unexpected}, missing_keys={disallowed_missing}"
                )
            return

        self.load_state_dict(payload)

    @staticmethod
    def _extract_protein_sequences(complex_data) -> Sequence[str]:
        sequences = getattr(complex_data, "protein_sequence", None)
        if sequences is None:
            raise ValueError("protein_context mode requires complex_data.protein_sequence")
        if isinstance(sequences, str):
            return [sequences]
        return [str(seq) for seq in sequences]

    def forward(self, batch_list, progress: float = 0.0):
        _, _, _, complex_data, _ = batch_list

        geom_embedding = None
        if self.protein_context_mode != "esm_only":
            z = complex_data.x[:, 0].long()
            pos = complex_data.pos.float()
            geom_embedding = self.gnn(z, pos, complex_data.batch)

        if self.protein_context_mode == "none":
            fused = geom_embedding
        else:
            protein_sequences = self._extract_protein_sequences(complex_data)
            protein_context = self.protein_context_encoder.encode_sequences(protein_sequences)

            if self.protein_context_mode == "esm_only":
                fused = protein_context
            else:
                projected_context = self.protein_context_projector(protein_context)
                fused = torch.cat([geom_embedding, projected_context], dim=-1)

        return self.head(fused).view(-1)
