from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.graph_components import build_dimenet_backbone, build_head, get_global_encoder_config
from models.protein_context import (
    ProteinContextProjector,
    build_protein_context_encoder,
    get_protein_context_mode,
)
from models.ligand_context import (
    build_ligand_context_encoder,
    get_ligand_context_mode,
)


class A1DimeNet(nn.Module):
    """
    DimeNet++ baseline with optional global protein and ligand context fusion.

    Genealogy:
    - the geometry branch descends from the original A1 DimeNet++ baseline,
    - the protein branch adds frozen ESM sequence embeddings,
    - the ligand branch adds frozen RDKit descriptor vectors,
    - the final prediction head performs late fusion over whichever branches
      are enabled by configuration.

    The class is intentionally conservative: each auxiliary context branch is
    projected into the same hidden width as the geometric embedding before
    concatenation. That keeps the fusion contract simple and makes ablations
    easier to interpret.
    """
    def __init__(
        self,
        config: dict,
        device: str,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.device_name = device
        self.global_encoder_cfg = get_global_encoder_config(config)
        self.hidden_channels = self.global_encoder_cfg.hidden_channels
        self.protein_context_mode = get_protein_context_mode(config)
        self.ligand_context_mode = get_ligand_context_mode(config)

        self.gnn = build_dimenet_backbone(self.global_encoder_cfg)

        self.protein_context_encoder = build_protein_context_encoder(config, device)
        self.protein_context_projector = None
        self.ligand_context_encoder = build_ligand_context_encoder(config, device)
        self.ligand_context_projector = None

        head_in_dim = 0
        if self.protein_context_mode != "esm_only":
            head_in_dim += self.hidden_channels

        if self.protein_context_mode != "none":
            if self.protein_context_encoder is None:
                raise ValueError(f"{self.protein_context_mode} requires a protein context encoder")
            if self.protein_context_mode == "esm_only":
                head_in_dim += self.protein_context_encoder.output_dim
            else:
                self.protein_context_projector = ProteinContextProjector(
                    in_dim=self.protein_context_encoder.output_dim,
                    out_dim=self.hidden_channels,
                )
                head_in_dim += self.hidden_channels

        if self.ligand_context_mode != "none":
            if self.ligand_context_encoder is None:
                raise ValueError(f"{self.ligand_context_mode} requires a ligand context encoder")
            self.ligand_context_projector = ProteinContextProjector(
                in_dim=self.ligand_context_encoder.output_dim,
                out_dim=self.hidden_channels,
            )
            head_in_dim += self.hidden_channels

        self.head = build_head(config, head_in_dim, self.hidden_channels, out_channels)

    def checkpoint_exclude_prefixes(self) -> tuple[str, ...]:
        """
        Return state-dict prefixes that should be omitted from lightweight checkpoints.

        Frozen ESM weights are reloaded from the source model definition and do
        not need to be duplicated in every experiment artifact.
        """
        prefixes: list[str] = []
        if self.protein_context_encoder is not None:
            prefixes.append("protein_context_encoder.model.")
        return tuple(prefixes)

    def build_checkpoint_payload(self) -> dict:
        """
        Build a checkpoint payload that omits known frozen weights when possible.

        Returns:
            Serializable checkpoint dictionary used by the trainer.
        """
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
        """
        Load either a lightweight payload or a raw `state_dict`.

        Args:
            payload: Checkpoint payload produced by `build_checkpoint_payload`
                or a plain `state_dict` for backwards compatibility.
        """
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
        """
        Extract batch-aligned protein sequences from the fused complex object.

        Args:
            complex_data: Batched PyG `Data` object produced by the tokenizer.

        Returns:
            List of protein sequences aligned with the batch dimension.
        """
        sequences = getattr(complex_data, "protein_sequence", None)
        if sequences is None:
            raise ValueError("protein_context mode requires complex_data.protein_sequence")
        if isinstance(sequences, str):
            return [sequences]
        return [str(seq) for seq in sequences]

    @staticmethod
    def _extract_ligand_smiles(complex_data) -> Sequence[str]:
        """
        Extract batch-aligned ligand SMILES strings from the fused complex object.

        Args:
            complex_data: Batched PyG `Data` object produced by the tokenizer.

        Returns:
            List of canonical or parser-produced ligand SMILES strings.
        """
        smiles = getattr(complex_data, "ligand_smiles", None)
        if smiles is None:
            raise ValueError("ligand_context mode requires complex_data.ligand_smiles")
        if isinstance(smiles, str):
            return [smiles]
        return [str(smi) for smi in smiles]

    def forward(self, batch_list, progress: float = 0.0):
        """
        Run the configured geometry/context branches and return affinity logits.

        Args:
            batch_list: Tuple-style batch emitted by `UniversalPDBBindDataset`
                and collated by the PyG dataloader.
            progress: Normalized training progress in `[0, 1]`. It is currently
                unused by the DimeNet path but kept for interface compatibility.

        Returns:
            One scalar prediction per complex in the batch.
        """
        _, _, _, complex_data, _ = batch_list

        fused_parts = []
        if self.protein_context_mode != "esm_only":
            z = complex_data.x[:, 0].long()
            pos = complex_data.pos.float()
            fused_parts.append(self.gnn(z, pos, complex_data.batch))

        if self.protein_context_mode != "none":
            protein_sequences = self._extract_protein_sequences(complex_data)
            protein_context = self.protein_context_encoder.encode_sequences(protein_sequences)

            if self.protein_context_mode == "esm_only":
                fused_parts.append(protein_context)
            else:
                projected_context = self.protein_context_projector(protein_context)
                fused_parts.append(projected_context)

        if self.ligand_context_mode != "none":
            ligand_smiles = self._extract_ligand_smiles(complex_data)
            ligand_context = self.ligand_context_encoder.encode_smiles_batch(ligand_smiles)
            fused_parts.append(self.ligand_context_projector(ligand_context))

        fused = fused_parts[0] if len(fused_parts) == 1 else torch.cat(fused_parts, dim=-1)

        return self.head(fused).view(-1)
