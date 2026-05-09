from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.a1 import A1DimeNet
from logger import log_warn
from models.graph_components import (
    build_dimenet_backbone,
    build_head,
    get_global_encoder_config,
    get_local_encoder_config,
    get_local_graph_config,
    get_local_graph_mode,
)


class A2DimeNet(A1DimeNet):
    """
    A2 model with an additional local DimeNet++ branch over a tighter subgraph.

    Genealogy:
    - inherits A1's global geometry/context fusion contract,
    - keeps the same protein and ligand context branches,
    - adds a second geometric branch operating on an interaction-focused
      ligand-pocket subgraph,
    - delegates the final combination of global and local representations to a
      configurable prediction head.

    The first revision intentionally keeps the local branch simple:
    radius-based node selection plus a second DimeNet++ encoder. Importance-
    based local weighting is left for later A2 iterations, but the config
    surface is now ready for that expansion.
    """

    def __init__(
        self,
        config: dict,
        device: str,
        out_channels: int = 1,
    ) -> None:
        super().__init__(config=config, device=device, out_channels=out_channels)
        self.global_encoder_cfg = get_global_encoder_config(config)
        self.local_graph_mode = get_local_graph_mode(config)
        self.local_graph_cfg = get_local_graph_config(config)
        self.local_encoder_cfg = get_local_encoder_config(config)

        self.local_gnn = None
        self.local_cutoff = None
        self.local_output_norm = None
        if self.local_encoder_cfg is not None and self.local_graph_mode != "none":
            self.local_gnn = build_dimenet_backbone(self.local_encoder_cfg)
            self.local_cutoff = float(self.local_graph_cfg.get("dist_threshold", 3.5))
            self.local_output_norm = nn.LayerNorm(self.local_encoder_cfg.hidden_channels)
        self.local_nonfinite_warned_ids: set[str] = set()

        head_in_dim = self._infer_global_head_input_dim()
        if self.local_gnn is not None:
            head_in_dim += self.local_encoder_cfg.hidden_channels
        self.head = build_head(config, head_in_dim, self.global_encoder_cfg.hidden_channels, out_channels)

    def _infer_global_head_input_dim(self) -> int:
        head_in_dim = 0
        if self.protein_context_mode != "esm_only":
            head_in_dim += self.hidden_channels
        if self.protein_context_mode != "none":
            if self.protein_context_mode == "esm_only":
                head_in_dim += self.protein_context_encoder.output_dim
            else:
                head_in_dim += self.hidden_channels
        if self.ligand_context_mode != "none":
            head_in_dim += self.hidden_channels
        return head_in_dim

    @staticmethod
    def _select_local_mask_for_graph(
        graph_x: torch.Tensor,
        graph_pos: torch.Tensor,
        local_cutoff: float,
    ) -> torch.Tensor:
        ligand_mask = graph_x[:, 3] > 0.5
        if not torch.any(ligand_mask):
            return torch.ones(graph_x.size(0), dtype=torch.bool, device=graph_x.device)

        pocket_mask = ~ligand_mask
        if not torch.any(pocket_mask):
            return ligand_mask

        ligand_pos = graph_pos[ligand_mask]
        pocket_pos = graph_pos[pocket_mask]
        distances = torch.cdist(pocket_pos, ligand_pos)
        close_pocket_mask = torch.any(distances <= local_cutoff, dim=1)

        selected = ligand_mask.clone()
        pocket_indices = torch.nonzero(pocket_mask, as_tuple=False).view(-1)
        selected[pocket_indices[close_pocket_mask]] = True
        return selected

    @staticmethod
    def _stabilize_local_coordinates(
        coords: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """
        Re-apply duplicate-coordinate stabilization on the selected local subgraph.

        The full interaction graph is already stabilized during parsing, but the
        tighter local branch can still surface numerically awkward coordinate
        patterns. Repeating the nudge here is cheap and helps keep the local
        DimeNet branch away from catastrophic explosions on a few outlier
        complexes.
        """
        if coords.size(0) < 2:
            return coords

        adjusted = coords.clone()
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=adjusted.dtype,
            device=adjusted.device,
        )
        directions = directions / directions.norm(dim=1, keepdim=True)

        seen: dict[tuple[float, float, float], int] = {}
        for idx in range(adjusted.size(0)):
            coord = adjusted[idx]
            key = tuple(round(float(value), 6) for value in coord.tolist())
            dup_count = seen.get(key, 0)
            if dup_count > 0:
                direction = directions[(dup_count - 1) % directions.size(0)]
                adjusted[idx] = coord + direction * (eps * dup_count)
            seen[key] = dup_count + 1

        return adjusted

    @staticmethod
    def _graph_pdb_id(complex_data, batch_id: int) -> str:
        pdb_ids = getattr(complex_data, "pdb_id", None)
        if isinstance(pdb_ids, (list, tuple)) and batch_id < len(pdb_ids):
            return str(pdb_ids[batch_id])
        if isinstance(pdb_ids, str):
            return pdb_ids
        return f"batch_{batch_id}"

    def _safe_local_zero(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.local_encoder_cfg.hidden_channels, device=device, dtype=torch.float32)

    def _encode_local_branch(self, complex_data) -> torch.Tensor:
        batch = complex_data.batch
        x = complex_data.x
        pos = complex_data.pos.float()
        outputs: list[torch.Tensor] = []

        unique_batches = torch.unique(batch, sorted=True)
        for batch_id in unique_batches.tolist():
            graph_mask = batch == batch_id
            graph_x = x[graph_mask]
            graph_pos = pos[graph_mask]
            selected = self._select_local_mask_for_graph(graph_x, graph_pos, self.local_cutoff)
            local_pos = self._stabilize_local_coordinates(graph_pos[selected])
            local_z = graph_x[selected, 0].long()

            if local_pos.size(0) < 2:
                outputs.append(self._safe_local_zero(device=graph_pos.device))
                continue

            local_batch = torch.zeros(local_pos.size(0), dtype=batch.dtype, device=batch.device)
            local_repr = self.local_gnn(local_z, local_pos, local_batch).view(-1)

            if not torch.isfinite(local_repr).all():
                pdb_id = self._graph_pdb_id(complex_data, batch_id)
                if pdb_id not in self.local_nonfinite_warned_ids:
                    self.local_nonfinite_warned_ids.add(pdb_id)
                    log_warn(
                        f"Local branch produced non-finite values for {pdb_id}; "
                        "falling back to a zero local representation for this complex.",
                        stage="MODEL"
                    )
                outputs.append(self._safe_local_zero(device=graph_pos.device))
                continue

            outputs.append(self.local_output_norm(local_repr))

        return torch.stack(outputs, dim=0)

    def forward(self, batch_list, progress: float = 0.0):
        _, _, _, complex_data, _ = batch_list

        fused_parts: list[torch.Tensor] = []
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
                fused_parts.append(self.protein_context_projector(protein_context))

        if self.ligand_context_mode != "none":
            ligand_smiles = self._extract_ligand_smiles(complex_data)
            ligand_context = self.ligand_context_encoder.encode_smiles_batch(ligand_smiles)
            fused_parts.append(self.ligand_context_projector(ligand_context))

        if self.local_gnn is not None:
            fused_parts.append(self._encode_local_branch(complex_data))

        fused = fused_parts[0] if len(fused_parts) == 1 else torch.cat(fused_parts, dim=-1)
        return self.head(fused).view(-1)
