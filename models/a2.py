from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.a1 import A1DimeNet
from logger import log_info, log_warn
from models.graph_components import (
    build_dimenet_backbone,
    get_global_encoder_config,
    get_local_encoder_config,
    get_local_graph_config,
    get_local_graph_mode,
)


class A2DimeNet(A1DimeNet):
    """
    A2 model: A1-style coarse branch plus an explicit local geometric branch.

    Genealogy:
    - inherits A1's global geometry/context fusion contract,
    - keeps the same protein and ligand context branches,
    - adds a second geometric branch operating on an interaction-focused
      ligand-pocket subgraph,
    - concatenates the coarse global representation with the local
      representation before the final readout.

    The first revision intentionally keeps the local branch simple:
    radius-based node selection plus a second DimeNet++ encoder. Importance-
    based local weighting is left for later A2/A3a iterations, but the model
    already exposes clean seams for that expansion.
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
        self.local_chemical_cfg = dict(config.get("model", {}).get("local_chemical_features", {}))
        self.local_chemical_enabled = bool(self.local_chemical_cfg.get("enabled", False))
        self.local_chemical_projector = None
        self.local_chemical_gate = None
        if self.local_encoder_cfg is not None and self.local_graph_mode != "none":
            self.local_gnn = build_dimenet_backbone(self.local_encoder_cfg)
            self.local_cutoff = float(self.local_graph_cfg.get("dist_threshold", 3.5))
            self.local_output_norm = nn.LayerNorm(self.local_encoder_cfg.hidden_channels)
            if self.local_chemical_enabled:
                self.local_chemical_projector = nn.Sequential(
                    nn.LazyLinear(self.local_encoder_cfg.hidden_channels),
                    nn.SiLU(),
                    nn.Linear(self.local_encoder_cfg.hidden_channels, self.local_encoder_cfg.hidden_channels),
                )
                self.local_chemical_gate = nn.Parameter(torch.zeros(1))
        self.local_nonfinite_warned_ids: set[str] = set()
        self.local_guard_activation_count = 0
        self.local_guard_activation_ids: set[str] = set()

        head_in_dim = self._infer_head_input_dim()
        self.head = A2DimeNet._build_head(self, head_in_dim, out_channels)

    def _infer_head_input_dim(self) -> int:
        """
        Return the dimensionality of the concatenated A2 readout input.

        This is the coarse global representation from A1 plus, when enabled,
        the local branch embedding.
        """
        head_in_dim = self._infer_global_head_input_dim()
        if self.local_gnn is not None:
            head_in_dim += self.local_encoder_cfg.hidden_channels
        return head_in_dim

    def _build_head(self, input_dim: int, out_channels: int = 1) -> nn.Module:
        """
        Build the A2 concat-readout head.

        The head itself stays deliberately small; the main architectural change
        in A2 is the existence of the local branch, not a more expressive MLP.
        """
        hidden_dim = max(self.global_encoder_cfg.hidden_channels // 2, 1)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    @staticmethod
    def _select_local_mask_for_graph(
        graph_x: torch.Tensor,
        graph_pos: torch.Tensor,
        local_cutoff: float,
    ) -> torch.Tensor:
        """
        Select the current A2 local subgraph by ligand-centered radius filtering.

        The mask always keeps ligand atoms. Pocket atoms are kept if they lie
        within `local_cutoff` of at least one ligand atom. This produces a
        coarse interaction zone rather than a sparse interaction explanation;
        later pair-scoring experiments are expected to refine this selection.
        """
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
        """Return a neutral local embedding for failed or degenerate subgraphs."""
        return torch.zeros(self.local_encoder_cfg.hidden_channels, device=device, dtype=torch.float32)

    def _summarize_local_chemical_features(
        self,
        graph_local_chemical_x: torch.Tensor | None,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.local_chemical_enabled or self.local_chemical_projector is None:
            return None
        if graph_local_chemical_x is None or graph_local_chemical_x.numel() == 0:
            return None
        selected_features = graph_local_chemical_x[selected_mask]
        if selected_features.numel() == 0:
            return None
        summary = selected_features.float().mean(dim=0, keepdim=True)
        return self.local_chemical_projector(summary).view(-1)

    def get_local_guard_summary(self) -> dict[str, object]:
        """Expose how often the temporary local non-finite fallback was used."""
        return {
            "activations": self.local_guard_activation_count,
            "pdb_ids": sorted(self.local_guard_activation_ids),
        }

    def log_local_guard_summary(self) -> None:
        """
        Emit one post-run summary for the temporary local-branch safeguard.

        This is mainly an experiment-debugging hook. Once the problematic
        complexes and numeric pathologies are understood well enough, this guard
        can likely be simplified or removed.
        """
        summary = self.get_local_guard_summary()
        activations = int(summary["activations"])
        pdb_ids = summary["pdb_ids"]
        if activations == 0:
            log_info(
                "Local branch guard summary -> no non-finite local fallback activations observed.",
                stage="MODEL"
            )
            return

        log_warn(
            f"Local branch guard summary -> activations={activations}, affected_pdb_ids={pdb_ids}",
            stage="MODEL"
        )

    def _encode_local_branch(self, complex_data) -> torch.Tensor:
        """
        Encode one local representation per complex in the current batch.

        The method deliberately operates per-complex rather than on one mixed
        local mega-batch so that:

        - each complex can define its own local subgraph,
        - numerical issues can be attributed back to a specific PDB id,
        - future pair-scoring or motif-scoring logic has a clean place to hook in.
        """
        batch = complex_data.batch
        x = complex_data.x
        pos = complex_data.pos.float()
        local_chemical_x = getattr(complex_data, "local_chemical_x", None)
        outputs: list[torch.Tensor] = []

        unique_batches = torch.unique(batch, sorted=True)
        for batch_id in unique_batches.tolist():
            graph_mask = batch == batch_id
            graph_x = x[graph_mask]
            graph_pos = pos[graph_mask]
            graph_local_chemical_x = None
            if local_chemical_x is not None:
                graph_local_chemical_x = local_chemical_x[graph_mask]
            selected = self._select_local_mask_for_graph(graph_x, graph_pos, self.local_cutoff)
            local_pos = self._stabilize_local_coordinates(graph_pos[selected])
            local_z = graph_x[selected, 0].long()

            if local_pos.size(0) < 2:
                outputs.append(self._safe_local_zero(device=graph_pos.device))
                continue

            # The local encoder sees one per-complex local graph at a time.
            local_batch = torch.zeros(local_pos.size(0), dtype=batch.dtype, device=batch.device)
            local_repr = self.local_gnn(local_z, local_pos, local_batch).view(-1)
            local_chem_repr = self._summarize_local_chemical_features(graph_local_chemical_x, selected)
            if local_chem_repr is not None:
                local_repr = local_repr + self.local_chemical_gate * local_chem_repr

            if not torch.isfinite(local_repr).all():
                pdb_id = self._graph_pdb_id(complex_data, batch_id)
                self.local_guard_activation_count += 1
                self.local_guard_activation_ids.add(pdb_id)
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
        """
        Predict affinity from a coarse global branch plus a local correction branch.

        In A2 the two branches are still merged by simple concatenation. A3
        keeps the same branch encoders but replaces this final combination rule
        with a more explicitly interpretable linear mixture of branch-level
        outputs.
        """
        _, _, _, complex_data, _ = batch_list

        fused_parts: list[torch.Tensor] = [self._encode_global_representation(complex_data)]
        if self.local_gnn is not None:
            fused_parts.append(self._encode_local_branch(complex_data))

        fused = fused_parts[0] if len(fused_parts) == 1 else torch.cat(fused_parts, dim=-1)
        return self.head(fused).view(-1)
