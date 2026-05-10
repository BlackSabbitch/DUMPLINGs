from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.a2 import A2DimeNet
from models.graph_components import get_a3_mixer_bias


class A3DimeNet(A2DimeNet):
    """
    A3 model: explicit coarse estimate plus explicit local correction estimate.

    Relative to A2, the global and local branches are left untouched. Only the
    final readout changes: each branch produces its own scalar prediction, and
    the model learns how strongly to weight both contributions.

    Current form:

    - `y_global = global_head(h_global)`
    - `y_local = local_head(h_local)`
    - `y = alpha * y_global + beta * y_local + gamma`

    The current revision keeps the linear coarse/local form but now exposes an
    explicit mixer bias `gamma`. This makes the offset term observable rather
    than forcing the local branch to impersonate a constant shift.
    """

    def __init__(
        self,
        config: dict,
        device: str,
        out_channels: int = 1,
        mixer_bias: bool | None = None,
    ) -> None:
        super().__init__(config=config, device=device, out_channels=out_channels)
        if out_channels != 1:
            raise ValueError("A3DimeNet currently supports out_channels=1 only.")
        if self.local_gnn is None or self.local_encoder_cfg is None:
            raise ValueError("A3DimeNet requires an active local branch.")

        self.mixer_bias = get_a3_mixer_bias(config, override=mixer_bias)
        self.head = self._build_head(out_channels=out_channels)

    def _build_global_head(self, out_channels: int = 1) -> nn.Module:
        """Build the scalar head that maps the coarse global branch to `y_global`."""
        global_in_dim = self._infer_global_head_input_dim()
        global_hidden = max(self.global_encoder_cfg.hidden_channels // 2, 1)
        return nn.Sequential(
            nn.Linear(global_in_dim, global_hidden),
            nn.SiLU(),
            nn.Linear(global_hidden, out_channels),
        )

    def _build_local_head(self, out_channels: int = 1) -> nn.Module:
        """Build the scalar head that maps the local branch to `y_local`."""
        local_in_dim = self.local_encoder_cfg.hidden_channels
        local_hidden = max(local_in_dim // 2, 1)
        return nn.Sequential(
            nn.Linear(local_in_dim, local_hidden),
            nn.SiLU(),
            nn.Linear(local_hidden, out_channels),
        )

    def _build_output_mixer(self, out_channels: int = 1) -> nn.Module:
        """
        Build the final linear mixer over branch-level scalar outputs.

        The mixer remains intentionally tiny and interpretable: two branch-level
        scalar inputs plus one explicit bias term.
        """
        return nn.Linear(2, out_channels, bias=self.mixer_bias)

    def _build_head(self, input_dim: int | None = None, out_channels: int = 1) -> nn.ModuleDict:
        """
        Assemble the full A3 readout block.

        A `ModuleDict` is used here because the "head" is no longer one simple
        MLP: it is a structured collection of semantically different subheads.
        """
        del input_dim
        return nn.ModuleDict(
            {
                "global": self._build_global_head(out_channels=out_channels),
                "local": self._build_local_head(out_channels=out_channels),
                "mixer": self._build_output_mixer(out_channels=out_channels),
            }
        )

    def _compute_branch_outputs(self, complex_data) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-complex scalar outputs of the coarse and local branches."""
        global_repr = self._encode_global_representation(complex_data)
        local_repr = self._encode_local_branch(complex_data)
        global_pred = self.head["global"](global_repr).view(-1)
        local_pred = self.head["local"](local_repr).view(-1)
        return global_pred, local_pred

    def get_history_payload(self) -> dict[str, float]:
        """
        Return per-epoch mixer coefficients for `history.json`.

        This keeps the training history aligned with the final test diagnostics:
        we can later inspect not only the best checkpoint's readout, but how the
        coarse/local mixture evolved during optimization.
        """
        mixer = self.head["mixer"]
        payload = {
            "alpha": float(mixer.weight[0, 0].item()),
            "beta": float(mixer.weight[0, 1].item()),
        }
        if mixer.bias is not None:
            payload["gamma"] = float(mixer.bias[0].item())
        return payload

    @staticmethod
    def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
        """Summarize one diagnostic tensor for JSON export."""
        flat = values.detach().view(-1).float()
        if flat.numel() == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "mean_abs": float("nan"),
            }

        return {
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "mean_abs": float(flat.abs().mean().item()),
        }

    def get_test_result_payload(
        self,
        loader,
        device: str,
        progress: float = 1.0,
    ) -> dict[str, Any]:
        """
        Export A3-specific diagnostics for `test_results.json`.

        These diagnostics are intentionally verbose in the current research
        phase. They help answer questions such as:

        - did one branch collapse?
        - is the local term really a correction rather than the dominant term?
        - how large are the effective contributions after weighting by the mixer?
        """
        self.eval()

        global_outputs: list[torch.Tensor] = []
        local_outputs: list[torch.Tensor] = []
        global_contribs: list[torch.Tensor] = []
        local_contribs: list[torch.Tensor] = []
        abs_ratio_values: list[torch.Tensor] = []

        mixer = self.head["mixer"]
        alpha = float(mixer.weight[0, 0].item())
        beta = float(mixer.weight[0, 1].item())
        gamma = float(mixer.bias[0].item()) if mixer.bias is not None else None

        with torch.no_grad():
            for batch in loader:
                batch = [
                    inp.to(device) if hasattr(inp, "to")
                    else {k: v.to(device) for k, v in inp.items()}
                    for inp in batch
                ]
                _, _, _, complex_data, _ = tuple(batch)
                global_pred, local_pred = self._compute_branch_outputs(complex_data)
                global_contrib = alpha * global_pred
                local_contrib = beta * local_pred

                global_outputs.append(global_pred.cpu())
                local_outputs.append(local_pred.cpu())
                global_contribs.append(global_contrib.cpu())
                local_contribs.append(local_contrib.cpu())
                # Ratio is tracked per sample so later analysis can ask whether
                # the local term is occasionally dominant even if its global
                # average looks modest.
                abs_ratio_values.append(local_contrib.abs().cpu() / (global_contrib.abs().cpu() + 1e-12))

        global_output_tensor = torch.cat(global_outputs, dim=0) if global_outputs else torch.empty(0)
        local_output_tensor = torch.cat(local_outputs, dim=0) if local_outputs else torch.empty(0)
        global_contrib_tensor = torch.cat(global_contribs, dim=0) if global_contribs else torch.empty(0)
        local_contrib_tensor = torch.cat(local_contribs, dim=0) if local_contribs else torch.empty(0)
        abs_ratio_tensor = torch.cat(abs_ratio_values, dim=0) if abs_ratio_values else torch.empty(0)

        mean_abs_global = float(global_contrib_tensor.abs().mean().item()) if global_contrib_tensor.numel() > 0 else float("nan")
        mean_abs_local = float(local_contrib_tensor.abs().mean().item()) if local_contrib_tensor.numel() > 0 else float("nan")
        aggregate_ratio = mean_abs_local / (mean_abs_global + 1e-12) if global_contrib_tensor.numel() > 0 else float("nan")

        return {
            "readout_diagnostics": {
                "type": "a3_linear_mixture",
                "mixer_has_bias": mixer.bias is not None,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "alpha_stats": self._tensor_stats(torch.tensor([alpha], dtype=torch.float32)),
                "beta_stats": self._tensor_stats(torch.tensor([beta], dtype=torch.float32)),
                "gamma_stats": self._tensor_stats(torch.tensor([gamma], dtype=torch.float32)) if gamma is not None else None,
                "global_branch_output": self._tensor_stats(global_output_tensor),
                "local_branch_output": self._tensor_stats(local_output_tensor),
                "global_contribution": self._tensor_stats(global_contrib_tensor),
                "local_contribution": self._tensor_stats(local_contrib_tensor),
                "local_to_global_abs_contribution_ratio": aggregate_ratio,
                "local_to_global_abs_contribution_ratio_stats": self._tensor_stats(abs_ratio_tensor),
            }
        }

    def forward(self, batch_list, progress: float = 0.0):
        """
        Predict affinity as a weighted sum of coarse and local scalar terms.

        This is the main architectural distinction of A3 relative to A2.
        """
        _, _, _, complex_data, _ = batch_list

        global_pred, local_pred = self._compute_branch_outputs(complex_data)
        global_pred = global_pred.view(-1, 1)
        local_pred = local_pred.view(-1, 1)
        branch_preds = torch.cat([global_pred, local_pred], dim=-1)
        return self.head["mixer"](branch_preds).view(-1)
