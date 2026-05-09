from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn
from torch_geometric.nn import DimeNetPlusPlus


@dataclass(frozen=True)
class EncoderConfig:
    """
    Normalized encoder configuration used by A1 and A2 graph branches.

    The project-level JSON keeps parser-facing and model-facing parameters in
    one place for the global branch. This helper converts that mixed structure
    into a compact representation the model classes can consume directly.
    """

    name: str
    hidden_channels: int
    cutoff: float
    max_num_neighbors: int
    num_blocks: int


def get_model_family(config: dict) -> str:
    """
    Return the selected high-level model family.

    Older configs did not carry `model.selected`, so A1 remains the implicit
    default for backwards compatibility.
    """

    return str(config.get("model", {}).get("selected", "A1"))


def get_global_encoder_mode(config: dict) -> str:
    return str(config["model"]["global_encoder"]["selected"])


def get_global_graph_mode(config: dict) -> str:
    return str(config["model"].get("global_graph", {}).get("selected", "interaction"))


def get_local_graph_mode(config: dict) -> str:
    return str(config["model"].get("local_graph", {}).get("selected", "none"))


def get_local_encoder_mode(config: dict) -> str:
    return str(config["model"].get("local_encoder", {}).get("selected", "none"))


def get_head_mode(config: dict) -> str:
    return str(config["model"].get("head", {}).get("selected", "global_local_concat"))


def get_global_graph_config(config: dict) -> dict[str, Any]:
    graph_section = config["model"].get("global_graph")
    if graph_section is None:
        legacy_entry = config["model"]["global_encoder"]["available"][get_global_encoder_mode(config)]
        legacy_params = legacy_entry.get("egnn_params", {})
        return {
            "dist_threshold": float(legacy_params.get("dist_threshold", 5.0)),
            "ca_only": bool(legacy_params.get("ca_only", False)),
        }
    mode = get_global_graph_mode(config)
    return dict(graph_section.get("available", {}).get(mode, {}))


def get_global_encoder_config(config: dict) -> EncoderConfig:
    entry = config["model"]["global_encoder"]["available"][get_global_encoder_mode(config)]
    params = entry.get("egnn_params", entry)
    return EncoderConfig(
        name=get_global_encoder_mode(config),
        hidden_channels=int(params.get("hidden_channels", 128)),
        cutoff=float(params.get("cutoff", params.get("dist_threshold", 5.0))),
        max_num_neighbors=int(params.get("max_num_neighbors", 32)),
        num_blocks=int(params.get("num_blocks", 3)),
    )


def get_local_graph_config(config: dict) -> dict[str, Any]:
    mode = get_local_graph_mode(config)
    available = config["model"].get("local_graph", {}).get("available", {})
    return dict(available.get(mode, {}))


def get_local_encoder_config(config: dict) -> EncoderConfig | None:
    mode = get_local_encoder_mode(config)
    if mode == "none":
        return None
    entry = config["model"]["local_encoder"]["available"][mode]
    return EncoderConfig(
        name=mode,
        hidden_channels=int(entry.get("hidden_channels", 128)),
        cutoff=float(entry.get("cutoff", 3.5)),
        max_num_neighbors=int(entry.get("max_num_neighbors", 32)),
        num_blocks=int(entry.get("num_blocks", 3)),
    )


def build_dimenet_backbone(
    encoder_cfg: EncoderConfig,
    out_channels: int | None = None,
) -> DimeNetPlusPlus:
    """
    Build the shared DimeNet++ backbone used by the global and local branches.
    """

    out_dim = encoder_cfg.hidden_channels if out_channels is None else out_channels
    return DimeNetPlusPlus(
        hidden_channels=encoder_cfg.hidden_channels,
        out_channels=out_dim,
        num_blocks=encoder_cfg.num_blocks,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=128,
        num_spherical=7,
        num_radial=6,
        cutoff=encoder_cfg.cutoff,
        max_num_neighbors=encoder_cfg.max_num_neighbors,
        envelope_exponent=5,
    )


def build_head(
    config: dict,
    input_dim: int,
    default_hidden_dim: int,
    out_channels: int = 1,
) -> nn.Module:
    """
    Build the prediction head selected in the config.

    `global_local_concat` is the conservative default used by A1 and the first
    A2 revision. `global_local_linear` is reserved for later linear-combination
    experiments, but it is already available so the config surface stays stable.
    """

    head_mode = get_head_mode(config)
    head_cfg = config["model"].get("head", {}).get("available", {}).get(head_mode, {})

    if head_mode == "global_local_linear":
        return nn.Linear(input_dim, out_channels)

    hidden_dim = int(head_cfg.get("hidden_dim", max(default_hidden_dim // 2, 1)))
    dropout = float(head_cfg.get("dropout", 0.0))
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_channels))
    return nn.Sequential(*layers)
