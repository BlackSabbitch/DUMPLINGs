from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

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
    default for backwards compatibility. The current research ladder is:

    - `A1`: one global graph encoder plus optional protein/ligand context
    - `A2`: A1 + explicit local geometric branch
    - `A3`: A2 + linear combination of branch-level scalar outputs
    """

    return str(config.get("model", {}).get("selected", "A1"))


def _parse_bool_override(name: str, raw_value: str) -> bool:
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean override for {name}: {raw_value!r}. "
        f"Use one of: 1/0, true/false, yes/no, on/off."
    )


def get_a3_mixer_bias(config: dict, override: bool | None = None) -> bool:
    """
    Resolve whether the A3 readout mixer should include an explicit bias.

    Resolution order is intentionally external-first so experiment launchers can
    toggle the setting without churning the main JSON config:

    1. explicit runtime override from the caller,
    2. `DUMPLING_A3_MIXER_BIAS` environment variable,
    3. optional `model.a3.mixer_bias` config entry,
    4. default `True`.
    """

    if override is not None:
        return bool(override)

    env_value = os.environ.get("DUMPLING_A3_MIXER_BIAS")
    if env_value is not None:
        return _parse_bool_override("DUMPLING_A3_MIXER_BIAS", env_value)

    model_a3 = config.get("model", {}).get("a3", {})
    if "mixer_bias" in model_a3:
        return bool(model_a3["mixer_bias"])

    return True


def get_global_encoder_mode(config: dict) -> str:
    return str(config["model"]["global_encoder"]["selected"])


def get_global_graph_mode(config: dict) -> str:
    return str(config["model"].get("global_graph", {}).get("selected", "interaction"))


def get_local_graph_mode(config: dict) -> str:
    return str(config["model"].get("local_graph", {}).get("selected", "none"))


def get_local_encoder_mode(config: dict) -> str:
    return str(config["model"].get("local_encoder", {}).get("selected", "none"))


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

    The project intentionally reuses the same geometric primitive for the
    global and local branches so architectural comparisons stay focused on
    *where* information is read from and *how* branches are combined, rather
    than on changes in the message-passing family itself.
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
