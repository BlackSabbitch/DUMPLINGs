from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

try:
    import pennylane as qml
except ImportError:  # pragma: no cover - exercised only in missing-dependency environments
    qml = None


def _build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation for VQC head: {name!r}")


def _normalize_hidden_layers(hidden_layers: Iterable[int] | int | None) -> list[int]:
    if hidden_layers is None:
        return []
    if isinstance(hidden_layers, int):
        return [int(hidden_layers)] if int(hidden_layers) > 0 else []
    normalized: list[int] = []
    for width in hidden_layers:
        width = int(width)
        if width > 0:
            normalized.append(width)
    return normalized


def _build_mlp(
    input_dim: int,
    hidden_layers: Iterable[int] | int | None,
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for width in _normalize_hidden_layers(hidden_layers):
        layers.append(nn.Linear(prev_dim, width))
        layers.append(_build_activation(activation))
        prev_dim = width
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class QuantumReUploadingLayer(nn.Module):
    """
    Small PennyLane-backed re-uploading layer for hybrid regression heads.

    The layer intentionally keeps the stabilizing ideas from earlier quantum
    experiments:

    - inputs are projected down to a small qubit count,
    - they are squashed and scaled before angle embedding,
    - the usable angle range grows gradually with training progress.

    What we deliberately do *not* keep is the old multi-block config surface:
    for the DUMPLINGs A2 experiments, this module is just one optional readout
    component, not a separate model family.
    """

    def __init__(
        self,
        in_dim: int,
        n_layers: int,
        *,
        backend: str = "default.qubit",
        rotation: str = "X",
        initial_rotation: str = "Y",
        entanglement: str = "strongly_entangling",
        input_scale: float = 0.01,
        start_scale: float = math.pi / 6.0,
        end_scale: float = math.pi,
    ) -> None:
        super().__init__()
        if qml is None:
            raise ImportError(
                "PennyLane is required for the VQC head but is not installed. "
                "Install `pennylane` or switch model.head.selected back to 'mlp'."
            )
        if in_dim <= 0:
            raise ValueError("QuantumReUploadingLayer requires in_dim > 0.")
        if n_layers <= 0:
            raise ValueError("QuantumReUploadingLayer requires n_layers > 0.")

        entanglement_mode = str(entanglement).strip().lower()
        if entanglement_mode not in {"strongly_entangling", "full"}:
            raise ValueError(
                f"Unsupported entanglement mode {entanglement!r}. "
                "Use 'strongly_entangling' (or legacy alias 'full')."
            )

        self.in_dim = int(in_dim)
        self.n_layers = int(n_layers)
        self.backend = str(backend)
        self.rotation = str(rotation)
        self.initial_rotation = str(initial_rotation)
        self.input_scale = float(input_scale)
        self.scale_start = float(start_scale)
        self.scale_end = float(end_scale)

        device = qml.device(self.backend, wires=self.in_dim)

        @qml.qnode(device, interface="torch")
        def circuit(
            inputs: torch.Tensor,
            entangling_weights: torch.Tensor,
            embedding_weights: torch.Tensor,
        ):
            qml.AngleEmbedding(
                features=inputs,
                wires=range(self.in_dim),
                rotation=self.initial_rotation,
            )
            for layer_idx in range(self.n_layers):
                qml.StronglyEntanglingLayers(
                    entangling_weights[layer_idx],
                    wires=range(self.in_dim),
                )
                qml.AngleEmbedding(
                    features=inputs * embedding_weights[layer_idx],
                    wires=range(self.in_dim),
                    rotation=self.rotation,
                )
            qml.StronglyEntanglingLayers(
                entangling_weights[-1],
                wires=range(self.in_dim),
            )
            return [qml.expval(qml.PauliZ(wire)) for wire in range(self.in_dim)]

        weight_shapes = {
            "entangling_weights": (self.n_layers + 1, 1, self.in_dim, 3),
            "embedding_weights": (self.n_layers, self.in_dim),
        }
        self.torch_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def current_scale(self, progress: float) -> float:
        progress = max(0.0, min(float(progress), 1.0))
        return self.scale_start + (self.scale_end - self.scale_start) * progress

    def forward(self, x: torch.Tensor, progress: float = 0.0) -> torch.Tensor:
        scale = self.current_scale(progress)
        x = x.float() * self.input_scale
        x = torch.tanh(x) * scale
        return self.torch_layer(x)


class VQCHead(nn.Module):
    """
    Hybrid A2 readout: classical bottleneck -> quantum layer -> scalar readout.

    The adapter remains classical on purpose. It performs the heavy dimension
    reduction from the fused A2 representation down to a small qubit count,
    while the VQC is used as a structured nonlinear readout rather than as a
    replacement for the whole model.
    """

    def __init__(
        self,
        input_dim: int,
        out_channels: int = 1,
        *,
        adapter_hidden_layers: Iterable[int] | int | None = None,
        adapter_activation: str = "Tanh",
        n_qubits: int = 6,
        n_layers: int = 2,
        backend: str = "default.qubit",
        rotation: str = "X",
        initial_rotation: str = "Y",
        entanglement: str = "strongly_entangling",
        input_scale: float = 0.01,
        start_scale: float = math.pi / 6.0,
        end_scale: float = math.pi,
        readout_hidden_dim: int | None = 16,
        readout_activation: str = "Tanh",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)

        if self.n_qubits <= 0:
            raise ValueError("VQCHead requires n_qubits > 0.")

        if adapter_hidden_layers is None:
            adapter_hidden_layers = [128, 64]

        self.adapter = _build_mlp(
            input_dim=self.input_dim,
            hidden_layers=adapter_hidden_layers,
            output_dim=self.n_qubits,
            activation=adapter_activation,
        )
        self.qlayer = QuantumReUploadingLayer(
            in_dim=self.n_qubits,
            n_layers=self.n_layers,
            backend=backend,
            rotation=rotation,
            initial_rotation=initial_rotation,
            entanglement=entanglement,
            input_scale=input_scale,
            start_scale=start_scale,
            end_scale=end_scale,
        )
        if readout_hidden_dim is None or int(readout_hidden_dim) <= 0:
            self.final_layer = nn.Linear(self.n_qubits, out_channels)
        else:
            self.final_layer = nn.Sequential(
                nn.Linear(self.n_qubits, int(readout_hidden_dim)),
                _build_activation(readout_activation),
                nn.Linear(int(readout_hidden_dim), out_channels),
            )

    def forward(self, x: torch.Tensor, progress: float = 0.0) -> torch.Tensor:
        x = self.adapter(x)
        x = self.qlayer(x, progress=progress)
        return self.final_layer(x)
