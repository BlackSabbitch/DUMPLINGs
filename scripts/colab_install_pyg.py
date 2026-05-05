#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import textwrap

import torch


def run(cmd: str) -> None:
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/content/.mplconfig")
    os.environ.setdefault("TMPDIR", "/content/.tmp")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    torch_base = torch.__version__.split("+")[0]
    cuda_tag = f"cu{torch.version.cuda.replace('.', '')}" if torch.version.cuda else "cpu"
    pyg_url = f"https://data.pyg.org/whl/torch-{torch_base}+{cuda_tag}.html"
    print("PyG wheel index:", pyg_url)

    run("pip uninstall -y pyg-lib torch-scatter torch-sparse torch-cluster torch-geometric || true")
    run(f"pip install pyg-lib torch-scatter torch-sparse torch-cluster -f {pyg_url}")
    run("pip install torch-geometric")

    import torch_sparse

    print("torch_sparse:", torch_sparse.__file__)
    if torch.cuda.is_available():
        row = torch.tensor([0, 1], device="cuda")
        col = torch.tensor([1, 0], device="cuda")
        sp = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=(2, 2))
        print("torch_sparse CUDA check OK:", sp.device())
    else:
        print("CUDA is not available; skipped torch_sparse CUDA smoke test.")

    print(textwrap.dedent("""
    Recommended Colab workflow:
    1. Keep the repo and training outputs in /content while running.
    2. Copy final runs/ and cached artifacts back to Drive after training.
    3. Avoid heavy read/write loops directly on mounted Google Drive.
    """))


if __name__ == "__main__":
    main()
