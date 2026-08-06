"""
Hardware selection for inference.

Priority: CUDA -> Apple MPS (with a sanity probe and CPU fallback) -> CPU.
"""

from __future__ import annotations

import torch


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        try:
            # Sanity-probe basic MPS ops so a broken driver falls back to CPU
            probe = torch.randn(1, 3, 64, 64, device="mps")
            (probe * 2.0).sum().item()
            return torch.device("mps")
        except Exception:
            pass

    return torch.device("cpu")
