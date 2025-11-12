"""
Optional GPU uplift: tiny U-Net training scaffold.

The module deliberately keeps the architecture lightweight so it can run on a
single GPU when available. By default the script skips execution if torch is
missing, while still documenting the training entrypoint.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - GPU optional path
    torch = None
    nn = None
    optim = None

LOGGER = logging.getLogger(__name__)


if nn is not None:

    class TinyUNet(nn.Module):  # type: ignore[misc]
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, out_channels, kernel_size=1),
            )

        def forward(self, x):  # type: ignore[override]
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

else:

    class TinyUNet:  # pragma: no cover - placeholder when torch missing
        pass


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny U-Net for spatial forecasts.")
    parser.add_argument("--data", type=Path, default=Path("data/proc/train.parquet"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--target", type=str, default="t2m")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args(args=args)


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if torch is None:
        LOGGER.warning("Torch not available; skipping U-Net training.")
        return

    cfg = parse_args(args)
    df = pd.read_parquet(cfg.data)

    spatial_dims = sorted(df[["lat", "lon"]].drop_duplicates().values.tolist())
    raise NotImplementedError(
        "Implement tensor conversion from tabular parquet before running U-Net training."
    )


if __name__ == "__main__":
    main()
