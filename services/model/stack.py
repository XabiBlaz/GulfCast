"""
Stacking utilities to combine baseline model outputs with optional deep models.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

LOGGER = logging.getLogger(__name__)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend model outputs via stacking.")
    parser.add_argument("--data", type=Path, default=Path("data/proc/val.parquet"))
    parser.add_argument("--models", type=str, required=True, help="Comma separated list of model joblib paths.")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--name", type=str, default="stack")
    return parser.parse_args(args=args)


def load_models(model_paths: List[Path]):
    return [joblib.load(path) for path in model_paths]


def run(cfg: argparse.Namespace) -> None:
    df = pd.read_parquet(cfg.data)
    features = [col for col in df.columns if col not in {"week_start", "lat", "lon", cfg.target}]
    X_val = df[features].values
    y_val = df[cfg.target].values

    model_paths = [Path(p.strip()) for p in cfg.models.split(",") if p.strip()]
    models = load_models(model_paths)

    preds = [model.predict(X_val) for model in models]
    stack_X = np.column_stack(preds)
    stack_model = LinearRegression().fit(stack_X, y_val)

    cfg.out.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out / f"{cfg.target}_{cfg.name}.joblib"
    joblib.dump(stack_model, out_path)

    metrics = {
        "mse": float(np.mean((stack_model.predict(stack_X) - y_val) ** 2)),
        "name": cfg.name,
        "models": [str(path) for path in model_paths],
    }
    (cfg.out / f"{cfg.target}_{cfg.name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Wrote stacked model to %s", out_path)


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = parse_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
