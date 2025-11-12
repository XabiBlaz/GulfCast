"""
Batch scoring utility to create grid-level forecasts for the API and risk layer.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score trained models and export parquet forecasts.")
    parser.add_argument("--data", type=Path, default=Path("data/proc/test.parquet"))
    parser.add_argument("--out", type=Path, default=Path("models/forecasts_latest.parquet"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--targets", type=str, default="t2m,precip")
    parser.add_argument("--quantile", type=float, default=0.9)
    return parser.parse_args(args=args)


def feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    discard = {"week_start", "lat", "lon", target}
    return [col for col in df.columns if col not in discard]


def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def score_target(df: pd.DataFrame, feature_cols: List[str], models_dir: Path, target: str, quantile: float) -> pd.DataFrame:
    X = df[feature_cols].values
    base_predictions: Dict[str, np.ndarray] = {}

    lr_model = load_model(models_dir / f"{target}_lr.joblib")
    if lr_model is not None:
        base_predictions["lr"] = lr_model.predict(X)

    rf_model = load_model(models_dir / f"{target}_rf.joblib")
    if rf_model is not None:
        base_predictions["rf"] = rf_model.predict(X)

    stack_model = load_model(models_dir / f"{target}_stack.joblib")

    if stack_model is not None and len(base_predictions) >= 2:
        stack_inputs = np.column_stack([base_predictions[name] for name in base_predictions])
        mean_pred = stack_model.predict(stack_inputs)
        mean_model_name = "stack"
    elif "rf" in base_predictions:
        mean_pred = base_predictions["rf"]
        mean_model_name = "rf"
    elif "lr" in base_predictions:
        mean_pred = base_predictions["lr"]
        mean_model_name = "lr"
    else:
        raise RuntimeError(f"No mean model available for target {target}")

    gbr_model = load_model(models_dir / f"{target}_gbr_q{quantile:.2f}.joblib")
    if gbr_model is None:
        raise RuntimeError(f"No quantile model found for {target} at q={quantile}")
    q_pred = gbr_model.predict(X)

    result = df[["week_start", "lat", "lon"]].copy()
    result["variable"] = target
    result["mean"] = mean_pred
    result["q90"] = q_pred
    result["mean_model"] = mean_model_name
    result["quantile_model"] = f"gbr_q{quantile:.2f}"
    return result


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ns = parse_args(args)
    df = pd.read_parquet(ns.data)
    df = df.dropna(subset=["lat", "lon"])

    targets = [t.strip() for t in ns.targets.split(",") if t.strip()]

    forecasts = []
    for target in targets:
        features = feature_columns(df, target)
        LOGGER.info("Scoring %s with %s features", target, len(features))
        forecasts.append(score_target(df, features, ns.models_dir, target, ns.quantile))

    forecast_df = pd.concat(forecasts, ignore_index=True)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_parquet(ns.out, index=False)
    LOGGER.info("Wrote forecast parquet to %s", ns.out)


if __name__ == "__main__":
    main()
