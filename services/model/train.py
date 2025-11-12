"""
Training entrypoint for tabular baseline models.

This script fits Linear Regression (mean), Quantile Regression (q=0.9),
RandomForest (mean), and GradientBoosting quantile models. It can optionally
assemble a stacking head that blends the base forecasts.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

ENSEMBLE_BASELINE = {
    't2m': 'temperature_ens_mean',
    'precip': 'precipitation_ens_mean',
}

@dataclass
class TrainConfig:
    data_root: Path
    output_root: Path
    models: List[str]
    targets: List[str]
    quantiles: List[float]
    features: str
    positional_encoding: str
    member_order: str
    seed: int = 42


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--data", type=Path, default=Path("data/proc"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--models", type=str, default="lr,rf,stack")
    parser.add_argument("--targets", type=str, default="t2m,precip")
    parser.add_argument("--quantiles", type=str, default="0.9")
    parser.add_argument("--features", type=str, default="ensemble_full")
    parser.add_argument("--posenc", type=str, default="pe")
    parser.add_argument("--member_order", type=str, default="original")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(args=args)


def load_split(root: Path, split: str) -> pd.DataFrame:
    path = root / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Required dataset missing: {path}")
    df = pd.read_parquet(path)
    return df


def feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    drop_cols = {"week_start", "lat", "lon", target}
    columns = [col for col in df.columns if col not in drop_cols]
    return columns


def train_linear_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X, y)
    return model


def train_rf_model(X: np.ndarray, y: np.ndarray, seed: int) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return rf


def train_quantile_model(X: np.ndarray, y: np.ndarray, quantile: float) -> Tuple[Optional[Pipeline], Optional[float]]:
    alphas = [0.0, 1e-6, 1e-4, 1e-2]
    last_error: Optional[Exception] = None
    for alpha in alphas:
        qr = QuantileRegressor(quantile=quantile, alpha=alpha)
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", qr),
            ]
        )
        try:
            pipe.fit(X, y)
            if alpha > 0:
                LOGGER.warning("QuantileRegressor converged after increasing alpha to %s", alpha)
            return pipe, alpha
        except Exception as exc:  # noqa: BLE001 - sklearn can raise different error types
            last_error = exc
            LOGGER.warning("QuantileRegressor failed with alpha=%s: %s", alpha, exc)
    LOGGER.error("QuantileRegressor could not converge for quantile %.2f: %s", quantile, last_error)
    return None, None


def train_quantile_gbr(X: np.ndarray, y: np.ndarray, quantile: float, seed: int) -> GradientBoostingRegressor:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=quantile, random_state=seed, n_estimators=400, max_depth=3)
    gbr.fit(X, y)
    return gbr


def evaluate_model(name: str, model, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    preds = model.predict(X_val)
    metrics = {
        "mse": float(mean_squared_error(y_val, preds)),
        "r2": float(r2_score(y_val, preds)),
        "name": name,
    }
    return metrics


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def train_for_target(cfg: TrainConfig, target: str) -> Dict[str, Dict[str, float]]:
    LOGGER.info("Training models for %s", target)
    train_df = load_split(cfg.data_root, "train")
    val_df = load_split(cfg.data_root, "val")

    dropped_rows = int(train_df[target].isna().sum())
    if dropped_rows:
        LOGGER.warning("Dropping %s rows with NaN target in training set for %s", dropped_rows, target)
        train_df = train_df.dropna(subset=[target])
    val_df = val_df.dropna(subset=[target])

    features = feature_columns(train_df, target)
    LOGGER.info("Using %s features", len(features))

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_val = val_df[features].values
    y_val = val_df[target].values

    metrics: Dict[str, Dict[str, float]] = {}

    models_requested = [m.strip() for m in cfg.models]

    base_predictions = {}

    if "lr" in models_requested:
        lr_model = train_linear_model(X_train, y_train)
        metrics["lr"] = evaluate_model("lr", lr_model, X_val, y_val)
        save_model(lr_model, cfg.output_root / f"{target}_lr.joblib")
        base_predictions["lr"] = lr_model.predict(X_val)

    if "rf" in models_requested:
        rf_model = train_rf_model(X_train, y_train, seed=cfg.seed)
        metrics["rf"] = evaluate_model("rf", rf_model, X_val, y_val)
        save_model(rf_model, cfg.output_root / f"{target}_rf.joblib")
        base_predictions["rf"] = rf_model.predict(X_val)

    for quantile in cfg.quantiles:
        if "lr" in models_requested:
            qr_model, alpha_used = train_quantile_model(X_train, y_train, quantile)
            if qr_model is None:
                LOGGER.warning("Skipping linear quantile model for %s (q=%.2f) due to convergence issues.", target, quantile)
            else:
                model_path = cfg.output_root / f"{target}_lr_q{quantile:.2f}.joblib"
                save_model(qr_model, model_path)
                meta_path = model_path.with_suffix(".meta.json")
                solver_name = qr_model.named_steps["regressor"].get_params().get("solver", "auto")
                meta = {
                    "target": target,
                    "quantile": quantile,
                    "alpha": alpha_used,
                    "solver": solver_name,
                    "dropped_rows": dropped_rows,
                    "n_features": len(features),
                }
                meta_path.write_text(json.dumps(meta, indent=2))
                q_preds = qr_model.predict(X_val)
                metrics[f"lr_q{quantile:.2f}"] = {
                    "pinball": float(mean_pinball_loss(y_val, q_preds, alpha=quantile)),
                    "name": f"lr_q{quantile:.2f}",
                    "alpha": alpha_used,
                    "dropped_rows": dropped_rows,
                }
        gbr_model = train_quantile_gbr(X_train, y_train, quantile, cfg.seed)
        save_model(gbr_model, cfg.output_root / f"{target}_gbr_q{quantile:.2f}.joblib")
        gbr_preds = gbr_model.predict(X_val)
        metrics[f"gbr_q{quantile:.2f}"] = {
            "pinball": float(mean_pinball_loss(y_val, gbr_preds, alpha=quantile)),
            "name": f"gbr_q{quantile:.2f}",
        }

    if "stack" in models_requested and len(base_predictions) >= 2:
        stack_features = np.column_stack(list(base_predictions.values()))
        stack_model = LinearRegression().fit(stack_features, y_val)
        stack_path = cfg.output_root / f"{target}_stack.joblib"
        save_model(stack_model, stack_path)
        preds = stack_model.predict(stack_features)
        metrics["stack"] = {
            "mse": mean_squared_error(y_val, preds),
            "r2": r2_score(y_val, preds),
            "name": "stack",
        }

    baseline_col = ENSEMBLE_BASELINE.get(target)
    if baseline_col and baseline_col in val_df.columns:
        baseline_preds = val_df[baseline_col].values
        metrics["ensemble_mean"] = {
            "mse": float(mean_squared_error(y_val, baseline_preds)),
            "r2": float(r2_score(y_val, baseline_preds)),
            "name": "ensemble_mean",
        }

    metrics_path = cfg.output_root / f"{target}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ns = parse_args(args)
    cfg = TrainConfig(
        data_root=ns.data,
        output_root=ns.out,
        models=[m.strip() for m in ns.models.split(",") if m.strip()],
        targets=[t.strip() for t in ns.targets.split(",") if t.strip()],
        quantiles=[float(q) for q in ns.quantiles.split(",") if q.strip()],
        features=ns.features,
        positional_encoding=ns.posenc,
        member_order=ns.member_order,
        seed=ns.seed,
    )
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for target in cfg.targets:
        all_metrics[target] = train_for_target(cfg, target)

    (cfg.output_root / "training_summary.json").write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
