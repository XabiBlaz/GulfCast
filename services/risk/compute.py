"""
Risk translation utilities.

This module maps model forecast tiles to asset-level hazard summaries and then
converts the hazards into losses via simple parametric curves. The outputs
include Expected Loss (EL), Value-at-Risk (95%), and Expected Shortfall (95%).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import yaml

from services.risk.metrics import expected_shortfall, value_at_risk
from services.risk.physrisk_adapter import run_physrisk
from services.risk.loss_models import (
    HazardSummary,
    heat_damage_fractions,
    precip_damage_fractions,
    fractions_to_currency,
)

LOGGER = logging.getLogger(__name__)

@dataclass
class Asset:
    id: str
    name: str
    lat: float
    lon: float
    value: float
    asset_type: str
    extra: Dict[str, str]


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute asset risk metrics from forecasts.")
    parser.add_argument("--portfolio", type=str, required=True, help="Comma separated list of asset CSV files.")
    parser.add_argument("--forecasts", type=Path, required=True, help="Parquet file with grid forecasts.")
    parser.add_argument("--curves", type=Path, default=Path("services/risk/curves.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/risk/latest.parquet"))
    parser.add_argument(
        "--method",
        type=str,
        default="parametric",
        choices=["parametric", "physrisk", "climada"],
        help="Risk method.",
    )
    parser.add_argument("--risk-config", type=Path, default=Path("config/risk.yaml"), help="Risk configuration YAML.")
    return parser.parse_args(args=args)


def load_assets(paths: Iterable[Path]) -> List[Asset]:
    assets: List[Asset] = []
    for path in paths:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            extra = {col: row[col] for col in df.columns if col not in {"id", "name", "lat", "lon", "value", "type"}}
            raw_value = row.get("value")
            if pd.isna(raw_value):
                fallback_capacity = row.get("capacity_mw", 1.0)
                raw_value = float(fallback_capacity) * 1_000
            assets.append(
                Asset(
                    id=str(row.get("id", row.get("name"))),
                    name=row.get("name", row.get("id", "asset")),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    value=float(raw_value),
                    asset_type=str(row.get("type", row.get("primary_fuel", "generic"))).lower(),
                    extra=extra,
                )
            )
    return assets


def load_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    expected_cols = {"lat", "lon", "variable", "mean", "q90"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Forecast parquet missing required columns: {missing}")
    return df


def nearest_hazard(asset: Asset, df: pd.DataFrame, variable: str) -> HazardSummary:
    subset = df[df["variable"] == variable]
    if subset.empty:
        raise ValueError(f"No forecast entries found for variable {variable}")

    distances = (subset["lat"] - asset.lat) ** 2 + (subset["lon"] - asset.lon) ** 2
    idx = distances.idxmin()
    row = subset.loc[idx]
    return HazardSummary(mean=float(row["mean"]), q90=float(row["q90"]))


def load_curves(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data


def load_risk_config(path: Path) -> Dict[str, object]:
    """Load risk configuration YAML or return defaults when missing."""
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            try:
                return yaml.safe_load(fh) or {}
            except Exception:  # pragma: no cover
                LOGGER.warning("Failed to parse %s; using defaults", path)
    return {
        "risk_method": "parametric",
        "hazards": ["heat", "drought", "flood"],
        "scenario": {"delta_T_C": 0.0, "flood_return_period": 20, "drought_scale": 1.0},
        "metrics": {"var_alpha": 0.95, "es_alpha": 0.95},
    }


def precip_loss(hazard: HazardSummary, asset: Asset, params: Dict[str, float]) -> Dict[str, float]:
    fractions = precip_damage_fractions(hazard, params)
    losses = fractions_to_currency(asset.value, fractions)
    return {
        "hazard_mean": hazard.mean,
        "hazard_q90": hazard.q90,
        **losses,
    }


def heat_loss(hazard: HazardSummary, asset: Asset, params: Dict[str, float]) -> Dict[str, float]:
    fractions = heat_damage_fractions(hazard, asset, params)
    losses = fractions_to_currency(asset.value, fractions)
    return {
        "hazard_mean": hazard.mean,
        "hazard_q90": hazard.q90,
        **losses,
    }


def _compute_parametric(
    assets: List[Asset], forecasts: pd.DataFrame, curves: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    rows = []
    for asset in assets:
        precip_params = curves["precip"].get(asset.asset_type, curves["precip"]["default"])
        heat_params = curves["heat"].get(asset.asset_type, curves["heat"]["default"])

        precip_hazard = nearest_hazard(asset, forecasts, "precip")
        heat_hazard = nearest_hazard(asset, forecasts, "t2m")

        precip_loss_metrics = precip_loss(precip_hazard, asset, precip_params)
        heat_loss_metrics = heat_loss(heat_hazard, asset, heat_params)

        total_el = precip_loss_metrics["EL"] + heat_loss_metrics["EL"]
        total_var = precip_loss_metrics["VaR95"] + heat_loss_metrics["VaR95"]
        total_es = precip_loss_metrics["ES95"] + heat_loss_metrics["ES95"]

        rows.append(
            {
                "asset_id": asset.id,
                "asset_name": asset.name,
                "asset_type": asset.asset_type,
                "lat": asset.lat,
                "lon": asset.lon,
                "precip_hazard_mean": precip_loss_metrics["hazard_mean"],
                "precip_hazard_q90": precip_loss_metrics["hazard_q90"],
                "heat_hazard_mean": heat_loss_metrics["hazard_mean"],
                "heat_hazard_q90": heat_loss_metrics["hazard_q90"],
                "EL": total_el,
                "VaR95": total_var,
                "ES95": total_es,
            }
        )
    df = pd.DataFrame(rows)
    df["risk_method"] = "parametric"
    df["hazards_used"] = "heat,precip"
    return df


def _compute_physrisk(
    assets: List[Asset], forecasts: pd.DataFrame, config: Dict[str, object]
) -> pd.DataFrame:
    return run_physrisk(assets, forecasts, config)


def compute_risk(
    assets: List[Asset],
    forecasts: pd.DataFrame,
    curves: Dict[str, Dict[str, Dict[str, float]]],
    method: str | None = None,
    config: Dict[str, object] | None = None,
) -> pd.DataFrame:
    """Dispatch to configured risk method; defaults to parametric."""
    method = method or "parametric"
    if method in {"climada", "physrisk", "os-climate"}:
        return _compute_physrisk(assets, forecasts, config or {})
    return _compute_parametric(assets, forecasts, curves)


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ns = parse_args(args)
    asset_paths = [Path(p.strip()) for p in ns.portfolio.split(",") if p.strip()]
    assets = load_assets(asset_paths)
    forecasts = load_forecasts(ns.forecasts)
    curves = load_curves(ns.curves)
    cfg = load_risk_config(ns.risk_config)

    df = compute_risk(assets, forecasts, curves, method=ns.method or cfg.get("risk_method", "parametric"), config=cfg)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ns.out, index=False)
    LOGGER.info("Wrote risk report to %s", ns.out)


if __name__ == "__main__":
    main()
