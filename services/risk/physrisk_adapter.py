"""
OS-Climate physrisk integration for GulfCast.

We build a lightweight hazard model on top of the subseasonal forecast tiles and
reuse the existing parametric loss curves via physrisk impact distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from services.risk.loss_models import (
    HazardSummary,
    heat_damage_fractions,
    precip_damage_fractions,
)
from services.risk.metrics import expected_shortfall, value_at_risk

LOGGER = logging.getLogger(__name__)


def _load_physrisk():
    try:
        from physrisk.kernel.assets import Asset as PhysAsset
        from physrisk.kernel.hazard_model import (
            HazardDataRequest,
            HazardDataResponse,
            HazardModel,
            HazardParameterDataResponse,
        )
        from physrisk.kernel.hazards import AirTemperature, Precipitation
        from physrisk.kernel.impact import calculate_impacts
        from physrisk.kernel.impact_distrib import ImpactDistrib, ImpactType
        from physrisk.kernel.vulnerability_model import (
            DictBasedVulnerabilityModels,
            VulnerabilityModelBase,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "physrisk-lib is not installed. Reinstall the project dependencies with `poetry install` "
            "or add the physrisk extra."
        ) from exc

    return {
        "PhysAsset": PhysAsset,
        "HazardModel": HazardModel,
        "HazardDataRequest": HazardDataRequest,
        "HazardParameterDataResponse": HazardParameterDataResponse,
        "HazardDataResponse": HazardDataResponse,
        "AirTemperature": AirTemperature,
        "Precipitation": Precipitation,
        "ImpactDistrib": ImpactDistrib,
        "ImpactType": ImpactType,
        "VulnerabilityModelBase": VulnerabilityModelBase,
        "DictBasedVulnerabilityModels": DictBasedVulnerabilityModels,
        "calculate_impacts": calculate_impacts,
    }


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series)


def _nearest_row(df: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    distances = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
    idx = distances.idxmin()
    return df.loc[idx]


def run_physrisk(
    assets: Sequence[Any],
    forecasts: pd.DataFrame,
    config: Dict[str, object],
) -> pd.DataFrame:
    """Compute losses via physrisk using subseasonal hazard tiles."""

    mods = _load_physrisk()
    PhysAsset = mods["PhysAsset"]
    HazardModel = mods["HazardModel"]
    HazardDataRequest = mods["HazardDataRequest"]
    HazardParameterDataResponse = mods["HazardParameterDataResponse"]
    AirTemperature = mods["AirTemperature"]
    Precipitation = mods["Precipitation"]
    ImpactDistrib = mods["ImpactDistrib"]
    ImpactType = mods["ImpactType"]
    VulnerabilityModelBase = mods["VulnerabilityModelBase"]
    DictBasedVulnerabilityModels = mods["DictBasedVulnerabilityModels"]
    calculate_impacts = mods["calculate_impacts"]

    physrisk_cfg = (config or {}).get("physrisk", {}) or {}
    mc_samples = int(physrisk_cfg.get("mc_samples", 4000))
    tail_probability = float(physrisk_cfg.get("tail_probability", 0.05))
    rng_seed = int(physrisk_cfg.get("rng_seed", 1234))
    scenario_id = str(physrisk_cfg.get("scenario_id", "subseasonal"))

    if forecasts.empty:
        raise ValueError("Forecast dataframe is empty; cannot run physrisk.")

    week_series = _ensure_datetime(forecasts["week_start"]) if "week_start" in forecasts.columns else None
    target_week = week_series.max() if week_series is not None else None

    class GulfPhysAsset(PhysAsset):
        def __init__(self, src: Any):
            super().__init__(src.lat, src.lon, id=str(src.id))
            self.name = src.name
            self.value = float(src.value)
            self.lat = src.lat
            self.lon = src.lon
            self.asset_type = src.asset_type
            self.extra = getattr(src, "extra", {}) or {}

    class SubseasonalHazardModel(HazardModel):
        def __init__(self, df: pd.DataFrame):
            self.df = df.copy()
            if week_series is not None:
                self.df = self.df.assign(week_start=_ensure_datetime(self.df["week_start"]))
            self.week = target_week

        def _subset(self, variable: str) -> pd.DataFrame:
            df = self.df[self.df["variable"] == variable]
            if self.week is not None and "week_start" in df.columns:
                df = df[df["week_start"] == self.week]
            return df

        def get_hazard_data(self, requests: Sequence[HazardDataRequest]):
            responses: Dict[HazardDataRequest, HazardParameterDataResponse] = {}
            for req in requests:
                variable = "t2m" if req.hazard_type is AirTemperature else "precip"
                subset = self._subset(variable)
                if subset.empty:
                    raise ValueError(f"No forecast data for variable '{variable}' to satisfy physrisk request.")
                row = _nearest_row(subset, req.latitude, req.longitude)
                mean = float(row["mean"])
                q90 = float(row["q90"])
                units = "K" if variable == "t2m" else "mm"
                responses[req] = HazardParameterDataResponse(
                    parameters=np.array([mean, q90], dtype=float),
                    units=units,
                    path=f"subseasonal:{variable}",
                )
            return responses

    class SubseasonalVulnerability(VulnerabilityModelBase):
        def __init__(self, hazard_type, indicator_id, builder):
            super().__init__(indicator_id=indicator_id, hazard_type=hazard_type, impact_type=ImpactType.damage)
            self._builder = builder

        def get_data_requests(self, asset: GulfPhysAsset, *, scenario: str, year: int):
            return HazardDataRequest(
                self.hazard_type,
                asset.longitude,
                asset.latitude,
                indicator_id=self.indicator_id,
                scenario=scenario,
                year=year,
            )

        def get_impact(self, asset: GulfPhysAsset, hazard_data: Sequence[mods["HazardDataResponse"]]):
            response = hazard_data[0]
            mean, q90 = response.parameters
            summary = HazardSummary(mean=float(mean), q90=float(q90))
            fractions = self._builder(asset, summary)
            return _fractions_to_distribution(
                ImpactDistrib,
                ImpactType,
                self.hazard_type,
                fractions,
                tail_probability,
                path=[response.path] if hasattr(response, "path") else ["subseasonal"],
            )

    def _heat_builder(asset: GulfPhysAsset, summary: HazardSummary):
        return heat_damage_fractions(summary, asset, physrisk_cfg.get("heat_params", {}))

    def _precip_builder(asset: GulfPhysAsset, summary: HazardSummary):
        return precip_damage_fractions(summary, physrisk_cfg.get("precip_params", {}))

    hazard_model = SubseasonalHazardModel(forecasts)
    temperature_model = SubseasonalVulnerability(AirTemperature, "t2m_weekly", _heat_builder)
    precip_model = SubseasonalVulnerability(Precipitation, "precip_weekly", _precip_builder)

    phys_assets = [GulfPhysAsset(asset) for asset in assets]
    if not phys_assets:
        return pd.DataFrame(columns=["asset_id", "EL", "VaR95", "ES95"])

    vuln_models = DictBasedVulnerabilityModels({GulfPhysAsset: [temperature_model, precip_model]})
    scenario_year = target_week.year if target_week is not None else 2016

    LOGGER.info("Running physrisk for %d assets (scenario=%s, year=%s)", len(phys_assets), scenario_id, scenario_year)
    impact_results = calculate_impacts(
        phys_assets,
        hazard_model,
        vuln_models,
        scenarios=[scenario_id],
        years=[scenario_year],
    )

    rng = np.random.default_rng(rng_seed)
    rows = []
    entries_by_asset: Dict[str, List[Any]] = {}
    hazard_meta: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for key, results in impact_results.items():
        asset_id = key.asset.id or key.asset.name
        entries_by_asset.setdefault(asset_id, []).extend(results)
        if results and results[0].hazard_data:
            params = getattr(results[0].hazard_data[0], "parameters", None)
            if params is not None:
                hazard_meta[(asset_id, key.hazard_type.__name__)] = (float(params[0]), float(params[1]))

    for asset in phys_assets:
        asset_entries = entries_by_asset.get(asset.id, [])
        if not asset_entries:
            continue
        loss_samples = _sample_losses(asset_entries, asset.value, mc_samples, rng)
        var = value_at_risk(loss_samples, 0.95)
        es = expected_shortfall(loss_samples, 0.95)
        rows.append(
            {
                "asset_id": asset.id,
                "asset_name": asset.name,
                "asset_type": asset.asset_type,
                "lat": asset.lat,
                "lon": asset.lon,
                "precip_hazard_mean": hazard_meta.get((asset.id, Precipitation.__name__), (np.nan, np.nan))[0],
                "precip_hazard_q90": hazard_meta.get((asset.id, Precipitation.__name__), (np.nan, np.nan))[1],
                "heat_hazard_mean": hazard_meta.get((asset.id, AirTemperature.__name__), (np.nan, np.nan))[0],
                "heat_hazard_q90": hazard_meta.get((asset.id, AirTemperature.__name__), (np.nan, np.nan))[1],
                "EL": float(np.mean(loss_samples)),
                "VaR95": float(var),
                "ES95": float(es),
                "risk_method": "physrisk",
                "hazards_used": "air_temperature,precipitation",
                "scenario_label": scenario_id,
                "week_start": str(target_week.date()) if target_week is not None else "",
            }
        )

    return pd.DataFrame(rows)


def _fractions_to_distribution(ImpactDistrib, ImpactType, hazard_type, fractions, tail_probability, path):
    """Create an ImpactDistrib with a two-bin representation."""
    eps = 1e-6
    mean = max(fractions.mean, 0.0)
    var95 = max(fractions.var95, mean + eps)
    bins = np.array([0.0, mean + eps, var95 + eps], dtype=float)
    tail_prob = np.clip(1.0 - tail_probability, 0.0, 1.0)
    probs = np.array([tail_prob, 1.0 - tail_prob], dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    return ImpactDistrib(hazard_type, bins, probs, path=path, impact_type=ImpactType.damage)


def _sample_losses(entries: Iterable[Any], asset_value: float, samples: int, rng: np.random.Generator) -> np.ndarray:
    draws = np.zeros(samples, dtype=float)
    for entry in entries:
        curve = entry.impact.to_exceedance_curve()
        uniforms = rng.random(samples)
        draws += curve.get_samples(uniforms)
    draws = np.clip(draws, 0.0, 1.0) * asset_value
    return draws
