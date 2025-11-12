"""
Shared loss curve utilities used by both the parametric and physrisk pipelines.

All damage calculations are expressed as fractions first so that we can either
scale them to absolute currency (parametric path) or feed them into the
physrisk distribution machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class HazardSummary:
    mean: float
    q90: float


@dataclass
class DamageFractions:
    """Damage ratios (0-1) for expected, VaR95, and ES95 outcomes."""

    mean: float
    var95: float
    es95: float


def _clip_fraction(value: float, max_value: float = 1.0) -> float:
    return float(np.clip(value, 0.0, max_value))


def _fraction_scales(params: Dict[str, float]) -> tuple[float, float]:
    var_scale = float(params.get("var_scale", 1.05))
    es_scale = float(params.get("es_scale", 1.20))
    return var_scale, es_scale


def precip_damage_fractions(hazard: HazardSummary, params: Dict[str, float]) -> DamageFractions:
    """Return fractional damage stats for precipitation-driven losses."""

    threshold = float(params.get("threshold", 150.0))
    slope = float(params.get("slope", 0.4))
    norm = float(params.get("intensity_norm", 100.0))
    max_damage = float(params.get("max_damage", 0.2))
    var_scale, es_scale = _fraction_scales(params)

    def damage(amount: float) -> float:
        excess = max(0.0, amount - threshold)
        if norm <= 0:
            return 0.0
        scaled = slope * (excess / norm)
        return _clip_fraction(scaled, max_damage)

    mean_damage = damage(hazard.mean)
    tail_damage = damage(hazard.q90)
    var95 = _clip_fraction(max(mean_damage, tail_damage * var_scale), max_damage)
    es95 = _clip_fraction(max(var95, tail_damage * es_scale), max_damage)
    return DamageFractions(mean=mean_damage, var95=var95, es95=es95)


def _heat_capacity(asset: Any, replacement_cost: float) -> float:
    extra = getattr(asset, "extra", {}) or {}
    raw_capacity = extra.get("capacity_mw")
    if raw_capacity is None or (isinstance(raw_capacity, float) and np.isnan(raw_capacity)):
        value = max(float(getattr(asset, "value", replacement_cost)), 1.0)
        raw_capacity = max(5.0, value / max(replacement_cost, 1.0))
    return float(raw_capacity)


def heat_damage_fractions(hazard: HazardSummary, asset: Any, params: Dict[str, float]) -> DamageFractions:
    """Return fractional derating stats for heat-driven capacity losses."""

    threshold = float(params.get("threshold_c", 35.0))
    slope = float(params.get("derate_slope", 0.02))
    max_damage = float(params.get("max_damage", 0.2))
    mw_value = float(params.get("value_per_mw_day", 200.0))
    duration_days = float(params.get("duration_days", 5.0))
    replacement_cost = float(params.get("replacement_cost_per_mw", 1_000_000))
    var_scale, es_scale = _fraction_scales(params)

    def derate(temp_c: float) -> float:
        excess = max(0.0, temp_c - threshold)
        scaled = slope * excess
        return _clip_fraction(scaled, max_damage)

    mean_derate = derate(_kelvin_to_c(hazard.mean))
    tail_derate = derate(_kelvin_to_c(hazard.q90))

    capacity = _heat_capacity(asset, replacement_cost)
    base_loss = capacity * mw_value * duration_days
    asset_value = max(float(getattr(asset, "value", base_loss)), 1.0)

    mean_frac = _clip_fraction((mean_derate * base_loss) / asset_value, max_damage)
    tail_frac = _clip_fraction((tail_derate * base_loss) / asset_value, max_damage)

    var95 = _clip_fraction(max(mean_frac, tail_frac * var_scale), max_damage)
    es95 = _clip_fraction(max(var95, tail_frac * es_scale), max_damage)
    return DamageFractions(mean=mean_frac, var95=var95, es95=es95)


def fractions_to_currency(asset_value: float, fractions: DamageFractions) -> Dict[str, float]:
    value = max(float(asset_value), 0.0)
    return {
        "EL": fractions.mean * value,
        "VaR95": fractions.var95 * value,
        "ES95": fractions.es95 * value,
    }


def _kelvin_to_c(value: float) -> float:
    return value - 273.15 if value > 200.0 else value
