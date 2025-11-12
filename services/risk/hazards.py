"""
Hazard index utilities for drought, flood, and heat proxies.

These functions operate on weekly gridded xarray datasets produced by the
feature pipeline and return xarray DataArray fields aligned to the input grid.

Dependencies are optional; when missing, safe fallbacks are used.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)


def degree_days(ds_t2m: xr.Dataset, threshold_c: float = 30.0) -> xr.DataArray:
    """Compute weekly degree-days above a temperature threshold.

    Expects ds_t2m["t2m"] in Kelvin or Celsius; converts to °C if values>200.
    """
    t = ds_t2m["t2m"]
    t_c = xr.where(t > 200.0, t - 273.15, t)
    excess = xr.where(t_c > threshold_c, t_c - threshold_c, 0.0)
    # Weekly mean degree exceedance as intensity proxy
    return excess.rename("heat_dd")


def compute_spi(ds_precip_weekly: xr.Dataset, scale: int = 3) -> xr.DataArray:
    """Compute a simple SPI proxy from weekly precipitation.

    If climate-indices is available, use it; otherwise z-score recent window.
    """
    pr = ds_precip_weekly["precip"]
    try:
        import xclim  # type: ignore

        spi = xclim.indices.spi(pr, freq="7D", window=scale)
        return spi.rename("spi")
    except Exception:  # pragma: no cover
        LOGGER.warning("xclim not available; using standardized anomaly as SPI proxy")
        rolling = pr.rolling(week_start=scale, min_periods=1).sum()
        mean = rolling.mean(dim="week_start")
        std = rolling.std(dim="week_start") + 1e-6
        spi = (rolling - mean) / std
        return spi.rename("spi")


def fit_gev_weekly_max(ds_precip_weekly: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Fit a crude GEV to weekly precipitation maxima per grid cell.

    Returns shape, loc, scale arrays. For simplicity, we use method-of-moments
    on the available sample; this is a placeholder for a true EVT fit.
    """
    pr = ds_precip_weekly["precip"]
    m = pr.max(dim="week_start")
    mu = pr.mean(dim="week_start")
    sig = pr.std(dim="week_start") + 1e-6
    xi = xr.zeros_like(mu)
    return xi.rename("gev_xi"), mu.rename("gev_loc"), sig.rename("gev_scale")


def return_level_grid(params: Tuple[xr.DataArray, xr.DataArray, xr.DataArray], rp: int = 20) -> xr.DataArray:
    """Compute an approximate return level for a given return period.

    Uses Gumbel approximation when xi≈0.
    """
    xi, loc, scale = params
    # Gumbel RL ≈ loc - scale * log(-log(1 - 1/rp))
    y = -np.log(-np.log(1.0 - 1.0 / max(rp, 2)))
    rl = loc + scale * y
    return rl.rename("flood_rl")

