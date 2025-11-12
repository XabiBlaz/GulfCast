"""
Adapters to bridge GulfCast xarray grids and CLIMADA Hazard/Exposures/Impact.

All imports of CLIMADA are lazy and optional: functions will raise a clear
ImportError with guidance if the package is missing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


def _require_climada():
    try:
        # Minimal imports to construct hazards/exposures
        from climada.hazard import Hazard  # type: ignore  # noqa: F401
        from climada.entity import Exposures  # type: ignore  # noqa: F401
        from climada.entity.impact_funcs.base import ImpactFunc  # type: ignore  # noqa: F401
        from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "CLIMADA is not installed. Install optional risk extras or set risk_method=parametric."
        ) from exc


def assets_to_exposures(assets_df: pd.DataFrame):
    _require_climada()
    from climada.entity import Exposures

    exp = Exposures()
    # Expect columns: id, lat, lon, value
    df = assets_df.rename(columns={"id": "exposure_id"}).copy()
    exp.set_from_dataframe(df, latitude="lat", longitude="lon")
    exp.gdf["value"] = df.get("value", pd.Series(np.ones(len(df))))
    return exp


def _grid_to_centroids(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = ds.coords["lat"].values
    lon = ds.coords["lon"].values
    # Flatten fields
    return lat, lon, np.array(np.meshgrid(lat, lon, indexing="ij"))


def build_heat_hazard(ds_t2m_weekly: xr.Dataset):
    _require_climada()
    from climada.hazard import Hazard

    t = ds_t2m_weekly["t2m"]
    t_c = xr.where(t > 200.0, t - 273.15, t)
    h = Hazard("HEAT")
    # Minimal scaffold: assign intensity as t_c weekly mean
    lat = t_c.coords["lat"].values
    lon = t_c.coords["lon"].values
    intens = t_c.values  # (week, lat, lon)
    h.centroids = np.array(np.meshgrid(lat, lon, indexing="ij")).reshape(2, -1).T
    h.intensity = intens.reshape(intens.shape[0], -1)
    return h


def build_drought_hazard(spi_grid: xr.DataArray):
    _require_climada()
    from climada.hazard import Hazard

    h = Hazard("DROUGHT")
    a = spi_grid
    lat = a.coords["lat"].values
    lon = a.coords["lon"].values
    intens = a.values
    h.centroids = np.array(np.meshgrid(lat, lon, indexing="ij")).reshape(2, -1).T
    h.intensity = intens.reshape(intens.shape[0], -1)
    return h


def build_flood_hazard(flood_rp_intensity: xr.DataArray):
    _require_climada()
    from climada.hazard import Hazard

    h = Hazard("FLOOD")
    a = flood_rp_intensity
    lat = a.coords["lat"].values
    lon = a.coords["lon"].values
    intens = a.values
    h.centroids = np.array(np.meshgrid(lat, lon, indexing="ij")).reshape(2, -1).T
    h.intensity = intens.reshape(intens.shape[0], -1)
    return h

