"""
Shared utilities for ingestion modules.

Centralizes target bounding boxes, temporal helpers, and catalog writing so
dataset-specific scripts remain light-weight.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import xarray as xr

from services.ingest.iri import open_iridl_dataset

LOGGER = logging.getLogger(__name__)

CONUS_BBOX = "-125,24,-66,50"


def chunk_dataset(ds: xr.Dataset, chunk_lat: int = 64, chunk_lon: int = 64) -> xr.Dataset:
    """Return dataset chunked along lat/lon dimensions for scalable writes.

    Requires dask; if unavailable, returns the dataset untouched with a warning.
    """
    try:
        import dask  # noqa: F401
    except ImportError:
        LOGGER.warning("Dask is not installed; writing unchunked dataset. Install 'dask[array]' for large bboxes.")
        return ds
    chunk_map = {}
    for dim in ds.dims:
        if "lat" in dim.lower():
            chunk_map[dim] = max(1, chunk_lat)
        elif "lon" in dim.lower():
            chunk_map[dim] = max(1, chunk_lon)
    if not chunk_map:
        return ds
    try:
        return ds.chunk(chunk_map)
    except ValueError as exc:
        LOGGER.warning("Chunking failed (%s); writing unchunked dataset.", exc)
        return ds


@dataclasses.dataclass(frozen=True)
class GulfBBox:
    west: float
    south: float
    east: float
    north: float

    @classmethod
    def from_string(cls, bbox: str) -> "GulfBBox":
        west, south, east, north = (float(part) for part in bbox.split(","))
        return cls(west=west, south=south, east=east, north=north)

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_coord_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for cand in candidates:
        if cand in ds.coords:
            return cand
    raise KeyError(f"None of coordinate candidates {candidates} present in dataset.")


def _wrap_lon(lon: float) -> float:
    return (lon + 360.0) % 360.0


def load_bbox(ds: xr.Dataset, bbox: GulfBBox) -> xr.Dataset:
    lon_name = _resolve_coord_name(ds, ("lon", "longitude", "X", "x"))
    lat_name = _resolve_coord_name(ds, ("lat", "latitude", "Y", "y"))

    west, south, east, north = bbox.west, bbox.south, bbox.east, bbox.north

    # Handle datasets that use 0..360 longitudes.
    lon_vals = ds[lon_name].values
    if lon_vals.max() > 180:
        west, east = _wrap_lon(west), _wrap_lon(east)

    # Preserve latitude order.
    lat_values = ds[lat_name].values
    lat_slice = slice(south, north) if lat_values[0] < lat_values[-1] else slice(north, south)

    return ds.sel({lon_name: slice(west, east), lat_name: lat_slice})


def open_remote_dataset(url: str) -> xr.Dataset:
    LOGGER.debug("Opening %s", url)
    if "iridl.ldeo.columbia.edu" in url:
        return open_iridl_dataset(url)
    return xr.open_dataset(url)


def generate_weekly_range(start_month: dt.date, end_month: dt.date) -> List[dt.date]:
    current = start_month
    weeks: List[dt.date] = []
    while current <= end_month:
        weeks.append(current)
        current += dt.timedelta(days=7)
    return weeks


def infer_lead_offsets(lead: str) -> List[int]:
    if lead != "wk34":
        raise ValueError(f"Unsupported lead specifier: {lead}")
    return [3, 4]


def to_member_id(init_date: dt.date, member: int) -> str:
    return f"{init_date:%Y%m%d}_lag{member}"


def write_catalog_row(catalog_path: Path, row: dict) -> None:
    if catalog_path.exists():
        df = pd.read_parquet(catalog_path)
    else:
        df = pd.DataFrame(columns=row.keys())
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(catalog_path, index=False)
