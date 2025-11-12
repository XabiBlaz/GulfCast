"""
ERA5 2 metre temperature ingest CLI.

Uses cdsapi for authenticated downloads. The CLI mirrors the signature used in
the project plan: it can aggregate hourly data to weekly means, crop to the Gulf
Coast bounding box, and persist both data arrays (Zarr) and a Parquet catalog.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

try:
    import cdsapi_v2 as cdsapi
except ImportError:  # pragma: no cover - optional dependency for offline dev
    import cdsapi

from services.ingest.shared import (
    GulfBBox,
    chunk_dataset,
    ensure_output_dir,
    generate_weekly_range,
    load_bbox,
    write_catalog_row,
)

LOGGER = logging.getLogger(__name__)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest ERA5 2m temperature.")
    parser.add_argument("--var", type=str, default="t2m", choices=["t2m"])
    parser.add_argument("--agg", type=str, default="weekly", choices=["weekly"])
    parser.add_argument("--bbox", type=str, default="-95,28,-89,31")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/raw/era5"))
    parser.add_argument("--catalog", type=str, default="catalog.parquet")
    parser.add_argument("--use-cache", action="store_true", help="Skip download if Zarr chunk already exists.")
    parser.add_argument("--chunk-lat", type=int, default=64)
    parser.add_argument("--chunk-lon", type=int, default=64)
    return parser.parse_args(args=args)


def retrieve_week(client: "cdsapi.Client", variable: str, week_start: dt.date, bbox: GulfBBox) -> xr.Dataset:
    week_end = week_start + dt.timedelta(days=6)
    tmp_dir = Path(tempfile.gettempdir()) / "gulfcast"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f"era5_t2m_{week_start:%Y%m%d}.nc"

    request = {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": [week_start.year, week_end.year],
        "month": [week_start.month, week_end.month],
        "day": [*(str(d).zfill(2) for d in range(1, 32))],
        "time": [f"{h:02d}:00" for h in range(24)],
        "format": "netcdf",
        "area": [bbox.north, bbox.west, bbox.south, bbox.east],
    }

    LOGGER.info("Requesting ERA5 week %s", week_start)
    client.retrieve("reanalysis-era5-single-levels", request, str(target))
    ds = xr.open_dataset(target)
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def aggregate_weekly_mean(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.resample(time="1W-MON", label="left", closed="left").mean()
    ds = ds.rename({"time": "week_start"})
    return ds


def process_range(args: argparse.Namespace) -> None:
    if cdsapi is None:
        raise ImportError("cdsapi is not installed. Install via `poetry install` with CDS credentials configured.")

    bbox = GulfBBox.from_string(args.bbox)
    client = cdsapi.Client()
    ensure_output_dir(args.out)
    catalog_path = args.out / args.catalog

    start_month = dt.datetime.strptime(args.start, "%Y-%m").date().replace(day=1)
    end_month = dt.datetime.strptime(args.end, "%Y-%m").date().replace(day=1)

    for week_start in generate_weekly_range(start_month, end_month):
        zarr_path = args.out / f"t2m-{week_start:%Y%m%d}.zarr"
        if args.use_cache and zarr_path.exists():
            LOGGER.info("Skipping existing %s", zarr_path)
            continue

        ds_hourly = retrieve_week(client, args.var, week_start, bbox)
        ds_weekly = aggregate_weekly_mean(ds_hourly)
        ds_weekly = load_bbox(ds_weekly, bbox)
        ds_weekly = chunk_dataset(ds_weekly, args.chunk_lat, args.chunk_lon)
        ds_weekly.to_zarr(zarr_path, mode="w")

        write_catalog_row(
            catalog_path,
            dict(
                variable="t2m",
                week_start=f"{week_start:%Y-%m-%d}",
                path=str(zarr_path),
                aggregation="weekly_mean",
            ),
        )


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    process_range(parse_args(args))


if __name__ == "__main__":
    main()
