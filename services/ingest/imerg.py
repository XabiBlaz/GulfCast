"""
IMERG daily precipitation ingest CLI.

The script downloads IMERG Final run daily accumulations, aggregates to weekly
sums, and writes both Zarr tiles and a catalog table. It is intentionally light
so it can be run in the Day 1 ingestion phase.
"""

from __future__ import annotations

import argparse
import os
import datetime as dt
import logging
import tempfile
from pathlib import Path
from typing import List, Sequence

import xarray as xr

try:
    import gcsfs
except ImportError:  # pragma: no cover
    gcsfs = None

from services.ingest.earthdata import EarthdataSession, download_file
from services.ingest.shared import (
    GulfBBox,
    chunk_dataset,
    ensure_output_dir,
    generate_weekly_range,
    load_bbox,
    write_catalog_row,
)

LOGGER = logging.getLogger(__name__)

IMERG_FILENAME_TEMPLATE = (
    "3B-DAY.MS.MRG.3IMERG.{year}{month:02d}{day:02d}-S000000-E235959.{version}.nc4"
)
IMERG_URI_TEMPLATE = "gs://gcp-public-data-nasa-gpm/IMERG_Final/{year}/{month:02d}/{filename}"
IMERG_EARTHDATA_TEMPLATE = "{base}/data/GPM_L3/GPM_3IMERGDF.07/{year}/{month:02d}/{filename}"


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest IMERG daily precipitation.")
    parser.add_argument("--agg", type=str, default="weekly", choices=["weekly"])
    parser.add_argument("--bbox", type=str, default="-95,28,-89,31")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/raw/imerg"))
    parser.add_argument("--catalog", type=str, default="catalog.parquet")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument(
        "--version",
        type=str,
        default="V07B",
        help="IMERG collection version tag (e.g., V07B).",
    )
    parser.add_argument(
        "--earthdata",
        action="store_true",
        help="Force download via NASA Earthdata (use if gcsfs is unavailable).",
    )
    parser.add_argument(
        "--earthdata-base",
        type=str,
        default="https://gpm1.gesdisc.eosdis.nasa.gov",
        help="Base URL for Earthdata downloads.",
    )
    parser.add_argument("--chunk-lat", type=int, default=64)
    parser.add_argument("--chunk-lon", type=int, default=64)
    return parser.parse_args(args=args)


def imerg_filename(date: dt.date, version: str) -> str:
    return IMERG_FILENAME_TEMPLATE.format(
        year=date.year,
        month=date.month,
        day=date.day,
        version=version,
    )


def open_imerg_daily(fs: "gcsfs.GCSFileSystem", date: dt.date, version: str) -> xr.Dataset:
    filename = imerg_filename(date, version)
    uri = IMERG_URI_TEMPLATE.format(year=date.year, month=date.month, filename=filename)
    LOGGER.info("Opening %s", uri)
    with fs.open(uri) as fh:
        ds = xr.open_dataset(fh)
        data = ds.load()
        ds.close()
    return data


def aggregate_weekly_sum(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.resample(time="1W-MON", label="left", closed="left").sum()
    ds = ds.rename({"time": "week_start"})
    return ds


def process_range(args: argparse.Namespace) -> None:
    bbox = GulfBBox.from_string(args.bbox)
    ensure_output_dir(args.out)
    catalog_path = args.out / args.catalog

    use_earthdata = args.earthdata or gcsfs is None
    fs = None
    if not use_earthdata:
        fs_token = {"username": None, "password": None}
        username = os.getenv("NASA_EARTHDATA_USERNAME")
        password = os.getenv("NASA_EARTHDATA_PASSWORD")
        if username and password:
            fs_token = {"username": username, "password": password}
        fs = gcsfs.GCSFileSystem(token=fs_token if username else "anon")

    start_month = dt.datetime.strptime(args.start, "%Y-%m").date().replace(day=1)
    end_month = dt.datetime.strptime(args.end, "%Y-%m").date().replace(day=1)

    earthdata_session = EarthdataSession() if use_earthdata else None

    for week_start in generate_weekly_range(start_month, end_month):
        week_dates = [week_start + dt.timedelta(days=d) for d in range(7)]
        zarr_path = args.out / f"pr-{week_start:%Y%m%d}.zarr"
        if args.use_cache and zarr_path.exists():
            LOGGER.info("Skipping existing %s", zarr_path)
            continue

        daily_arrays: List[xr.Dataset] = []
        if use_earthdata:
            with tempfile.TemporaryDirectory(prefix="imerg") as tmpdir:
                for date in week_dates:
                    filename = imerg_filename(date, args.version)
                    url = IMERG_EARTHDATA_TEMPLATE.format(
                        base=args.earthdata_base,
                        year=date.year,
                        month=date.month,
                        filename=filename,
                    )
                    dest = Path(tmpdir) / Path(url).name
                    LOGGER.info("Downloading %s", url)
                    download_file(url, dest, session=earthdata_session)
                    ds = xr.open_dataset(dest)
                    data = ds.load()
                    ds.close()
                    daily_arrays.append(data)
        else:
            daily_arrays = [open_imerg_daily(fs, date, args.version) for date in week_dates]  # type: ignore[arg-type]

        ds = xr.concat(daily_arrays, dim="time")
        ds_week = aggregate_weekly_sum(ds)
        ds_week = load_bbox(ds_week, bbox)
        ds_week = chunk_dataset(ds_week, args.chunk_lat, args.chunk_lon)
        ds_week.to_zarr(zarr_path, mode="w")

        write_catalog_row(
            catalog_path,
            dict(
                variable="precip",
                week_start=f"{week_start:%Y-%m-%d}",
                path=str(zarr_path),
                aggregation="weekly_sum",
            ),
        )


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    process_range(parse_args(args))


if __name__ == "__main__":
    main()
