"""
Asset ingestion utilities (power plants & ports).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

# ✅ Correct URLs
POWER_PLANT_URL = (
    "https://raw.githubusercontent.com/wri/global-power-plant-database/master/"
    "output_database/global_power_plant_database.csv"
)
# Official NGA World Port Index CSV (script-friendly)
WPI_URL = (
    "https://msi.nga.mil/api/publications/download"
    "?key=16920959/SFH00000/UpdatedPub150.csv&type=download"
)

def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and filter asset datasets.")
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--bbox", type=str, default="-95,28,-89,31")  # west,south,east,north
    return parser.parse_args(args=args)

def _bbox_tuple(bbox: str):
    west, south, east, north = (float(v) for v in bbox.split(","))
    return west, south, east, north

def ingest_power_plants(out_dir: Path, bbox: str) -> Path:
    LOGGER.info("Downloading power plant database…")
    # Mixed dtypes in some cols -> avoid DtypeWarning on Windows
    df = pd.read_csv(POWER_PLANT_URL, low_memory=False)
    west, south, east, north = _bbox_tuple(bbox)
    mask = (
        df["longitude"].between(west, east)
        & df["latitude"].between(south, north)
    )
    keep = df.loc[mask].copy()
    keep.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)
    keep["type"] = keep["primary_fuel"].str.lower().fillna("unknown")

    out_path = out_dir / "plants.csv"
    cols = ["gppd_idnr", "name", "lat", "lon", "capacity_mw", "primary_fuel", "type"]
    keep[cols].to_csv(out_path, index=False)
    LOGGER.info("Wrote %s (%d rows)", out_path, len(keep))
    return out_path

def ingest_ports(out_dir: Path, bbox: str) -> Path:
    LOGGER.info("Downloading World Port Index CSV…")
    ports = pd.read_csv(WPI_URL)

    # Normalize column names we need (some envs change case/spacing)
    rename_map = {
        "World Port Index Number": "wpi_number",
        "Main Port Name": "name",
        "Latitude": "lat",
        "Longitude": "lon",
    }
    # Create a lowercase->original map to be robust to variations
    lc = {c.lower(): c for c in ports.columns}
    need = {k: lc.get(k.lower(), k) for k in rename_map.keys()}
    ports = ports.rename(columns={need[k]: v for k, v in rename_map.items() if need.get(k)})

    for col in ["lat", "lon"]:
        if col not in ports.columns:
            raise ValueError(f"Expected '{col}' in WPI CSV but did not find it. Columns: {list(ports.columns)}")

    west, south, east, north = _bbox_tuple(bbox)
    mask = ports["lon"].between(west, east) & ports["lat"].between(south, north)
    keep = ports.loc[mask, ["wpi_number", "name", "lat", "lon"]].copy()
    keep["name"] = keep["name"].fillna("Unknown Port")
    keep["type"] = "port"

    out_path = out_dir / "ports.csv"
    keep[["wpi_number", "name", "lat", "lon", "type"]].to_csv(out_path, index=False)
    LOGGER.info("Wrote %s (%d rows)", out_path, len(keep))
    return out_path

def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ns = parse_args(args)
    ns.out.mkdir(parents=True, exist_ok=True)
    ingest_power_plants(ns.out, ns.bbox)
    ingest_ports(ns.out, ns.bbox)

if __name__ == "__main__":
    main()
