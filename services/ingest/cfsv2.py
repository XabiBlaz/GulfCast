# services/ingest/cfsv2.py
from __future__ import annotations
import argparse, datetime as dt, logging
from pathlib import Path
from typing import Sequence, List
import pandas as pd
import xarray as xr
import numpy as np
from services.ingest.shared import (
    GulfBBox, ensure_output_dir, generate_weekly_range, infer_lead_offsets,
    load_bbox, open_remote_dataset, to_member_id, write_catalog_row
)

LOGGER = logging.getLogger(__name__)

# Map CLI to IRIDL variable names
VARIABLE_ALIASES = {
    "pr":  "pr",   # total precip flux (kg m^-2 s^-1)
    "t2m": "ts",   # surface temperature (K)
}

def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest CFSv2 subseasonal hindcasts (IRIDL).")
    p.add_argument("--vars", type=str, default="pr,t2m", help="Comma separated variables (pr,t2m).")
    p.add_argument("--lead", type=str, default="wk34", choices=["wk34"], help="Target lead (wk34 -> forecast_period 3 & 4).")
    p.add_argument("--bbox", type=str, default="-95,28,-89,31", help="west,south,east,north")
    p.add_argument("--start", type=str, required=True, help="Start month YYYY-MM (Mondays will be generated)")
    p.add_argument("--end", type=str, required=True, help="End month YYYY-MM")
    p.add_argument("--out", type=Path, default=Path("data/raw/cfsv2"))
    p.add_argument("--catalog", type=str, default="catalog.parquet")
    p.add_argument("--chunk-weeks", type=int, default=4)
    p.add_argument("--lagged-members", type=int, default=4, help="How many lagged inits as ensemble members")
    return p.parse_args(args=args)

def _iridl_root_url(var: str) -> str:
    # OPeNDAP endpoint on IRIDL
    return f"https://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubC/.NCEP/.CFSv2/.forecast/.{var}/dods"


def _coord_names(ds: xr.Dataset) -> dict:
    coords = set(ds.coords)

    def _first(*candidates):
        for cand in candidates:
            if cand in coords:
                return cand
        return None

    name_init = _first("init", "T", "S", "time")
    name_lead = _first("forecast_period", "Lead", "L")
    name_member = _first("members", "M")
    name_x = _first("X", "lon", "longitude")
    name_y = _first("Y", "lat", "latitude")

    missing = [label for label, value in {"init": name_init, "lead": name_lead, "member": name_member, "x": name_x, "y": name_y}.items() if value is None]
    if missing:
        raise KeyError(f"Could not resolve coord names (missing {missing}); have coords={list(ds.coords)}")

    return dict(init=name_init, lead=name_lead, member=name_member, x=name_x, y=name_y)

def _open_and_select(var: str, init_date: dt.date, lead_week: int, member_idx: int, bbox: GulfBBox) -> xr.Dataset:
    url = _iridl_root_url(var)
    LOGGER.info("Opening %s", url)
    ds = open_remote_dataset(url)
    names = _coord_names(ds)

    # Convert init date to pandas Timestamp to match coord dtype
    init_ts = pd.to_datetime(init_date)

    # Select; use nearest init in case weekly Mondays don't exactly match archive days
    da = ds[var].sel({names["init"]: init_ts}, method="nearest")

    lead_index = ds[names["lead"]]
    if np.issubdtype(lead_index.dtype, np.timedelta64):
        target = np.timedelta64((lead_week + 1) * 7, "D")
        da = da.sel({names["lead"]: target}, method="nearest")
    else:
        da = da.sel({names["lead"]: lead_week}, method="nearest")
    da = da.sel({names["member"]: member_idx}, method="nearest")

    # Subset bbox (handles 0..360 and lat direction)
    ds_sel = da.to_dataset(name=var)
    ds_sel = load_bbox(ds_sel, bbox)

    # Unit fixes to match your downstream expectations
    if var == "ts":
        ds_sel[var] = ds_sel[var] - 273.15
        ds_sel[var].attrs["units"] = "degC"
    elif var == "pr":
        # pr is flux (kg m^-2 s^-1). For week-3/4  (14 days), convert to total mm.
        ds_sel[var] = ds_sel[var] * (14 * 24 * 3600.0)
        ds_sel[var].attrs["units"] = "mm"

    member_name = to_member_id(init_date, member_idx)
    if "member" in ds_sel.dims:
        ds_sel = ds_sel.assign_coords(member=("member", [member_name]))
    else:
        ds_sel = ds_sel.expand_dims(dim={"member": [member_name]})

    if "lead_week" in ds_sel.dims:
        ds_sel = ds_sel.assign_coords(lead_week=("lead_week", [lead_week]))
    else:
        ds_sel = ds_sel.expand_dims(dim={"lead_week": [lead_week]})

    # Standardize lon/lat names for consistency
    coord_map = {}
    if "X" in ds_sel.coords: coord_map["X"] = "lon"
    if "Y" in ds_sel.coords: coord_map["Y"] = "lat"
    if coord_map: ds_sel = ds_sel.rename(coord_map)
    return ds_sel

def process_variable(
    var_cli: str, start_month: dt.date, end_month: dt.date, bbox: GulfBBox,
    chunk_weeks: int, lagged_members: int, lead: str, output_root: Path, catalog_path: Path
) -> None:
    iridl_var = VARIABLE_ALIASES.get(var_cli, var_cli)
    lead_offsets = infer_lead_offsets(lead)
    weeks: List[dt.date] = generate_weekly_range(start_month, end_month)
    ensure_output_dir(output_root)

    for week_start in weeks:
        init_dates = [week_start - dt.timedelta(days=7 * lag) for lag in range(lagged_members)]
        for lead_week in lead_offsets:
            member_datasets = []
            for member_idx, init_date in enumerate(init_dates):
                ds = _open_and_select(iridl_var, init_date, lead_week, member_idx, bbox)
                member_datasets.append(ds)

            stacked = xr.concat(member_datasets, dim="member")
            chunk_dir = output_root / iridl_var / f"{week_start:%Y%m%d}-wk{lead_week}"
            LOGGER.info("Writing %s", chunk_dir)
            stacked.to_zarr(chunk_dir, mode="w")

            write_catalog_row(
                catalog_path,
                dict(
                    variable=iridl_var,
                    target_week_start=f"{week_start:%Y-%m-%d}",
                    lead_week=lead_week,
                    members=len(member_datasets),
                    path=str(chunk_dir),
                ),
            )

def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ns = parse_args(args)
    bbox = GulfBBox.from_string(ns.bbox)
    start_month = dt.datetime.strptime(ns.start, "%Y-%m").date().replace(day=1)
    end_month   = dt.datetime.strptime(ns.end, "%Y-%m").date().replace(day=1)
    catalog_path = ns.out / ns.catalog
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    variables = [v.strip() for v in ns.vars.split(",")]
    LOGGER.info("Requested variables: %s", variables)
    for var_cli in variables:
        process_variable(
            var_cli, start_month, end_month, bbox,
            ns.chunk_weeks, ns.lagged_members, ns.lead,
            ns.out, catalog_path
        )

if __name__ == "__main__":
    main()
