"""
Feature engineering pipeline for the GulfCast stack.

The builder stitches together raw Zarr tiles from the ingest stage, constructs
member-wise tensors, lags, and positional encodings, and emits parquet datasets
ready for tabular models plus optional tensor dumps for the convolutional path.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import xarray as xr


LOGGER = logging.getLogger(__name__)

VARIABLE_ALIASES: Dict[str, List[str]] = {
    "precip": ["precipitation", "precipitationCal", "precipitationCalAvg", "MWprecipitation"],
}


def _normalize_weeks(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.normalize()


def _compute_common_weeks(
    cfsv2_catalog: pd.DataFrame, era5_catalog: pd.DataFrame, imerg_catalog: pd.DataFrame
) -> tuple[List[pd.Timestamp], pd.Timestamp]:
    cfsv2_weeks = _normalize_weeks(cfsv2_catalog["target_week_start"])
    era5_weeks = _normalize_weeks(era5_catalog["week_start"])
    imerg_weeks = _normalize_weeks(imerg_catalog["week_start"])

    common = sorted(set(cfsv2_weeks).intersection(era5_weeks).intersection(imerg_weeks))
    if not common:
        raise ValueError("No common week_start found across cfsv2, era5, and imerg catalogs.")
    latest = common[-1]
    return common, latest


def _contiguous_split_weeks(weeks: List[pd.Timestamp]) -> Dict[str, List[pd.Timestamp]]:
    weeks = sorted(weeks)
    if not weeks:
        return {"train": [], "val": [], "test": []}
    n = len(weeks)
    n_train = max(1, int(np.floor(n * 0.6)))
    n_val = max(1, int(np.floor(n * 0.2)))
    remaining = n - (n_train + n_val)
    if remaining <= 0:
        # ensure we always have some test weeks by borrowing from val/train
        remaining = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train = max(1, n_train - 1)
    train_weeks = weeks[:n_train]
    val_weeks = weeks[n_train : n_train + n_val]
    test_weeks = weeks[n_train + n_val :]
    return {"train": train_weeks, "val": val_weeks, "test": test_weeks}


def _parse_split_spec(spec: Dict[str, List], available_weeks: List[pd.Timestamp]) -> Dict[str, List[pd.Timestamp]]:
    available_weeks = sorted(available_weeks)
    available_set = set(available_weeks)
    resolved: Dict[str, List[pd.Timestamp]] = {k: [] for k in ("train", "val", "test")}
    for split_name, entries in spec.items():
        bucket: List[pd.Timestamp] = []
        for entry in entries:
            if isinstance(entry, int):
                bucket.extend([wk for wk in available_weeks if wk.year == entry])
                continue
            ts = pd.to_datetime(entry).normalize()
            if ts in available_set:
                bucket.append(ts)
            else:
                LOGGER.warning("Split spec entry %s is not present in synchronized weeks; skipping.", entry)
        target_name = split_name if split_name in resolved else "train"
        resolved[target_name].extend(bucket)
    for name in resolved:
        resolved[name] = sorted(set(resolved[name]))
    return resolved


def _resolve_variable_name(ds: xr.Dataset, variable: str) -> str:
    """
    Return the actual data variable name, falling back to known aliases.
    """
    if variable in ds.data_vars:
        return variable
    for alias in VARIABLE_ALIASES.get(variable, []):
        if alias in ds.data_vars:
            LOGGER.debug("Mapping variable %s -> %s", variable, alias)
            return alias
    raise KeyError(
        f"No variable named '{variable}'. Available variables: {sorted(ds.data_vars)}"
    )


@dataclass
class FeatureBuilderConfig:
    raw_root: Path
    output_root: Path
    members: str = "full"
    lags: int = 1
    positional_encoding: str = "pe"
    member_order: str = "original"
    seed: int = 42
    split_spec: Dict[str, List] | None = None


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature tables from raw Zarr tiles.")
    parser.add_argument("--inputs", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/proc"))
    parser.add_argument("--members", type=str, default="full", choices=["full", "meanstd"])
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--posenc", type=str, default="pe", choices=["pe", "latlon", "none"])
    parser.add_argument("--member-order", type=str, default="original", choices=["original", "sorted", "shuffled"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-spec", type=Path, help="Optional JSON file with train/val/test year lists.")
    return parser.parse_args(args=args)


def positional_encoding(lat: np.ndarray, lon: np.ndarray, num_frequencies: int = 4) -> np.ndarray:
    """
    Compute sinusoidal positional encodings for lat / lon coordinates.
    """

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    pe_list = []
    for k in range(num_frequencies):
        freq = 2.0 ** k
        pe_list.append(np.sin(lat_rad * freq))
        pe_list.append(np.cos(lat_rad * freq))
        pe_list.append(np.sin(lon_rad * freq))
        pe_list.append(np.cos(lon_rad * freq))
    return np.stack(pe_list, axis=-1)


def _standardise_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    if "lon" not in df.columns:
        for candidate in ("longitude", "Longitude", "x", "X"):
            if candidate in df.columns:
                rename_map[candidate] = "lon"
                break
    if "lat" not in df.columns:
        for candidate in ("latitude", "Latitude", "y", "Y"):
            if candidate in df.columns:
                rename_map[candidate] = "lat"
                break
    df = df.rename(columns=rename_map)
    if "lon" in df.columns:
        lon = df["lon"].astype(float)
        df["lon"] = np.where(lon > 180, lon - 360, lon)
    return df


def flatten_dataset(
    ds: xr.Dataset,
    variable: str,
    week_start: pd.Timestamp,
    member_order: str,
    seed: int,
) -> tuple[pd.DataFrame, List[str]]:
    resolved_var = _resolve_variable_name(ds, variable)
    data_var = ds[resolved_var]
    members = list(data_var.coords.get("member", []))

    if members:
        if member_order == "sorted":
            members = sorted(members)
        elif member_order == "shuffled":
            rng = np.random.default_rng(seed)
            members = list(members)
            rng.shuffle(members)

        merged: pd.DataFrame | None = None
        member_columns: List[str] = []
        for idx, member in enumerate(members):
            member_data = data_var.sel(member=member)
            member_df = member_data.to_dataframe().reset_index()
            if "lead_week" in member_df.columns:
                member_df = member_df.drop(columns=["lead_week"])
            member_df = _standardise_lat_lon(member_df)
            member_df["week_start"] = week_start
            feature_name = f"{variable}_m{idx:02d}"
            member_df = member_df[["week_start", "lat", "lon", resolved_var]].rename(columns={resolved_var: feature_name})
            member_columns.append(feature_name)
            merged = (
                member_df
                if merged is None
                else merged.merge(member_df, on=["week_start", "lat", "lon"], how="left")
            )
        assert merged is not None
        return merged, member_columns

    df = data_var.to_dataframe().reset_index()
    if "lead_week" in df.columns:
        df = df.drop(columns=["lead_week"])
    df = _standardise_lat_lon(df)
    df["week_start"] = week_start
    df = df[["week_start", "lat", "lon", resolved_var]].rename(columns={resolved_var: variable})
    return df, []


def load_catalog(root: Path, dataset: str) -> pd.DataFrame:
    catalog_path = root / dataset / "catalog.parquet"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found for {dataset}: {catalog_path}")
    return pd.read_parquet(catalog_path)


def add_lag_features(df: pd.DataFrame, group_cols: List[str], target_cols: List[str], lags: int) -> pd.DataFrame:
    df = df.sort_values(group_cols + ["week_start"])
    for lag in range(1, lags + 1):
        for col in target_cols:
            df[f"{col}_lag{lag}"] = df.groupby(group_cols)[col].shift(lag)
    return df


def build_feature_table(cfg: FeatureBuilderConfig) -> None:
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    cfsv2_catalog = load_catalog(cfg.raw_root, "cfsv2")
    era5_catalog = load_catalog(cfg.raw_root, "era5")
    imerg_catalog = load_catalog(cfg.raw_root, "imerg")

    cfsv2_catalog = cfsv2_catalog.assign(week_start=_normalize_weeks(cfsv2_catalog["target_week_start"]))
    era5_catalog = era5_catalog.assign(week_start=_normalize_weeks(era5_catalog["week_start"]))
    imerg_catalog = imerg_catalog.assign(week_start=_normalize_weeks(imerg_catalog["week_start"]))

    common_weeks, latest_common_week = _compute_common_weeks(cfsv2_catalog, era5_catalog, imerg_catalog)
    LOGGER.info("Latest common week available across datasets: %s", latest_common_week.date())
    common_set = set(common_weeks)

    cfsv2_catalog = cfsv2_catalog[cfsv2_catalog["week_start"].isin(common_set)].reset_index(drop=True)
    era5_catalog = era5_catalog[era5_catalog["week_start"].isin(common_set)].reset_index(drop=True)
    imerg_catalog = imerg_catalog[imerg_catalog["week_start"].isin(common_set)].reset_index(drop=True)

    frames: List[pd.DataFrame] = []
    ensemble_feature_map: Dict[str, List[str]] = {}

    for _, row in cfsv2_catalog.iterrows():
        ds = xr.open_zarr(row["path"])
        variable = row["variable"]
        week_start = pd.to_datetime(row["week_start"])
        df, member_cols = flatten_dataset(
            ds=ds,
            variable=variable,
            week_start=week_start,
            member_order=cfg.member_order,
            seed=cfg.seed,
        )

        if member_cols:
            ensemble_feature_map.setdefault(variable, []).extend(member_cols)
            df[f"{variable}_ens_mean"] = df[member_cols].mean(axis=1)
            df[f"{variable}_ens_std"] = df[member_cols].std(axis=1)
            if cfg.members == "meanstd":
                df = df.drop(columns=member_cols)

        frames.append(df)

    cfsv2_df = pd.concat(frames, ignore_index=True)

    era5_frames = []
    for _, row in era5_catalog.iterrows():
        ds = xr.open_zarr(row["path"])
        week_start = pd.to_datetime(row["week_start"])
        df, _ = flatten_dataset(
            ds=ds,
            variable="t2m",
            week_start=week_start,
            member_order=cfg.member_order,
            seed=cfg.seed,
        )
        era5_frames.append(df)
    era5_df = pd.concat(era5_frames, ignore_index=True)

    imerg_frames = []
    for _, row in imerg_catalog.iterrows():
        ds = xr.open_zarr(row["path"])
        week_start = pd.to_datetime(row["week_start"])
        df, _ = flatten_dataset(
            ds=ds,
            variable="precip",
            week_start=week_start,
            member_order=cfg.member_order,
            seed=cfg.seed,
        )
        df["lat"] = df["lat"].round(0)
        df["lon"] = df["lon"].round(0)
        df = df.groupby(["week_start", "lat", "lon"], as_index=False).mean()
        imerg_frames.append(df)
    imerg_df = pd.concat(imerg_frames, ignore_index=True)

    feature_df = (
        cfsv2_df.merge(era5_df, on=["week_start", "lat", "lon"], how="left", suffixes=("", "_era5"))
        .merge(imerg_df, on=["week_start", "lat", "lon"], how="left", suffixes=("", "_imerg"))
    )

    lag_cols = [col for col in feature_df.columns if col in {"t2m", "precip"}]
    feature_df = add_lag_features(feature_df, group_cols=["lat", "lon"], target_cols=lag_cols, lags=cfg.lags)

    if cfg.positional_encoding == "pe":
        pe = positional_encoding(feature_df["lat"].values, feature_df["lon"].values)
        for idx in range(pe.shape[1]):
            feature_df[f"pe_{idx}"] = pe[:, idx]
    elif cfg.positional_encoding == "latlon":
        feature_df["lat_norm"] = (feature_df["lat"] - feature_df["lat"].mean()) / feature_df["lat"].std()
        feature_df["lon_norm"] = (feature_df["lon"] - feature_df["lon"].mean()) / feature_df["lon"].std()

    feature_df = feature_df.dropna()

    available_weeks = sorted(feature_df["week_start"].unique())
    if not available_weeks:
        raise ValueError("No synchronized weeks remain after preprocessing; cannot build splits.")

    if cfg.split_spec is None:
        split_weeks = _contiguous_split_weeks(available_weeks)
        LOGGER.info(
            "Auto-derived contiguous splits (weeks): train=%d, val=%d, test=%d",
            len(split_weeks["train"]),
            len(split_weeks["val"]),
            len(split_weeks["test"]),
        )
    else:
        split_weeks = _parse_split_spec(cfg.split_spec, available_weeks)
        if all(len(weeks) == 0 for weeks in split_weeks.values()):
            LOGGER.warning("Provided split spec matched no weeks; falling back to contiguous splits.")
            split_weeks = _contiguous_split_weeks(available_weeks)

    assigned_weeks = set().union(*split_weeks.values())
    missing_weeks = set(available_weeks) - assigned_weeks
    if missing_weeks:
        LOGGER.warning(
            "Excluding %d weeks that were not assigned to any split (earliest missing: %s).",
            len(missing_weeks),
            min(missing_weeks),
        )

    for split, weeks in split_weeks.items():
        out_path = cfg.output_root / f"{split}.parquet"
        if not weeks:
            LOGGER.info("No weeks assigned to %s split; writing empty DataFrame to %s", split, out_path)
            feature_df.head(0).to_parquet(out_path, index=False)
            continue
        mask = feature_df["week_start"].isin(weeks)
        split_df = feature_df[mask]
        LOGGER.info("Writing %s rows spanning %d weeks to %s", len(split_df), len(set(weeks)), out_path)
        split_df.to_parquet(out_path, index=False)

    meta = cfg.output_root / "metadata.json"
    meta_payload = {
        "members": cfg.members,
        "lags": cfg.lags,
        "positional_encoding": cfg.positional_encoding,
        "member_order": cfg.member_order,
        "seed": cfg.seed,
        "latest_common_week": latest_common_week.strftime("%Y-%m-%d"),
        "synchronized_weeks": len(available_weeks),
        "splits": {k: [wk.strftime("%Y-%m-%d") for wk in v] for k, v in split_weeks.items()},
        "ensemble_members": {k: sorted(set(v)) for k, v in ensemble_feature_map.items()},
    }
    meta.write_text(json.dumps(meta_payload, indent=2))


def main(args: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ns = parse_args(args)
    cfg = FeatureBuilderConfig(
        raw_root=ns.inputs,
        output_root=ns.out,
        members=ns.members,
        lags=ns.lags,
        positional_encoding=ns.posenc,
        member_order=ns.member_order,
        seed=ns.seed,
        split_spec=json.loads(ns.split_spec.read_text()) if ns.split_spec else None,
    )
    build_feature_table(cfg)


if __name__ == "__main__":
    main()
