"""
FastAPI application exposing forecast, risk, insights, and templated reporting endpoints.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
try:
    from openai import OpenAI  # type: ignore
except ImportError:  # OpenAI is optional for CLI-only usage
    OpenAI = None  # type: ignore
from pydantic import BaseModel, Field

from services.risk.compute import compute_risk, load_assets, load_curves, load_forecasts

app = FastAPI(title="GulfCast API", version="0.1.0")


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PROC_DIR = DATA_DIR / "proc"
METADATA_PATH = PROC_DIR / "metadata.json"
RISK_PATH = DATA_DIR / "risk" / "latest.parquet"
TRAINING_SUMMARY_PATH = MODELS_DIR / "training_summary.json"
RISK_CONFIG_PATH = Path("config/risk.yaml")
OPENAI_MODEL = os.getenv("OPENAI_INSIGHTS_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_INSIGHTS_MAX_TOKENS", "600"))

LOGGER = logging.getLogger(__name__)


class AssetIn(BaseModel):
    id: str
    lat: float
    lon: float
    value: Optional[float] = None
    type: Optional[str] = Field(default=None, alias="asset_type")
    name: Optional[str] = None


class ForecastResponse(BaseModel):
    grid: Optional[dict] = None
    assets: Optional[List[dict]] = None


class RiskRequest(BaseModel):
    portfolio: List[AssetIn]
    curve_override: Optional[dict] = None


class RiskResponse(BaseModel):
    assets: List[dict]
    summary: dict


class ReportResponse(BaseModel):
    framework: str
    content: str


class SplitSummary(BaseModel):
    name: str
    weeks: int
    example_weeks: List[str]


class ForecastStats(BaseModel):
    records: int
    bbox: Dict[str, float]
    variables: List[str]


class RiskStats(BaseModel):
    assets: int
    el_mean: float
    el_max: float
    var95_mean: float
    es95_mean: float


class RiskByTypeEntry(BaseModel):
    asset_type: str
    count: int
    el_total: float
    var95_total: float
    es95_total: float


class HazardSummaryEntry(BaseModel):
    variable: str
    mean_mean: float
    mean_q90: float
    max_q90: float
    unit: str


class HazardSeriesEntry(BaseModel):
    week: str
    variable: str
    q90: float


class TopAssetEntry(BaseModel):
    asset_id: str
    asset_name: str
    asset_type: str
    EL: float
    VaR95: float
    ES95: float


class ModelScoreEntry(BaseModel):
    target: str
    model: str
    metric: str
    value: float


class MetricDoc(BaseModel):
    name: str
    description: str
    direction: str


class InsightResponse(BaseModel):
    latest_common_week: str
    feature_config: Dict[str, str]
    split_summary: List[SplitSummary]
    forecast_stats: ForecastStats
    risk_stats: RiskStats
    training_notes: str
    ablation_guidance: str
    forecast_methodology: str
    risk_methodology: str
    hazard_summary: List[HazardSummaryEntry]
    top_assets: List[TopAssetEntry]
    hazard_series: List[HazardSeriesEntry]
    risk_by_type: List[RiskByTypeEntry]
    report_content: str
    project_overview: str
    model_scores: List[ModelScoreEntry]
    metric_docs: List[MetricDoc]
    model_overview: str


@lru_cache(maxsize=1)
def _get_openai_client() -> Optional[OpenAI]:
    if OpenAI is None:
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def _extract_openai_text(response) -> Optional[str]:
    try:
        chunks: List[str] = []
        for item in getattr(response, "output", []):
            for block in getattr(item, "content", []):
                text = getattr(block, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip() if chunks else None
    except Exception:  # pragma: no cover - defensive
        return None


def _fallback_report(framework: str, context: Dict[str, object]) -> str:
    latest_week = context.get("latest_week", "N/A")
    top_asset = ""
    top_assets = context.get("top_assets") or []
    if top_assets:
        leader = top_assets[0]
        top_asset = (
            f" Highest exposure: {leader.get('asset_name')} ({leader.get('asset_type')}) "
            f"with EL ≈ ${leader.get('EL'):,.0f}."
        )
    return (
        f"{framework.upper()} climate risk note: {context.get('forecast_records', 0)} grid cells are synchronized "
        f"through week {latest_week}, reinforcing monitoring of tail hazards across the active portfolio."
        f"{top_asset}"
    )


def generate_llm_report(framework: str, context: Dict[str, object]) -> str:
    client = _get_openai_client()
    if client is None:
        return _fallback_report(framework, context)

    system_prompt = (
        f"You are a senior climate-risk analyst preparing a professional, insightful {framework.upper()} disclosure. "
        "Use the provided JSON (forecast coverage, hazard summaries, risk_by_type, top_assets, scenario notes) to craft "
        "two tight paragraphs that first describe the key subseasonal forecast signals and then explain how they flow "
        "through the current risk translation into EL/VaR/ES. Follow with a short bullet list of asset-level watch "
        "actions that reference the quantified losses or hazards. Keep the tone executive-ready."
    )
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": json.dumps(context, indent=2, default=str)}]},
            ],
            temperature=0.2,
            max_output_tokens=OPENAI_MAX_TOKENS,
        )
        text = _extract_openai_text(response)
        if text:
            return text
    except Exception as exc:  # pragma: no cover - network failures
        LOGGER.warning("OpenAI report generation failed: %s", exc)
    return _fallback_report(framework, context)


@lru_cache(maxsize=1)
def load_forecast_tiles() -> pd.DataFrame:
    path = MODELS_DIR / "forecasts_latest.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "No forecast parquet found. Run the modeling pipeline to populate models/forecasts_latest.parquet."
        )
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError("Metadata file missing. Build features to generate data/proc/metadata.json.")
    return json.loads(METADATA_PATH.read_text())


@lru_cache(maxsize=1)
def load_risk_report() -> pd.DataFrame:
    if not RISK_PATH.exists():
        raise FileNotFoundError("Risk report missing. Run services.risk.compute to populate data/risk/latest.parquet.")
    return pd.read_parquet(RISK_PATH)


@lru_cache(maxsize=1)
def load_training_summary() -> Optional[dict]:
    if not TRAINING_SUMMARY_PATH.exists():
        return None
    return json.loads(TRAINING_SUMMARY_PATH.read_text())


def convert_variable_value(variable: str, value: float) -> float:
    if value is None:
        return 0.0
    var = str(variable).lower()
    if var.startswith("t"):
        return float(value) - 273.15
    return float(value)


def sample_assets_from_grid(assets: List[AssetIn], grid_df: pd.DataFrame) -> List[dict]:
    rows = []
    for asset in assets:
        asset_dict = asset.model_dump()
        for var in ["precip", "t2m"]:
            subset = grid_df[grid_df["variable"] == var]
            distances = (subset["lat"] - asset.lat) ** 2 + (subset["lon"] - asset.lon) ** 2
            idx = distances.idxmin()
            record = subset.loc[idx]
            asset_dict[f"{var}_mean"] = convert_variable_value(var, record["mean"])
            asset_dict[f"{var}_q90"] = convert_variable_value(var, record["q90"])
        rows.append(asset_dict)
    return rows


@app.post("/forecast", response_model=ForecastResponse)
def forecast_endpoint(
    date: dt.date = Query(..., description="Target forecast date"),
    lead: str = Query("wk34", description="Lead window identifier."),
    assets: Optional[List[AssetIn]] = None,
) -> ForecastResponse:
    try:
        grid_df = load_forecast_tiles()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    base_response = {"grid": {"updated_at": dt.datetime.utcnow().isoformat(), "records": len(grid_df)}}
    asset_payload = sample_assets_from_grid(assets or [], grid_df) if assets else None
    return ForecastResponse(grid=base_response, assets=asset_payload)


@app.post("/risk", response_model=RiskResponse)
def risk_endpoint(request: RiskRequest) -> RiskResponse:
    grid_df = load_forecast_tiles()
    temp_path = DATA_DIR / "tmp" / "api_assets.csv"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asset.model_dump() for asset in request.portfolio]).to_csv(temp_path, index=False)

    curves_path = Path("services/risk/curves.yaml")
    curves = load_curves(curves_path)
    if request.curve_override:
        curves.update(request.curve_override)

    asset_objs = load_assets([temp_path])
    risk_df = compute_risk(asset_objs, grid_df, curves)

    summary = {
        "EL_total": float(risk_df["EL"].sum()),
        "VaR95_total": float(risk_df["VaR95"].sum()),
        "ES95_total": float(risk_df["ES95"].sum()),
    }

    response = RiskResponse(assets=risk_df.to_dict(orient="records"), summary=summary)
    temp_path.unlink(missing_ok=True)
    return response


@app.get("/report", response_model=ReportResponse)
def report_endpoint(
    framework: str = Query("CSRD", description="Reporting framework (CSRD | Pillar3)"),
) -> ReportResponse:
    grid_df = load_forecast_tiles()
    latest_week = str(grid_df.get("week_start", pd.Series()).max())[:10]
    context = {
        "framework": framework,
        "latest_week": latest_week,
        "forecast_records": len(grid_df),
        "top_assets": [],
        "risk_summary": {},
    }
    content = generate_llm_report(framework, context)
    return ReportResponse(framework=framework.upper(), content=content)


@app.get("/insights", response_model=InsightResponse)
def insights_endpoint() -> InsightResponse:
    metadata = load_metadata()
    grid_df = load_forecast_tiles()
    risk_df = load_risk_report()
    risk_method_raw = (
        str(risk_df["risk_method"].iloc[0])
        if not risk_df.empty and "risk_method" in risk_df.columns
        else "parametric"
    )
    risk_method_key = risk_method_raw.lower()

    bbox = {
        "lat_min": float(grid_df["lat"].min()),
        "lat_max": float(grid_df["lat"].max()),
        "lon_min": float(grid_df["lon"].min()),
        "lon_max": float(grid_df["lon"].max()),
    }
    forecast_stats = ForecastStats(
        records=len(grid_df),
        bbox=bbox,
        variables=sorted({str(v) for v in grid_df["variable"].unique()}),
    )
    risk_stats = RiskStats(
        assets=len(risk_df),
        el_mean=float(risk_df["EL"].mean()),
        el_max=float(risk_df["EL"].max()),
        var95_mean=float(risk_df["VaR95"].mean()),
        es95_mean=float(risk_df["ES95"].mean()),
    )

    split_summary = []
    splits = metadata.get("splits", {})
    for name in ("train", "val", "test"):
        weeks = splits.get(name, [])
        split_summary.append(
            SplitSummary(name=name, weeks=len(weeks), example_weeks=weeks[:3])
        )

    feature_config = {
        "ensemble_mode": metadata.get("members", "full"),
        "positional_encoding": metadata.get("positional_encoding", "pe"),
        "lags": str(metadata.get("lags", 1)),
        "latest_common_week": metadata.get("latest_common_week", "NA"),
        "member_order": metadata.get("member_order", "original"),
    }

    training_notes = (
        "Feature pipeline synchronizes cfsv2, era5, and imerg grids through "
        f"{feature_config['latest_common_week']}. "
        f"Lag horizon={feature_config['lags']}, positional encoding={feature_config['positional_encoding']}, "
        f"ensemble setting={feature_config['ensemble_mode']}."
    )
    ablation_guidance = (
        "Use the ablation switches to compare full-ensemble tensors against mean/std compression, "
        "toggle positional encodings (pe vs lat/lon vs none), and experiment with member ordering "
        "to mirror the Beyond Ensemble Averages study."
    )
    forecast_methodology = (
        "Forecast tiles blend CFSv2 ensemble members ({member_order}) with ERA5 reanalysis and IMERG precipitation, "
        "aggregated to weekly means/sums with {lags} lag(s) and {posenc} positional encoding to capture spatial context. "
        "Each week_start marks the verifying week for the subseasonal week-3/4 lead."
    ).format(
        member_order=feature_config["member_order"],
        lags=feature_config["lags"],
        posenc=feature_config["positional_encoding"],
    )
    if risk_method_key == "physrisk":
        risk_methodology = (
            "Losses are computed with the OS-Climate physrisk engine: subseasonal hazards feed custom vulnerability "
            "models to generate fractional damage distributions, which are Monte Carlo sampled into EL/VaR95/ES95."
        )
    else:
        risk_methodology = (
            "Asset losses derive from services/risk/curves.yaml: precipitation excess above dynamic thresholds feeds "
            "a capped damage curve, while heat stress derates capacity via temperature exceedance, both translated into "
            "EL/VaR/ES with conservative tail multipliers."
        )
    risk_methodology += " EL is the mean loss, VaR95 marks the 95th percentile, and ES95 averages the tail beyond VaR."

    hazard_summary = []
    for var in sorted({str(v) for v in grid_df["variable"].unique()}):
        subset = grid_df[grid_df["variable"] == var]
        if subset.empty:
            continue
        unit = "°C" if str(var).lower().startswith("t") else "mm"
        hazard_summary.append(
            HazardSummaryEntry(
                variable=var,
                mean_mean=convert_variable_value(var, subset["mean"].mean()),
                mean_q90=convert_variable_value(var, subset["q90"].mean()),
                max_q90=convert_variable_value(var, subset["q90"].max()),
                unit=unit,
            )
            )

    hazard_series = []
    if "week_start" in grid_df.columns:
        series_df = (
            grid_df.groupby(["week_start", "variable"])["q90"]
            .mean()
            .reset_index()
            .sort_values("week_start")
            .tail(30)
        )
        for _, row in series_df.iterrows():
            hazard_series.append(
                HazardSeriesEntry(
                    week=str(row["week_start"])[:10],
                    variable=str(row["variable"]),
                    q90=convert_variable_value(row["variable"], row["q90"]),
                )
            )

    hazard_payload = [entry.model_dump() for entry in hazard_summary]

    risk_by_type = []
    if not risk_df.empty:
        group_df = risk_df.groupby("asset_type").agg(
            count=("asset_id", "count"),
            el_total=("EL", "sum"),
            var_total=("VaR95", "sum"),
            es_total=("ES95", "sum"),
        )
        for asset_type, row in group_df.sort_values("el_total", ascending=False).iterrows():
            risk_by_type.append(
                RiskByTypeEntry(
                    asset_type=asset_type,
                    count=int(row["count"]),
                    el_total=float(row["el_total"]),
                    var95_total=float(row["var_total"]),
                    es95_total=float(row["es_total"]),
                )
            )

    top_assets = []
    if not risk_df.empty:
        top_assets_df = risk_df.nlargest(10, "EL")[["asset_id", "asset_name", "asset_type", "EL", "VaR95", "ES95"]]
        top_assets = [
            TopAssetEntry(
                asset_id=row["asset_id"],
                asset_name=row["asset_name"],
                asset_type=row["asset_type"],
                EL=float(row["EL"]),
                VaR95=float(row["VaR95"]),
                ES95=float(row["ES95"]),
            )
            for _, row in top_assets_df.iterrows()
        ]

    top_asset_payload = [
        {
            "asset_id": asset.asset_id,
            "asset_name": asset.asset_name,
            "asset_type": asset.asset_type,
            "EL": float(asset.EL),
            "VaR95": float(asset.VaR95),
            "ES95": float(asset.ES95),
        }
        for asset in top_assets
    ]

    report_context = {
        "framework": "CSRD",
        "latest_week": feature_config["latest_common_week"],
        "forecast_records": forecast_stats.records,
        "risk_summary": {
            "assets": risk_stats.assets,
            "portfolio_EL_mean": float(np.nan_to_num(risk_stats.el_mean)),
            "portfolio_VaR95_mean": float(np.nan_to_num(risk_stats.var95_mean)),
            "portfolio_ES95_mean": float(np.nan_to_num(risk_stats.es95_mean)),
            "risk_method": risk_method_raw,
        },
        "top_assets": top_asset_payload,
        "hazards": hazard_payload,
    }
    report_content = generate_llm_report("CSRD", report_context)
    project_overview = (
        "GulfCast ingests Copernicus ERA5 reanalysis, IMERG precipitation, and CFSv2 ensembles across CONUS, "
        "builds synchronized weekly features with positional encodings and lagged predictors, trains linear/RF/stack "
        "baselines plus quantile heads, and routes forecasts into the risk engine for asset-level loss metrics."
    )
    model_overview = (
        "Model vs data panels benchmark the linear regression, random forest, stacking, and 0.90 quantile heads on "
        "held-out weeks. Lower MSE/Pinball indicates tighter fit, while R² near 1.0 signals strong skill. These "
        "forecasts feed the risk translation module; there is no separate ML model for losses."
    )

    model_scores = []
    training_summary = load_training_summary()
    if training_summary:
        for target, model_dict in training_summary.items():
            for model_name, info in model_dict.items():
                if "mse" in info:
                    model_scores.append(
                        ModelScoreEntry(
                            target=target,
                            model=model_name,
                            metric="MSE",
                            value=float(info["mse"]),
                        )
                    )
                if "pinball" in info:
                    label = info.get("name", f"{model_name}_pinball")
                    model_scores.append(
                        ModelScoreEntry(
                            target=target,
                            model=label,
                            metric="Pinball",
                            value=float(info["pinball"]),
                        )
                    )
                if "r2" in info:
                    model_scores.append(
                        ModelScoreEntry(
                            target=target,
                            model=model_name,
                            metric="R2",
                            value=float(info["r2"]),
                        )
        )

    metric_docs = [
        MetricDoc(
            name="MSE",
            description="Mean Squared Error between forecast and validation truth; penalizes large deviations.",
            direction="Lower is better",
        ),
        MetricDoc(
            name="R2",
            description="Coefficient of determination measuring explained variance on validation data.",
            direction="Higher (closer to 1) is better",
        ),
        MetricDoc(
            name="Pinball",
            description="Quantile (τ=0.90) loss capturing calibration of the upper-tail forecasts.",
            direction="Lower is better",
        ),
    ]

    return InsightResponse(
        latest_common_week=feature_config["latest_common_week"],
        feature_config=feature_config,
        split_summary=split_summary,
        forecast_stats=forecast_stats,
        risk_stats=risk_stats,
        training_notes=training_notes,
        ablation_guidance=ablation_guidance,
        forecast_methodology=forecast_methodology,
        risk_methodology=risk_methodology,
        hazard_summary=hazard_summary,
        top_assets=top_assets,
        hazard_series=hazard_series,
        risk_by_type=risk_by_type,
        report_content=report_content,
        project_overview=project_overview,
        model_scores=model_scores,
        metric_docs=metric_docs,
        model_overview=model_overview,
    )


@app.get("/forecast/weeks")
def forecast_weeks() -> dict:
    meta = load_metadata()
    try:
        grid_df = load_forecast_tiles()
        week_series = pd.to_datetime(grid_df.get("week_start", pd.Series(dtype="datetime64[ns]"))).dropna()
        weeks = sorted({str(w.date()) for w in week_series})
        weeks_by_var = {}
        if "variable" in grid_df.columns:
            tmp = grid_df[["variable", "week_start"]].dropna()
            tmp["week_start"] = pd.to_datetime(tmp["week_start"])
            for var, sub in tmp.groupby("variable"):
                weeks_by_var[str(var)] = sorted({str(w.date()) for w in sub["week_start"]})
        next_week = None
        if weeks:
            last = pd.to_datetime(weeks[-1])
            next_week = str((last + pd.Timedelta(days=7)).date())
    except Exception:
        weeks, next_week, weeks_by_var = [], None, {}
    return {
        "splits": meta.get("splits", {}),
        "available_weeks": weeks,
        "weeks_by_variable": weeks_by_var,
        "next_forecast_week": next_week,
    }


@app.get("/forecast/map")
def get_forecast_map(week: str, var: str = "t2m", mode: str = "weekly"):
    """
    Return gridded forecast data for mapping (lat/lon/value).
    """
    try:
        df = load_forecast_tiles()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No forecasts available")
    
    # Filter by week and variable
    subset = df[
        (df["week_start"] == pd.Timestamp(week)) & 
        (df["variable"] == var)
    ]
    
    if subset.empty:
        # Fall back to latest available week for this variable
        var_data = df[df["variable"] == var]
        if var_data.empty:
            raise HTTPException(status_code=404, detail=f"No data for variable {var}")
        latest = var_data["week_start"].max()
        subset = df[(df["week_start"] == latest) & (df["variable"] == var)]
        week = str(latest)[:10]
    
    # Use 'mean' or 'q90' column based on mode
    value_col = "q90" if mode == "quantile" else "mean"
    if value_col not in subset.columns:
        raise HTTPException(status_code=400, detail=f"Column {value_col} not found")
    
    # Handle duplicates: group by lat/lon and take mean
    subset = subset.groupby(["lat", "lon"], as_index=False).agg({
        value_col: "mean",
        "mean_model": "first",  # Keep first model name for reference
        "quantile_model": "first"
    })
    
    # Build response with native Python types
    grid = subset[["lat", "lon", value_col]].copy()
    grid.rename(columns={value_col: "value"}, inplace=True)
    
    # Convert numpy types to Python float
    grid["lat"] = grid["lat"].astype(float)
    grid["lon"] = grid["lon"].astype(float)
    grid["value"] = grid["value"].astype(float)
    
    # Convert temperature from Kelvin to Celsius if needed
    if var == "t2m" and grid["value"].max() > 200:
        grid["value"] = grid["value"] - 273.15

    return {
        "week": week,
        "variable": var,
        "mode": mode,
        "grid": grid.to_dict(orient="records")
    }


@app.get("/risk/hazards")
def risk_hazards() -> dict:
    cfg = {}
    if RISK_CONFIG_PATH.exists():
        try:
            import yaml  # local import

            cfg = yaml.safe_load(RISK_CONFIG_PATH.read_text()) or {}
        except Exception:
            cfg = {}
    return {
        "risk_method": cfg.get("risk_method", "parametric"),
        "hazards": cfg.get("hazards", ["heat", "drought", "flood"]),
        "scenario": cfg.get("scenario", {}),
    }


@app.get("/risk/latest")
def risk_latest(limit: int = 500) -> dict:
    df = load_risk_report()
    df = df.sort_values("EL", ascending=False)
    if limit and limit > 0:
        df = df.head(limit)
    records = df.to_dict(orient="records")
    meta = {
        "risk_method": df["risk_method"].iloc[0] if "risk_method" in df.columns and not df.empty else "parametric",
        "hazards_used": df["hazards_used"].iloc[0] if "hazards_used" in df.columns and not df.empty else "",
    }
    return {"assets": records, "meta": meta}
