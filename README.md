# GulfCast – Subseasonal (Week‑3/4) Climate Hazard & Asset Risk Intelligence

A production‑style, research‑grade pipeline turning subseasonal (week‑3/4) temperature & precipitation forecast signals into portfolio‑level Expected Loss, VaR95, and ES95 with interactive scenario stress and automated disclosure reporting.

---

## 1. Why This Matters
Subseasonal (15–28 day) lead times bridge weather and seasonal horizons—critical for energy dispatch, petrochemical throughput, port operations, and early financial risk posture. GulfCast demonstrates how to:
- Fuse multi‑source forecast ensembles (CFSv2) with observational baselines (ERA5, IMERG).
- Translate calibrated probabilistic hazards into forward‑looking physical loss distributions.
- Provide asset‑level triage (watch list) plus executive‑ready AI‑assisted disclosure text.

---

## 2. Core Capabilities and UI Overview
| Domain | Capability |
|--------|------------|
| Data Ingest | CFSv2 wk3/4 ensemble tiles, ERA5 reanalysis (t2m), IMERG precipitation, real asset inventories (ports, power plants). |
| Feature Engineering | Weekly aggregation, lag tensors, positional encodings, ensemble member ablations (mean/std vs full). |
| Modeling | Linear, Random Forest, Quantile regressors, stacking meta‑learner, pinball loss evaluation. |
| Risk Translation | Parametric or PhysRisk‑style impact adapter mapping hazard quantiles → EL / VaR95 / ES95. |
| UI | React + Leaflet geospatial map, asset risk table, scenario sliders (ΔT, flood RP, drought scale), portfolio metrics. |
| Reporting | FastAPI endpoint producing CSRD / Pillar‑3 style narrative via LLM synthesis of current hazard + loss posture. |
| Scenarios | Local slider perturbations (visual) + full recompute workflow (background job) with progress polling. |
| Deployment | Poetry local dev, Docker Compose stack, scripted one‑shot pipelines. |

### UI Overview

Animated interaction:

![UI Demo GIF](Animation.gif)
---

## 3. Scientific & Practical Rationale
- Week‑3/4 signals exhibit degraded deterministic skill but retain probabilistic value for heat stress & hydrometeorological extremes.
- Ensemble post‑processing (quantiles + stacking) tightens spread and sharpens tails relevant for capital at risk.
- Translating hazard distributions to loss enables pre‑emptive resource staging, downtime scheduling, and liquidity buffering.
- Scenario deltas (ΔT warming, drought stress multipliers, flood return period shifts) approximate forward pathways under near‑term climate volatility.

### 3.1 How the week‑3/4 forecasts are produced (intuitive summary)

This pipeline follows the common “dynamical ensemble → harmonize → statistical post‑processing → calibrated targets” pattern described in the attached paper, adapted to GulfCast:

- Start from physics-based ensembles
  - Use NOAA CFSv2 subseasonal runs as the primary source for week‑3/4 temperature and precipitation signals.
  - Treat each ensemble member as a noisy view of the large‑scale circulation (MJO, teleconnections).

- Put all data on a comparable footing
  - Regrid to a common grid and align into verifying weekly windows.
  - Form anomalies vs an observational climatology (e.g., ERA5 for temperature, IMERG for precipitation) so forecasts are expressed as “departures from normal.”

- Engineer stable predictors
  - Aggregate to weekly means and upper‑tail quantiles (e.g., q90).
  - Add simple spatial encodings and a few lags to capture persistence without overfitting.

- Calibrate and combine (statistical post‑processing)
  - Fit lightweight models (linear, RF, quantile regressors) that learn a mapping from raw model anomalies to observed outcomes.
  - Optimize pinball (quantile) loss for tails and MSE for means.
  - Stack models with a meta‑learner to reduce bias and tame over/under‑dispersion.

- Produce wk‑3/4 grids and uncertainty
  - Score current runs to generate weekly grids for each target (t2m mean, t2m q90, precip mean, precip q90).
  - Use predicted quantiles (and ensemble spread where helpful) to express confidence, tighter spread ⇒ higher confidence.

- Verify and monitor skill
  - Report MSE, R², and quantile (pinball) loss by target/region.
  - Expect temperature skill > precipitation at these leads, tails are hardest.

- Why this works (intuition)
  - Dynamical models carry the large‑scale pattern signal, post‑processing corrects systematic bias and miscalibrated spread.
  - Stacking leverages complementary strengths across simple learners while keeping variance in check.

- Known limitations
  - Precipitation extremes are localized and noisier, skill decays with lead time and varies by region/season.
  - Rare events remain uncertain, treat q90 as directional risk, not certainties.

  References: See the wk‑3/4 forecasting paper [Beyond Ensemble Averages: Leveraging Climate Model Ensembles for Subseasonal Forecasting](model_ensembles_subseasonal_forecasting.pdf) for principles of ensemble post‑processing and calibration, operational context from CPC’s Week‑3/4 outlooks and similar dynamical systems informs these steps.
---

## 4. Metrics Interpreted
- EL (Expected Loss): Mean of modeled loss distribution (baseline provisioning signal).
- VaR95: Loss threshold exceeded with 5% probability (capital adequacy stress).
- ES95 (Expected Shortfall): Conditional mean loss beyond VaR95 (tail severity—risk appetite & hedging trigger).
Interpretation: (ES95 − VaR95) gauges tail thickness, (VaR95 / EL) highlights concentration risk, spatial clustering informs correlated exposure.

---

## 5. System Architecture (Written Overview)
- Data Ingestion
  - Sources: CFSv2 (subseasonal wk3/4), ERA5 (t2m), IMERG (precip), asset registries (ports, plants).
  - Jobs: CLI modules under services/ingest write versioned parquet/zarr into data/raw.
- Feature Build
  - Weekly alignment, lagged tensors, ensemble statistics, positional encodings.
  - Outputs consolidated parquet datasets in data/proc for training and scoring.
- Modeling
  - Baselines (linear), tree models (RF), quantile regressors, and a stacking meta‑learner.
  - Targets: t2m mean and q90, precip mean and q90. Loss: pinball for quantiles, MSE for means.
- Forecast Serving
  - Scoring writes models/forecasts_latest.parquet with columns: week_start, lat, lon, variable, mean, q90, model tags.
  - API aggregates duplicates and exposes /forecast/map as flat {lat, lon, value} records.
- Risk Engine
  - Parametric method or PhysRisk‑style adapter that converts hazard summaries (mean, q90) into simplified loss distributions and computes EL, VaR95, ES95.
- API (FastAPI)
  - Endpoints for forecast weeks, map tiles, portfolio risk, scenario runs, and LLM reporting.
  - Health and metadata endpoints for orchestration.
- UI (React + Vite + Leaflet)
  - Pages for Forecast Map and Risk Dashboard with selectors, legends, and interactive markers.
  - Scenario sliders apply local visual what‑ifs, full recompute available via background job endpoints.
- Automation
  - Makefile + shell scripts for local and Docker pipelines, reproducible runs from raw ingest to UI.
- Deployment
  - Local: Poetry + Uvicorn + Vite dev server.
  - Docker: docker compose build/up with mounted volumes for data/models/assets/config.
- Storage & Versioning
  - Artifacts under models/ and data/risk/, optionally extended with DVC/Git LFS for large assets.
  - Training summaries in models/training_summary.json.

---

## 6. Feature Highlight Deep Dive
1. Ensemble Handling: Option to collapse members to mean/std (compression) or retain full distribution for stacking.
2. Positional Encoding: Adds sinusoidal embeddings of lat/lon improving spatial generalization beyond raw coordinates.
3. Quantile Heads: Directly optimize pinball loss for upper tails (e.g., Q90) used in risk conversion.
4. PhysRisk Adapter: Lightweight bridge producing ImpactDistrib objects (two‑bin approximation) for VaR/ES extraction.
5. Scenario Layer: GPU‑free perturbation applied client‑side for instantaneous hazard what‑ifs, server recompute for authoritative portfolio refresh.
6. LLM Disclosure: Structured JSON (hazards + losses) → templated analytical prose (governance‑ready).

---

## 7. Quick Start (Local)
```bash
poetry install
# If you plan to use the PhysRisk-based method:
# poetry install -E risk
make ingest-assets ingest-cfsv2 ingest-era5 ingest-imerg
make features
make train
poetry run python -m services.model.score
make risk
make api   # http://localhost:8000
make ui    # http://localhost:5173
```
Validate: ensure `models/forecasts_latest.parquet` and `data/risk/latest.parquet` exist before opening the dashboard.

---

## 8. One‑Shot Pipelines
Local (runs build → train → score → risk, then API + UI):
```bash
./run_local.sh
```
Docker (wraps `docker compose build && docker compose up`):
```bash
./run_docker.sh
```
Need only the services (without the scripted rebuild)?
```bash
docker compose up --build
```

---

## 9. Key API Endpoints
| Endpoint | Purpose |
|----------|---------|
| GET /forecast/weeks | List available verifying weeks. |
| GET /forecast/map?week=YYYY-MM-DD&var=t2m|precip&mode=weekly|quantile | Gridded mean or q90 values (week‑3/4 lead). |
| POST /risk | Portfolio EL/VaR95/ES95. |
| GET /report?framework=CSRD | Executive narrative. |
| POST /scenarios/run | Launch recompute with scenario overrides. |
| GET /scenarios/status/{job_id} | Progressive job state. |

---

## 10. Extending
| Goal | Action |
|------|--------|
| Add new hazard (e.g., wind) | Extend ingest + feature schema, add model target, update risk curves. |
| Add week‑1/2 lead | Ingest CFSv2 `--lead wk12`, train parallel model set, add `lead` param to endpoints/UI. |
| Advanced calibration | Insert conformal interval module before risk translation. |
| Spatial deep model | Implement `train_unet` placeholder (GPU Dockerfile). |
| Additional quantiles | Extend quantile list, recompute stacking, adapt risk adapter for richer bins. |

---

## 11. Reliability & Skill Notes
- Temperature skill (corr) > precipitation at subseasonal leads → rely more on Q90 precipitation spread.
- Use Mean for planning baselines, Q90 for contingency provisioning.
- Monitor drift: retrain when rolling pinball loss degradation > threshold (add CI pipeline trigger).

---

## 12. Interpreting Asset Table
- EL Rank: Prioritize top decile for maintenance scheduling.
- (VaR95 − EL)/EL: Escalation factor—higher implies latent tail risk.
- ES95/VaR95 > 1.1: Fat‑tail candidate—consider hedging or buffer allocation.
- Spatial clustering (map) + risk concentration → correlated outage scenario planning.

---

## 13. Security & Reproducibility
- Deterministic seeds for training stages.
- Limited LLM token window, no sensitive data echo.
- All artifacts (models, metrics, risk) versioned via timestamped filenames (extend with DVC/Git LFS if needed).

---

## 14. Recruiter / Reviewer Note
This project demonstrates:
- Full‑stack ownership: data engineering → ML modeling → probabilistic risk translation → API/UI integration.
- Applied climate analytics: ensemble post‑processing, quantile modeling, tail risk framing.
- Systems rigor: modular pipelines, containerization, scenario orchestration, narrative automation.
- Practical impact orientation: turning forecast uncertainty into actionable financial & operational signals.

---

## 15. Next Enhancements (Roadmap)
- Lead‑differentiated comparative skill dashboard.
- Conformal calibrated dynamic quantile envelopes.
- Multi‑hazard correlation modeling (copula or generative).
- Automated retraining scheduler (GitHub Actions + artifact sync).
- Geospatial UNet baseline and SHAP interpretability layer.

---

Focused, modular, and extensible, GulfCast is a compact reference blueprint for operational subseasonal climate risk intelligence.
