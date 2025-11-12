#!/usr/bin/env bash
#
# run_local.sh
# Convenience wrapper to rebuild features, train models, score forecasts,
# compute risk, and launch the API + UI locally with one command.
#
# Tunable parameters (override via environment variables before running):
#   MEMBERS=full|meanstd
#   POSENC=pe|latlon|none
#   LAGS=1
#   MEMBER_ORDER=original|sorted|shuffled
#   METHOD=physrisk|parametric
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MEMBERS="${MEMBERS:-full}"
POSENC="${POSENC:-pe}"
LAGS="${LAGS:-1}"
MEMBER_ORDER="${MEMBER_ORDER:-original}"
METHOD="${METHOD:-physrisk}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

# Optional ingestion examples (uncomment if you need to refresh raw data)
# log "Ingesting ERA5..."
# poetry run python -m services.ingest.era5 --var t2m --agg weekly \
#   --bbox "-125,24,-66,50" --start 2016-01 --end 2016-03 --out data/raw/era5

# log "Ingesting IMERG..."
# poetry run python -m services.ingest.imerg --agg weekly \
#   --bbox "-125,24,-66,50" --start 2016-01 --end 2016-03 \
#   --out data/raw/imerg --chunk-lat 64 --chunk-lon 64

# log "Ingesting CFSv2..."
# poetry run python -m services.ingest.cfsv2 --var t2m --agg weekly \
#   --bbox "-125,24,-66,50" --start 2016-01 --end 2016-03 --out data/raw/cfsv2

log "Building features (members=${MEMBERS}, posenc=${POSENC}, lags=${LAGS})..."
poetry run python -m services.features.build \
  --inputs data/raw \
  --out data/proc \
  --members "${MEMBERS}" \
  --lags "${LAGS}" \
  --posenc "${POSENC}"

log "Training models..."
poetry run python -m services.model.train \
  --data data/proc \
  --out models \
  --models lr,rf,stack \
  --targets t2m,precip \
  --quantiles 0.9 \
  --features "ensemble_${MEMBERS}" \
  --posenc "${POSENC}" \
  --member_order "${MEMBER_ORDER}"

log "Scoring forecasts..."
poetry run python -m services.model.score \
  --data data/proc/test.parquet \
  --out models/forecasts_latest.parquet \
  --models-dir models \
  --targets t2m,precip \
  --quantile 0.9

log "Computing risk with method=${METHOD}..."
poetry run python -m services.risk.compute \
  --portfolio assets/plants.csv,assets/ports.csv \
  --forecasts models/forecasts_latest.parquet \
  --out data/risk/latest.parquet \
  --method "${METHOD}"

log "Starting FastAPI backend on :8000..."
poetry run uvicorn services.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
trap 'log "Stopping FastAPI (pid=${API_PID})" && kill "${API_PID}"' EXIT

log "Starting Vite dev server for the UI..."
cd services/ui
npm install
npm run dev
