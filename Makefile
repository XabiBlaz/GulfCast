SHELL := /usr/bin/env bash

.PHONY: install ingest-assets ingest-cfsv2 ingest-era5 ingest-imerg features train score risk api ui docker-build docker-up docker-down lint format

install:
	poetry install

ingest-assets:
	poetry run python -m services.ingest.assets --out assets --bbox="-125,24,-66,50"

ingest-cfsv2:
	poetry run python -m services.ingest.cfsv2 --vars pr,t2m --lead wk34 --bbox="-125,24,-66,50" --start 2016-01 --end 2022-12 --out data/raw/cfsv2

ingest-era5:
	python -m services.ingest.era5  --start 2016-01 --end 2016-03 --bbox="-125,24,-66,50" --out data/raw/era5 --chunk-lat 64 --chunk-lon 64

ingest-imerg:
	python -m services.ingest.imerg --start 2016-01 --end 2016-03 --bbox="-125,24,-66,50" --out data/raw/imerg --chunk-lat 64 --chunk-lon 64

features: # Pick the ablation settings if wanted
	python -m services.features.build --inputs data/raw --out data/proc --members full --lags 1 --posenc pe

train:
	python -m services.model.train --data data/proc --out models --models lr,rf,stack --targets t2m,precip --quantiles 0.9 --features ensemble_full --posenc pe --member_order original

score:
	python -m services.model.score --data data/proc/test.parquet --out models/forecasts_latest.parquet --models-dir models --targets t2m,precip --quantile 0.9

risk:
	python -m services.risk.compute --portfolio assets/plants.csv,assets/ports.csv --forecasts models/forecasts_latest.parquet --method physrisk --out data/risk/latest.parquet

api:
	poetry run uvicorn services.api.main:app --host 0.0.0.0 --port 8000

ui:
	cd services/ui && npm install && npm run dev

### IF USING DOCKER ###

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Run full pipeline in Docker
docker-pipeline:
	docker-compose exec api poetry run python -m services.ingest.era5 --var t2m,tp --agg weekly --bbox "-125,24,-66,50" --start 2016-01 --end 2016-03 --out data/raw/era5
	docker-compose exec api poetry run python -m services.ingest.imerg --agg weekly --bbox "-125,24,-66,50" --start 2016-01 --end 2016-03 --out data/raw/imerg
	docker-compose exec api poetry run python -m services.ingest.assets --out assets --bbox "-125,24,-66,50"
	docker-compose exec api poetry run python -m services.features.build --inputs data/raw --out data/proc --members full --lags 1 --posenc pe
	docker-compose exec api poetry run python -m services.model.train --data data/proc --out models --models lr,rf,stack --targets t2m,precip --quantiles 0.9
	docker-compose exec api poetry run python -m services.model.score --data data/proc/test.parquet --out models/forecasts_latest.parquet --models-dir models --targets t2m,precip --quantile 0.9
	docker-compose exec api poetry run python -m services.risk.compute --assets assets/infrastructure.parquet --forecasts models/forecasts_latest.parquet --curves config/risk.yaml --out data/risk/latest.parquet --method physrisk

lint:
	poetry run ruff check services

format:
	poetry run black services
