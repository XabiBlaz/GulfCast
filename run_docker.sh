#!/usr/bin/env bash
#
# run_docker.sh
# Build and launch the full stack via docker-compose with one command.
# Assumes docker-compose.yml already defines api/ui services.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

log "Building docker images..."
docker compose build

log "Starting dockerized services (Ctrl+C to stop)..."
docker compose up
