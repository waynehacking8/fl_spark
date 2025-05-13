#!/usr/bin/env bash
set -e

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$ROOT_DIR"

printf "\n[FT] Patching shards to simulate 30% data loss...\n"
python fault_tolerance/ft_shard_loss_30/patch_shards.py

BASE_COMPOSE="original/docker-compose.yml"
OVERRIDE="fault_tolerance/ft_shard_loss_30/docker-compose.override.yml"

docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" down -v || true

docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" up -d fl-server fl-participant-{1..16}

echo "[FT] Stack running. Check logs." 