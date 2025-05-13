#!/usr/bin/env bash
set -e

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$ROOT_DIR"

# 先 patch shards
printf "\n[FT] Patching shards to simulate 10% data loss...\n"
python fault_tolerance/ft_shard_loss_10/patch_shards.py

BASE_COMPOSE="original/docker-compose.yml"
OVERRIDE="fault_tolerance/ft_shard_loss_10/docker-compose.override.yml"

printf "\n[FT] Cleaning previous stack (shard loss 10%)...\n"
docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" down -v || true

printf "\n[FT] Starting stack with shard loss 10%...\n"
docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" up -d fl-server fl-participant-{1..16}

printf "[FT] Stack running. Monitor with 'docker logs -f fl-server | cat' .\n" 