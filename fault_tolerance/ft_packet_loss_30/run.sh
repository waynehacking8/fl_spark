#!/usr/bin/env bash
set -e

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$ROOT_DIR"

BASE_COMPOSE="original/docker-compose.yml"
OVERRIDE="fault_tolerance/ft_packet_loss_30/docker-compose.override.yml"

printf "\n[FT] Cleaning previous stack (packet loss 30%)...\n"
docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" down -v || true

printf "\n[FT] Building participant images with NET_ADMIN...\n"
docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" build --pull --no-cache fl-participant-{1..16} || true

printf "\n[FT] Starting stack with 30% packet loss...\n"
docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" up -d fl-server fl-participant-{1..16}

printf "[FT] Stack running. Monitor with 'docker logs -f fl-server | cat' .\n" 