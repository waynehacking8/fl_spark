#!/usr/bin/env bash
set -e

# Root of project
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$ROOT_DIR"

BASE_COMPOSE="original/docker-compose.yml"
OVERRIDE="fault_tolerance/ft_node_loss_multi/docker-compose.override.yml"

printf "\n[FT] Cleaning previous stack (multi-node loss)...\n"
docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" down -v || true

printf "\n[FT] Starting baseline stack...\n"
docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" up -d fl-server fl-participant-{1..16}

# 等待指定時間後注入多節點故障
DELAY=${DELAY:-240} # 預設等到 Round 10 左右
printf "[FT] Sleeping $DELAY s before fault injection...\n"
sleep "$DELAY"

TARGETS=(fl-participant-5 fl-participant-6 fl-participant-11 fl-participant-12)
printf "[FT] Injecting multi node loss: stopping ${TARGETS[*]} ...\n"
for t in "${TARGETS[@]}"; do
  docker stop "$t" || true
done

printf "[FT] Fault injected. Monitor with 'docker logs -f fl-server | cat' ...\n" 