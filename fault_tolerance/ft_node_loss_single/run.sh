#!/usr/bin/env bash
set -e

# 保證 network 存在（避免 Docker race condition）
# docker network create original_federated_net 2>/dev/null || true

# Root of project
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
cd "$ROOT_DIR"

# Compose files
BASE_COMPOSE="original/docker-compose.yml"
OVERRIDE="fault_tolerance/ft_node_loss_single/docker-compose.override.yml"

# 1. 先確保 baseline network 乾淨
printf "\n[FT] Cleaning previous stack...\n"
docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" down -v || true

# 2. 啟動 baseline 服務 (傳統 FL server + 16 participants)
printf "\n[FT] Starting baseline stack...\n"
docker compose -f "$BASE_COMPOSE" -f "$OVERRIDE" up -d fl-server fl-participant-{1..16}

# 3. 等待指定回合時間（粗估 5 round ≈ 120s，可視需要調整）
DELAY=${DELAY:-120}
printf "[FT] Sleeping $DELAY s before fault injection...\n"
sleep "$DELAY"

# 4. 故障注入：停止一個 participant
TARGET=${TARGET:-fl-participant-9}
printf "[FT] Injecting node loss: stopping $TARGET ...\n"
docker stop "$TARGET"

printf "[FT] Fault injected. Monitoring with 'docker logs -f fl-server | cat' ...\n" 