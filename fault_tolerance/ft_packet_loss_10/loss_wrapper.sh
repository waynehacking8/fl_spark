#!/usr/bin/env bash
set -e

# 進容器後 add netem packet loss
RATE=${LOSS_RATE:-10}
# 若 tc 已存在規則先刪除
(tc qdisc del dev eth0 root || true)
# 加入 netem
 tc qdisc add dev eth0 root netem loss ${RATE}%

# 執行原本 participant 腳本，所有參數照轉
exec python /app/traditional_code/participant.py "$@" 