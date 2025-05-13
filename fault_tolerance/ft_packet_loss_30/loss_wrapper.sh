#!/usr/bin/env bash
set -e

RATE=${LOSS_RATE:-30}
(tc qdisc del dev eth0 root || true)
tc qdisc add dev eth0 root netem loss ${RATE}%
exec python /app/traditional_code/participant.py "$@" 