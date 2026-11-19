#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
# Restart the hub in the background and wait until it answers.
# Stops the old one first and waits for its port to be free: starting too early fails
# with "address already in use" and leaves no hub running.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=${HANDHUB_LOG:-${HANDHUB_DATA:-$HOME/.local/share/handhub}/hub.log}
mkdir -p "$(dirname "$LOG")"

for pid in $(pgrep -f "[p]ython -m hub$" || true); do kill "$pid" 2>/dev/null || true; done
for _ in $(seq 1 40); do
    pgrep -f "[p]ython -m hub$" >/dev/null || break
    sleep 0.25
done
for pid in $(pgrep -f "[p]ython -m hub$" || true); do kill -9 "$pid" 2>/dev/null || true; done
for _ in $(seq 1 40); do
    ss -ltn 2>/dev/null | grep -q ":8765 " || break
    sleep 0.25
done
# camera streamers left behind on the phones would keep their cameras busy
for dev in $(adb devices 2>/dev/null | awk 'NR>1 && $2=="device"{print $1}'); do
    adb -s "$dev" shell pkill -f scrcpy-server 2>/dev/null || true
done

setsid nohup .venv/bin/python -m hub >"$LOG" 2>&1 </dev/null &
for _ in $(seq 1 80); do
    if curl -s -m 1 -o /dev/null http://127.0.0.1:8765/api/config; then echo "hub is up"; exit 0; fi
    sleep 0.5
done
echo "hub did not come up, see $LOG" >&2
exit 1
