#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
# Put the headset phone's adb on Wi-Fi, so its USB port (and the laptop's) is free.
#
# The phone is the hotspot, so its address is the laptop's default gateway.
#   adb-wifi.sh          connect to the phone over Wi-Fi (works until the phone reboots)
#   adb-wifi.sh --setup  first time, or after a phone reboot: the phone must be on USB
set -euo pipefail

PORT=5555
gw=$(ip route | awk '/default/{print $3; exit}')
[ -n "$gw" ] || { echo "no default gateway: is the laptop on the phone's hotspot?" >&2; exit 1; }

if [ "${1:-}" = "--setup" ]; then
    # The headset phone is the USB device whose hotspot address is our gateway.
    serial=""
    for s in $(adb devices | awk 'NR>1 && $2=="device" && $1 !~ /:/{print $1}'); do
        if adb -s "$s" shell ip -4 addr 2>/dev/null | grep -q "inet $gw/"; then serial=$s; break; fi
    done
    [ -n "$serial" ] || { echo "no USB phone with address $gw found" >&2; exit 1; }
    adb -s "$serial" tcpip $PORT
    sleep 3
fi

adb connect "$gw:$PORT"
adb -s "$gw:$PORT" get-state >/dev/null && echo "Headset phone is on adb over Wi-Fi: $gw:$PORT (the USB cable can be unplugged)."
