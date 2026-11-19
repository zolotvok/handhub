#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
# Connect the phone's WiVRn client to the WiVRn server on this laptop.
#
# The Cardboard has no controllers, so the client's in-VR lobby cannot be
# operated. This starts the client with a connect link over adb instead.
#
#   wivrn-connect.sh           connect over Wi-Fi (phone must already be paired)
#   wivrn-connect.sh --pair    also pair: enables pairing and passes the PIN
#   wivrn-connect.sh --usb     stream over the USB cable (adb reverse) instead
#   wivrn-connect.sh --restart stop the client first: needed when the picture stays black
#                              after the screen was off or the app was in the background
set -euo pipefail

APP=org.meumeu.wivrn.local
PORT=9757
pair=0 usb=0 restart=0
for a in "$@"; do
    case "$a" in
        --pair) pair=1 ;;
        --usb) usb=1 ;;
        --restart) restart=1 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

pgrep -x wivrn-server >/dev/null || { echo "wivrn-server is not running: start it with 'DRI_PRIME=1 wivrn-dashboard'" >&2; exit 1; }
# Several phones may be on adb (fixed cameras): pick the one that has the client installed.
serial=""
for s in $(adb devices | awk 'NR>1 && $2=="device"{print $1}'); do
    if adb -s "$s" shell pm path "$APP" 2>/dev/null | grep -q package:; then serial=$s; break; fi
done
[ -n "$serial" ] || { echo "no phone with $APP on adb (USB debugging on, cable connected?)" >&2; exit 1; }
adb() { command adb -s "$serial" "$@"; }

if [ "$restart" = 1 ]; then
    adb shell am force-stop "$APP"
    sleep 1
fi

if [ "$usb" = 1 ]; then
    adb reverse tcp:$PORT tcp:$PORT >/dev/null
    host=127.0.0.1
else
    host=$(ip -4 route get 1.1.1.1 | grep -oP 'src \K[0-9.]+')
fi

auth=""
if [ "$pair" = 1 ]; then
    pin=$(wivrnctl pair -d 2 | grep -oE '[0-9]{6}' | head -1)
    [ -n "$pin" ] || { echo "could not get a PIN from wivrnctl" >&2; exit 1; }
    auth=":$pin@"
fi

adb shell am start -a android.intent.action.VIEW -d "wivrn+tcp://$auth$host:$PORT" "$APP" >/dev/null
echo "Sent connect link for $host:$PORT to the phone."
