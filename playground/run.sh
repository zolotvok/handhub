#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
# Run the HandHub playground. Extra arguments are passed through to Godot.
#   ./run.sh            VR mode (OpenXR; falls back to Direct if OpenXR is unavailable), DRI_PRIME=1
#   ./run.sh --direct   desktop window, no OpenXR (HANDHUB_DIRECT=1 and --xr-mode off)
# Godot binary: GODOT (default: godot4 or godot from the PATH).
# Hub address: HANDHUB_URL (default ws://127.0.0.1:8765/ws), HANDHUB_STATUS_PORT (default 39571).
set -euo pipefail

# Godot 4.4 or newer: set GODOT to the binary, or have godot4 / godot on the PATH.
GODOT=${GODOT:-$(command -v godot4 || command -v godot || true)}
[ -n "$GODOT" ] || { echo "Godot not found: install Godot 4 or set GODOT=/path/to/godot" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

args=()
direct=0
for a in "$@"; do
	if [[ "$a" == "--direct" ]]; then direct=1; else args+=("$a"); fi
done

if (( direct )); then
	export HANDHUB_DIRECT=1
	# project.godot enables OpenXR, so the engine would otherwise start the OpenXR runtime itself
	args=(--xr-mode off "${args[@]}")
else
	export DRI_PRIME=1
fi

exec "$GODOT" --path "$HERE" "${args[@]}"
