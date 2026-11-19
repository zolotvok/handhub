# HandHub playground

A small Godot 4 (4.7, Mobile renderer) test room for the HandHub optical hand tracker. It draws the
tracked hands as 21-joint skeletons and lets you grab, throw and poke things with them. There are no
controllers: everything is driven by the hub's hand data. Built entirely in code from primitive meshes
and flat colours (no assets, no addons).

## Run

```
./run.sh --direct     # desktop window: RMB held = mouse-look, WASD = move, Esc = quit
./run.sh              # VR mode (OpenXR / WiVRn), DRI_PRIME=1; falls back to Direct if OpenXR is unavailable
```

Extra arguments go to Godot (`./run.sh --direct --headless --quit-after 300`). `--direct` also passes
`--xr-mode off`, because `project.godot` enables OpenXR and the engine would otherwise start the OpenXR
runtime itself before any script runs. If you start Godot yourself with only `HANDHUB_DIRECT=1`, the
script skips OpenXR but the engine may already have created an OpenXR instance.

Mode is chosen at startup: VR if the OpenXR interface initialises (and `HANDHUB_DIRECT` is not `1`),
otherwise Direct. The room works with no hub running; the client retries every second.

## Environment variables

| variable | default | meaning |
| --- | --- | --- |
| `HANDHUB_URL` | `ws://127.0.0.1:8765/ws` | hub WebSocket (30 Hz JSON snapshots) |
| `HANDHUB_STATUS_PORT` | `39571` | UDP port on 127.0.0.1 for the 40-byte `HHD1` head-pose packets (~30 Hz) |
| `HANDHUB_DIRECT` | unset | `1` = desktop mode, do not try OpenXR |
| `HANDHUB_DEBUG` | unset | `1` = print a 1 Hz state line, object positions, and grab / release / button events |

## What is where

Hand positions arrive in the hub's body frame (metres, x right, y up, -z forward, origin at the eyes) and
are mapped to the world as `head_position + Basis(UP, yaw_offset) * p_body`; the head is the XR camera
(VR) or the desktop camera (Direct). The head pose goes back to the hub over UDP so its "Align" works.

Stations (the player starts at the room centre facing -z; the room is 4 x 4 m with a 1 m floor grid):

- **Hands**: per tracked hand, 21 joints and 21 bones tinted by confidence (green >= 0.6, amber >= 0.35,
  red below), a billboard "L"/"R" label and a larger wrist joint. Hidden when `tracked` is false or `lm` is null.
  Tests tracking quality, latency and the body-to-world mapping (yaw offset).
- **Table, 1.4 x 0.7 m, top at 0.85 m, centre z = -0.6**: six grabbable objects (3 cubes of 8 cm, 3 balls of
  4 cm radius). Closing the hand until the grip click bit (2) sets while the palm (mean of landmarks
  0, 5, 9, 13, 17) is within 12 cm of an object grabs the nearest one; it keeps its pose relative to the hand.
  Releasing (bit cleared, or hand lost > 0.3 s) throws it with the palm velocity of the last ~0.1 s. Objects near
  a hand glow; held ones glow dimmer. Anything below y = -2 or farther than 6 m respawns on the table.
  Tests grip-click reliability and hand pose/velocity quality.
- **Three push buttons** on the table's far side (index fingertip, landmark 8, inside the button's Area3D
  volume = pressed; cap depresses, colour changes, counter label above): RESET puts all objects back,
  LIGHT dims/brightens the sun, COUNT only counts. Tests fingertip position accuracy and press repeatability.
- **Hand mirror**, 1.5 m ahead at eye height: both hands enlarged x3, wrist-relative, left copy on the left and
  right copy on the right, reflected like a real mirror (fingers pointing away from you point towards you
  in the image). Tests finger curl and shape without looking down.
- **Status panel**, left of the table, turned towards the start position: mode, fps, hub connection and snapshot
  rate, yaw offset, and per hand tracked / confidence / trigger and grip bars / pressed buttons / held object.

## Testing without a camera

`tools/fake_hub.py` is a stand-in hub (needs the `websockets` package, e.g. the hub's `.venv`):

```
python tools/fake_hub.py              # ws://127.0.0.1:8798/ws, UDP 39579
HANDHUB_URL=ws://127.0.0.1:8798/ws HANDHUB_STATUS_PORT=39579 ./run.sh --direct
```

It plays two synthetic hands (right hand circles down to the blue cube and grabs, carries and releases it every
4 s; left hand's index finger presses the LIGHT button), reports how many valid `HHD1` packets per second it
receives, and has `--yaw RAD`, `--calm` (hands hover 35 cm ahead / 35 cm below the eyes) and `--dropout T,DUR`
(lose the right hand) options.
