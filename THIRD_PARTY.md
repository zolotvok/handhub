# Third-party components

HandHub itself is GPL-3.0-or-later. Nothing below is copied into this repository except
where stated; the rest is installed, downloaded at run time, or patched in a separate
checkout. All of these licences are compatible with distributing HandHub under the GPLv3.

## Patched (patches in this repository)

| Component | Licence | Notes |
|---|---|---|
| [WiVRn](https://github.com/WiVRn/WiVRn) | GPL-3.0-or-later | `runtime/wivrn-client-mali/*.patch` modify its Android client. The patches are derived from WiVRn and are under the same licence. WiVRn's source is not included. |

## Python dependencies (installed by pip)

| Package | Licence |
|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 |
| [OpenCV](https://opencv.org/) (`opencv-python-headless`) | Apache-2.0 |
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [FastAPI](https://fastapi.tiangolo.com/) | MIT |
| [Uvicorn](https://www.uvicorn.org/) | BSD-3-Clause |
| [PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause; its wheels bundle FFmpeg libraries (LGPL-2.1-or-later, some builds GPL) |
| [pySerial](https://github.com/pyserial/pyserial) | BSD-3-Clause |
| [websockets](https://github.com/python-websockets/websockets), [httpx](https://www.python-httpx.org/), [pytest](https://pytest.org/) (development only) | BSD-3-Clause, BSD-3-Clause, MIT |

## Downloaded at run time (into `~/.cache/handhub`)

| File | From | Licence |
|---|---|---|
| `hand_landmarker.task`, `pose_landmarker_lite.task` | Google's MediaPipe model storage | Apache-2.0 |
| `scrcpy-server-v5.0.1` | [Genymobile/scrcpy](https://github.com/Genymobile/scrcpy) releases; pushed to the phone to stream its camera | Apache-2.0 |

## Used alongside, not distributed

| Component | Licence | Role |
|---|---|---|
| [Godot Engine](https://godotengine.org/) | MIT | Runs the playground |
| [Monado](https://monado.freedesktop.org/) | BSL-1.0 | OpenXR runtime on the phone |
| [OpenXR Runtime Broker](https://github.com/KhronosGroup/OpenXR-SDK-Source) | Apache-2.0 | Selects the runtime on Android |
| Android Debug Bridge (`adb`) | Apache-2.0 | Reaches the phones |

## Design references

The Pico camera's serial protocol in `hub/cameras/pico_serial.py` was written against the
author's own firmware and helper scripts, which are not part of this repository.

The One Euro filter in `hub/filters.py` is an independent implementation of the algorithm
published by Casiez, Roussel and Vogel (CHI 2012).
