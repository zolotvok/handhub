#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Turn a raw recording into a side-by-side video of the tracked hands and body under two settings.

    .venv/bin/python tools/render_replay.py REC.pkl \\
        --left "before:fusion.POSITION_FILTER=False" --right "after:" --out OUT.mp4 \\
        [--start S] [--end S] [--fps 30] [--still T OUT.png]... [--with-headset]

REC.pkl is a raw recording (POST /api/record/raw). It holds the cameras' detections only, so
the video shows the result of the tracking, not camera pictures. The recording is replayed
offline through the hub's fusion once for each side, exactly as tools/evaluate.py does it, and
the hub's state is sampled FPS times per second of recording time. A side is NAME:KEY=VAL,...
where KEY is "fusion.CONSTANT" or "handfilter.CONSTANT" and VAL is a Python literal;
"NAME:" alone means the defaults. By default the camera "headset" and the "_hmd" events are
left out, as in evaluate.py; --with-headset keeps them.

Each side gets a panel titled with its NAME: a front view as seen from the laptop webcam
(room x to the right, room y up) and a small top view (the player at the bottom, the webcam at
the top). The body is a grey stick figure, the left hand is blue and the right hand orange. A
hand that is not currently tracked is not drawn. Behind each hand its palm centre leaves a
trail that fades over 0.8 s, so a jump shows as a long straight line. The scale and framing
are chosen once from the hands and the body down to the knees, of both sides, and never change
during the video (the shins are cropped by the edge of the view).

Under each panel two counters run for each hand, counted the way evaluate.py counts them
(over consecutive updates of that hand less than 0.3 s apart): "shape flips" (the hand's shape
changed by more than 3 cm) and "palm jumps > 5 cm" (the palm centre moved more than 5 cm
between two updates). A counter lights up for 0.3 s each time it goes up. With --start the
counters and the elapsed time start from zero at that moment of the recording.

--start and --end are seconds from the start of the recording (default: the whole recording).
--still T OUT.png (repeatable) also saves the frame nearest to second T of the recording as a
PNG. The video is H.264 in an MP4 file; without a working PyAV or libx264 it is written with
OpenCV's "mp4v" codec and a note on stderr says so.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import io
import json
import pickle
import re
import socket
import sys
import tempfile
import time as real_time
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hub import driver_link, fusion, geometry as g, handfilter  # noqa: E402
from hub.cameras import runner  # noqa: E402
from hub.config import Config  # noqa: E402

MODULES = {"fusion": fusion, "handfilter": handfilter}
MAX_GAP_S = 0.3        # updates further apart than this are not compared (as in evaluate.py)
FLIP_M = 0.03          # a hand's shape changing by more than this between updates is a flip
JUMP_M = 0.05          # the palm centre moving more than this between updates is a jump
VISIBLE_S = 0.3        # a tracked hand not updated for this long is not drawn
TRAIL_S = 0.8
FLASH_S = 0.3

W, H = 1920, 1080
PANEL_W = W // 2
TITLE_Y = 66
VIEW = (14, 90, 932, 785)       # front view inside a panel: x, y, width, height
INSET = (620, 12, 300, 540)     # top view inside the view, on its right: x, y, width, height
FRONT_W = 600                   # width of the part of the view left of the inset, where the front view is fitted
FRAMED_BODY = 27                # body points used for the framing: those above the ankles (see Framing)
TOP_HEAD = 50                   # pixels above the webcam (z = 0) in the top view, for its labels
STRIP_Y = 890                   # counters from here to the bottom
GRID_M = 0.25                   # grid spacing in metres
FRONT_THICK, TOP_THICK = 5, 3   # line widths of hands and body in pixels

BG = (18, 22, 30)
VIEW_BG = (23, 28, 38)
BORDER = (52, 60, 76)
GRID = (36, 43, 56)
GRID_STRONG = (50, 58, 74)
BODY = (150, 157, 170)
TEXT = (228, 232, 240)
DIM = (140, 148, 162)
HAND_COLORS = ((90, 170, 255), (255, 170, 70))     # left, right (RGB)
HAND_NAMES = ("Left hand", "Right hand")
FONT = cv2.FONT_HERSHEY_DUPLEX

BODY_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28)]
HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
              (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
              (0, 5), (5, 9), (9, 13), (13, 17), (17, 0)]
HEAD_RADIUS_M = 0.09


class FakeTime:
    """Stands in for the time module: the recording's own clock."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return int(self.now * 1e9)

    def strftime(self, *a):
        return real_time.strftime(*a)


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---- variants (as in evaluate.py) ---------------------------------------------------------

# Overrides are separated by commas, but a value may hold commas ([1, 2]): a comma only
# separates when another "module.CONSTANT=" follows.
_OVERRIDE_SPLIT = re.compile(r",(?=\s*[A-Za-z_]\w*\.\w+\s*=)")


def parse_variant(text: str) -> tuple[str, dict]:
    """"NAME:fusion.A=1,handfilter.B=2.5" -> ("NAME", {("fusion", "A"): 1, ("handfilter", "B"): 2.5})."""
    name, colon, body = text.partition(":")
    name = name.strip()
    if not colon or not name:
        raise ValueError(f"variant {text!r}: expected NAME:KEY=VAL,KEY=VAL (NAME: alone is fine)")
    overrides: dict = {}
    for item in (_OVERRIDE_SPLIT.split(body) if body.strip() else []):
        key, eq, raw = item.partition("=")
        module, dot, const = key.strip().partition(".")
        if not eq or not dot:
            raise ValueError(f"variant {name!r}: {item.strip()!r} is not KEY=VAL with KEY like fusion.CONSTANT")
        if module not in MODULES:
            raise ValueError(f"variant {name!r}: unknown module {module!r} (use {' or '.join(MODULES)})")
        if not hasattr(MODULES[module], const):
            raise ValueError(f"variant {name!r}: hub/{module}.py has no {const}")
        try:
            value = ast.literal_eval(raw.strip())
        except (ValueError, SyntaxError):
            raise ValueError(f"variant {name!r}: cannot read {raw.strip()!r} as a Python literal") from None
        overrides[(module, const)] = value
    return name, overrides


def apply_overrides(originals: dict, overrides: dict) -> None:
    """Put every constant any variant touches back as it was, then apply these overrides."""
    for (module, const), value in originals.items():
        setattr(MODULES[module], const, value)
    for (module, const), value in overrides.items():
        original = originals[(module, const)]
        if isinstance(original, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)  # "ACCEL_SIGMA=4" means 4.0
        setattr(MODULES[module], const, value)


# ---- replay -------------------------------------------------------------------------------

def replay(data: dict, with_headset: bool, times: np.ndarray, t0: float) -> dict:
    """Replay the recording once and sample the hub's state at `times` (seconds from t0).

    A sample is the state after all events up to its time. Returns arrays over the samples:
    "lm" (n, 2, 21, 3) hand landmarks in room axes and "body" (n, 33, 3), NaN where something
    is not tracked or not drawn; "trail" per hand (times, palm centres); "flips" and "jumps",
    per hand, the times of the updates that count (those from times[0] on)."""
    n = len(times)
    lm = np.full((n, 2, 21, 3), np.nan)
    body = np.full((n, 33, 3), np.nan)
    trail: list = [[], []]
    counted = {"flips": [[], []], "jumps": [[], []]}
    prev: list = [None, None]       # per hand: (time, palm centre, shape) of its last update
    last_palm: list = [None, None]

    clock = FakeTime()
    fusion.time = clock
    runner.time = clock
    if with_headset:
        driver_link.time = clock  # else the headset link looks dead and the head camera is left out

    cfg_data = copy.deepcopy(data["config"])
    cfg_data["driver"]["port"], cfg_data["driver"]["status_port"] = free_port(), free_port()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(cfg_data))
        config = Config(path)
        hub = fusion.Hub(config=config, start_camera=False)
        try:
            cfg = config.get()

            def visible(h) -> bool:
                return h.state.tracked and clock.now - h.last_seen < VISIBLE_S

            def sample(i: int) -> None:
                clock.now = t0 + times[i]
                for k, h in enumerate(hub.hands):
                    if visible(h):
                        lm[i, k] = h.lm_room
                snap = hub._body_snapshot(cfg)
                if snap["tracked"]:
                    for j, p in enumerate(snap["points"]):
                        if p is not None:
                            body[i, j] = p

            nxt = 0
            for now, name, msg in data["events"]:
                rel = now - t0
                while nxt < n and times[nxt] < rel:
                    sample(nxt)
                    nxt += 1
                if nxt >= n:
                    break
                clock.now = now
                if name == "_hmd":
                    if with_headset:
                        hub.link.hmd_history.append(msg)
                        hub.link.hmd_valid, hub.link.last_status_t = True, now
                    continue
                if name == "headset" and not with_headset:
                    continue
                cam = hub.cams[name]
                frame = cam.on_message(msg)
                if frame is None:
                    continue
                if cam.last_head is not None:
                    hub.process_head(name, frame[1], cam.last_head, cfg)
                hub.process_frame(*frame, cfg, cam=name)
                hub._expire(cfg)
                for k, h in enumerate(hub.hands):
                    if h.state.tracked and h.last_t == frame[0] / 1e9:  # this picture updated the hand
                        palm = h.lm_ref[g.PALM_IDS].mean(axis=0)
                        shape = h.lm_ref - palm
                        p = prev[k]
                        if p is not None and rel - p[0] < MAX_GAP_S and rel >= times[0]:
                            if np.linalg.norm(shape - p[2], axis=1).mean() > FLIP_M:
                                counted["flips"][k].append(rel)
                            if np.linalg.norm(palm - p[1]) > JUMP_M:
                                counted["jumps"][k].append(rel)
                        prev[k] = (rel, palm, shape)
                    if visible(h):
                        c = h.lm_room[g.PALM_IDS].mean(axis=0)
                        if last_palm[k] is None or not np.array_equal(c, last_palm[k]):
                            trail[k].append((rel, c))
                            last_palm[k] = c
            while nxt < n:
                sample(nxt)
                nxt += 1
        finally:
            hub.link.close()
    return {
        "lm": lm, "body": body,
        "trail": [(np.array([e[0] for e in tr]), np.array([e[1] for e in tr]).reshape(-1, 3)) for tr in trail],
        "flips": [np.array(v) for v in counted["flips"]], "jumps": [np.array(v) for v in counted["jumps"]],
    }


def replay_quietly(data: dict, with_headset: bool, times: np.ndarray, t0: float) -> dict:
    """The libraries may print and numpy may warn: keep it off the terminal unless it fails."""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return replay(data, with_headset, times, t0)
    except Exception:
        sys.stderr.write(captured.getvalue())
        raise


# ---- framing ------------------------------------------------------------------------------

class Framing:
    """Where room coordinates land in a view: fixed for the whole video.

    Room axes are the player's: x to the player's right, y up, and the player stands at
    positive z in front of the webcam (at z = 0), facing -z. So the front view, seen from the
    webcam, which faces the player, has room x running from the right of the picture to its
    left, and the top view (the webcam at the top, the player below it, x to the right) is not
    flipped. The ranges are percentiles over the hand points and the body points down to the knees,
    of both sides, so a few wild points do not shrink the picture (the rest of such a point's line
    is cut off at the edge). The ankles are left out: they are seen in a few frames only and
    would make the hands, which are what matters here, a third smaller; the shins are cropped."""

    def __init__(self, sides: list):
        pts = np.concatenate([a.reshape(-1, 3) for s in sides for a in (s["lm"], s["body"][:, :FRAMED_BODY])])
        pts = pts[~np.isnan(pts).any(axis=1)]
        lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
        hi[1] += HEAD_RADIUS_M   # the head is a circle around its centre point
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        vh = VIEW[3]
        self.front_scale = min(FRONT_W / (hi[0] - lo[0]), vh / (hi[1] - lo[1]))
        self.front_centre = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
        _, _, iw, ih = INSET
        self.top_scale = min((iw - 20) / (hi[0] - lo[0]), (ih - TOP_HEAD - 40) / max(hi[2], 0.5))
        self.top_x = (lo[0] + hi[0]) / 2

    def front(self, p: np.ndarray) -> np.ndarray:
        """(N, 3) room points -> (N, 2) pixels in the front view."""
        vh = VIEW[3]
        cx, cy = self.front_centre
        return np.stack([FRONT_W / 2 - (p[:, 0] - cx) * self.front_scale, vh / 2 - (p[:, 1] - cy) * self.front_scale], axis=1)

    def top(self, p: np.ndarray) -> np.ndarray:
        _, _, iw, _ = INSET
        return np.stack([(p[:, 0] - self.top_x) * self.top_scale + iw / 2, TOP_HEAD + p[:, 2] * self.top_scale], axis=1)


# ---- drawing ------------------------------------------------------------------------------

SH = 4   # sub-pixel bits for line and circle coordinates, so nothing jitters by whole pixels


def fp(p) -> tuple[int, int]:
    return int(round(float(p[0]) * (1 << SH))), int(round(float(p[1]) * (1 << SH)))


def line(img, a, b, color, thick: float) -> None:
    if np.isfinite(a).all() and np.isfinite(b).all() and np.abs(np.concatenate([a, b])).max() < 20000:
        cv2.line(img, fp(a), fp(b), color, max(1, round(thick)), cv2.LINE_AA, SH)


def dot(img, c, r: float, color, thick: int = -1) -> None:
    if np.isfinite(c).all() and np.abs(c).max() < 20000:
        cv2.circle(img, fp(c), int(r * (1 << SH)), color, thick, cv2.LINE_AA, SH)


def blend(a, b, f: float) -> tuple:
    """a at f = 0, b at f = 1."""
    return tuple(int(round(x + (y - x) * f)) for x, y in zip(a, b))


def text(img, s: str, x: float, y: float, scale: float, color, thick: int = 2, anchor: str = "l") -> None:
    (w, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    x -= {"l": 0, "c": w / 2, "r": w}[anchor]
    cv2.putText(img, s, (int(round(x)), int(round(y))), FONT, scale, color, thick, cv2.LINE_AA)


def grid_lines(lo: float, hi: float):
    """Multiples of GRID_M between lo and hi, with whether each is a multiple of 0.5 m."""
    for k in range(int(np.ceil(lo / GRID_M)), int(np.floor(hi / GRID_M)) + 1):
        yield k * GRID_M, k % 2 == 0


def scale_bar(img, x: float, y: float, length: float, size: float, label_right: bool = False) -> None:
    """A bar 0.5 m long (`length` pixels) with its left end at x, labelled above it or to its right."""
    tick = 8 * size
    line(img, np.array([x, y]), np.array([x + length, y]), TEXT, 3)
    for tx in (x, x + length):
        line(img, np.array([tx, y - tick]), np.array([tx, y + tick]), TEXT, 3)
    if label_right:
        text(img, "0.5 m", x + length + 10, y + 8 * size, size, TEXT, 2)
    else:
        text(img, "0.5 m", x + length / 2, y - tick - 8, size, TEXT, 2, "c")


def front_base(fr: Framing) -> np.ndarray:
    """The front view without any tracking: background, grid and the scale bar."""
    _, _, vw, vh = VIEW
    img = np.empty((vh, vw, 3), np.uint8)
    img[:] = VIEW_BG
    cx, cy = fr.front_centre
    half_x, half_y = FRONT_W / 2 / fr.front_scale, vh / 2 / fr.front_scale
    for x, strong in grid_lines(cx - half_x, cx + half_x):
        px = fr.front(np.array([[x, cy, 0.0]]))[0, 0]
        line(img, np.array([px, 0.0]), np.array([px, vh]), GRID_STRONG if strong else GRID, 1)
    for y, strong in grid_lines(cy - half_y, cy + half_y):
        py = fr.front(np.array([[cx, y, 0.0]]))[0, 1]
        line(img, np.array([0.0, py]), np.array([vw, py]), GRID_STRONG if strong else GRID, 1)
    scale_bar(img, vw - 28 - 0.5 * fr.front_scale, vh - 36, 0.5 * fr.front_scale, 0.9)
    return img


def top_base(fr: Framing) -> np.ndarray:
    _, _, iw, ih = INSET
    img = np.empty((ih, iw, 3), np.uint8)
    img[:] = VIEW_BG
    half_x = iw / 2 / fr.top_scale
    for x, strong in grid_lines(fr.top_x - half_x, fr.top_x + half_x):
        px = fr.top(np.array([[x, 0.0, 0.0]]))[0, 0]
        line(img, np.array([px, 0.0]), np.array([px, ih]), GRID_STRONG if strong else GRID, 1)
    for z, strong in grid_lines(0.0, (ih - TOP_HEAD) / fr.top_scale):
        py = fr.top(np.array([[0.0, 0.0, z]]))[0, 1]
        line(img, np.array([0.0, py]), np.array([iw, py]), GRID_STRONG if strong else GRID, 1)
    cam = fr.top(np.array([[0.0, 0.0, 0.0]]))[0]
    cv2.fillPoly(img, [np.round(cam + np.array([[-9, -12], [9, -12], [0, 0]])).astype(np.int32)], DIM, cv2.LINE_AA)
    scale_bar(img, 14, ih - 20, 0.5 * fr.top_scale, 0.8, label_right=True)
    text(img, "top view", 10, 28, 0.85, TEXT, 2)
    text(img, "webcam", iw - 10, 28, 0.7, DIM, 1, "r")
    return img


def draw_trail(img, project, tt: np.ndarray, pp: np.ndarray, now: float, color, thick: float) -> None:
    """The palm centre over the last TRAIL_S seconds, fading to the background with age."""
    lo = np.searchsorted(tt, now - TRAIL_S)
    hi = np.searchsorted(tt, now, side="right")
    if hi - lo < 2:
        return
    px = project(pp[lo:hi])
    for a in range(hi - lo - 1):
        if tt[lo + a + 1] - tt[lo + a] < MAX_GAP_S:   # a gap is a hand lost and found again, not a jump
            age = now - tt[lo + a + 1]
            line(img, px[a], px[a + 1], blend(VIEW_BG, color, max(0.0, 1 - age / TRAIL_S) ** 0.7), thick)


def draw_figure(img, project, px_per_m: float, lm: np.ndarray, body: np.ndarray, trails: list, now: float,
                thick: float) -> None:
    """One view: the hand trails, then the body, then the hands on top. NaN points are not drawn."""
    for k in range(2):
        draw_trail(img, project, *trails[k], now, HAND_COLORS[k], thick * 0.9)
    px = project(body)
    for a, b in BODY_BONES:
        line(img, px[a], px[b], BODY, thick * 1.4)
    ears = [i for i in (7, 8) if np.isfinite(px[i]).all()]
    dot(img, px[ears].mean(axis=0) if ears else px[0], HEAD_RADIUS_M * px_per_m, BODY, round(thick * 1.4))
    for k in range(2):
        if np.isnan(lm[k]).any():
            continue
        px = project(lm[k])
        for a, b in HAND_LINES:
            line(img, px[a], px[b], HAND_COLORS[k], thick)
        for p in px:
            dot(img, p, thick * 0.9, blend(HAND_COLORS[k], TEXT, 0.3))


def draw_counters(frame: np.ndarray, x0: int, side: dict, now: float, elapsed: float) -> None:
    """The table under a panel: per hand the running counts, lit for FLASH_S after each rise."""
    cols = (x0 + 360, x0 + 590)
    text(frame, "shape flips", cols[0], 930, 0.85, DIM, 2, "c")
    text(frame, "palm jumps > 5 cm", cols[1], 930, 0.85, DIM, 2, "c")
    for k in range(2):
        y = 998 + k * 64
        text(frame, HAND_NAMES[k], x0 + 30, y, 1.1, HAND_COLORS[k], 2)
        for cx, key in zip(cols, ("flips", "jumps")):
            times = side[key][k]
            n = int(np.searchsorted(times, now, side="right"))
            lit = n > 0 and now - times[n - 1] < FLASH_S
            if lit:
                cv2.rectangle(frame, (cx - 68, y - 46), (cx + 68, y + 14), HAND_COLORS[k], -1)
            text(frame, str(n), cx, y, 1.7, BG if lit else HAND_COLORS[k], 3, "c")
    text(frame, "elapsed", x0 + 925, 930, 0.85, DIM, 2, "r")
    text(frame, f"{elapsed:.1f} s", x0 + 925, 998, 1.5, TEXT, 2, "r")


def draw_panel(frame: np.ndarray, p: int, name: str, side: dict, i: int, now: float, elapsed: float,
               fr: Framing, bases: tuple) -> None:
    x0 = p * PANEL_W
    vx, vy, vw, vh = VIEW
    ix, iy, iw, ih = INSET
    # each view is drawn on its own array, which also cuts off whatever leaves it
    front = bases[0].copy()
    draw_figure(front, fr.front, fr.front_scale, side["lm"][i], side["body"][i], side["trail"], now, FRONT_THICK)
    top = bases[1].copy()
    draw_figure(top, fr.top, fr.top_scale, side["lm"][i], side["body"][i], side["trail"], now, TOP_THICK)
    front[iy:iy + ih, ix:ix + iw] = top
    cv2.rectangle(front, (ix, iy), (ix + iw - 1, iy + ih - 1), BORDER, 2, cv2.LINE_AA)
    frame[vy:vy + vh, x0 + vx:x0 + vx + vw] = front
    cv2.rectangle(frame, (x0 + vx, vy), (x0 + vx + vw - 1, vy + vh - 1), BORDER, 2, cv2.LINE_AA)
    text(frame, name, x0 + PANEL_W / 2, TITLE_Y, 2.0, TEXT, 3, "c")
    draw_counters(frame, x0, side, now, elapsed)


# ---- video --------------------------------------------------------------------------------

class VideoWriter:
    """H.264 through PyAV; if that does not work, OpenCV's mp4v. Frames are RGB uint8."""

    def __init__(self, path: Path, fps: int):
        self.cv = None
        try:
            import av
            self.av = av
            self.container = av.open(str(path), "w", options={"movflags": "+faststart"})
            self.stream = self.container.add_stream("libx264", rate=fps)
            self.stream.width, self.stream.height, self.stream.pix_fmt = W, H, "yuv420p"
            self.stream.options = {"crf": "19", "preset": "medium"}
            cc = self.stream.codec_context
            cc.colorspace = cc.color_primaries = cc.color_trc = 1   # BT.709, which players assume for HD
        except Exception as e:
            sys.stderr.write(f"PyAV/H.264 is not usable ({e}); writing {path} with OpenCV's mp4v codec instead\n")
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            if not self.cv.isOpened():
                raise RuntimeError(f"cannot write {path}")

    def write(self, rgb: np.ndarray) -> None:
        if self.cv is not None:
            self.cv.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            return
        frame = self.av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p", dst_colorspace="ITU709")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.cv is not None:
            self.cv.release()
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


# ---- main ---------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Render a recording as a side-by-side video of two tracking settings.",
                                 epilog='example: --left "before:fusion.POSITION_FILTER=False" --right "after:"')
    ap.add_argument("recording", metavar="REC.pkl", help="a raw recording (POST /api/record/raw)")
    ap.add_argument("--left", required=True, metavar="NAME:KEY=VAL,...", help="the left panel's setting")
    ap.add_argument("--right", required=True, metavar="NAME:KEY=VAL,...", help="the right panel's setting")
    ap.add_argument("--out", required=True, metavar="OUT.mp4")
    ap.add_argument("--start", type=float, default=0.0, metavar="S", help="seconds from the recording's start")
    ap.add_argument("--end", type=float, metavar="S", help="seconds from the recording's start (default: the end)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--still", nargs=2, action="append", default=[], metavar=("T", "OUT.png"),
                    help="also save the frame nearest second T of the recording as a PNG (repeatable)")
    ap.add_argument("--with-headset", action="store_true", help='keep the "headset" camera and the "_hmd" events')
    args = ap.parse_args()

    try:
        variants = [parse_variant(args.left), parse_variant(args.right)]
        stills = [(float(t), Path(p)) for t, p in args.still]
    except ValueError as e:
        ap.error(str(e))
    path = Path(args.recording).expanduser()
    if not path.is_file():
        ap.error(f"no such recording: {path}")
    if args.fps < 1:
        ap.error("--fps must be at least 1")
    with open(path, "rb") as f:
        data = pickle.load(f)
    t0 = data["events"][0][0]
    duration = data["events"][-1][0] - t0
    start, end = args.start, min(args.end if args.end is not None else duration, duration)
    if not 0 <= start < end:
        ap.error(f"--start and --end must satisfy 0 <= start < end <= {duration:.1f} (seconds)")
    # frames every 1/fps; the last one at the very end, so the counters hold everything
    times = np.minimum(start + np.arange(int(np.ceil((end - start) * args.fps - 1e-9)) + 1) / args.fps, end)
    still_at = {}
    for t, p in stills:
        i = int(round((t - start) * args.fps))
        if not 0 <= i < len(times):
            ap.error(f"--still {t:g}: outside the video ({start:g} to {end:g} s)")
        still_at.setdefault(i, []).append(p)

    # what every constant either side touches was before the first one was applied
    originals = {key: getattr(MODULES[key[0]], key[1]) for _, ov in variants for key in ov}
    sides = []
    try:
        for _, overrides in variants:
            apply_overrides(originals, overrides)
            sides.append(replay_quietly(data, args.with_headset, times, t0))
    finally:
        apply_overrides(originals, {})

    fr = Framing(sides)
    bases = front_base(fr), top_base(fr)
    out = Path(args.out).expanduser()
    writer = VideoWriter(out, args.fps)
    try:
        frame = np.empty((H, W, 3), np.uint8)
        for i, now in enumerate(times):
            frame[:] = BG
            cv2.line(frame, (PANEL_W, 24), (PANEL_W, H - 24), BORDER, 2, cv2.LINE_AA)
            for p, ((name, _), side) in enumerate(zip(variants, sides)):
                draw_panel(frame, p, name, side, i, now, now - start, fr, bases)
            writer.write(frame)
            for still in still_at.get(i, []):
                cv2.imwrite(str(still), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.close()

    for (name, _), side in zip(variants, sides):
        n = [[len(side[key][k]) for key in ("flips", "jumps")] for k in range(2)]
        print(f"{name}: left flips {n[0][0]}, palm jumps > 5 cm {n[0][1]}; right flips {n[1][0]}, palm jumps > 5 cm {n[1][1]}")
    print(f"wrote {out}: {len(times)} frames at {args.fps} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
