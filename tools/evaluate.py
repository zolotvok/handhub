#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Score the hub's fusion on raw recordings, for several settings side by side.

    .venv/bin/python tools/evaluate.py REC.pkl [REC2.pkl ...] \\
        [--variant NAME:KEY=VAL,KEY=VAL ...]... [--with-headset] [--json OUT.json]

Each recording (POST /api/record/raw) is replayed offline through the hub's fusion once
per variant; replays run one after another. A variant is a name and a list of overrides:
KEY is "fusion.CONSTANT" or "handfilter.CONSTANT" (a module-level constant of
hub/fusion.py or hub/handfilter.py) and VAL is a Python literal (True, 0.35, 12, ...).
"--variant default:" is a variant without overrides; with no --variant at all, that one
variant, called "default", is run. Before every replay each constant that any variant
touches is put back to its original value, so variants do not leak into one another.

There is no ground truth, so the scores are about steadiness. They are computed per hand
(L = index 0, R = index 1) over consecutive updates of that hand less than 0.3 s apart:
  flips    updates where the hand's shape changed by more than 3 cm (mean over the 21
           landmarks of the distance moved, relative to the palm centre, which is the
           mean of landmarks 0, 5, 9, 13, 17);
  p5, p10  steps of the palm centre over 5 cm and over 10 cm;
  tw       twitchiness: 90th percentile of the change of the step length from one step
           to the next, in cm;
  w5, w10  the same step counts for the wrist (landmark 0);
  n        number of updates of the hand.
And, over everything the hub tracked, how well the estimate held before a picture
arrived predicts where that picture finds the hand: the distance between the hand,
projected into the camera, and the detection it gets matched to, as a share of the
picture width: median, 90th percentile, share over 5 % and over 10 %, and the count.

By default the camera named "headset" and the "_hmd" events are left out, as if the
head camera were not there; --with-headset keeps them.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import io
import itertools
import json
import pickle
import re
import socket
import sys
import tempfile
import time as real_time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hub import driver_link, fusion, geometry as g, handfilter  # noqa: E402
from hub.cameras import runner  # noqa: E402
from hub.config import Config  # noqa: E402

MODULES = {"fusion": fusion, "handfilter": handfilter}
MAX_GAP_S = 0.3        # updates further apart than this are not compared
FLIP_M = 0.03          # a hand's shape changing by more than this between updates is a flip
HAND_NAMES = ("left", "right")


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


# ---- variants ---------------------------------------------------------------------------

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


# ---- replay -----------------------------------------------------------------------------

def predict_misses(hub, cfg: dict, cam_cfgs: dict, name: str, msg: dict, now: float, ext: dict) -> list:
    """How far the hands as estimated before this picture land from the detections in it.

    Hands tracked and seen in the last 0.3 s are matched one to one to the detections
    (the best pairing for two hands); the distance, from the palm centre projected into
    this camera to the detection's palm centre, is a share of the picture width."""
    dets = msg["hands"]
    if name not in ext or not dets:
        return []
    tracked = [h for h in hub.hands if h.state.tracked and now - h.last_seen < 0.3]
    if not tracked or len(dets) < len(tracked):
        return []
    size = msg["size"]
    k = g.intrinsics_from_hfov(size[0], size[1], cam_cfgs[name]["hfov_deg"])
    r, t = ext[name]
    det_pts = [np.asarray(d[2], float)[g.PALM_IDS].mean(axis=0) for d in dets]
    proj = []
    for h in tracked:
        c = r.T @ (h.lm_ref[g.PALM_IDS].mean(axis=0) - t)
        if c[2] <= 0.05:
            return []  # behind the camera: no projection to compare
        proj.append(c[:2] / c[2] * k[0, 0] + k[:2, 2])
    cost = np.array([[np.linalg.norm(p - d) for d in det_pts] for p in proj]) / size[0]
    if len(proj) == 2:
        best = min(itertools.permutations(range(len(det_pts)), 2), key=lambda pm: cost[0, pm[0]] + cost[1, pm[1]])
        return [float(cost[0, best[0]]), float(cost[1, best[1]])]
    return [float(cost[0].min())]


def replay(data: dict, with_headset: bool) -> tuple[dict, list]:
    """One replay of a loaded recording with the constants as they are now.

    Returns ({hand index: [(time, landmarks 21x3), ...]}, [prediction misses])."""
    clock = FakeTime()
    fusion.time = clock
    runner.time = clock
    if with_headset:
        driver_link.time = clock  # else the headset link looks dead and the head camera is left out

    cfg_data = copy.deepcopy(data["config"])
    cfg_data["driver"]["port"], cfg_data["driver"]["status_port"] = free_port(), free_port()
    rows: dict = {0: [], 1: []}
    pre: list = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(cfg_data))
        config = Config(path)
        hub = fusion.Hub(config=config, start_camera=False)
        try:
            cfg = config.get()
            cam_cfgs = hub.camera_configs(cfg)
            for now, name, msg in data["events"]:
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
                # before this picture is used: does the estimate so far say where its hands are?
                pre.extend(predict_misses(hub, cfg, cam_cfgs, name, msg, now, hub.extrinsics(cfg)))
                hub.process_frame(*frame, cfg, cam=name)
                hub._expire(cfg)
                for i, h in enumerate(hub.hands):
                    if h.state.tracked and h.last_t == frame[0] / 1e9:  # this picture updated the hand
                        rows[i].append((now, h.lm_ref.copy()))
        finally:
            hub.link.close()
    return rows, pre


# ---- scores -----------------------------------------------------------------------------

def hand_scores(rows: list) -> dict:
    out = {"updates": len(rows), "flips": 0, "palm5": 0, "palm10": 0, "twitch_p90_cm": None, "wrist5": 0, "wrist10": 0}
    if len(rows) < 2:
        return out
    t = np.array([r[0] for r in rows])
    lm = np.stack([r[1] for r in rows])  # updates x 21 x 3
    close = np.diff(t) < MAX_GAP_S
    palm = lm[:, g.PALM_IDS].mean(axis=1)
    palm_step = np.linalg.norm(np.diff(palm, axis=0), axis=1)[close]
    wrist_step = np.linalg.norm(np.diff(lm[:, 0], axis=0), axis=1)[close]
    shape = lm - palm[:, None, :]
    shape_step = np.linalg.norm(np.diff(shape, axis=0), axis=2).mean(axis=1)[close]
    twitch = np.abs(np.diff(palm_step)) * 100.0
    out.update(
        flips=int((shape_step > FLIP_M).sum()),
        palm5=int((palm_step > 0.05).sum()), palm10=int((palm_step > 0.10).sum()),
        wrist5=int((wrist_step > 0.05).sum()), wrist10=int((wrist_step > 0.10).sum()),
        twitch_p90_cm=round(float(np.percentile(twitch, 90)), 2) if twitch.size else None)
    return out


def predict_scores(pre: list) -> dict:
    a = np.asarray(pre, float)
    if a.size == 0:
        return {"median_pct": None, "p90_pct": None, "over5_pct": None, "over10_pct": None, "n": 0}
    return {"median_pct": round(float(np.median(a)) * 100, 2), "p90_pct": round(float(np.percentile(a, 90)) * 100, 2),
            "over5_pct": round(float(np.mean(a > 0.05)) * 100, 1), "over10_pct": round(float(np.mean(a > 0.10)) * 100, 1),
            "n": int(a.size)}


def evaluate(data: dict, with_headset: bool) -> dict:
    """Replay quietly (the libraries may print, numpy may warn) and score the result."""
    started = real_time.monotonic()
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            rows, pre = replay(data, with_headset)
    except Exception:
        sys.stderr.write(captured.getvalue())
        raise
    out = {nm: hand_scores(rows[i]) for i, nm in enumerate(HAND_NAMES)}
    out["predict"] = predict_scores(pre)
    out["elapsed_s"] = round(real_time.monotonic() - started, 1)
    return out


# ---- output -----------------------------------------------------------------------------

LEGEND = (
    "L/R = left/right hand, over updates less than 0.3 s apart.  flips = updates where the hand's shape changed >3 cm;"
    " p5/p10 = palm-centre steps >5/>10 cm;\n"
    "tw = 90th percentile of the change of step length between steps (cm);  w5/w10 = the same step counts for the wrist;"
    "  n = updates.\n"
    "next-picture miss = where the estimate so far puts the hand vs where the next picture finds it, % of picture width:"
    " median, p90, share >5 %, share >10 %, count."
)


def cell(v, width: int, prec: int = 0) -> str:
    return "-".rjust(width) if v is None else f"{v:{width}.{prec}f}" if prec else f"{v:{width}d}"


def hand_cells(h: dict) -> str:
    return " ".join((cell(h["flips"], 5), cell(h["palm5"], 4), cell(h["palm10"], 4), cell(h["twitch_p90_cm"], 5, 1),
                     cell(h["wrist5"], 4), cell(h["wrist10"], 4), cell(h["updates"], 5)))


def header(rec_w: int, var_w: int) -> str:
    hand = "flips   p5  p10    tw   w5  w10     n"
    return (f"{'recording':<{rec_w}} {'variant':<{var_w}} | L: {hand} | R: {hand} | next-picture miss: "
            f"{'med':>5} {'p90':>5} {'>5%':>5} {'>10%':>5} {'n':>6}")


def line(rec: str, variant: str, res: dict, rec_w: int, var_w: int) -> str:
    p = res["predict"]
    return (f"{rec:<{rec_w}} {variant:<{var_w}} | L: {hand_cells(res['left'])} | R: {hand_cells(res['right'])} | "
            f"{' ' * 19}{cell(p['median_pct'], 5, 1)} {cell(p['p90_pct'], 5, 1)} {cell(p['over5_pct'], 5, 1)} "
            f"{cell(p['over10_pct'], 5, 1)} {cell(p['n'], 6)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the hub's fusion on raw recordings, for several settings.",
                                 epilog="example: --variant old:fusion.POSITION_FILTER=False "
                                        "--variant new:handfilter.ACCEL_SIGMA=4.0,fusion.FILTER_RANGE_SHARE=0.35")
    ap.add_argument("recordings", nargs="+", metavar="REC.pkl", help="raw recordings (POST /api/record/raw)")
    ap.add_argument("--variant", action="append", default=[], metavar="NAME:KEY=VAL,KEY=VAL",
                    help="a set of constant overrides to compare (repeatable); KEY is fusion.X or handfilter.X")
    ap.add_argument("--with-headset", action="store_true", help='keep the "headset" camera and the "_hmd" events')
    ap.add_argument("--json", metavar="OUT.json", help="also write all numbers here")
    args = ap.parse_args()

    try:
        variants = [parse_variant(v) for v in args.variant] or [("default", {})]
    except ValueError as e:
        ap.error(str(e))
    names = [n for n, _ in variants]
    if len(set(names)) != len(names):
        ap.error("variant names must be different")
    recordings = [Path(p).expanduser() for p in args.recordings]
    for p in recordings:
        if not p.is_file():
            ap.error(f"no such recording: {p}")

    # what every constant any variant touches was before the first one was applied
    originals = {key: getattr(MODULES[key[0]], key[1]) for _, ov in variants for key in ov}

    rec_w = max(len("recording"), *(len(p.stem) for p in recordings))
    var_w = max(len("variant"), *(len(n) for n in names))
    print(LEGEND)
    print(header(rec_w, var_w))
    results = []

    def save() -> None:
        if args.json:
            doc = {"with_headset": args.with_headset, "results": results}
            Path(args.json).expanduser().write_text(json.dumps(doc, indent=1, default=repr) + "\n")

    try:
        for path in recordings:
            with open(path, "rb") as f:
                data = pickle.load(f)
            for name, overrides in variants:
                apply_overrides(originals, overrides)
                res = evaluate(data, args.with_headset)
                print(line(path.stem, name, res, rec_w, var_w), flush=True)
                results.append({"recording": path.stem, "path": str(path), "variant": name,
                                "overrides": {f"{m}.{c}": v for (m, c), v in overrides.items()}, **res})
                save()
    finally:
        apply_overrides(originals, {})
    return 0


if __name__ == "__main__":
    sys.exit(main())
