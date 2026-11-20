# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""The hub core: camera results -> hand poses, gestures -> UDP packets and UI state.

Stage 1: one camera, single-view PnP. Multi-camera triangulation plugs in at
`_process_frame` (stage 2).
"""
from __future__ import annotations

import math
from collections import deque
import threading
from multiprocessing.connection import wait as wait_connections
import time

import numpy as np

from dataclasses import dataclass

from . import geometry as g
from . import headcam as hc
from . import multiview as mv
from .cameras.runner import CameraRunner
from .config import Config
from .driver_link import FLAG_FOLLOW_HMD_POSITION, DriverLink, HandState
from .filters import OneEuro, OneEuroQuat
from .gestures import GestureDetector

REST_TARGET = np.array([0.0, -0.35, -0.35])  # body frame, where "hands in front of the chest" maps
HEAD_HALF_WIDTH = 0.075  # eye-centre to temple, metres
TEMPLE_Z = 0.02  # temples sit slightly behind the eyes (+z is backwards)
VIEW_MAX_AGE_S = 0.07  # a view older than this is not combined with a newer one
# The head camera is fitted to the fixed cameras all the time (see _headfit_solve).
HEADFIT_WINDOW_S = 45.0      # sightings older than this are forgotten
HEADFIT_EVERY_S = 3.0        # how often the fit is redone
HEADFIT_MIN_SIGHTINGS = 60
HEADFIT_GOOD_DEG = 3.0       # a fit worse than this is not used
HEADFIT_LOST_DEG = 7.0       # the fit in use disagrees this much with fresh sightings: drop it
HEADFIT_MAX_AGE_S = 300.0    # without fresh sightings the headset's heading may have drifted
HEADFIT_FULL_TURN_DEG = 25.0  # the head must have turned this much to tell A from B
HEADFIT_PAIR_MAX_AGE_S = 0.12
VIEW_PREDICT_MAX_S = 0.10    # a view is carried forward along its own motion by at most this long
# Different cameras, and different ways of combining them, put the same hand a few
# centimetres apart, and which of them reports changes from frame to frame. The hand
# must not hop between their opinions: on a change of source it stays where it was and
# then drifts to the new source's place over this long.
HANDOVER_TAU_S = 0.4
HANDOVER_MAX_M = 0.20        # further apart than this is a real correction, taken at once
HEAD_SHAPE_HOLD_S = 0.25     # fingers stay as the head camera last saw them for this long
TRIANGULATION_MAX_RMS_PX = 20.0
HEAD_IDX = [0, 2, 5, 7, 8, 9, 10]  # head landmarks within the 33 pose landmarks
SINGLE_KEEP_RANGE_S = 0.6  # a single view keeps the hand's last known distance this long
HAND_MAX_SPEED_MPS = 5.0
SIDE_PRIORITY = "body"  # which evidence names a hand first: "body" (the arm) or "shape" (its 3D shape)
ASSOC_MAX_JUMP_M = 0.30  # a hand found within this of where it was is the same hand
ASSOC_MAX_JUMP_PX = 0.15  # the same, in one camera's picture, as a share of its width
HAND_FIT_MAX_RMS_M = 0.035  # how far triangulated landmarks may be from a rigid hand
HAND_WRIST_SLACK_M = 1.0  # the pull toward the skeleton wrist is effectively off: the replay showed it adds leaps
HAND_WRIST_SLACK_GOOD_M = 1.0
# (landmark a, landmark b, factor to torso length): shoulders-to-hips, upper arms, forearms.
BODY_SCALE_SEGMENTS = [(11, 23, 1.0), (12, 24, 1.0), (11, 13, 1.7), (12, 14, 1.7), (13, 15, 1.8), (14, 16, 1.8)]
UPPER_BODY = (0, 11, 12, 13, 14, 15, 16)  # nose, shoulders, elbows, wrists
HAND_MAX_EYE_RATIO = 7.5
WRIST_TRUST_DIST = 0.14  # of the image width: closer than this, a hand belongs to that wrist
HAND_MAX_BODY_RATIO = 0.65  # hand length / torso length; a real hand is about 0.35
WRIST_MATCH_MAX_AGE_S = 0.5
WRIST_MATCH_MAX_DIST = 0.30  # of the image width
AGREEMENT_BAD_PX = 25.0
AGREEMENT_LANDMARKS = [0, 2, 5, 7, 8, 11, 12]  # nose, eyes, ears, shoulders
# (from, to, longest plausible length in metres), ordered from the torso outward.
BODY_BONE_LIMITS = [(11, 13, 0.42), (13, 15, 0.40), (12, 14, 0.42), (14, 16, 0.40),
                    (11, 23, 0.75), (12, 24, 0.75), (23, 25, 0.60), (25, 27, 0.58), (24, 26, 0.60), (26, 28, 0.58),
                    (27, 29, 0.20), (27, 31, 0.30), (28, 30, 0.20), (28, 32, 0.30)]
BODY_MAX_RMS_PX = 40.0  # the skeleton is for display and coarse anchoring: be lenient
BODY_MIN_VISIBILITY = 0.5
BODY_HOLD_S = 0.6
HEAD_VIEW_MAX_AGE_S = 0.25
HEAD_MIN_VISIBILITY = 0.6
HEAD_HOLD_S = 1.5  # keep using the last head position this long after losing it
EYE_SPACING_M = 0.063
EYE_TO_MOUTH_M = 0.066
CALIBRATION_MAX_RMS_PX = 12.0
PRIMARY = "webcam"  # the camera that defines the body frame


@dataclass
class View:
    """One camera's latest observation of one hand."""
    t: float
    k: np.ndarray
    image: np.ndarray  # (21, 2) pixels
    world: np.ndarray  # (21, 3) MediaPipe metric landmarks
    score: float
    ext: tuple | None = None  # (R, t) at this moment, for a camera that moves
    vel: np.ndarray | None = None  # (21, 2) pixels per second, from this camera's previous frame

    def image_at(self, t: float) -> np.ndarray:
        """The landmarks where they would be at time t. The cameras do not expose at the
        same instant: a hand moving at 1 m/s is 5 cm further on in the 50 ms between two
        of them, which combining the pictures as they are turns into a wrong distance."""
        if self.vel is None:
            return self.image
        return self.image + self.vel * min(max(t - self.t, 0.0), VIEW_PREDICT_MAX_S)


class HandTrack:
    def __init__(self):
        self.state = HandState()
        self.pos_f = OneEuro()
        self.rot_f = OneEuroQuat()
        self.gestures = GestureDetector()
        self.last_seen = 0.0
        self.last_t = 0.0  # capture time of the last frame used
        self.mode: str | None = None  # "triangulated" | "single"
        self.views_used: list = []
        self.raw_pos = np.zeros(3)
        self.lm_body = np.zeros((21, 3))
        self.lm_room = np.zeros((21, 3))
        self.lm_ref = np.zeros((21, 3))  # in the webcam's frame
        self.centre_fixed: np.ndarray | None = None  # palm centre from the fixed cameras only
        self.centre_t = -1e9
        self.path_key: tuple | None = None   # which cameras, combined how, gave the last pose
        self.carry = np.zeros(3)             # see HANDOVER_TAU_S
        self.head_shape: np.ndarray | None = None  # landmarks about the palm centre, as the head camera saw them
        self.head_shape_t = -1e9
        self.last_go = None                  # the gestures read from the head camera's view
        self.leaps = 0  # consecutive frames skipped as implausible leaps
        self.wrong_side = 0.0  # running evidence that this is really the other hand
        self.confidence = 0.0
        self.reproj_px = 0.0
        self.height_px = 0.0
        self.pinch = 1.0

    def lose(self) -> None:
        self.state = HandState()
        self.pos_f.reset()
        self.rot_f.reset()
        self.gestures.reset()
        self.confidence = 0.0
        self.mode, self.views_used = None, []


class Hub:
    def __init__(self, config: Config | None = None, start_camera: bool = True):
        self.config = config or Config()
        cfg = self.config.get()
        self.link = DriverLink(cfg["driver"]["host"], cfg["driver"]["port"], cfg["driver"]["status_port"])
        self.hands = [HandTrack(), HandTrack()]  # 0 = left, 1 = right
        self.lock = threading.Lock()
        self.cams = {PRIMARY: CameraRunner(PRIMARY)} | {n: CameraRunner(n) for n in cfg["extra_cameras"]}
        self._start_camera = start_camera
        self.views: dict = {}  # (camera, hand index) -> View
        self.side_stats = {"groups": 0, "shape": 0, "body": 0, "shape_and_body_agree": 0, "shape_and_body_differ": 0}
        self.dets: dict = {}  # camera -> (time, [(View, votes right)]) of its latest frame
        self.head_views: dict = {}  # camera -> (t, k, head pixels, head visibility)
        self.body_views: dict = {}  # camera -> (t, k, pixels (33, 2), visibility (33,))
        self.body_pts = np.zeros((33, 3))  # room axes, relative to the webcam
        self.body_seen = np.full(33, -1e9)
        self.body_views_used: list = []
        self.torso_yaw = 0.0  # room axes, radians about +y; 0 = facing the webcam
        self.torso_seen = -1e9
        self.floor_samples: deque = deque(maxlen=300)
        self.floor_y: float | None = None  # room y of the floor (the webcam is y = 0)
        # Eye heights seen lately: when no camera pair sees the feet, the floor is taken to
        # be the user's own eye height below where their eyes are when standing upright.
        self.eye_y_samples: deque = deque(maxlen=3600)
        self.head_hand_ratios = [deque(maxlen=600), deque(maxlen=600)]  # see _learn_head_hand_scale
        self.body_ok = False  # the triangulated body currently looks like a body
        self.r_room = g.BASE_CAM_TO_BODY
        self.body_world: dict = {}  # camera -> (t, pose world landmarks (33, 3), visibility)
        self.recentre_seq = 0
        self.raw_log: list | None = None  # raw recording in progress
        self.raw_until = 0.0
        self.raw_last: str | None = None
        # The head camera's fit: {"a", "b", "off", "err_deg", "used", "t"} or None. Its
        # mount (b, off) is remembered in the config; a (the VR runtime's heading against
        # the room) is found afresh in every session.
        self.headfit: dict | None = None
        self.headfit_note = "waiting for a hand that the fixed cameras and the head camera both see"
        self.head_sightings: deque = deque(maxlen=4000)  # (time, sighting no., d, x, m, h)
        self._sighting_no = 0
        self.headfit_mount_ok = False  # the mount was confirmed in this run, with the head turning
        self.headfit_serial = 0  # goes up with every new fit: a refit is a change of source for the hands
        self._headfit_next = 0.0
        self._headfit_busy = False
        self.headfit_sync = not start_camera  # replay and tests: solve in the calling thread
        self.agreement: dict = {}  # camera -> smoothed agreement with the others, pixels at 640 wide
        self.head_pos = np.zeros(3)  # eyes, in the webcam's frame
        self.head_f = OneEuro(min_cutoff=1.0, beta=0.5)  # steady when still, quick when crouching
        self.head_rot = np.array([1.0, 0.0, 0.0, 0.0])  # room axes, w x y z
        self.head_rot_f = OneEuroQuat(min_cutoff=0.6, beta=0.05)
        self.head_rot_t = 0.0
        self.head_rot_seen = -1e9
        self.head_t = 0.0
        self.head_seen = -1e9
        self.head_mode: str | None = None
        self.head_views_used: list = []
        self.collector: mv.PairCollector | None = None
        self.collect_until = 0.0
        self.calib_result: dict | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="hub", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        for cam in self.cams.values():
            cam.stop()
        self.link.close()

    @staticmethod
    def camera_configs(cfg: dict) -> dict:
        """Per-camera worker configs by name. The primary webcam is a V4L2 device."""
        return {PRIMARY: cfg["camera"] | {"type": "v4l2"}} | cfg["extra_cameras"]

    # ---- main loop -------------------------------------------------------
    def _run(self) -> None:
        last_sync = 0.0
        while self._running:
            cfg = self.config.get()
            now = time.monotonic()
            if self._start_camera and now - last_sync > 0.5:
                last_sync = now
                for name, cam_cfg in self.camera_configs(cfg).items():
                    self.cams[name].sync(cam_cfg)
            period = 1.0 / max(cfg["driver"]["min_send_hz"], 1)
            by_conn = {c.conn: c for c in self.cams.values() if c.conn is not None}
            messages = []
            if by_conn:
                try:
                    for conn in wait_connections(list(by_conn), timeout=period):
                        try:
                            messages.append((by_conn[conn], conn.recv()))
                        except (EOFError, OSError):
                            pass  # worker died; sync() restarts it
                except OSError:
                    time.sleep(period)
            else:
                time.sleep(period)
            with self.lock:
                if self.raw_log is not None:
                    now_m = time.monotonic()
                    for cam, msg in messages:
                        if "error" not in msg:
                            self.raw_log.append((now_m, cam.name, {k: v for k, v in msg.items() if k != "jpeg"}))
                    if self.link.hmd_history:
                        self.raw_log.append((now_m, "_hmd", self.link.hmd_history[-1]))
                    if now_m >= self.raw_until:
                        self._raw_finish()
                for cam, msg in messages:
                    frame = cam.on_message(msg)
                    if frame is not None:
                        if cam.last_head is not None:
                            self.process_head(cam.name, frame[1], cam.last_head, cfg)
                        self.process_frame(*frame, cfg, cam=cam.name)
                if self.collector is not None and time.monotonic() >= self.collect_until:
                    self._finish_calibration(cfg)
                self._expire(cfg)
                self.link.poll()
                self._send(cfg)

    def extrinsics(self, cfg: dict, trusted_only: bool = False) -> dict:
        """Calibrated cameras: {name: (R, t)} into the webcam's frame.

        With trusted_only, a camera that currently disagrees with the others (it was
        moved, or its calibration is poor) is left out, so it cannot pull the hands and
        the head around; it comes back by itself when it agrees again.
        """
        out = {PRIMARY: (np.eye(3), np.zeros(3))}
        for name, c in cfg["extra_cameras"].items():
            if c["enabled"] and c["calibrated"] and not c.get("moving"):
                if trusted_only and (self.agreement.get(name) or 0.0) > AGREEMENT_BAD_PX:
                    continue
                out[name] = (np.array(c["rot"]).reshape(3, 3), np.array(c["tvec"]))
        return out

    def _process_body(self, cam: str, t: float, size, px: np.ndarray, vis: np.ndarray, cfg: dict,
                      world: np.ndarray | None = None) -> None:
        """Whole-body skeleton (33 pose landmarks) triangulated from the calibrated
        cameras, in room axes relative to the webcam. It is for display: seeing the
        tracked body next to the hands shows where the tracking is wrong."""
        k = g.intrinsics_from_hfov(size[0], size[1], self.camera_configs(cfg)[cam]["hfov_deg"])
        self.body_views[cam] = (t, k, px, vis)
        if self.collector is not None and world is not None and not self.camera_configs(cfg)[cam].get("fixed"):
            # The whole body as a calibration target: it covers far more of the room
            # than a hand, and every camera sees it.
            w = np.asarray(world, float)
            segs = [(a, b, float(np.linalg.norm(w[a] - w[b]))) for a, b in mv.BODY_SEGMENTS]
            self.collector.add(cam, "body", t, px, vis > 0.75, segs, size[0], list(self.cams), max_dt_s=0.12)
        ext = self.extrinsics(cfg)
        fresh = {n: v for n, v in self.body_views.items() if n in ext and t - v[0] <= HEAD_VIEW_MAX_AGE_S}
        if cam not in fresh or len(fresh) < 2:
            return
        c = cfg["camera"]
        r_cb = g.cam_to_body_rotation(c["yaw_deg"], c["pitch_deg"], c["roll_deg"])
        now = time.monotonic()
        for i in range(33):
            views = [(v[1], *ext[n], v[2][i:i + 1], 1.0) for n, v in fresh.items() if v[3][i] > BODY_MIN_VISIBILITY]
            sol = mv.triangulate_robust(views, BODY_MAX_RMS_PX)
            if sol is None:
                continue
            p_room = r_cb @ sol[0][0]
            fresh_pt = now - self.body_seen[i] < BODY_HOLD_S
            self.body_pts[i] = self.body_pts[i] * 0.5 + p_room * 0.5 if fresh_pt else p_room
            self.body_seen[i] = now

        # Calibration health: how well each pair of cameras agrees on steady body points
        # (head and shoulders). A camera whose best agreement is poor has been moved
        # since it was calibrated, or was calibrated badly.
        # Only judged on a body that triangulated into a plausible shape: a camera that
        # sees the person from behind guesses the hidden face and shoulders, and those
        # guesses would make every camera look wrong.
        names = list(fresh) if self._body_plausible(now) else []
        best = {n: None for n in names}
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                va, vb = fresh[names[a]], fresh[names[b]]
                idx = [i for i in AGREEMENT_LANDMARKS if va[3][i] > 0.7 and vb[3][i] > 0.7]
                if len(idx) < 3:
                    continue
                _, rms = mv.triangulate([(va[1], *ext[names[a]], va[2][idx], 1.0),
                                         (vb[1], *ext[names[b]], vb[2][idx], 1.0)])
                # Compare across resolutions: express the error for a 640-pixel-wide image.
                rms *= 640.0 / max(va[1][0, 2], vb[1][0, 2]) / 2.0
                for n in (names[a], names[b]):
                    best[n] = rms if best[n] is None else min(best[n], rms)
        self._update_agreement(best)
        self.body_views_used = sorted(fresh, key=list(self.cams).index)
        self.r_room = r_cb
        # Which way the torso faces (from the shoulders) and where the floor is (from the
        # feet). A VR app uses these to keep its world lined up with the room all the
        # time: the headset's own heading drifts, and its height is not known at all.
        if now - self.body_seen[11] < BODY_HOLD_S and now - self.body_seen[12] < BODY_HOLD_S:
            right = self.body_pts[12] - self.body_pts[11]
            right[1] = 0.0
            if 0.2 < float(np.linalg.norm(right)) < 0.6:
                fwd = np.cross([0.0, 1.0, 0.0], right)
                self.torso_yaw = math.atan2(-fwd[0], -fwd[2])
                self.torso_seen = now
        feet = [self.body_pts[i][1] for i in (27, 28, 29, 30, 31, 32) if now - self.body_seen[i] < BODY_HOLD_S]
        if feet:
            self.floor_samples.append(min(feet) - 0.05)
            self.floor_y = float(np.percentile(self.floor_samples, 20))
        pts, ok = self.body_pts, now - self.body_seen < BODY_HOLD_S
        self.body_ok = False
        if ok[[0, 11, 12]].all():
            shoulders = float(np.linalg.norm(pts[11] - pts[12]))
            neck = float(np.linalg.norm(pts[0] - (pts[11] + pts[12]) / 2.0))
            self.body_ok = 0.22 <= shoulders <= 0.60 and 0.10 <= neck <= 0.45

    def _body_plausible(self, now: float) -> bool:
        pts, ok = self.body_pts, now - self.body_seen < BODY_HOLD_S
        if not ok[[0, 11, 12]].all():
            return False
        shoulders = float(np.linalg.norm(pts[11] - pts[12]))
        neck = float(np.linalg.norm(pts[0] - (pts[11] + pts[12]) / 2.0))
        return 0.22 <= shoulders <= 0.60 and 0.10 <= neck <= 0.45

    def _update_agreement(self, best: dict) -> None:
        """Slowly track each camera's best agreement with another camera (pixels at 640 wide)."""
        for n, r in best.items():
            if r is not None:
                old = self.agreement.get(n)
                # A camera starts out trusted: one bad first reading must not exclude it.
                self.agreement[n] = min(r, 15.0) if old is None else old * 0.93 + min(r, 150.0) * 0.07

    def process_head(self, cam: str, size, head: tuple, cfg: dict) -> None:
        """Locate the head (eyes) in the webcam frame from the cameras that see it.

        With two or more calibrated views the visible head landmarks are triangulated.
        The webcam alone gives a rough distance from the eye spacing.
        """
        t_ns, body_px, body_vis, body_world = head
        t = t_ns / 1e9
        self._process_body(cam, t, size, np.asarray(body_px, float), np.asarray(body_vis, float), cfg, body_world)
        self.body_world[cam] = (t, np.asarray(body_world, float), np.asarray(body_vis, float))
        px, vis, world = body_px[HEAD_IDX], body_vis[HEAD_IDX], body_world[HEAD_IDX]
        k = g.intrinsics_from_hfov(size[0], size[1], self.camera_configs(cfg)[cam]["hfov_deg"])
        self.head_views[cam] = (t, k, np.asarray(px, float), np.asarray(vis, float))
        ext = self.extrinsics(cfg, trusted_only=True)
        if cam in ext:
            # Orientation: from the camera that sees the face. Every camera's estimate is
            # brought into the webcam's frame, then into room axes.
            r_head = g.head_rotation(world, vis)
            if r_head is not None:
                c = cfg["camera"]
                r_room = g.cam_to_body_rotation(c["yaw_deg"], c["pitch_deg"], c["roll_deg"]) @ ext[cam][0] @ r_head
                self.head_rot = self.head_rot_f(g.quat_from_matrix(r_room), t) if t > self.head_rot_t else self.head_rot
                self.head_rot_t = max(t, self.head_rot_t)
                self.head_rot_seen = time.monotonic()
        fresh = {n: v for n, v in self.head_views.items() if n in ext and t - v[0] <= HEAD_VIEW_MAX_AGE_S}
        if cam not in fresh:
            return

        points = {}  # landmark index -> 3D point
        if len(fresh) >= 2:
            for i in range(5):
                views = [(v[1], *ext[n], v[2][i:i + 1], 1.0) for n, v in fresh.items() if v[3][i] > HEAD_MIN_VISIBILITY]
                if len(views) >= 2:
                    p, rms = mv.triangulate(views)
                    if rms <= TRIANGULATION_MAX_RMS_PX:
                        points[i] = p[0]
        mode, used = "triangulated", sorted(fresh, key=list(self.cams).index)
        if not points:
            # One camera: distance from the apparent size of the face. Turning the head
            # shrinks the eye spacing and nodding shrinks the eye-to-mouth height, so the
            # nearer of the two estimates (the less foreshortened one) is used.
            v = fresh.get(PRIMARY)
            if v is None or min(v[3][1], v[3][2]) <= HEAD_MIN_VISIBILITY:
                return
            f = v[1][0, 0]
            eyes_mid = (v[2][1] + v[2][2]) / 2.0
            estimates = []
            eye_px = float(np.linalg.norm(v[2][1] - v[2][2]))
            if eye_px >= 4.0:
                estimates.append(f * EYE_SPACING_M / eye_px)
            if min(v[3][5], v[3][6]) > HEAD_MIN_VISIBILITY:
                mouth_px = float(np.linalg.norm((v[2][5] + v[2][6]) / 2.0 - eyes_mid))
                if mouth_px >= 4.0:
                    estimates.append(f * EYE_TO_MOUTH_M / mouth_px)
            if not estimates:
                return
            z = min(estimates)
            if not 0.2 <= z <= 4.0:
                return
            mid = mv.normalize_px((v[2][1] + v[2][2]) / 2.0, v[1])
            points = {1: np.array([mid[0] * z, mid[1] * z, z])}
            points[2] = points[1]
            mode, used = "single", [PRIMARY]

        if 1 in points and 2 in points:
            eyes = (points[1] + points[2]) / 2.0
        elif 3 in points and 4 in points:
            eyes = (points[3] + points[4]) / 2.0  # between the ears: a few cm behind the eyes
        else:
            eyes = np.mean(list(points.values()), axis=0)
        self.head_pos = self.head_f(eyes, t) if t > self.head_t else self.head_pos
        self.head_t = max(t, self.head_t)
        self.head_seen = time.monotonic()
        c = cfg["camera"]
        self.eye_y_samples.append(float(
            (g.cam_to_body_rotation(c["yaw_deg"], c["pitch_deg"], c["roll_deg"]) @ self.head_pos)[1]))
        self.head_mode, self.head_views_used = mode, used

    def camera_position(self, cfg: dict, r_cb: np.ndarray) -> np.ndarray:
        """The webcam's position in the body frame: from the tracked head if there is
        one (the body frame's origin is the eyes), else the configured position."""
        if cfg["tracking"]["head_tracking"] and time.monotonic() - self.head_seen < HEAD_HOLD_S:
            return -(r_cb @ self.head_pos)
        return np.array(cfg["camera"]["pos"])

    def _which_hands(self, cam: str, t: float, size, dets: list, cam_cfg: dict) -> list:
        """Left or right for each detected hand: True = right, None = not this person's.

        The hand model's own label is unreliable (it flips for the back of the hand and
        at odd angles, differently in each camera), and mismatched labels make the
        cameras combine two different hands. The body tracker has no such problem, so
        when it has fresh wrists in the same image each hand goes to the nearer wrist.
        """
        by_label = [(handed == "Right") != bool(cam_cfg["swap_hands"]) for handed, *_ in dets]
        body = self.body_views.get(cam)
        if not dets or body is None or t - body[0] > WRIST_MATCH_MAX_AGE_S:
            return by_label
        _, _, px, vis = body
        if min(vis[15], vis[16]) < 0.5:
            return by_label
        width = float(size[0])
        # The hand model sometimes "finds" a huge hand in a torso or a piece of clothing.
        # A real hand is about a third of the torso's length; compare each detection with
        # the longest body segment visible (any of them bounds the person's size in the image).
        seg = [float(np.linalg.norm(px[a] - px[b])) * k for a, b, k in BODY_SCALE_SEGMENTS
               if min(vis[a], vis[b]) > 0.6]
        body_px = max(seg) if seg else None
        # With only the face in view, the eyes give the scale: a hand at the head's distance
        # is about three eye-spacings long, and twice that when reaching toward the camera.
        eye_px = float(np.linalg.norm(px[2] - px[5])) if min(vis[2], vis[5]) > 0.7 else None
        # Distance from each hand (wrist and middle knuckle) to the body's left/right wrist.
        d = np.array([[min(np.linalg.norm(np.asarray(image[j], float) - px[w]) for j in (0, 9))
                       for w in (15, 16)] for _, _, image, _ in dets]) / width
        if float(np.linalg.norm(px[15] - px[16])) / width < 0.06:
            return by_label  # wrists together (clapping, holding something): cannot tell
        out = []
        for i, (dl, dr) in enumerate(d):
            image = np.asarray(dets[i][2], float)
            hand_px = float(np.linalg.norm(image[12] - image[0]))  # wrist to middle fingertip
            near_body = min(float(np.linalg.norm(image[0] - px[j])) for j in UPPER_BODY if vis[j] > 0.5) / width \
                if any(vis[j] > 0.5 for j in UPPER_BODY) else 0.0
            if body_px is not None and hand_px > HAND_MAX_BODY_RATIO * body_px:
                out.append(None)  # far too big for this person: not a hand
            elif body_px is None and eye_px is not None and eye_px > 6.0 and hand_px > HAND_MAX_EYE_RATIO * eye_px:
                out.append(None)  # far too big for this face: not a hand
            elif near_body > WRIST_MATCH_MAX_DIST:
                out.append(None)  # nowhere near this person: someone else's hand
            elif min(dl, dr) > WRIST_TRUST_DIST:
                # Near the person but not at either wrist: the body tracker has the arm
                # wrong (it happens with unusual poses), so fall back to the hand's own label.
                out.append(by_label[i])
            else:
                out.append(bool(dr < dl))
        if len(dets) == 2 and out[0] is not None and out[0] == out[1]:
            # Both detections claim the same wrist. Either they are the two hands close
            # together, or the hand model reported one hand twice. Try the split with the
            # smaller total; a detection that would land far from its wrist while sitting
            # on the other one is a duplicate and is dropped.
            straight, crossed = d[0, 0] + d[1, 1], d[0, 1] + d[1, 0]
            out = [False, True] if straight <= crossed else [True, False]
            for i in range(2):
                mine, other = (d[i, 1], d[i, 0]) if out[i] else (d[i, 0], d[i, 1])
                if mine > max(0.10, 2.5 * other):
                    out[i] = None
            if out[0] is None and out[1] is None:  # keep the better one on the shared wrist
                keep = int(np.argmin(d.min(axis=1)))
                out[keep] = bool(d[keep, 1] < d[keep, 0])
        return out

    def process_frame(self, t_ns: int, size, dets: list, cfg: dict, cam: str = PRIMARY) -> None:
        """Take one camera's detections and update the hands it sees.

        dets: (handedness, score, image_px (21,2), world (21,3)) per detected hand.
        """
        cam_cfg = self.camera_configs(cfg)[cam]
        k = g.intrinsics_from_hfov(size[0], size[1], cam_cfg["hfov_deg"])
        t = t_ns / 1e9

        labels = self._which_hands(cam, t, size, dets, cam_cfg)
        best: dict[int, tuple] = {}
        for is_right, (handed, score, image, world) in zip(labels, dets):
            if is_right is None:
                continue
            idx = 1 if is_right else 0
            if idx not in best or score > best[idx][0]:
                best[idx] = (score, image, world)

        ext = self.extrinsics(cfg, trusted_only=True)
        ext_all = self.extrinsics(cfg)
        pose = None
        if cam_cfg.get("moving"):
            self._headfit_collect(cam, t, k, dets)
            pose = self._headcam_pose(cam_cfg, t)
            if pose is None:
                return  # not calibrated, or the head or the headset's orientation is unknown

        # Every hand this camera reports, with this camera's own opinion of which hand it
        # is. The opinion is only a vote: which detections in different cameras are the
        # same physical hand is decided from geometry in _associate().
        mine = []
        t_prev, before = self.dets.get(cam, (-1e9, []))
        for is_right, (handed, score, image, world) in zip(labels, dets):
            if is_right is not None:
                v = View(t, k, np.asarray(image, float), np.asarray(world, float), float(score), pose)
                # How the hand moves in this camera's picture, from the nearest hand in its
                # previous frame (a camera's own left/right label is not steady enough).
                if 0.0 < t - t_prev < 0.12 and before:
                    old = min((b[0] for b in before), key=lambda b: float(np.linalg.norm(b.image[0] - v.image[0])))
                    if float(np.linalg.norm(old.image[0] - v.image[0])) < 0.12 * size[0]:
                        v.vel = (v.image - old.image) / (t - t_prev)
                mine.append((v, bool(is_right)))
        self.dets[cam] = (t, mine)

        for idx, (score, image, world) in best.items():
            image = np.asarray(image, float)
            self.views[(cam, idx)] = View(t, k, image, np.asarray(world, float), float(score), pose)
            # Calibration uses one hand at a time and ignores the left/right label, which
            # cameras can disagree on (back of the hand, odd angles): a camera that sees
            # exactly one hand is taken to see "the" hand.
            if self.collector is not None and len(best) == 1 and len(dets) == 1 and not (cam_cfg.get("fixed") or cam_cfg.get("moving")):
                self.collector.add(cam, "hand", t, image, np.ones(21, bool),
                                   mv.hand_segments(world, cfg["tracking"]["hand_scale"]), size[0], list(self.cams))
            if cam in ext_all:
                self._hand_agreement(cam, idx, t, ext_all)
        if cam in ext or pose is not None:
            self._associate(t, ext, cfg)

    def _associate(self, t: float, ext: dict, cfg: dict) -> None:
        """Decide which detections across the cameras are the same physical hand, which
        hand each group is, and update the hands.

        Cameras disagree about left and right (the label flips for the back of the hand,
        and the body tracker that helps is unreliable with a headset on the face). So
        detections are grouped by geometry: two detections in different cameras belong
        together if they triangulate into one rigid hand. Each group is then named by
        where the hands were a moment ago, and only failing that by the cameras' votes.
        """
        items = []  # (camera, view, votes right)
        for cam, (t_cam, lst) in self.dets.items():
            if t - t_cam > VIEW_MAX_AGE_S:
                continue
            for v, right in lst:
                if cam in ext or v.ext is not None:
                    items.append((cam, v, right))
        if not items:
            return

        def pose_of(i):
            cam, v, _ = items[i]
            return v.ext if v.ext is not None else ext[cam]

        # Pairs of detections from different cameras that fit one rigid hand.
        pairs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0] == items[j][0]:
                    continue
                vi, vj = items[i][1], items[j][1]
                pts, rms = mv.triangulate([(vi.k, *pose_of(i), vi.image_at(t), 1.0), (vj.k, *pose_of(j), vj.image_at(t), 1.0)])
                if rms > TRIANGULATION_MAX_RMS_PX:
                    continue
                lead = vi if vi.score >= vj.score else vj
                _, _, fit = g.fit_rigid(lead.world * cfg["tracking"]["hand_scale"], pts)
                if fit <= HAND_FIT_MAX_RMS_M:
                    pairs.append((rms, i, j, pts))
        group = list(range(len(items)))  # union-find, never two detections of one camera in a group
        members = {i: [i] for i in group}
        shape = {}  # group -> triangulated landmarks of its best pair
        for rms, i, j, w in sorted(pairs, key=lambda p: p[0]):
            gi, gj = group[i], group[j]
            if gi == gj or {items[m][0] for m in members[gi]} & {items[m][0] for m in members[gj]}:
                continue
            for m in members[gj]:
                group[m] = gi
            members[gi] += members.pop(gj)
            shape.setdefault(gi, w)

        cands = []  # (members, 3D wrist or None)
        side = {}  # id(members) -> True (right) / False (left), where the 3D shape tells
        for gi, ms in members.items():
            pts3 = shape.get(gi)
            cands.append((ms, None if pts3 is None else pts3[0]))
            if pts3 is not None:
                side[id(ms)] = g.hand_side(pts3)
        cands.sort(key=lambda c: (-len(c[0]), -max(items[m][1].score for m in c[0])))

        # One-handed play (the other hand holds the headset to the face): there is no
        # question of which hand a detection is. Whatever is at the head is the holding
        # hand and is left out; the best of the rest is the free hand, always.
        free = cfg["tracking"]["free_hand"]
        if free in ("left", "right"):
            idx = 1 if free == "right" else 0
            h = self.hands[idx]
            head_ok = time.monotonic() - self.head_seen < HEAD_HOLD_S
            near = max(float(cfg["tracking"]["ignore_hand_near_head_m"]), 0.22)
            best = None
            for ms, w in cands:
                if w is None:  # one camera: a rough place from the hand's size is enough here
                    cam, v, _ = items[ms[0]]
                    sol = g.solve_hand_pnp(v.world, v.image, v.k, cfg["tracking"]["hand_scale"])
                    if sol is None:
                        continue
                    r, c = pose_of(ms[0])
                    w = r @ sol[0][0] + c
                    rough = True
                else:
                    rough = False
                # the holding hand: at the head (for a rough place, judge by direction
                # from the camera too, since its distance is unreliable)
                if head_ok:
                    d_head = float(np.linalg.norm(w - self.head_pos))
                    if rough:
                        cam, v, _ = items[ms[0]]
                        r, c = pose_of(ms[0])
                        hp = r.T @ (self.head_pos - c)
                        if hp[2] > 0.05:
                            px = hp[:2] / hp[2] * v.k[0, 0] + v.k[:2, 2]
                            head_px = v.k[0, 0] * 0.20 / hp[2]  # 20 cm at the head's distance
                            if float(np.linalg.norm(v.image[g.PALM_IDS].mean(axis=0) - px)) < head_px:
                                continue
                    elif d_head < near:
                        continue
                jump = float(np.linalg.norm(w - h.lm_ref[0])) if (h.state.tracked and t - h.last_t < 0.5 and not rough) else 0.0
                key = (-len(ms), jump, -max(items[m][1].score for m in ms))
                if best is None or key < best[0]:
                    best = (key, ms)
            if best is not None and t > h.last_t:
                self._fuse(idx, t, ext, cfg, {items[m][0]: items[m][1] for m in best[1]})
            return

        # The skeleton's wrists, when the body is tracked in 3D: each arm ends in its own hand.
        now_m = time.monotonic()
        body_wrists = None
        if self.body_ok and now_m - self.body_seen[15] < 0.4 and now_m - self.body_seen[16] < 0.4:
            wl, wr = self.r_room.T @ self.body_pts[15], self.r_room.T @ self.body_pts[16]
            if float(np.linalg.norm(wl - wr)) > 0.20:
                body_wrists = (wl, wr)

        def by_body(ms, w):
            """Which arm this hand is on: 1 right, 0 left, None if not clear."""
            if body_wrists is None:
                return None
            if w is not None:
                d = [float(np.linalg.norm(w - bw)) for bw in body_wrists]
                limit = 0.22
            else:  # one camera: compare in its picture
                cam, v, _ = items[ms[0]]
                r, c = pose_of(ms[0])
                d = []
                for bw in body_wrists:
                    p = r.T @ (bw - c)
                    if p[2] <= 0.05:
                        return None
                    d.append(float(np.linalg.norm(v.image[0] - (p[:2] / p[2] * v.k[0, 0] + v.k[:2, 2]))) / (2.0 * v.k[0, 2]))
                limit = 0.12
            i = int(d[1] < d[0])
            return i if d[i] < limit and d[i] < 0.6 * d[1 - i] else None

        # Name each group. Continuity first: a hand is where it was a moment ago.
        prev = {i: h for i, h in enumerate(self.hands) if h.state.tracked and 0.0 <= t - h.last_t < 0.5}
        assigned: dict = {}
        for ms, w in cands:
            dist = {}
            for i, h in prev.items():
                if w is not None:
                    dist[i] = float(np.linalg.norm(w - h.lm_ref[0])) / ASSOC_MAX_JUMP_M
                else:  # one camera: compare in its picture
                    cam, v, _ = items[ms[0]]
                    r, c = pose_of(ms[0])
                    p = r.T @ (h.lm_ref[0] - c)
                    if p[2] > 0.05:
                        px = p[:2] / p[2] * v.k[0, 0] + v.k[:2, 2]
                        dist[i] = float(np.linalg.norm(v.image[0] - px)) / (ASSOC_MAX_JUMP_PX * 2.0 * v.k[0, 2])
            near = sorted((d, i) for i, d in dist.items() if d < 1.0)
            idx = None
            geo = side.get(id(ms))
            arm = by_body(ms, w)
            if geo is not None and arm is not None:
                self.side_stats["shape_and_body_agree" if int(geo) == arm else "shape_and_body_differ"] += 1
            if arm is not None:
                self.side_stats["body"] += 1
            if geo is not None:
                self.side_stats["shape"] += 1
            self.side_stats["groups"] += 1
            if (near and near[0][1] not in assigned and near[0][0] < 0.4
                    and (len(near) == 1 or near[1][0] > 2.5 * near[0][0])):
                # Right where one hand was a moment ago and well away from the other: it is
                # that hand, whatever the skeleton or a label says in this one frame. (They
                # are wrong for a frame now and then, which would make the two hands trade
                # places and back. A hand truly on the wrong arm for a second or so is put
                # right by _check_sides.)
                idx = near[0][1]
            elif (near and near[0][1] in assigned and near[0][0] < 0.4
                  and (len(near) == 1 or near[1][0] > 2.5 * near[0][0])):
                # A second sighting of a hand already placed in this frame (one camera's
                # view of it that did not join the others): it is not the other hand.
                continue
            elif arm is not None and arm not in assigned and SIDE_PRIORITY == "body":
                # The arm it is attached to: the skeleton knows its left from its right.
                idx = arm
            elif geo is not None and int(geo) not in assigned:
                # Seen in 3D by two cameras, the hand's own shape says which hand it is.
                # That beats everything else, and it corrects a swap the moment both
                # cameras see the hand again.
                idx = int(geo)
            elif arm is not None and arm not in assigned:
                idx = arm
            elif near and near[0][1] not in assigned and (len(near) == 1 or near[1][0] > 1.5 * near[0][0]):
                idx = near[0][1]
            if idx is None:  # no clear continuity: the cameras' votes, weighted by confidence
                vote = sum((1.0 if items[m][2] else -1.0) * items[m][1].score for m in ms)
                idx = 1 if vote > 0 else 0
                if idx in assigned:
                    idx = 1 - idx if (1 - idx) not in assigned and (1 - idx) not in prev else None
            if idx is not None and idx not in assigned:
                assigned[idx] = ms

        for idx, ms in assigned.items():
            if t > self.hands[idx].last_t:
                self._fuse(idx, t, ext, cfg, {items[m][0]: items[m][1] for m in ms})
        self._check_sides(t)

    def _check_sides(self, t: float) -> None:
        """Catch left and right having been swapped.

        Naming hands by where they were a moment ago keeps them steady, but it also keeps
        a mistake: once the two are swapped they would stay swapped. The body tracker's
        wrists do know their sides, so a hand that keeps sitting clearly at the other
        side's wrist (for about half a second, with the wrists apart) is renamed.
        """
        now = time.monotonic()
        if not self.body_ok or now - self.body_seen[15] > 0.3 or now - self.body_seen[16] > 0.3:
            return
        wl, wr = self.r_room.T @ self.body_pts[15], self.r_room.T @ self.body_pts[16]
        if float(np.linalg.norm(wl - wr)) < 0.25:
            return
        for i, h in enumerate(self.hands):
            if not h.state.tracked or t - h.last_t > 0.2:
                continue
            own, other = (wr, wl) if i == 1 else (wl, wr)
            d_own, d_other = float(np.linalg.norm(h.lm_ref[0] - own)), float(np.linalg.norm(h.lm_ref[0] - other))
            h.wrong_side = 0.9 * h.wrong_side + 0.1 * float(d_other < 0.6 * d_own and d_other < 0.25)
        l, r = self.hands
        if max(l.wrong_side, r.wrong_side) > 0.8 and min(l.wrong_side if l.state.tracked else 1.0,
                                                          r.wrong_side if r.state.tracked else 1.0) > 0.3:
            self.hands[0], self.hands[1] = r, l
            l.wrong_side = r.wrong_side = 0.0

    # ---- head-mounted camera ---------------------------------------------
    def _headcam_pose(self, cam_cfg: dict, t: float):
        """(R, centre) of the head camera in the webcam's frame at time t, or None."""
        fit = self._headfit_usable()
        if fit is None or time.monotonic() - self.head_seen > HEAD_HOLD_S:
            return None
        q = self.link.hmd_rot_at(t) if self.link.connected and self.link.hmd_valid else None
        if q is None:
            return None
        return hc.camera_extrinsics(fit["a"], fit["b"], fit["off"], g.matrix_from_quat(q), self.head_pos)

    def _headfit_usable(self):
        fit = self.headfit
        if fit is None or fit["err_deg"] > HEADFIT_GOOD_DEG or time.monotonic() - fit["t"] > HEADFIT_MAX_AGE_S:
            return None
        return fit

    def _headcam_which_hands(self, dets: list, k: np.ndarray, size, pose: tuple, cam_cfg: dict) -> dict:
        """Left/right for the head camera's detections: {hand index: (score, image, world)}.

        A hand already tracked goes by where it should appear in this picture. Otherwise:
        two hands are left and right as seen (your own hands rarely cross), one hand goes
        by the model's label."""
        r, c = pose
        now = time.monotonic()
        proj = {}
        for i, h in enumerate(self.hands):
            if h.state.tracked and now - h.last_seen < 0.25:
                cam_pt = r.T @ (h.lm_ref[0] - c)
                if cam_pt[2] > 0.05:
                    proj[i] = cam_pt[:2] / cam_pt[2] * k[0, 0] + k[:2, 2]
        out: dict = {}
        order = sorted(dets, key=lambda d: float(np.asarray(d[2])[0][0]))  # left to right in the picture
        for n, (handed, score, image, world) in enumerate(order):
            wrist = np.asarray(image, float)[0]
            near = [(float(np.linalg.norm(wrist - p)), i) for i, p in proj.items()]
            near = [x for x in near if x[0] < 0.18 * size[0]]
            if near:
                idx = min(near)[1]
            elif len(order) == 2:
                idx = n
            else:
                idx = 1 if (handed == "Right") != bool(cam_cfg["swap_hands"]) else 0
            if idx not in out or score > out[idx][0]:
                out[idx] = (score, image, world)
        return out

    def calibrate_headcam_start(self, seconds: float = 0.0) -> dict:
        """Forget what is known about the head camera and learn it again (after the phone
        was taken out of the viewer and put back differently, say). Learning needs no
        ceremony: it happens whenever a hand is in view of two fixed cameras and the head
        camera, and goes faster if the head turns meanwhile."""
        with self.lock:
            if not (self.link.connected and self.link.hmd_valid):
                return {"ok": False, "error": "no headset orientation: the VR runtime and a VR app must be running"}
            self.headfit = None
            self.headfit_mount_ok = False
            self.head_sightings.clear()
            self.headfit_note = "learning: hold a hand where the cameras and the headset see it, and look around"
        for name, c in self.config.get()["extra_cameras"].items():
            if c.get("moving"):
                self.config.update({"extra_cameras": {name: {"calibrated": False}}})
        return {"ok": True, "seconds": seconds}

    def _headfit_collect(self, cam: str, t: float, k: np.ndarray, dets: list) -> None:
        """Note where the head camera sees each palm, beside where the fixed cameras put
        the palms at that moment, and refit the head camera now and then."""
        now = time.monotonic()
        q = self.link.hmd_rot_at(t) if self.link.connected and self.link.hmd_valid else None
        if q is not None and now - self.head_seen < 0.3:
            m, k_inv = g.matrix_from_quat(q), np.linalg.inv(k)
            # Which hand a sighting is must be known beforehand. Left to the fit, a still
            # pair of hands fools it: "the left hand's picture is the right hand" is just
            # as consistent as the truth, with the room turned by the angle between the
            # hands. Seen from one's own head, the left hand is the one on the left; a
            # single hand goes by the hand tracker's label.
            order = sorted(dets, key=lambda det: float(np.asarray(det[2])[0][0]))
            for n, det in enumerate(order):
                if len(order) == 2:
                    idx = n
                elif len(order) == 1:
                    idx = 1 if (det[0] == "Right") != bool(self.camera_configs(self.config.get())[cam]["swap_hands"]) else 0
                else:
                    continue
                hnd, lm = self.hands[idx], np.asarray(det[2], float)
                w = 16 if idx == 1 else 15
                if hnd.centre_fixed is not None and abs(t - hnd.centre_t) < HEADFIT_PAIR_MAX_AGE_S:
                    px, where = lm[g.PALM_IDS, :2].mean(axis=0), hnd.centre_fixed.copy()
                elif self.body_ok and now - self.body_seen[w] < 0.15:
                    # No palm from two fixed cameras (often so for one of the hands): the
                    # skeleton's wrist is coarser but there nearly always, and sightings of
                    # both hands, far apart in the picture, are what pins the camera down.
                    px, where = lm[0, :2], self.r_room.T @ self.body_pts[w]
                else:
                    continue
                d = k_inv @ np.array([px[0], px[1], 1.0])
                self._sighting_no += 1
                self.head_sightings.append(
                    (now, self._sighting_no, d / np.linalg.norm(d), where, m, self.head_pos.copy()))
        if now >= self._headfit_next and not self._headfit_busy:
            self._headfit_next = now + HEADFIT_EVERY_S
            rows = [r for r in self.head_sightings if now - r[0] < HEADFIT_WINDOW_S]
            if len({r[1] for r in rows}) >= HEADFIT_MIN_SIGHTINGS:
                self._headfit_busy = True
                args = (cam, rows, self.headfit, self.r_room.copy())
                if self.headfit_sync:
                    self._headfit_solve(*args)
                else:
                    threading.Thread(target=self._headfit_solve, args=args, daemon=True).start()

    def _headfit_solve(self, cam: str, rows: list, current, r_room: np.ndarray) -> None:
        """Runs beside the tracking (it can take a second the first time)."""
        try:
            grp = np.array([r[1] for r in rows])
            d, x, m, h = (np.array([r[i] for r in rows]) for i in (2, 3, 4, 5))
            turned = hc.head_spread_deg(list(m[:: max(1, len(m) // 60)]))
            c = self.config.get()["extra_cameras"][cam]
            stored = (np.array(c["rot"]).reshape(3, 3), np.array(c["tvec"])) if c["calibrated"] else None
            # The runtime's world and the room both have "up" about right, so A is a turn
            # about the vertical, give or take; every heading is tried when none is known.
            headings = [r_room.T @ g.rot_y(y) for y in np.radians(np.arange(0, 360, 30))]
            full = turned >= HEADFIT_FULL_TURN_DEG
            # With the head still, a wrong mount and a wrong heading can cancel out for the
            # one direction being looked in, and fall apart at the next glance aside. So
            # the mount, remembered from last time or not, counts for nothing until a fit
            # with the head turning has confirmed it; after that the heading alone is
            # followed while the head is still.
            if not full and not (self.headfit_mount_ok and current is not None):
                self.headfit_note = "learning: look around a little (up, down and sideways) with a hand in view"
                return
            grid = [(a, b, np.zeros(3)) for a in headings for b in hc.MOUNT_GUESSES]
            if current is not None:
                inits = [(current["a"], current["b"], current["off"])]
            elif stored is not None:
                inits = [(a, stored[0], stored[1]) for a in headings]
            else:
                inits = grid
            fit = hc.solve_mount(d, x, m, h, grp, inits, fix_mount=not full)
            if full and inits is not grid and (fit is None or fit["err_deg"] > HEADFIT_GOOD_DEG):
                fit = hc.solve_mount(d, x, m, h, grp, grid)  # what was remembered no longer holds
            mount = stored if current is None else (current["b"], current["off"])
            now = time.monotonic()
            if fit is not None and fit["err_deg"] <= HEADFIT_GOOD_DEG and fit["used"] >= HEADFIT_MIN_SIGHTINGS * 0.6:
                fit["t"], fit["turned_deg"] = now, round(turned)
                first_mount = mount is None or not c["calibrated"]
                moved = mount is not None and hc.rotation_angle_deg(mount[0], fit["b"]) > 1.5
                with self.lock:
                    self.headfit = fit
                    self.headfit_mount_ok = self.headfit_mount_ok or full
                    self.headfit_serial += 1
                    self.headfit_note = "tracking"
                if full and (first_mount or moved):  # remember how the camera sits in the headset
                    self.config.update({"extra_cameras": {cam: {
                        "calibrated": True, "align": fit["a"].reshape(9).tolist(),
                        "rot": fit["b"].reshape(9).tolist(), "tvec": fit["off"].tolist()}}})
            elif current is not None:
                # No better fit. Is the one in use still right for what is seen now?
                e, _ = hc.bearing_errors(current["a"], current["b"], current["off"], d, x, m, h)
                best = [float(e[grp == n].min()) for n in np.unique(grp)]
                if float(np.percentile(best, 50)) > HEADFIT_LOST_DEG:
                    with self.lock:
                        self.headfit = None
                        self.headfit_mount_ok = False
                        self.headfit_note = "lost (headset reseated or its heading reset): learning again"
        finally:
            self._headfit_busy = False

    def _hand_agreement(self, cam: str, idx: int, t: float, ext: dict) -> None:
        """Calibration health from a hand seen by this camera and another at once. Hand
        landmarks are only reported when the hand is really visible, so they are a
        sounder check than the body's."""
        mine = self.views[(cam, idx)]
        best = {}
        for (n, i), v in self.views.items():
            if i != idx or n == cam or n not in ext or t - v.t > 0.04:
                continue
            _, rms = mv.triangulate([(mine.k, *ext[cam], mine.image, 1.0), (v.k, *ext[n], v.image, 1.0)])
            rms *= 640.0 / max(mine.k[0, 2], v.k[0, 2]) / 2.0
            for name in (cam, n):
                best[name] = rms if name not in best else min(best[name], rms)
        self._update_agreement(best)

    def _fuse(self, idx: int, t: float, ext: dict, cfg: dict, fresh: dict) -> None:
        """Hand pose from the views (camera -> View) that _associate() found to show this hand."""
        cam, trk = cfg["camera"], cfg["tracking"]
        if not fresh:
            return
        ext = {n: (v.ext if v.ext is not None else ext[n]) for n, v in fresh.items()}
        lead = max(fresh, key=lambda n: fresh[n].score)  # best view: gestures and single-view fallback
        pts_ref, err, mode, used = None, 0.0, "single", [lead]
        # Where the palm is by the fixed cameras alone: what the head camera is fitted to.
        steady = [n for n, v in fresh.items() if v.ext is None]
        if len(steady) >= 2:
            solc = mv.triangulate_robust(
                [(fresh[n].k, *ext[n], fresh[n].image_at(t)[g.PALM_IDS], fresh[n].score) for n in steady], 12.0)
            if solc is not None:
                self.hands[idx].centre_fixed, self.hands[idx].centre_t = solc[0].mean(axis=0), t
        scale_of = {n: trk["hand_scale"] for n in fresh}
        # The head camera sees the hand from close by and from the wearer's side: nothing
        # shows the fingers better. The fixed cameras stand far off and around: nothing
        # places the hand in the room better. So with the head camera on the hand, its
        # view gives the hand's shape and orientation, and the fixed cameras (or, with
        # only one of them, that one and the head camera) give where the palm is.
        head = next((n for n, v in fresh.items() if v.ext is not None), None)
        if head is not None:
            lead = head
            scale_of[head] = trk["hand_scale"] * self._head_hand_scale(idx)
            hv = fresh[head]
            solh = g.solve_hand_pnp(hv.world, hv.image, hv.k, scale_of[head])
            if solh is not None:
                shape_ref = solh[0] @ ext[head][0].T + ext[head][1]
                self._learn_head_hand_scale(idx, shape_ref[0], ext[head][1])
                # A hand cannot be further from the wearer's face than the arm is long.
                reach = 0.46 * trk["user_height_m"] if trk["user_height_m"] > 0.5 else 0.85
                ray = shape_ref[0] - ext[head][1]
                rng = float(np.linalg.norm(ray))
                if rng > reach:
                    shape_ref = shape_ref - ray / rng * (rng - reach)
                fixed = [n for n in fresh if n != head]
                for names in ([fixed] if len(fixed) >= 2 else []) + ([list(fresh)] if len(fresh) >= 2 else []):
                    # The head camera and a camera it is facing look along the same line
                    # from its two ends: together they say nothing about how far along.
                    if len(names) == 2 and head in names:
                        other = names[0] if names[1] == head else names[1]
                        to_hand = [g.normalize(shape_ref[0] - ext[n][1]) for n in (head, other)]
                        if abs(float(to_hand[0] @ to_hand[1])) > math.cos(math.radians(25.0)):
                            continue
                    sol = mv.triangulate_robust(
                        [(fresh[n].k, *ext[n], fresh[n].image_at(t)[g.PALM_IDS], fresh[n].score) for n in names],
                        TRIANGULATION_MAX_RMS_PX)
                    if sol is None:
                        continue
                    centre = sol[0].mean(axis=0)
                    moved = shape_ref + (centre - shape_ref[g.PALM_IDS].mean(axis=0))
                    # the palm must still be in front of the head camera, about where it saw it
                    if (float(np.linalg.norm(moved[0] - shape_ref[0])) < 0.35
                            and float(np.linalg.norm(moved[0] - ext[head][1])) < reach + 0.10):
                        pts_ref, err, mode = moved, sol[1], "triangulated"
                        used = sorted({names[i] for i in sol[2]} | {head}, key=list(self.cams).index)
                        break
            used = used if pts_ref is not None else [lead]
        if pts_ref is None and len(fresh) >= 2:
            names = list(fresh)
            sol = mv.triangulate_robust([(fresh[n].k, *ext[n], fresh[n].image_at(t), fresh[n].score) for n in names],
                                        TRIANGULATION_MAX_RMS_PX)
            # No agreeing set of views means a wrong hand was matched, or fast motion
            # between unsynchronized frames: fall back to the best single view then.
            if sol is not None and float(np.min((sol[0] - ext[lead][1]) @ ext[lead][0][:, 2])) > 0.05:
                # Triangulating each landmark separately lets small errors bend and stretch
                # the hand. A hand is rigid over one frame: take its shape from the best
                # view's own hand model and only its place and orientation from the
                # triangulated points. A poor fit means the views did not show the same
                # thing, and the single-view path below takes over.
                model = fresh[lead].world * trk["hand_scale"]
                r_fit, t_fit, fit_rms = g.fit_rigid(model, sol[0])
                if fit_rms <= HAND_FIT_MAX_RMS_M:
                    pts_ref, err, mode = model @ r_fit.T + t_fit, sol[1], "triangulated"
                    used = sorted((names[i] for i in sol[2]), key=list(self.cams).index)
        if pts_ref is None:
            v = fresh[lead]
            sol = g.solve_hand_pnp(v.world, v.image, v.k, scale_of[lead])
            if sol is None:
                return
            r, tv = ext[lead]
            pts_ref, err = sol[0] @ r.T + tv, sol[1]
            # One camera gives the hand's direction well but its distance poorly (it is
            # guessed from the hand's size). When the body skeleton has this wrist in 3D,
            # slide the hand along the camera's line of sight to the wrist's distance.
            # If the wrist itself is not in the skeleton, the elbow, hip or shoulder of the
            # same side still bounds the error to an arm's length instead of metres.
            anchors = (16, 14, 24, 12) if idx == 1 else (15, 13, 23, 11)
            now_m = time.monotonic()
            anchor = next((a for a in anchors if now_m - self.body_seen[a] < BODY_HOLD_S), None)
            ray = pts_ref[0] - tv
            rng_hand = float(np.linalg.norm(ray))
            rng_new, new_mode = None, mode
            hp = self.hands[idx]
            if hp.state.tracked and 0.0 < t - hp.last_t < SINGLE_KEEP_RANGE_S:
                # Best of all: where the hand was a moment ago. Its distance from this camera
                # changes little between frames, so keep it and let this view move the hand
                # sideways. This is what stops the hand leaping when a second camera drops out.
                rng_new, new_mode = float(np.linalg.norm(hp.lm_ref[0] - tv)), "single+kept"
                if anchor in (15, 16) and self.body_ok:  # drift slowly toward the skeleton's wrist
                    rng_body = float(np.linalg.norm(self.r_room.T @ self.body_pts[anchor] - tv))
                    rng_new = 0.9 * rng_new + 0.1 * rng_body
            elif anchor is not None and self.body_ok:
                rng_new, new_mode = float(np.linalg.norm(self.r_room.T @ self.body_pts[anchor] - tv)), "single+body"
            if rng_new is not None and rng_hand > 0.05 and 0.3 < rng_new / rng_hand < 3.0:
                pts_ref = pts_ref + ray / rng_hand * (rng_new - rng_hand)
                mode = new_mode

        # A hand cannot leave its arm. If the solution puts the hand's wrist away from
        # the skeleton's wrist, it is wrong (bad distance, or two cameras combining
        # different things): move the hand back onto the wrist, keeping its shape. A
        # well-agreeing triangulation gets more slack, because the skeleton updates
        # less often and trails a fast-moving hand.
        w = 16 if idx == 1 else 15
        if self.body_ok and time.monotonic() - self.body_seen[w] < BODY_HOLD_S:
            off = self.r_room.T @ self.body_pts[w] - pts_ref[0]
            slack = HAND_WRIST_SLACK_GOOD_M if mode == "triangulated" and err <= 8.0 else HAND_WRIST_SLACK_M
            if float(np.linalg.norm(off)) > slack:
                # Pull it back gradually: the skeleton's wrist is itself only roughly right,
                # and jumping onto it in one frame would look like the hand twitching.
                pts_ref = pts_ref + 0.35 * off
                mode = mode + "+snapped"

        # Without a head strap, one hand holds the headset to the face. A hand wrapped
        # around a phone is not something the hand tracker can read, and it is not a
        # hand to play with: leave it untracked rather than report nonsense.
        near = float(trk["ignore_hand_near_head_m"])
        if near > 0 and time.monotonic() - self.head_seen < HEAD_HOLD_S:
            palm = pts_ref[g.PALM_IDS].mean(axis=0)
            if float(np.linalg.norm(palm - self.head_pos)) < near:
                return

        # Fingers: the head camera's view of them, while it has one. A frame from the
        # fixed cameras in between moves the hand but must not swap in their coarser idea
        # of its shape, or the fingers would flicker between the two.
        hp = self.hands[idx]
        from_head = head is not None and lead == head
        if from_head:
            hp.head_shape, hp.head_shape_t = pts_ref - pts_ref[g.PALM_IDS].mean(axis=0), t
        elif hp.head_shape is not None and 0.0 <= t - hp.head_shape_t < HEAD_SHAPE_HOLD_S:
            pts_ref = hp.head_shape + pts_ref[g.PALM_IDS].mean(axis=0)
        held_shape = not from_head and hp.head_shape is not None and 0.0 <= t - hp.head_shape_t < HEAD_SHAPE_HOLD_S

        # Change of source: stay put, then drift (see HANDOVER_TAU_S).
        key = (mode.split("+")[0], tuple(used), self.headfit_serial if head is not None else 0)
        if hp.state.tracked and 0.0 < t - hp.last_t < 0.3:
            hp.carry = hp.carry * math.exp(-(t - hp.last_t) / HANDOVER_TAU_S)
            if key != hp.path_key:
                step = hp.lm_ref[0] - (pts_ref[0] + hp.carry)
                hp.carry = hp.carry + step if float(np.linalg.norm(step)) < HANDOVER_MAX_M else np.zeros(3)
        else:
            hp.carry = np.zeros(3)
        hp.path_key = key
        pts_ref = pts_ref + hp.carry

        # A hand does not teleport. A sudden leap is almost always one bad frame (a wrong
        # match, a camera dropping out): skip it. If the new place persists for a few
        # frames it is real (tracking had been wrong, or was just re-acquired) and is taken.
        hp = self.hands[idx]
        if hp.state.tracked and 0.0 < t - hp.last_t < 0.3:
            speed = float(np.linalg.norm(pts_ref[0] - hp.lm_ref[0])) / (t - hp.last_t)
            if speed > HAND_MAX_SPEED_MPS and hp.leaps < 3:
                hp.leaps += 1
                hp.last_seen = time.monotonic()
                return
        hp.leaps = 0

        r_cb = g.cam_to_body_rotation(cam["yaw_deg"], cam["pitch_deg"], cam["roll_deg"])
        pts = g.cam_to_body(pts_ref, r_cb, self.camera_position(cfg, r_cb))
        pos, rot = g.controller_pose(pts, trk["roll_offset_deg"], trk["pitch_offset_deg"], trk["grip_offset"])

        h = self.hands[idx]
        v = fresh[lead]
        for f in (h.pos_f, h.rot_f.f):
            f.min_cutoff, f.beta, f.d_cutoff = (cfg["smoothing"][n] for n in ("min_cutoff", "beta", "d_cutoff"))
        h.raw_pos, h.lm_body = pos, pts
        h.lm_ref = pts_ref
        h.lm_room = pts_ref @ r_cb.T  # room axes relative to the webcam, like the body and cameras
        h.reproj_px = err
        h.mode, h.views_used = mode, used
        h.height_px = float(np.ptp(v.image[:, 1]))
        h.confidence = v.score * g_clamp(1.0 - err / 25.0)
        if held_shape and h.last_go is not None:
            go = h.last_go  # pinch and grip, too, are read from the head camera while it sees the hand
        else:
            go = h.gestures.update(v.world, cfg["gestures"], h.confidence >= trk["min_confidence"])
            h.last_go = go if from_head else None
        h.pinch = go.pinch
        h.last_seen = time.monotonic()
        h.last_t = t
        h.state = HandState(
            tracked=True, buttons=go.buttons,
            pos=h.pos_f(pos, t), rot=h.rot_f(g.quat_from_matrix(rot), t),
            trigger=go.trigger, grip=go.grip, curl=list(go.curl),
            lm=(pts - pos) @ rot,  # landmarks in the controller's local frame
        )

    # ---- camera-to-camera calibration -------------------------------------
    def calibrate_cameras_start(self, seconds: float = 30.0) -> dict:
        with self.lock:
            online = [n for n, c in self.cams.items() if c.public_stats()["online"]]
            if len(online) < 2:
                return {"ok": False, "error": "at least two cameras must be online"}
            self.collector = mv.PairCollector()
            self.collect_until = time.monotonic() + seconds
            self.calib_result = None
        return {"ok": True, "seconds": seconds}

    def _finish_calibration(self, cfg: dict) -> None:
        """Solve every camera pair that shared enough samples, chain to the webcam, save."""
        collector, self.collector = self.collector, None
        cam_cfgs = self.camera_configs(cfg)
        sizes = {n: c.stats["size"] for n, c in self.cams.items()}
        hfov = {n: float(cam_cfgs[n]["hfov_deg"]) for n in self.cams}
        usable = {k: v for k, v in collector.samples.items() if sizes[k[0]] and sizes[k[1]]}
        try:  # keep the raw samples: they allow re-solving and diagnosing offline
            import pickle
            from .config import CACHE_DIR
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_DIR / "last_calibration.pkl", "wb") as f:
                pickle.dump({"samples": usable, "sizes": sizes, "hfov": hfov}, f)
        except OSError:
            pass

        def intrinsics(name: str, fov: float) -> np.ndarray:
            return g.intrinsics_from_hfov(*sizes[name], fov)

        pair_info = {(a, b): mv.solve_samples(samples, intrinsics(a, hfov[a]), intrinsics(b, hfov[b]))
                     for (a, b), samples in usable.items()}
        pairs = {k: (v["r"], v["t"]) for k, v in pair_info.items()
                 if v["ok"] and v["rms_px"] <= CALIBRATION_MAX_RMS_PX}
        ext = mv.chain_extrinsics(pairs, PRIMARY)
        # The pair solutions are each a little off and need not fit together. Refine all
        # cameras at once on every shared observation, then set the scale from all the
        # known body and hand segment lengths together.
        rms_before = rms_after = None
        if len(ext) > 1:
            ext, _, rms_before, rms_after = mv.refine_joint(usable, intrinsics, hfov, ext, PRIMARY,
                                                            optimize_fov=False)
            scale = mv.metric_scale(usable, intrinsics, hfov, ext)
            if scale is not None:
                ext = {n: (r, t * scale) for n, (r, t) in ext.items()}
        cameras, patch = {}, {}
        for name in cfg["extra_cameras"]:
            if cfg["extra_cameras"][name].get("fixed") or cfg["extra_cameras"][name].get("moving"):
                continue  # placed by other means, not solved here
            mine = {k: v for k, v in pair_info.items() if name in k}
            if name in ext:
                r, tv = ext[name]
                used = [v for k, v in mine.items() if k in pairs]
                patch[name] = {"calibrated": True, "rot": r.reshape(9).tolist(), "tvec": tv.tolist()}
                cameras[name] = {"ok": True,
                                 "rms_px": round(rms_after if rms_after is not None else max(v["rms_px"] for v in used), 2),
                                 "samples": sum(v["samples"] for v in mine.values()),
                                 "distance_m": round(float(np.linalg.norm(tv)), 2)}
            elif not cam_cfgs[name]["enabled"]:
                continue
            elif not mine:
                cameras[name] = {"ok": False, "error": "never saw a hand at the same time as another camera"}
            else:
                v = max(mine.values(), key=lambda x: x.get("samples", 0))
                why = v.get("error") or f"error too large ({v['rms_px']:.1f} px): check the field of view setting"
                cameras[name] = {"ok": False, "error": why, "samples": v.get("samples", 0)}
        if patch:
            self.config.update({"extra_cameras": patch})
            for name in patch:
                self.agreement.pop(name, None)  # judge the new calibration afresh
        ok = any(c["ok"] for c in cameras.values())
        self.calib_result = {"ok": ok, "cameras": cameras,
                             "refined_rms_px": None if rms_after is None else [round(rms_before, 2), round(rms_after, 2)]}
        if not ok:
            self.calib_result["error"] = "no camera could be placed relative to the webcam"

    def _expire(self, cfg: dict) -> None:
        hold = cfg["tracking"]["lost_hold_ms"] / 1000.0
        now = time.monotonic()
        for h in self.hands:
            if h.state.tracked and now - h.last_seen > hold:
                h.lose()

    def _send(self, cfg: dict) -> None:
        d = cfg["driver"]
        flags = FLAG_FOLLOW_HMD_POSITION if d["follow_hmd_position"] else 0
        self.link.send(d["yaw_offset"], flags, self.hands[0].state, self.hands[1].state)

    # ---- calibration -----------------------------------------------------
    def _shift_camera(self, measured: np.ndarray, target: np.ndarray) -> dict:
        """Move the camera position so a point measured at `measured` lands on `target`."""
        cam_pos = np.array(self.config.get()["camera"]["pos"])
        new = cam_pos + (np.asarray(target, float) - np.asarray(measured, float))
        self.config.update({"camera": {"pos": new.tolist()}})
        return {"ok": True, "pos": new.tolist()}

    def calibrate_rest(self) -> dict:
        """Shift the camera position so the current hand pose maps to the rest target."""
        with self.lock:
            pts = [h.raw_pos for h in self.hands if h.state.tracked]
        if not pts:
            return {"ok": False, "error": "no hand is tracked"}
        return self._shift_camera(np.mean(pts, axis=0), REST_TARGET)

    def calibrate_head(self) -> dict:
        """Anchor the hands to the head: thumb tips touching the temples, palms to the camera.

        The temples are at eye height, HEAD_HALF_WIDTH to each side and slightly
        behind the eyes, so the thumb tips give the body-frame origin directly.
        With one hand tracked, that side's temple is used.
        """
        with self.lock:
            tips = {i: h.lm_body[g.THUMB_TIP].copy() for i, h in enumerate(self.hands) if h.state.tracked}
        if not tips:
            return {"ok": False, "error": "no hand is tracked"}
        if len(tips) == 2:
            measured, x = np.mean(list(tips.values()), axis=0), 0.0
        else:
            (i, measured), = tips.items()
            x = -HEAD_HALF_WIDTH if i == 0 else HEAD_HALF_WIDTH
        out = self._shift_camera(measured, np.array([x, 0.0, TEMPLE_Z]))
        out["hands_used"] = len(tips)
        return out

    def calibrate_level(self) -> dict:
        """Find which way is up from the standing body, and set the webcam's tilt.

        The room's axes come from the webcam, which usually looks up or down at an
        angle; uncorrected, the whole 3D model leans by that angle. The line from the
        hips to the shoulders of a person standing upright is vertical.
        """
        with self.lock:
            now = time.monotonic()
            ok = bool((now - self.body_seen[[11, 12, 23, 24]] < BODY_HOLD_S).all()) and self.body_ok
            pts = self.body_pts[[11, 12, 23, 24]].copy()
            r_room = self.r_room.copy()
        if not ok:
            return {"ok": False, "error": "shoulders and hips must be seen by two cameras: stand upright in view"}
        up_room = (pts[0] + pts[1]) / 2.0 - (pts[2] + pts[3]) / 2.0
        if float(np.linalg.norm(up_room)) < 0.25:
            return {"ok": False, "error": "body not measured well enough"}
        v = g.BASE_CAM_TO_BODY @ (r_room.T @ g.normalize(up_room))  # up, in the untilted camera axes
        roll = math.atan2(v[0], v[1])
        pitch = math.atan2(-v[2], math.hypot(v[0], v[1]))
        self.eye_y_samples.clear()  # the room's axes are about to change
        self.config.update({"camera": {"pitch_deg": math.degrees(pitch), "roll_deg": math.degrees(roll),
                                       "yaw_deg": 0.0}})
        out = {"ok": True, "pitch_deg": round(math.degrees(pitch), 1), "roll_deg": round(math.degrees(roll), 1)}

        # Size: the calibration's scale comes from the body tracker's idea of limb
        # lengths, which runs small. A standing person of known height fixes it: the
        # nose is about 12 cm below the top of the head, the ankle joint 8 cm above the floor.
        cfg = self.config.get()
        height = float(cfg["tracking"]["user_height_m"])
        with self.lock:
            now = time.monotonic()
            nose_ok = now - self.body_seen[0] < BODY_HOLD_S
            ankles = [self.body_pts[i].copy() for i in (27, 28) if now - self.body_seen[i] < BODY_HOLD_S]
            nose = self.body_pts[0].copy()
        if height > 0.5 and nose_ok and ankles:
            measured = float(np.mean([np.linalg.norm(nose - a) for a in ankles]))
            factor = (height - 0.20) / measured if measured > 0.3 else 1.0
            if 0.6 < factor < 1.7 and abs(factor - 1.0) > 0.02:
                patch = {n: {"tvec": (np.array(c["tvec"]) * factor).tolist()}
                         for n, c in cfg["extra_cameras"].items() if c["calibrated"]}
                self.config.update({"extra_cameras": patch})
                out["scale_factor"] = round(factor, 3)
            out["measured_nose_to_ankle_m"] = round(measured, 2)
        elif height > 0.5:
            out["scale_note"] = "size not corrected: nose and an ankle must be seen by two cameras"
        return out

    def calibrate_align(self) -> dict:
        with self.lock:
            ok = self.link.connected and self.link.hmd_valid
            q = self.link.hmd_rot.copy()
        if not ok:
            return {"ok": False, "error": "no HMD pose from the runtime"}
        yaw = g.yaw_offset_from_hmd(q)
        self.config.update({"driver": {"yaw_offset": yaw}})
        return {"ok": True, "yaw_offset": yaw}

    def debug_views(self) -> dict:
        """Raw per-camera observations and calibration, for diagnosing the geometry."""
        cfg = self.config.get()
        with self.lock:
            ext = self.extrinsics(cfg)
            return {
                "extrinsics": {n: {"r": r.tolist(), "t": t.tolist()} for n, (r, t) in ext.items()},
                "body": {n: {"t": v[0], "k": v[1].tolist(), "px": v[2].tolist(), "vis": v[3].tolist()}
                         for n, v in self.body_views.items()},
                "hands": {f"{n}/{i}": {"t": v.t, "k": v.k.tolist(), "px": v.image.tolist(), "score": v.score}
                          for (n, i), v in self.views.items()},
            }

    # ---- raw recording ----------------------------------------------------
    def record_raw_start(self, seconds: float = 60.0) -> dict:
        """Record everything the camera processes report (detections, not pictures), so
        the fusion can be re-run and tuned later without anyone standing in front of
        the cameras."""
        with self.lock:
            self.raw_log, self.raw_until = [], time.monotonic() + seconds
        return {"ok": True, "seconds": seconds}

    def _raw_finish(self) -> None:
        import pickle

        from .config import DATA_DIR

        log, self.raw_log = self.raw_log, None
        out = DATA_DIR / "recordings"
        out.mkdir(parents=True, exist_ok=True)
        path = out / time.strftime("raw-%H%M%S.pkl")
        with open(path, "wb") as f:
            pickle.dump({"config": self.config.get(), "events": log}, f)
        self.raw_last = str(path)

    def request_recentre(self) -> dict:
        self.recentre_seq += 1
        return {"ok": True, "recentre_seq": self.recentre_seq}

    def _head_hand_scale(self, idx: int) -> float:
        """How much bigger or smaller this hand is than the hand tracker assumes, learnt
        from the moments when the skeleton's wrist told how far the hand really was."""
        r = self.head_hand_ratios[idx]
        return float(np.clip(np.median(r), 0.7, 1.4)) if len(r) >= 30 else 1.0

    def _learn_head_hand_scale(self, idx: int, wrist_ref: np.ndarray, cam_centre: np.ndarray) -> None:
        """One camera judges a hand's distance by its size, so a hand larger than the
        tracker's model looks nearer than it is. The skeleton's wrist, measured by the
        fixed cameras, gives the true distance from the head camera: the ratio of the
        two, gathered over time, is the size of this user's hand."""
        w = 16 if idx == 1 else 15
        if not self.body_ok or time.monotonic() - self.body_seen[w] > 0.15:
            return
        seen = float(np.linalg.norm(wrist_ref - cam_centre)) / self._head_hand_scale(idx)
        true = float(np.linalg.norm(self.r_room.T @ self.body_pts[w] - cam_centre))
        if 0.15 < seen < 1.2 and 0.15 < true < 1.2 and 0.5 < true / seen < 2.0:
            self.head_hand_ratios[idx].append(true / seen)

    def _floor(self, cfg: dict):
        """Room height of the floor: the user's eye height (their height less 11 cm) below
        their standing eye level, or, with no height given, under the feet two cameras see."""
        # The height comes first when it is known: a camera that cannot see the feet still
        # reports where it guesses them to be, and a floor found from such feet can sit
        # half a metre too high.
        height = float(cfg["tracking"]["user_height_m"])
        if height > 0.5 and len(self.eye_y_samples) >= 60:
            return round(float(np.percentile(self.eye_y_samples, 90)) - (height - 0.11), 3)
        return None if self.floor_y is None else round(self.floor_y, 3)

    def _world_yaw(self, cfg: dict):
        """Angle (radians, about +y) turning the VR runtime's world into room axes, or None."""
        fit = self._headfit_usable()
        if fit is None:
            return None
        c = cfg["camera"]
        r = g.cam_to_body_rotation(c["yaw_deg"], c["pitch_deg"], c["roll_deg"]) @ fit["a"]
        fwd = r @ np.array([0.0, 0.0, -1.0])
        return round(math.atan2(-fwd[0], -fwd[2]), 4)

    def _body_snapshot(self, cfg: dict) -> dict:
        """The body as a stick figure in room axes relative to the webcam.

        Triangulated points are used when they form a plausible body. Otherwise the
        figure is one camera's own 3D pose estimate (right proportions and posture, but
        an estimated shape) placed at the tracked head position.
        """
        now = time.monotonic()
        ok = now - self.body_seen < BODY_HOLD_S
        pts = self.body_pts
        plausible = self.body_ok and bool(ok[[0, 11, 12]].all())
        if plausible:
            # A limb cannot be longer than a limb. A point that makes a bone implausibly
            # long was triangulated from views that did not show the same thing: drop the
            # far end (working outward from the torso, so one bad point does not hide the rest).
            ok = ok.copy()
            for a, b, longest in BODY_BONE_LIMITS:
                if ok[a] and ok[b] and float(np.linalg.norm(pts[a] - pts[b])) > longest:
                    ok[b] = False
                elif not ok[a]:
                    ok[b] = False
            return {"tracked": True, "source": "triangulated", "views": list(self.body_views_used),
                    "points": [np.round(p, 3).tolist() if o else None for p, o in zip(pts, ok)]}

        ext = self.extrinsics(cfg)
        t_now = time.monotonic_ns() / 1e9
        cands = {n: v for n, v in self.body_world.items() if n in ext and t_now - v[0] < BODY_HOLD_S}
        if not cands or now - self.head_seen > HEAD_HOLD_S:
            return {"tracked": False, "source": None, "views": [], "points": [None] * 33}
        name = max(cands, key=lambda n: float(cands[n][2].mean()))
        _, world, vis = cands[name]
        c = cfg["camera"]
        r = g.cam_to_body_rotation(c["yaw_deg"], c["pitch_deg"], c["roll_deg"]) @ ext[name][0]
        room = world @ r.T
        room = room - (room[2] + room[5]) / 2.0 + r @ ext[name][0].T @ self.head_pos  # eyes at the head
        return {"tracked": True, "source": "single", "views": [name],
                "points": [np.round(p, 3).tolist() if v > 0.3 else None for p, v in zip(room, vis)]}

    # ---- UI state --------------------------------------------------------
    def snapshot(self) -> dict:
        cfg = self.config.get()
        cam_cfgs = self.camera_configs(cfg)
        with self.lock:
            now_t = time.monotonic_ns() / 1e9
            cameras = {name: cam.public_stats() | {
                "enabled": bool(cam_cfgs[name]["enabled"]), "type": cam_cfgs[name]["type"],
                "calibrated": name == PRIMARY or bool(cam_cfgs[name]["calibrated"]),
                "sees": [(name, i) in self.views and now_t - self.views[(name, i)].t < 0.3 for i in range(2)],
            } for name, cam in self.cams.items()}
            c = cfg["camera"]
            r_room = g.cam_to_body_rotation(c["yaw_deg"], c["pitch_deg"], c["roll_deg"])
            for name, (r, tv) in self.extrinsics(cfg).items():
                # Where each calibrated camera stands and looks, in room axes relative to the webcam.
                a = self.agreement.get(name)
                cameras[name]["agreement_px"] = None if a is None else round(a, 1)
                cameras[name]["pos_room"] = np.round(r_room @ tv, 3).tolist()
                cameras[name]["dir_room"] = np.round(r_room @ r[:, 2], 3).tolist()
            hands = []
            for h in self.hands:
                s = h.state
                hands.append({
                    "tracked": s.tracked, "buttons": s.buttons,
                    "pos": np.round(s.pos, 4).tolist(), "rot": np.round(s.rot, 4).tolist(),
                    "trigger": round(s.trigger, 3), "grip": round(s.grip, 3),
                    "curl": [round(c, 3) for c in s.curl], "pinch": round(h.pinch, 3),
                    "confidence": round(h.confidence, 3), "reproj_px": round(h.reproj_px, 2),
                    "views": list(h.views_used), "mode": h.mode, "residual_px": round(h.reproj_px, 2),
                    "height_px": round(h.height_px), "lm": np.round(h.lm_body, 4).tolist() if s.tracked else None,
                    "lm_room": np.round(h.lm_room, 4).tolist() if s.tracked else None,
                })
            return {
                "camera": cameras[PRIMARY],  # kept for clients that know one camera
                "cameras": cameras,
                "head": {"tracked": time.monotonic() - self.head_seen < HEAD_HOLD_S,
                         "enabled": bool(cfg["tracking"]["head_tracking"]),
                         "pos_cam": np.round(self.head_pos, 3).tolist(),
                         # Head relative to the webcam in room axes (x right, y up, -z forward).
                         "pos_room": np.round(g.cam_to_body_rotation(
                             cfg["camera"]["yaw_deg"], cfg["camera"]["pitch_deg"], cfg["camera"]["roll_deg"])
                             @ self.head_pos, 3).tolist(),
                         "rot": np.round(self.head_rot, 4).tolist(),  # room axes, w x y z
                         "rot_tracked": time.monotonic() - self.head_rot_seen < HEAD_HOLD_S,
                         "mode": self.head_mode, "views": list(self.head_views_used)},
                "body": self._body_snapshot(cfg),
                # For a VR app: a request to re-centre (the number goes up each time), and,
                # once the headset camera is calibrated, the turn about the vertical that
                # lines the VR runtime's world up with the room the cameras measure in.
                "game": {"recentre_seq": self.recentre_seq, "world_yaw": self._world_yaw(cfg),
                         "torso_yaw": round(self.torso_yaw, 4),
                         "torso_tracked": time.monotonic() - self.torso_seen < BODY_HOLD_S and self.body_ok,
                         "floor_y": self._floor(cfg)},
                "head_hand_scale": [round(self._head_hand_scale(i), 3) for i in range(2)],
                "head_hand_samples": [len(r) for r in self.head_hand_ratios],
                "headcam": {
                    "collecting": self.headfit is None,
                    "seconds_left": 0.0,
                    "samples": len({r[1] for r in self.head_sightings}),
                    "note": self.headfit_note,
                    "result": None if self.headfit is None else {
                        "ok": True, "rotation_error_deg": round(self.headfit["err_deg"], 1),
                        "used": self.headfit["used"], "head_turn_deg": self.headfit.get("turned_deg"),
                        "age_s": round(time.monotonic() - self.headfit["t"], 1),
                        "offset_from_eyes_m": np.round(self.headfit["off"], 3).tolist()},
                },
                "multicam": {
                    "collecting": self.collector is not None,
                    "seconds_left": round(max(self.collect_until - time.monotonic(), 0.0), 1) if self.collector else 0.0,
                    "pairs": self.collector.counts() if self.collector else {},
                    "result": self.calib_result,
                },
                "camera_pos": cfg["camera"]["pos"],
                "hands": hands,
                "driver": {"connected": self.link.connected, "rx_error": self.link.rx_error},
                "hmd": {"valid": self.link.connected and self.link.hmd_valid,
                        "pos": self.link.hmd_pos.tolist(), "rot": self.link.hmd_rot.tolist()},
                "yaw_offset": cfg["driver"]["yaw_offset"],
            }


def g_clamp(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)
