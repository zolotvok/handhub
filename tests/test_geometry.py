# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
import math

import numpy as np

from hub import geometry as g


def _rand_rot(rng):
    return g.rot_x(rng.uniform(-3, 3)) @ g.rot_y(rng.uniform(-3, 3)) @ g.rot_z(rng.uniform(-3, 3))


def test_cam_to_body_zero_angles_is_base_rotation():
    r = g.cam_to_body_rotation()
    assert np.allclose(r, np.diag([-1.0, -1.0, 1.0]))
    assert np.isclose(np.linalg.det(r), 1.0)


def test_point_in_front_of_camera_maps_to_body():
    r = g.cam_to_body_rotation()
    body = g.cam_to_body(np.array([0.0, 0.0, 0.5]), r, np.array([0.0, 0.0, -0.6]))
    assert np.allclose(body, [0.0, 0.0, -0.1])


def test_camera_axes_map_to_body_axes():
    r = g.cam_to_body_rotation()
    origin = np.zeros(3)
    assert np.allclose(g.cam_to_body(np.array([1.0, 0, 0]), r, origin), [-1, 0, 0])
    assert np.allclose(g.cam_to_body(np.array([0, 1.0, 0]), r, origin), [0, -1, 0])


def test_quat_identity_convention():
    assert np.allclose(g.quat_from_matrix(np.eye(3)), [1, 0, 0, 0])
    assert np.allclose(g.matrix_from_quat([1, 0, 0, 0]), np.eye(3))


def test_quat_matches_axis_angle_convention():
    a = 0.5
    q = g.quat_from_matrix(g.rot_x(a))
    assert np.allclose(q, [math.cos(a / 2), math.sin(a / 2), 0, 0], atol=1e-9)


def test_quat_matrix_round_trip_random_rotations():
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = _rand_rot(rng)
        q = g.quat_from_matrix(r)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(g.matrix_from_quat(q), r, atol=1e-9)


def test_yaw_offset_from_pure_yaw():
    for a in (-2.0, -0.5, 0.3, 1.5, 2.8):
        yaw = g.yaw_offset_from_hmd(g.quat_from_matrix(g.rot_y(a)))
        assert np.allclose(g.rot_y(yaw) @ [0, 0, -1], g.rot_y(a) @ [0, 0, -1], atol=1e-9)


def test_yaw_offset_ignores_pitch():
    for a in (-2.0, -0.5, 0.3, 1.5):
        r = g.rot_y(a) @ g.rot_x(0.3)
        yaw = g.yaw_offset_from_hmd(g.quat_from_matrix(r))
        fwd = r @ [0, 0, -1]
        flat = np.array([fwd[0], fwd[2]]) / np.hypot(fwd[0], fwd[2])
        yaw_fwd = g.rot_y(yaw) @ [0, 0, -1]
        assert np.allclose([yaw_fwd[0], yaw_fwd[2]], flat * np.hypot(yaw_fwd[0], yaw_fwd[2]), atol=1e-9)
        assert np.isclose(yaw, a, atol=1e-9)


def test_intrinsics_from_hfov():
    k = g.intrinsics_from_hfov(640, 480, 70)
    f = 320 / math.tan(math.radians(35))
    assert np.isclose(k[0, 0], f) and np.isclose(k[1, 1], f)
    assert k[0, 2] == 320 and k[1, 2] == 240
    assert k[2, 2] == 1 and k[0, 1] == 0


def test_solve_hand_pnp_recovers_pose():
    rng = np.random.default_rng(0)
    world = rng.uniform(-0.04, 0.04, (21, 3))  # non-planar, within ~8 cm
    k = g.intrinsics_from_hfov(640, 480, 70)
    r = g.rot_z(0.2) @ g.rot_x(-0.3) @ g.rot_y(0.4)
    t = np.array([0.02, -0.01, 0.5])
    cam = world @ r.T + t
    image = (k @ (cam / cam[:, 2:3]).T).T[:, :2]

    pts, err = g.solve_hand_pnp(world, image, k)
    assert np.allclose(pts, cam, atol=2e-3)
    assert err < 0.5


def test_controller_pose_identity_for_canonical_hand():
    pts = np.zeros((21, 3))
    pts[0] = [0, 0, 0]  # wrist
    pts[5] = [0, 0.03, -0.09]  # index MCP, thumb side up
    pts[9] = [0, 0, -0.09]  # middle MCP, straight forward (-z)
    pts[13] = [0, -0.01, -0.09]
    pts[17] = [0, -0.03, -0.09]  # pinky MCP
    pos, r = g.controller_pose(pts)

    assert np.allclose(r, np.eye(3), atol=1e-9)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(pos, pts[[0, 5, 9, 13, 17]].mean(axis=0))


def test_level_angles_make_a_tilted_cameras_up_vertical():
    """The pitch/roll that calibrate_level derives map the measured up direction to +y."""
    import math

    # A camera looking up by 35 degrees and rolled 8 degrees sees "up" as this, in its own axes.
    true_r = g.cam_to_body_rotation(0.0, -35.0, 8.0)
    up_cam = true_r.T @ np.array([0.0, 1.0, 0.0])
    v = g.BASE_CAM_TO_BODY @ up_cam
    roll = math.atan2(v[0], v[1])
    pitch = math.atan2(-v[2], math.hypot(v[0], v[1]))
    r = g.cam_to_body_rotation(0.0, math.degrees(pitch), math.degrees(roll))
    assert np.allclose(r @ up_cam, [0.0, 1.0, 0.0], atol=1e-9)


def test_fit_rigid_recovers_pose_and_ignores_bending():
    rng = np.random.default_rng(4)
    model = rng.normal(0, 0.04, (21, 3))
    r_true, t_true = g.rot_y(0.7) @ g.rot_x(-0.4), np.array([0.2, -0.1, 1.3])
    target = model @ r_true.T + t_true
    r, t, rms = g.fit_rigid(model, target)
    assert np.allclose(r, r_true, atol=1e-9) and np.allclose(t, t_true, atol=1e-9) and rms < 1e-9
    assert abs(np.linalg.det(r) - 1.0) < 1e-9

    bent = target + rng.normal(0, 0.01, target.shape)      # landmark noise bends the hand
    r2, t2, rms2 = g.fit_rigid(model, bent)
    fitted = model @ r2.T + t2
    # the fitted hand keeps the model's exact shape
    d_model = np.linalg.norm(model[:, None] - model[None], axis=-1)
    d_fit = np.linalg.norm(fitted[:, None] - fitted[None], axis=-1)
    assert np.allclose(d_model, d_fit, atol=1e-9)
    assert 0.005 < rms2 < 0.03


def test_hand_side_tells_a_hand_from_its_mirror_image():
    # A right hand, palm toward +z, fingers up (+y) and curled toward the palm, thumb on the palm side.
    pts = np.zeros((21, 3))
    for f, x in enumerate((0.03, 0.01, -0.01, -0.03)):          # index .. pinky knuckles
        mcp = 5 + 4 * f
        pts[mcp] = [x, 0.09, 0.0]
        for j in range(1, 4):
            pts[mcp + j] = [x, 0.09 + 0.025 * j, 0.012 * j]      # curling toward +z
    pts[1:5] = [[0.03, 0.02, 0.01], [0.05, 0.04, 0.02], [0.06, 0.06, 0.03], [0.065, 0.08, 0.035]]
    assert g.hand_side(pts) is True
    mirrored = pts * np.array([-1.0, 1.0, 1.0])
    assert g.hand_side(mirrored) is False
    # any rigid motion keeps the answer
    r = g.rot_y(1.1) @ g.rot_x(-0.6)
    assert g.hand_side(pts @ r.T + [0.3, -0.2, 1.0]) is True
    assert g.hand_side(mirrored @ r.T + [0.3, -0.2, 1.0]) is False
    # a flat hand cannot be told
    flat = pts.copy()
    flat[:, 2] = 0.0
    assert g.hand_side(flat) is None
