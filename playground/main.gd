# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
extends Node3D
## HandHub playground. Picks VR (OpenXR) or Direct (desktop) mode, builds a ~4x4 m room in code and
## wires the hub client, hand skeletons, grabbable objects, push buttons, hand mirror and status panel.
##
## Env: HANDHUB_DIRECT=1 (skip OpenXR), HANDHUB_URL, HANDHUB_STATUS_PORT, HANDHUB_DEBUG=1.

const ROOM_HALF := 2.0
const WALL_H := 2.6
const TABLE_TOP_Y := 0.85
const TABLE_Z := -0.6
const OBJECT_Z := -0.45
const BUTTON_Z := -0.78
const EYE_H := 1.6
const MOVE_SPEED := 1.6
const MOUSE_SENS := 0.003
const WEBCAM_AHEAD := 1.3   # metres from the room's centre to the webcam, toward the table

var _http: HTTPRequest
var _base_pos := Vector3(0, EYE_H, 0)   # desktop camera without the head offset
var _head_ref := Vector3.ZERO
var _head_ref_set := false
var _origin: XROrigin3D
var _shots_dir := ""
var _shot_t := 0.0
var _seen_recentre_seq := -1
var _yaw_settled := false
var _yaw_far_t := 0.0  # how long the torso and the view have disagreed widely
var _head_offset := Vector3.ZERO
var _head_quat := Quaternion.IDENTITY
var _head_quat_ref := Quaternion.IDENTITY
var _head_follow := true
var _recentre_left := 0.0
var _head_msg := ""
var _calib_label: Label3D
var _calib_left := 0.0
var _calib_msg := ""
const FOV_MIN := 40.0
const FOV_MAX := 140.0

var vr_mode := false
var debug := false

var camera: Camera3D
var client: HubClient
var grab: GrabManager
var mirror: HandMirror
var panel: StatusPanel
var sun: DirectionalLight3D
var visuals: Array[HandVisual] = []
var buttons: Array[PushButton] = []

var _mats: Dictionary = {}
var _yaw := 0.0
var _pitch := 0.0
var _status_t := 0.0
var _debug_t := 0.0
var _light_on := true


func _ready() -> void:
	process_priority = 100   # run after XRCamera3D has taken this frame's head pose
	debug = OS.get_environment("HANDHUB_DEBUG") == "1"
	_calib_label = Label3D.new()   # big prompt in front of the player during calibration
	_calib_label.position = Vector3(0, 1.95, -1.3)
	_calib_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_calib_label.pixel_size = 0.003
	_calib_label.font_size = 48
	_calib_label.outline_size = 12
	_calib_label.modulate = Color(1.0, 0.9, 0.3)
	add_child(_calib_label)
	_http = HTTPRequest.new()
	_http.timeout = 3.0
	add_child(_http)
	_http.request_completed.connect(_on_calibration_done)
	_shots_dir = OS.get_environment("HANDHUB_SHOTS")
	if DisplayServer.get_name() == "headless":
		_shots_dir = ""   # nothing is drawn without a display
	if _shots_dir != "":
		DirAccess.make_dir_recursive_absolute(_shots_dir)
	if OS.get_environment("HANDHUB_SCREENSHOT") != "":
		_save_screenshot(OS.get_environment("HANDHUB_SCREENSHOT"))
	if DisplayServer.get_name() == "headless":
		Engine.max_fps = 60  # headless would otherwise spin; keeps frame counts ~ wall time for tests
	vr_mode = _try_start_vr()
	print("[main] mode: %s" % ("VR (OpenXR)" if vr_mode else "Direct (desktop)"))

	_build_environment()
	_build_room()
	_build_table()
	_build_player()

	client = HubClient.new()
	add_child(client)
	for i in 2:
		var hv := HandVisual.new()
		hv.setup(i == 1)
		add_child(hv)
		visuals.append(hv)

	grab = GrabManager.new()
	grab.client = client
	grab.debug = debug
	add_child(grab)
	_build_objects()
	_build_buttons()

	mirror = HandMirror.new()
	mirror.position = Vector3(0, 1.5, -1.5)
	add_child(mirror)

	panel = StatusPanel.new()
	var panel_pos := Vector3(-1.3, 1.5, -0.9)
	panel.position = panel_pos
	var to_eye := Vector3(0, EYE_H, 0) - panel_pos
	panel.rotation.y = atan2(to_eye.x, to_eye.z)   # +Z (the text side) faces the start position
	add_child(panel)


func _exit_tree() -> void:
	if debug and client != null and grab != null:
		print("[summary] snapshots=%d status_udp_sent=%d grabs=%d releases=%d" % [
			client.snapshots_total, client.status_sent_total, grab.grab_count, grab.release_count])


# ---------------------------------------------------------------- mode selection

func _try_start_vr() -> bool:
	if OS.get_environment("HANDHUB_DIRECT") == "1":
		print("[main] HANDHUB_DIRECT=1: not trying OpenXR")
		return false
	if DisplayServer.get_name() == "headless":
		print("[main] headless display: not trying OpenXR")
		return false
	var xr := XRServer.find_interface("OpenXR")
	if xr == null:
		print("[main] OpenXR interface not available")
		return false
	if not xr.is_initialized() and not xr.initialize():
		print("[main] OpenXR failed to initialise")
		return false
	get_viewport().use_xr = true
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	return true


func _build_player() -> void:
	if vr_mode:
		_origin = XROrigin3D.new()   # room centre, facing -z; the HMD pose moves the camera under it
		add_child(_origin)
		var cam := XRCamera3D.new()
		_origin.add_child(cam)
		_origin.current = true
		camera = cam
	else:
		camera = Camera3D.new()
		camera.position = Vector3(0, EYE_H, 0)
		# Vertical field of view. Wide by default: without a headset the desktop
		# window is the only view. HANDHUB_FOV overrides; mouse wheel or +/- adjust.
		var fov_env := OS.get_environment("HANDHUB_FOV")
		camera.fov = clampf(fov_env.to_float(), FOV_MIN, FOV_MAX) if fov_env.is_valid_float() else 100.0
		camera.near = 0.02
		add_child(camera)
	camera.current = true


# ---------------------------------------------------------------- world

func _mat(c: Color) -> StandardMaterial3D:
	if not _mats.has(c):
		var m := StandardMaterial3D.new()
		m.albedo_color = c
		m.roughness = 0.9
		_mats[c] = m
	return _mats[c]


## A box centred at pos. solid=true makes it a StaticBody3D with a matching collider.
func _box(parent: Node, size: Vector3, pos: Vector3, c: Color, solid := false) -> Node3D:
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = size
	mi.mesh = bm
	mi.material_override = _mat(c)
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	if not solid:
		mi.position = pos
		parent.add_child(mi)
		return mi
	var body := StaticBody3D.new()
	body.position = pos
	var cs := CollisionShape3D.new()
	var bs := BoxShape3D.new()
	bs.size = size
	cs.shape = bs
	body.add_child(cs)
	body.add_child(mi)
	parent.add_child(body)
	return body


func _build_environment() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.45, 0.62, 0.78)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.8, 0.82, 0.9)
	env.ambient_light_energy = 0.7
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	sun = DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-55, -30, 0)
	sun.light_energy = 1.0
	sun.shadow_enabled = false
	add_child(sun)


func _build_room() -> void:
	var span := ROOM_HALF * 2.0
	_box(self, Vector3(span + 0.2, 0.1, span + 0.2), Vector3(0, -0.05, 0), Color(0.30, 0.33, 0.36), true)
	var line := Color(0.62, 0.68, 0.74)
	for k in range(-2, 3):   # 1 m grid, 2 cm wide lines just above the floor
		_box(self, Vector3(0.02, 0.004, span), Vector3(k, 0.002, 0), line)
		_box(self, Vector3(span, 0.004, 0.02), Vector3(0, 0.002, k), line)
	var wall_y := WALL_H * 0.5
	var t := 0.1
	_box(self, Vector3(span + 0.2, WALL_H, t), Vector3(0, wall_y, -ROOM_HALF - t * 0.5), Color(0.28, 0.36, 0.52), true)  # front of the player
	_box(self, Vector3(span + 0.2, WALL_H, t), Vector3(0, wall_y, ROOM_HALF + t * 0.5), Color(0.42, 0.42, 0.44), true)
	_box(self, Vector3(t, WALL_H, span), Vector3(-ROOM_HALF - t * 0.5, wall_y, 0), Color(0.40, 0.33, 0.30), true)
	_box(self, Vector3(t, WALL_H, span), Vector3(ROOM_HALF + t * 0.5, wall_y, 0), Color(0.30, 0.40, 0.33), true)
	# start-position marker
	var disc := MeshInstance3D.new()
	var dm := CylinderMesh.new()
	dm.top_radius = 0.25
	dm.bottom_radius = 0.25
	dm.height = 0.005
	dm.radial_segments = 24
	disc.mesh = dm
	disc.material_override = _mat(Color(0.9, 0.7, 0.2))
	disc.position = Vector3(0, 0.004, 0)
	add_child(disc)


func _build_table() -> void:
	var wood := Color(0.55, 0.38, 0.22)
	_box(self, Vector3(1.4, 0.05, 0.7), Vector3(0, TABLE_TOP_Y - 0.025, TABLE_Z), wood, true)
	for sx in [-0.65, 0.65]:
		for sz in [-0.3, 0.3]:
			_box(self, Vector3(0.06, TABLE_TOP_Y - 0.05, 0.06), Vector3(sx, (TABLE_TOP_Y - 0.05) * 0.5, TABLE_Z + sz), wood.darkened(0.25))


func _build_objects() -> void:
	var specs := [
		["Cube_red", "cube", Color(0.90, 0.20, 0.20), -0.5],
		["Ball_yellow", "ball", Color(0.95, 0.85, 0.15), -0.3],
		["Cube_green", "cube", Color(0.20, 0.75, 0.30), -0.1],
		["Ball_cyan", "ball", Color(0.15, 0.80, 0.85), 0.1],
		["Cube_blue", "cube", Color(0.25, 0.35, 0.90), 0.3],
		["Ball_magenta", "ball", Color(0.85, 0.25, 0.75), 0.5],
	]
	for s in specs:
		var o := Grabbable.new()
		o.name = s[0]
		o.setup(s[1], s[2], Transform3D(Basis.IDENTITY, Vector3(s[3], TABLE_TOP_Y + 0.041, OBJECT_Z)))
		add_child(o)
		grab.add_object(o)


func _build_buttons() -> void:
	var specs := [
		["RESET", Color(0.70, 0.15, 0.15), Color(1.0, 0.45, 0.35), -0.3],
		["LIGHT", Color(0.65, 0.55, 0.10), Color(1.0, 0.95, 0.40), 0.0],
		["COUNT", Color(0.15, 0.35, 0.70), Color(0.45, 0.75, 1.0), 0.3],
	]
	for s in specs:
		var b := PushButton.new()
		b.position = Vector3(s[3], TABLE_TOP_Y, BUTTON_Z)
		add_child(b)
		b.setup(s[0], s[1], s[2])
		buttons.append(b)
		b.pressed.connect(_on_button.bind(b))


func _on_button(b: PushButton) -> void:
	if debug:
		print("[button] %s pressed (count=%d)" % [b.title, b.count])
	match b.title:
		"RESET":
			grab.reset_all()
		"LIGHT":
			_light_on = not _light_on
			sun.light_energy = 1.0 if _light_on else 0.2


# ---------------------------------------------------------------- per frame

# The user's body as the cameras measure it, drawn from the shoulders down (the head would
# only block the view): it shows at a glance whether the game and the room line up.
const BODY_BONES := [[11, 12, 2], [23, 24, 2], [11, 23, 0], [12, 24, 1],
		[11, 13, 0], [13, 15, 0], [12, 14, 1], [14, 16, 1],
		[23, 25, 0], [25, 27, 0], [27, 31, 0], [24, 26, 1], [26, 28, 1], [28, 32, 1]]
var _bones: Array[MeshInstance3D] = []


func _build_body() -> void:
	var colours := [Color(0.35, 0.65, 1.0), Color(1.0, 0.65, 0.25), Color(0.9, 0.9, 0.95)]   # left, right, middle
	for b in BODY_BONES:
		var m := MeshInstance3D.new()
		var box := BoxMesh.new()
		box.size = Vector3(0.03, 0.03, 1.0)
		m.mesh = box
		var mat := StandardMaterial3D.new()
		mat.albedo_color = colours[b[2]]
		mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		m.material_override = mat
		m.visible = false
		add_child(m)
		_bones.append(m)


func _room_to_world(p: Vector3) -> Vector3:
	var floor_h := client.floor_y if client.floor_known else client.head_room.y - EYE_H
	return Vector3(p.x, p.y - floor_h, p.z - WEBCAM_AHEAD)


func _update_body() -> void:
	if _bones.is_empty():
		_build_body()
	for i in BODY_BONES.size():
		var b: Array = BODY_BONES[i]
		var m := _bones[i]
		var ok: bool = client.body_tracked and client.body_points.size() > 32 \
				and client.body_points[b[0]] != null and client.body_points[b[1]] != null
		m.visible = ok
		if not ok:
			continue
		var p0 := _room_to_world(client.body_points[b[0]])
		var p1 := _room_to_world(client.body_points[b[1]])
		var d := p1 - p0
		var length := d.length()
		if length < 0.02:
			m.visible = false
			continue
		var z := d / length
		var up := Vector3.UP if absf(z.y) < 0.95 else Vector3.RIGHT
		var x := up.cross(z).normalized()
		m.global_transform = Transform3D(Basis(x, z.cross(x), z * length), (p0 + p1) * 0.5)


func _process(delta: float) -> void:
	if not vr_mode:
		_move(delta)
	_follow_head(delta)
	client.ignore_yaw = vr_mode and _head_ref_set   # the play space is turned to the room instead
	client.update(delta, camera.global_transform)

	for i in 2:
		var h := client.hands[i]
		visuals[i].update_pose(h.lm_world, h.confidence, h.valid)
	mirror.update_hands(client)
	_update_body()
	grab.step(delta)

	var tips: Array[Vector3] = []
	for h in client.hands:
		if h.valid:
			tips.append(h.tip_world)
	for b in buttons:
		b.test_points(tips)

	if _calib_left > 0.0:
		_calib_left -= delta
		if _calib_left <= 0.0:
			_finish_head_calibration()

	if _recentre_left > 0.0:
		_calib_label.text = _head_line()
	elif _calib_left > 0.0 or _calib_msg != "":
		_calib_label.text = _calibration_line()
	else:
		_calib_label.text = _head_msg

	if _shots_dir != "":
		_shot_t += delta
		if _shot_t >= 0.33:
			_shot_t = 0.0
			_save_shot()

	_status_t += delta
	if _status_t >= 0.1:
		_status_t = 0.0
		panel.set_text(_status_text())
	if debug:
		_debug_t += delta
		if _debug_t >= 1.0:
			_debug_t = 0.0
			_debug_print()


func _move(delta: float) -> void:
	var d := Vector2.ZERO
	if Input.is_physical_key_pressed(KEY_W): d.y -= 1.0
	if Input.is_physical_key_pressed(KEY_S): d.y += 1.0
	if Input.is_physical_key_pressed(KEY_A): d.x -= 1.0
	if Input.is_physical_key_pressed(KEY_D): d.x += 1.0
	if d == Vector2.ZERO:
		return
	d = d.normalized()
	var p := _base_pos + Basis(Vector3.UP, _yaw) * Vector3(d.x, 0, d.y) * (MOVE_SPEED * delta)
	var lim := ROOM_HALF - 0.2
	_base_pos = Vector3(clampf(p.x, -lim, lim), p.y, clampf(p.z, -lim, lim))


# The view moves like the tracked head, so leaning and crouching work: you can bend
# down to the table. Nothing follows until a reference is taken: H starts a countdown
# (time to step away from the keyboard and stand in place, looking at the laptop); the
# head position at its end maps to the standing spot. T switches following off and on.
#
# VR: a phone headset only knows which way it is turned, not where it is. The cameras
# know where the head is, so the play space is moved by the head's displacement. At the
# reference moment the play space is also turned so that "looking at the laptop" faces
# the table, which lines the headset's world up with the room the cameras measure in.
# Desktop: the camera itself is moved and turned.
func _follow_head(delta: float) -> void:
	# A re-centre can be asked for from the hub's web page (no need to reach the
	# keyboard), and in VR one starts by itself as soon as the cameras see the head.
	if client.recentre_seq >= 0:
		if _seen_recentre_seq < 0:
			_seen_recentre_seq = client.recentre_seq
		elif client.recentre_seq != _seen_recentre_seq:
			_seen_recentre_seq = client.recentre_seq
			_recentre_left = 8.0 if vr_mode else 3.0
			_head_msg = ""
	if vr_mode:
		_recentre_left = 0.0   # VR lines itself up continuously, see _follow_room()
	if _recentre_left > 0.0:
		_recentre_left -= delta
		if _recentre_left <= 0.0:
			if client.head_tracked:
				_head_ref = client.head_room
				_head_quat_ref = client.head_rot if client.head_rot_tracked else Quaternion.IDENTITY
				_head_ref_set = true
				_head_offset = Vector3.ZERO
				_head_msg = "head reference set"
				if vr_mode and not client.world_yaw_known:
					var fwd := -camera.global_transform.basis.z
					fwd.y = 0.0
					if fwd.length() > 0.2:
						_origin.rotation.y -= atan2(-fwd.x, -fwd.z)
			else:
				_head_msg = "head not seen: no reference"
	var offset := Vector3.ZERO
	var quat := Quaternion.IDENTITY
	if _head_ref_set and _head_follow:
		offset = _head_offset
		quat = _head_quat
		if client.head_tracked:
			offset = client.head_room - _head_ref
		if client.head_rot_tracked:
			quat = client.head_rot * _head_quat_ref.inverse()
	_head_offset = _head_offset.lerp(offset, clampf(delta * 8.0, 0.0, 1.0))
	if vr_mode:
		_follow_room(delta)
		return
	_head_quat = _head_quat.slerp(quat, clampf(delta * 8.0, 0.0, 1.0))
	camera.position = _base_pos + _head_offset
	camera.quaternion = Quaternion(Vector3.UP, _yaw) * _head_quat * Quaternion(Vector3.RIGHT, _pitch)


# VR, continuously: keep the game's world lined up with the real room.
#
# Where: the cameras give the head's place in the room all the time, so the view is put
# there outright (no reference pose): the webcam sits WEBCAM_AHEAD metres ahead of the
# room's centre, and eye height is measured from the floor the cameras see under the feet.
# Which way: a phone headset's heading starts arbitrary and drifts. The cameras see which
# way the torso faces, and on average you look where your torso points, so the world is
# turned slowly to make the two agree: quickly when they are far apart (start-up, or
# after the phone lost its bearings), very slowly otherwise so that looking over a
# shoulder for a while does not drag the world round.
func _follow_room(delta: float) -> void:
	if not _head_follow:
		return
	if client.torso_tracked and not client.world_yaw_known:
		var fwd := -camera.global_transform.basis.z
		fwd.y = 0.0
		if fwd.length() > 0.2:
			var err := wrapf(client.torso_yaw - atan2(-fwd.x, -fwd.z), -PI, PI)
			# Fast only at start-up, or when the two have disagreed widely for a while: a
			# single bad torso reading, or a glance aside, must not swing the room.
			_yaw_far_t = _yaw_far_t + delta if absf(err) > deg_to_rad(50.0) else 0.0
			var rate := 1.0 / 120.0
			if not _yaw_settled or _yaw_far_t > 3.0:
				rate = 1.5
			if absf(err) < deg_to_rad(8.0):
				_yaw_settled = true
				_yaw_far_t = 0.0
			_origin.rotation.y += err * clampf(delta * rate, 0.0, 1.0)
	elif client.world_yaw_known:
		_origin.rotation.y = client.world_yaw
	if client.head_tracked:
		var eye_h := (client.head_room.y - client.floor_y) if client.floor_known else EYE_H
		var target := Vector3(client.head_room.x, clampf(eye_h, 0.3, 2.3), client.head_room.z - WEBCAM_AHEAD)
		# the play space goes where it puts the headset's own (small) position on the target
		var want := target - _origin.basis * camera.position
		_origin.position = _origin.position.lerp(want, clampf(delta * 25.0, 0.0, 1.0))
		_head_ref_set = true


func _head_line() -> String:
	if _recentre_left > 0.0:
		return "STAND IN PLACE, LOOK AT THE LAPTOP: %d" % ceili(_recentre_left)
	if not _head_ref_set:
		return "H: follow my head (3 s countdown)"
	return "head follow %s  (H: re-centre, T: on/off)" % ("ON" if _head_follow else "off")


func _unhandled_input(event: InputEvent) -> void:
	var key := event as InputEventKey
	if key != null and key.pressed and key.keycode == KEY_ESCAPE:
		get_tree().quit()
		return
	if key != null and key.pressed and not key.echo and key.keycode == KEY_H:
		# second press cancels; VR gets longer, the headset has to go on first
		_recentre_left = 0.0 if _recentre_left > 0.0 else (8.0 if vr_mode else 3.0)
		_head_msg = ""
		return
	if key != null and key.pressed and not key.echo and key.keycode == KEY_T:
		_head_follow = not _head_follow
		return
	if key != null and key.pressed and not key.echo and key.keycode == KEY_C:
		_start_head_calibration()
		return
	if vr_mode:
		return
	if key != null and key.pressed and key.keycode in [KEY_PLUS, KEY_EQUAL, KEY_KP_ADD]:
		_set_fov(camera.fov - 5.0)
		return
	if key != null and key.pressed and key.keycode in [KEY_MINUS, KEY_KP_SUBTRACT]:
		_set_fov(camera.fov + 5.0)
		return
	var mb := event as InputEventMouseButton
	if mb != null and mb.pressed and mb.button_index == MOUSE_BUTTON_WHEEL_UP:
		_set_fov(camera.fov - 5.0)
		return
	if mb != null and mb.pressed and mb.button_index == MOUSE_BUTTON_WHEEL_DOWN:
		_set_fov(camera.fov + 5.0)
		return
	if mb != null and mb.button_index == MOUSE_BUTTON_RIGHT:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if mb.pressed else Input.MOUSE_MODE_VISIBLE
		return
	var mm := event as InputEventMouseMotion
	if mm != null and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		_yaw -= mm.relative.x * MOUSE_SENS
		_pitch = clampf(_pitch - mm.relative.y * MOUSE_SENS, -1.5, 1.5)


# Head-anchor calibration (C key): after a countdown, ask the hub to place the
# thumb tips at the temples. Fixes hands that appear offset, e.g. behind the player.
func _start_head_calibration() -> void:
	if _calib_left > 0.0:
		_calib_left = 0.0
		_calib_msg = "calibration cancelled"
		return
	_calib_left = 3.0
	_calib_msg = ""


func _finish_head_calibration() -> void:
	_calib_msg = "calibrating..."
	var base := client.url.replace("ws://", "http://").replace("wss://", "https://").trim_suffix("/ws")
	var err := _http.request(base + "/api/calibrate/head", [], HTTPClient.METHOD_POST, "")
	if err != OK:
		_calib_msg = "calibration: cannot reach hub"


func _on_calibration_done(result: int, code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	var data: Variant = JSON.parse_string(body.get_string_from_utf8())
	if result != HTTPRequest.RESULT_SUCCESS or code != 200 or not (data is Dictionary):
		_calib_msg = "calibration failed (hub error %d)" % code
	elif data.get("ok", false):
		_calib_msg = "calibrated (%d hand%s)" % [int(data.get("hands_used", 0)), "" if int(data.get("hands_used", 0)) == 1 else "s"]
	else:
		_calib_msg = "calibration failed: %s" % str(data.get("error", "?"))
	print("[main] ", _calib_msg)


func _calibration_line() -> String:
	if _calib_left > 0.0:
		return "THUMBS ON TEMPLES, PALMS FORWARD: %d" % ceili(_calib_left)
	return _calib_msg if _calib_msg != "" else "C: calibrate hands to head"


# Diagnostics: save what the player sees a few times a second (HANDHUB_SHOTS=<folder>),
# named by the system clock so the pictures can be lined up with the camera recordings.
func _save_shot() -> void:
	var img := get_viewport().get_texture().get_image()
	if img == null:
		return
	if img.get_width() > 960:
		img.resize(960, int(960.0 * img.get_height() / img.get_width()), Image.INTERPOLATE_BILINEAR)
	img.save_jpg("%s/%.2f.jpg" % [_shots_dir, Time.get_unix_time_from_system()], 0.8)


func _set_fov(v: float) -> void:
	camera.fov = clampf(v, FOV_MIN, FOV_MAX)
	print("[main] fov: %d" % int(camera.fov))


func _save_screenshot(path: String) -> void:
	for i in 90:
		await get_tree().process_frame
	get_viewport().get_texture().get_image().save_png(path)
	print("[main] screenshot saved: ", path)


# ---------------------------------------------------------------- status / debug

static func _bar(v: float) -> String:
	var n := clampi(roundi(v * 10.0), 0, 10)
	return "[" + "#".repeat(n) + "-".repeat(10 - n) + "]"


func _status_text() -> String:
	var t := PackedStringArray()
	t.append("HANDHUB PLAYGROUND")
	t.append("mode %s   fps %d" % ["VR" if vr_mode else "Direct", Engine.get_frames_per_second()])
	if client.connected:
		t.append("hub  connected  %d snap/s" % client.snapshot_rate)
	else:
		t.append("hub  NOT connected")
	t.append("yaw  %.1f deg  udp>%d" % [rad_to_deg(client.yaw_offset), client.status_port])
	t.append(_calibration_line())
	t.append("head %s" % ("seen by cameras" if client.head_tracked else "NOT seen"))
	t.append(_head_line())
	for i in 2:
		var h := client.hands[i]
		t.append("")
		t.append("%s %s  conf %s" % ["LEFT " if i == 0 else "RIGHT", "tracked" if h.valid else "LOST   ",
				"%.2f" % h.confidence if h.valid else "--"])
		t.append(" trig %s" % (("%s %.2f" % [_bar(h.trigger), h.trigger]) if h.valid else "--"))
		t.append(" grip %s" % (("%s %.2f" % [_bar(h.grip), h.grip]) if h.valid else "--"))
		t.append(" btn  %s  held %s" % [h.button_text() if h.valid else "--", grab.held_name(i)])
	return "\n".join(t)


func _debug_print() -> void:
	var parts := PackedStringArray()
	for i in 2:
		var h := client.hands[i]
		if h.valid:
			parts.append("%s tracked conf=%.2f grip=%.2f btn=%d palm=%s tip=%s" % [
				"LR"[i], h.confidence, h.grip, h.buttons, GrabManager._fmt(h.palm_world), GrabManager._fmt(h.tip_world)])
		else:
			parts.append("%s not-tracked" % "LR"[i])
	print("[dbg] t=%.1f fps=%d hub=%s snaps=%d udp_sent=%d head=%s | %s" % [
		Time.get_ticks_msec() / 1000.0, Engine.get_frames_per_second(), client.connected, client.snapshots_total,
		client.status_sent_total, GrabManager._fmt(camera.global_position), " | ".join(parts)])
	var objs := PackedStringArray()
	for o in grab.objects:
		objs.append("%s%s=%s" % [o.name, "*" if o.holder >= 0 else "", GrabManager._fmt(o.global_position)])
	print("[dbg]   objects: " + " ".join(objs))
