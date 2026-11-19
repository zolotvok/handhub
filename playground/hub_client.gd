# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
class_name HubClient
extends Node
## Talks to the HandHub.
##  - WebSocket in: JSON snapshots (30 Hz) with both hands in the hub's "body frame".
##  - UDP out: 40-byte "HHD1" head-pose packets (30 Hz) so the hub's "Align" works.
## Body frame: metres, x right, y up, -z forward, origin at the eyes (follows head position,
## not head rotation).  World: p_world = head_position + Basis(UP, yaw_offset) * p_body.

const RECONNECT_S := 1.0
const CONNECT_TIMEOUT_S := 3.0
const STALE_S := 0.5            # no snapshot for this long -> hands are considered lost
const STATUS_PERIOD := 1.0 / 30.0
const PALM_IDS := [0, 5, 9, 13, 17]
const BTN_NAMES := ["TRIG", "GRIP", "A", "B"]

## One hand. Body-frame values come straight from the hub; *_world are refreshed every frame.
class HandData extends RefCounted:
	var tracked := false
	var has_lm := false
	var valid := false             # tracked, has landmarks and data is fresh: safe to draw / use
	var buttons := 0
	var trigger := 0.0
	var grip := 0.0
	var pinch := 0.0
	var confidence := 0.0
	var pos_body := Vector3.ZERO
	var rot_body := Quaternion.IDENTITY
	var lm_body := PackedVector3Array()
	var lm_world := PackedVector3Array()
	var pos_world := Vector3.ZERO
	var rot_world := Quaternion.IDENTITY
	var xf := Transform3D.IDENTITY # hand (controller-like) frame in world: pos/rot from the hub
	var palm_world := Vector3.ZERO # mean of landmarks 0,5,9,13,17
	var tip_world := Vector3.ZERO  # index fingertip (landmark 8)

	func _init() -> void:
		lm_body.resize(21)
		lm_world.resize(21)

	func button_text() -> String:
		var names: PackedStringArray = []
		for i in BTN_NAMES.size():
			if buttons & (1 << i):
				names.append(BTN_NAMES[i])
		return " ".join(names) if not names.is_empty() else "-"

var url := "ws://127.0.0.1:8765/ws"
var status_port := 39571
var hands: Array[HandData] = [HandData.new(), HandData.new()]   # [0] left, [1] right
var yaw_offset := 0.0
var recentre_seq := -1            # goes up when the hub asks for a re-centre
var world_yaw_known := false      # the hub knows how the VR world is turned relative to the room
var world_yaw := 0.0
var torso_tracked := false         # the cameras see which way the torso faces
var torso_yaw := 0.0              # room axes, radians; 0 = facing the webcam
var floor_known := false
var floor_y := 0.0                # height of the floor in room axes (webcam = 0)
var ignore_yaw := false            # set when the world is already turned to the room's axes
var head_tracked := false         # the hub's cameras see the head
var head_room := Vector3.ZERO     # head relative to the webcam, room axes, metres
var head_rot_tracked := false
var head_rot := Quaternion.IDENTITY   # head orientation, room axes
var connected := false            # WebSocket open
var snapshots_total := 0
var snapshot_rate := 0            # snapshots received during the last full second
var status_sent_total := 0

var _ws: WebSocketPeer = null
var _connect_started_us := 0
var _closed_at_us := 0
var _last_snapshot_us := 0
var _rate_window_us := 0
var _rate_count := 0
var _udp := PacketPeerUDP.new()
var _status_accum := 0.0
var _status_seq := 0


func _ready() -> void:
	var u := OS.get_environment("HANDHUB_URL")
	if u != "":
		url = u
	var p := OS.get_environment("HANDHUB_STATUS_PORT")
	if p.is_valid_int():
		status_port = p.to_int()
	_udp.set_dest_address("127.0.0.1", status_port)
	_closed_at_us = Time.get_ticks_usec() - int(RECONNECT_S * 1e6)  # connect right away
	print("[hub] url=%s status_udp=127.0.0.1:%d" % [url, status_port])


func _exit_tree() -> void:
	if _ws != null:
		_ws.close()
	_udp.close()


## Call once per frame: services the socket, maps hands into the world and sends the head pose.
func update(delta: float, head_xf: Transform3D) -> void:
	_poll_socket()
	_update_world(head_xf.origin)
	_send_status(delta, head_xf)


func is_fresh() -> bool:
	return connected and (Time.get_ticks_usec() - _last_snapshot_us) < int(STALE_S * 1e6)


# ---------------------------------------------------------------- WebSocket

func _poll_socket() -> void:
	var now := Time.get_ticks_usec()
	if _ws == null:
		if now - _closed_at_us < int(RECONNECT_S * 1e6):
			return
		_ws = WebSocketPeer.new()
		var err := _ws.connect_to_url(url)
		_connect_started_us = now
		if err != OK:
			_ws = null
			_closed_at_us = now
		return

	_ws.poll()
	match _ws.get_ready_state():
		WebSocketPeer.STATE_OPEN:
			if not connected:
				connected = true
				print("[hub] connected to %s" % url)
			var latest := ""
			while _ws.get_available_packet_count() > 0:
				var pkt := _ws.get_packet()
				if _ws.was_string_packet():
					latest = pkt.get_string_from_utf8()   # only the newest snapshot matters
			if latest != "":
				_parse_snapshot(latest, now)
		WebSocketPeer.STATE_CONNECTING:
			if now - _connect_started_us > int(CONNECT_TIMEOUT_S * 1e6):
				_ws.close()
		WebSocketPeer.STATE_CLOSED:
			if connected:
				print("[hub] connection closed")
			connected = false
			_ws = null
			_closed_at_us = now   # reconnect RECONNECT_S from now


func _parse_snapshot(text: String, now_us: int) -> void:
	var json := JSON.new()
	if json.parse(text) != OK:
		return
	var d: Variant = json.data
	if typeof(d) != TYPE_DICTIONARY:
		return
	yaw_offset = float(d.get("yaw_offset", 0.0))
	var gm: Variant = d.get("game", null)
	if gm is Dictionary:
		recentre_seq = int(gm.get("recentre_seq", 0))
		var wy: Variant = gm.get("world_yaw", null)
		world_yaw_known = wy != null
		if world_yaw_known:
			world_yaw = float(wy)
		torso_tracked = bool(gm.get("torso_tracked", false))
		torso_yaw = float(gm.get("torso_yaw", 0.0))
		var fy: Variant = gm.get("floor_y", null)
		floor_known = fy != null
		if floor_known:
			floor_y = float(fy)
	var hd: Variant = d.get("head", null)
	head_tracked = hd is Dictionary and bool(hd.get("tracked", false)) and bool(hd.get("enabled", true))
	if head_tracked:
		var hp: Variant = hd.get("pos_room", null)
		if hp is Array and hp.size() == 3:
			head_room = Vector3(float(hp[0]), float(hp[1]), float(hp[2]))
		else:
			head_tracked = false
	head_rot_tracked = false
	if hd is Dictionary and bool(hd.get("rot_tracked", false)) and bool(hd.get("enabled", true)):
		var hr: Variant = hd.get("rot", null)
		if hr is Array and hr.size() == 4:
			head_rot = Quaternion(float(hr[1]), float(hr[2]), float(hr[3]), float(hr[0])).normalized()
			head_rot_tracked = true
	var hl: Variant = d.get("hands", [])
	if hl is Array:
		for i in mini(2, hl.size()):
			if typeof(hl[i]) == TYPE_DICTIONARY:
				_parse_hand(hands[i], hl[i])
	_last_snapshot_us = now_us
	snapshots_total += 1
	_rate_count += 1
	if now_us - _rate_window_us >= 1000000:
		snapshot_rate = _rate_count
		_rate_count = 0
		_rate_window_us = now_us


func _parse_hand(h: HandData, d: Dictionary) -> void:
	h.tracked = bool(d.get("tracked", false))
	h.buttons = int(d.get("buttons", 0))
	h.trigger = float(d.get("trigger", 0.0))
	h.grip = float(d.get("grip", 0.0))
	h.pinch = float(d.get("pinch", 0.0))
	h.confidence = float(d.get("confidence", 0.0))
	var p: Variant = d.get("pos", null)
	if p is Array and p.size() >= 3:
		h.pos_body = Vector3(p[0], p[1], p[2])
	var r: Variant = d.get("rot", null)   # (w, x, y, z)
	if r is Array and r.size() >= 4:
		var q := Quaternion(float(r[1]), float(r[2]), float(r[3]), float(r[0]))
		h.rot_body = q.normalized() if q.length_squared() > 1e-8 else Quaternion.IDENTITY
	var lm: Variant = d.get("lm", null)
	h.has_lm = false
	if lm is Array and lm.size() >= 21:
		h.has_lm = true
		for i in 21:
			var v: Variant = lm[i]
			if v is Array and v.size() >= 3:
				h.lm_body[i] = Vector3(v[0], v[1], v[2])
			else:
				h.has_lm = false
				break


# ---------------------------------------------------------------- body -> world

func _update_world(head_pos: Vector3) -> void:
	var yaw_b := Basis(Vector3.UP, 0.0 if ignore_yaw else yaw_offset)
	var yaw_q := Quaternion(Vector3.UP, 0.0 if ignore_yaw else yaw_offset)
	var fresh := is_fresh()
	for h in hands:
		h.valid = fresh and h.tracked and h.has_lm
		if not h.valid:
			continue
		h.pos_world = head_pos + yaw_b * h.pos_body
		h.rot_world = (yaw_q * h.rot_body).normalized()
		h.xf = Transform3D(Basis(h.rot_world), h.pos_world)
		var palm := Vector3.ZERO
		for i in 21:
			h.lm_world[i] = head_pos + yaw_b * h.lm_body[i]
		for id: int in PALM_IDS:
			palm += h.lm_world[id]
		h.palm_world = palm / PALM_IDS.size()
		h.tip_world = h.lm_world[8]


# ---------------------------------------------------------------- UDP head pose

func _send_status(delta: float, head_xf: Transform3D) -> void:
	_status_accum += delta
	if _status_accum < STATUS_PERIOD:
		return
	_status_accum = minf(_status_accum - STATUS_PERIOD, STATUS_PERIOD)  # keeps the mean at 30 Hz
	_status_seq += 1
	var q := head_xf.basis.get_rotation_quaternion()
	var b := StreamPeerBuffer.new()   # little-endian by default
	b.put_data("HHD1".to_ascii_buffer())
	b.put_u32(_status_seq)
	b.put_u32(1)                      # hmd_valid
	b.put_float(head_xf.origin.x)
	b.put_float(head_xf.origin.y)
	b.put_float(head_xf.origin.z)
	b.put_float(q.w)
	b.put_float(q.x)
	b.put_float(q.y)
	b.put_float(q.z)
	_udp.put_packet(b.data_array)
	status_sent_total += 1
