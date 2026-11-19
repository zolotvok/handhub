# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
class_name GrabManager
extends Node
## Grab rule (no controllers): when a hand's grip-click bit (2) becomes set and the palm is
## within GRAB_RADIUS of an object's centre, the nearest such object is frozen and follows the
## hand with the relative transform it had at grab time. Releasing (bit cleared, or the hand
## lost for > LOST_RELEASE_S) unfreezes it with the palm's recent velocity so it can be thrown.

const GRAB_RADIUS := 0.12
const LOST_RELEASE_S := 0.3
const VELOCITY_WINDOW_S := 0.1
const MAX_THROW_SPEED := 8.0
const BTN_GRIP := 2
const RESPAWN_BELOW_Y := -2.0
const RESPAWN_BEYOND := 6.0

class HandState extends RefCounted:
	var held: Grabbable = null
	var rel := Transform3D.IDENTITY      # object transform in the hand frame at grab time
	var last_xf := Transform3D.IDENTITY  # last valid hand frame
	var prev_grip := false
	var lost_time := 0.0
	var history: Array = []              # [time_s, palm position] samples

var client: HubClient
var objects: Array[Grabbable] = []
var debug := false
var grab_count := 0
var release_count := 0
var _hs: Array[HandState] = [HandState.new(), HandState.new()]


func add_object(o: Grabbable) -> void:
	objects.append(o)


## Release anything that is held and put every object back to its start position.
func reset_all() -> void:
	for i in 2:
		if _hs[i].held != null:
			_release(i, false)
	for o in objects:
		o.reset_to_start()
	if debug:
		print("[grab] reset_all")


func step(delta: float) -> void:
	var now := Time.get_ticks_usec() / 1e6
	var near_obj: Array[Grabbable] = [null, null]
	for i in 2:
		var h := client.hands[i]
		var s := _hs[i]
		if h.valid:
			s.lost_time = 0.0
			s.last_xf = h.xf
			s.history.append([now, h.palm_world])
			while s.history.size() > 1 and now - float(s.history[0][0]) > 0.25:
				s.history.pop_front()
			var grip := (h.buttons & BTN_GRIP) != 0
			if s.held == null:
				near_obj[i] = _nearest(h.palm_world)
				if grip and not s.prev_grip and near_obj[i] != null:
					_grab(i, near_obj[i], h.palm_world)
			elif not grip:
				_release(i, true)
			s.prev_grip = grip
		else:
			s.history.clear()
			s.prev_grip = false
			s.lost_time += delta
			if s.held != null and s.lost_time > LOST_RELEASE_S:
				_release(i, true)
		if s.held != null:
			s.held.global_transform = s.last_xf * s.rel

	for o in objects:
		if o.holder >= 0:
			o.set_glow(Grabbable.Glow.HELD)
		elif o == near_obj[0] or o == near_obj[1]:
			o.set_glow(Grabbable.Glow.NEAR)
		else:
			o.set_glow(Grabbable.Glow.NONE)
		if o.holder < 0 and (o.global_position.y < RESPAWN_BELOW_Y or o.global_position.length() > RESPAWN_BEYOND):
			o.reset_to_start()
			if debug:
				print("[grab] respawn %s" % o.name)


func held_name(i: int) -> String:
	return String(_hs[i].held.name) if _hs[i].held != null else "-"


func _nearest(palm: Vector3) -> Grabbable:
	var best: Grabbable = null
	var best_d := GRAB_RADIUS
	for o in objects:
		if o.holder >= 0:
			continue
		var d := palm.distance_to(o.global_position)
		if d <= best_d:
			best_d = d
			best = o
	return best


func _grab(i: int, o: Grabbable, palm: Vector3) -> void:
	var s := _hs[i]
	s.held = o
	s.rel = s.last_xf.affine_inverse() * o.global_transform
	o.holder = i
	o.freeze = true
	o.linear_velocity = Vector3.ZERO
	o.angular_velocity = Vector3.ZERO
	grab_count += 1
	if debug:
		print("[grab] t=%.2f hand=%s obj=%s dist=%.3f palm=%s" % [Time.get_ticks_msec() / 1000.0, "LR"[i], o.name, palm.distance_to(o.global_position), _fmt(palm)])


func _release(i: int, throw: bool) -> void:
	var s := _hs[i]
	var o := s.held
	s.held = null
	o.holder = -1
	var v := _release_velocity(s) if throw else Vector3.ZERO
	o.freeze = false
	o.sleeping = false
	o.linear_velocity = v
	o.angular_velocity = Vector3.ZERO
	release_count += 1
	if debug:
		print("[grab] t=%.2f release hand=%s obj=%s vel=%s pos=%s" % [Time.get_ticks_msec() / 1000.0, "LR"[i], o.name, _fmt(v), _fmt(o.global_position)])


## Average palm velocity over the last VELOCITY_WINDOW_S.
func _release_velocity(s: HandState) -> Vector3:
	var h := s.history
	if h.size() < 2:
		return Vector3.ZERO
	var last: Array = h[h.size() - 1]
	var first: Array = last
	for k in range(h.size() - 1, -1, -1):
		if float(last[0]) - float(h[k][0]) > VELOCITY_WINDOW_S:
			break
		first = h[k]
	var dt := float(last[0]) - float(first[0])
	if dt < 0.02:
		return Vector3.ZERO
	var p_last: Vector3 = last[1]
	var p_first: Vector3 = first[1]
	return ((p_last - p_first) / dt).limit_length(MAX_THROW_SPEED)


static func _fmt(v: Vector3) -> String:
	return "(%.3f, %.3f, %.3f)" % [v.x, v.y, v.z]
