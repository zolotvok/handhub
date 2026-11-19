# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
class_name PushButton
extends Node3D
## A big push button standing on a surface (origin = surface point under the button).
## "Pressed" while an index fingertip is inside its trigger volume (an Area3D with a cylinder
## shape; fingertips are points, so containment is tested against that shape directly, which
## needs no physics frame and no probe bodies). Shows a counter Label3D.

signal pressed   # emitted once per press (rising edge)

const BASE_H := 0.02
const CAP_H := 0.03
const CAP_R := 0.05
const TRAVEL := 0.015
const AREA_R := 0.055
const AREA_H := 0.075   # from the surface up to 2.5 cm above the cap's rest top

var count := 0
var is_down := false
var title := ""

var _cap := MeshInstance3D.new()
var _mat := StandardMaterial3D.new()
var _area := Area3D.new()
var _shape := CylinderShape3D.new()
var _counter := Label3D.new()
var _idle := Color.WHITE
var _down := Color.WHITE
var _depth := 0.0       # 0 = rest, 1 = fully depressed (smoothed)


func setup(p_title: String, idle: Color, down: Color) -> void:
	title = p_title
	_idle = idle
	_down = down

	var base := MeshInstance3D.new()
	var bm := CylinderMesh.new()
	bm.top_radius = 0.065
	bm.bottom_radius = 0.065
	bm.height = BASE_H
	bm.radial_segments = 24
	bm.rings = 1
	base.mesh = bm
	var base_mat := StandardMaterial3D.new()
	base_mat.albedo_color = Color(0.12, 0.12, 0.14)
	base.material_override = base_mat
	base.position = Vector3(0, BASE_H * 0.5, 0)
	add_child(base)

	var cm := CylinderMesh.new()
	cm.top_radius = CAP_R
	cm.bottom_radius = CAP_R
	cm.height = CAP_H
	cm.radial_segments = 24
	cm.rings = 1
	_cap.mesh = cm
	_mat.albedo_color = idle
	_mat.emission_enabled = true
	_mat.emission = idle
	_mat.emission_energy_multiplier = 0.1
	_cap.material_override = _mat
	add_child(_cap)

	_shape.radius = AREA_R
	_shape.height = AREA_H
	var cs := CollisionShape3D.new()
	cs.shape = _shape
	_area.add_child(cs)
	_area.position = Vector3(0, AREA_H * 0.5, 0)
	_area.monitoring = false      # purely a volume definition; see class comment
	_area.monitorable = false
	add_child(_area)

	_counter.font_size = 40
	_counter.pixel_size = 0.0014
	_counter.position = Vector3(0, 0.14, 0.0)
	_counter.outline_size = 8
	add_child(_counter)
	_refresh_text()
	_apply_depth()


## Feed the fingertip world positions each frame; emits `pressed` on the rising edge.
func test_points(points: Array[Vector3]) -> void:
	var inside := false
	var inv := _area.global_transform.affine_inverse()
	for p in points:
		var l := inv * p
		if Vector2(l.x, l.z).length() <= _shape.radius and absf(l.y) <= _shape.height * 0.5:
			inside = true
			break
	if inside and not is_down:
		count += 1
		_refresh_text()
		pressed.emit()
	is_down = inside


func _process(delta: float) -> void:
	var target := 1.0 if is_down else 0.0
	_depth = move_toward(_depth, target, delta * 12.0)
	_apply_depth()


func _apply_depth() -> void:
	_cap.position = Vector3(0, BASE_H + CAP_H * 0.5 - TRAVEL * _depth, 0)
	var c := _idle.lerp(_down, _depth)
	_mat.albedo_color = c
	_mat.emission = c
	_mat.emission_energy_multiplier = 0.1 + 0.9 * _depth


func _refresh_text() -> void:
	_counter.text = "%s\npressed: %d" % [title, count]
