# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
class_name HandVisual
extends Node3D
## One hand skeleton: 21 joint spheres + 21 thin bones, tinted by tracking confidence.
## Points are given in this node's local space (the node itself stays at identity for the
## real hands; the mirror reuses it with size_mult = 3).

const BONES := [
	[0, 1], [1, 2], [2, 3], [3, 4],
	[0, 5], [5, 6], [6, 7], [7, 8],
	[5, 9], [9, 10], [10, 11], [11, 12],
	[9, 13], [13, 14], [14, 15], [15, 16],
	[13, 17], [17, 18], [18, 19], [19, 20],
	[0, 17],
]
const COLOR_GOOD := Color(0.15, 0.85, 0.25)   # confidence >= 0.6
const COLOR_FAIR := Color(1.0, 0.70, 0.10)    # >= 0.35
const COLOR_POOR := Color(0.95, 0.20, 0.18)   # below
const JOINT_RADIUS := 0.008
const BONE_RADIUS := 0.0035

var _joints: Array[MeshInstance3D] = []
var _bones: Array[MeshInstance3D] = []
var _label: Label3D
var _mat := StandardMaterial3D.new()
var _band := -1
var _size_mult := 1.0


## is_right picks the "L"/"R" label; size_mult scales joint/bone thickness (positions are not scaled).
func setup(is_right: bool, size_mult: float = 1.0) -> void:
	_size_mult = size_mult
	_mat.roughness = 0.6
	_mat.emission_enabled = true
	_mat.emission_energy_multiplier = 0.35
	_set_band(0)

	var sphere := SphereMesh.new()
	sphere.radius = JOINT_RADIUS * size_mult
	sphere.height = sphere.radius * 2.0
	sphere.radial_segments = 10
	sphere.rings = 5
	var cyl := CylinderMesh.new()          # unit height, scaled per bone along its axis
	cyl.top_radius = BONE_RADIUS * size_mult
	cyl.bottom_radius = BONE_RADIUS * size_mult
	cyl.height = 1.0
	cyl.radial_segments = 6
	cyl.rings = 1

	for i in 21:
		var j := MeshInstance3D.new()
		j.mesh = sphere
		j.material_override = _mat
		j.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		if i == 0:
			j.scale = Vector3.ONE * 1.6       # the wrist is bigger so it is easy to spot
		add_child(j)
		_joints.append(j)
	for i in BONES.size():
		var b := MeshInstance3D.new()
		b.mesh = cyl
		b.material_override = _mat
		b.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		add_child(b)
		_bones.append(b)

	_label = Label3D.new()
	_label.text = "R" if is_right else "L"
	_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	_label.no_depth_test = true
	_label.font_size = 64
	_label.pixel_size = 0.0004 * size_mult
	_label.outline_size = 12
	_label.modulate = Color(1.0, 0.45, 0.2) if is_right else Color(0.3, 0.7, 1.0)
	add_child(_label)
	visible = false


## points: 21 positions in this node's local space. Hidden when show_hand is false.
func update_pose(points: PackedVector3Array, confidence: float, show_hand: bool) -> void:
	if visible != show_hand:
		visible = show_hand
	if not show_hand:
		return
	_set_band(0 if confidence >= 0.6 else (1 if confidence >= 0.35 else 2))
	for i in 21:
		_joints[i].position = points[i]
	for i in BONES.size():
		_place_bone(_bones[i], points[BONES[i][0]], points[BONES[i][1]])
	_label.position = points[0] + Vector3(0.0, 0.045 * _size_mult, 0.0)


func _place_bone(mi: MeshInstance3D, a: Vector3, b: Vector3) -> void:
	var d := b - a
	var dist := d.length()
	if dist < 1e-5:
		mi.visible = false
		return
	mi.visible = true
	var y := d / dist
	var ref := Vector3.RIGHT if absf(y.x) < 0.9 else Vector3.FORWARD
	var x := y.cross(ref).normalized()
	var z := x.cross(y)
	mi.transform = Transform3D(Basis(x, y * dist, z), (a + b) * 0.5)


func _set_band(band: int) -> void:
	if band == _band:
		return
	_band = band
	var c := COLOR_GOOD if band == 0 else (COLOR_FAIR if band == 1 else COLOR_POOR)
	_mat.albedo_color = c
	_mat.emission = c
