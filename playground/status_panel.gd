# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
class_name StatusPanel
extends Node3D
## Dark wall panel with a monospace Label3D. Faces +Z; the caller orients it towards the player.

const SIZE := Vector2(0.78, 0.76)

var _label := Label3D.new()


func _ready() -> void:
	var bg := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(SIZE.x, SIZE.y, 0.02)
	bg.mesh = bm
	var m := StandardMaterial3D.new()
	m.albedo_color = Color(0.05, 0.06, 0.08)
	m.roughness = 1.0
	bg.material_override = m
	add_child(bg)

	var font := SystemFont.new()
	font.font_names = PackedStringArray(["monospace", "DejaVu Sans Mono", "Noto Sans Mono"])
	_label.font = font
	_label.font_size = 30
	_label.pixel_size = 0.0013
	_label.outline_size = 0
	_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	_label.vertical_alignment = VERTICAL_ALIGNMENT_TOP
	_label.position = Vector3(-SIZE.x * 0.5 + 0.03, SIZE.y * 0.5 - 0.025, 0.012)
	_label.modulate = Color(0.85, 0.95, 0.85)
	add_child(_label)


func set_text(t: String) -> void:
	_label.text = t
