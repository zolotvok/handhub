# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
class_name HandMirror
extends Node3D
## A wall "mirror": enlarged (x3) copies of both hand skeletons so finger poses can be checked
## without looking down. Each copy uses landmark positions relative to its own wrist (world
## orientation, so it turns with the hand) and sits at a fixed anchor: the left hand on the left,
## the right hand on the right, like a real mirror seen from the front.
##
## Mirroring: x is mirrored as asked and the image is also turned 180 degrees about the vertical
## axis, which together equal a reflection in the panel plane (z -> -z): fingers pointing away
## from you in the room point towards you in the mirror, so the image never pokes into the wall.
## To get the plain "x mirrored only" variant, set FLIP to Vector3(-1, 1, 1) and move the panel.

const SCALE := 3.0
const FLIP := Vector3(1, 1, -1)
const PANEL := Vector2(1.5, 0.95)
const ANCHORS := [Vector3(-0.38, -0.30, 0.03), Vector3(0.38, -0.30, 0.03)]   # [left, right]

var _hands: Array[HandVisual] = []


func _ready() -> void:
	var m := StandardMaterial3D.new()
	m.albedo_color = Color(0.07, 0.09, 0.12)
	m.roughness = 0.4
	var panel := MeshInstance3D.new()
	var pm := BoxMesh.new()
	pm.size = Vector3(PANEL.x, PANEL.y, 0.03)
	panel.mesh = pm
	panel.material_override = m
	add_child(panel)

	var frame_mat := StandardMaterial3D.new()
	frame_mat.albedo_color = Color(0.75, 0.65, 0.3)
	for r in [Rect2(0, PANEL.y * 0.5, PANEL.x + 0.06, 0.03), Rect2(0, -PANEL.y * 0.5, PANEL.x + 0.06, 0.03),
			Rect2(-PANEL.x * 0.5, 0, 0.03, PANEL.y), Rect2(PANEL.x * 0.5, 0, 0.03, PANEL.y)]:
		var f := MeshInstance3D.new()
		var fm := BoxMesh.new()
		fm.size = Vector3(r.size.x, r.size.y, 0.05)
		f.mesh = fm
		f.material_override = frame_mat
		f.position = Vector3(r.position.x, r.position.y, 0.01)
		add_child(f)

	var title := Label3D.new()
	title.text = "HAND MIRROR  x3"
	title.font_size = 36
	title.pixel_size = 0.0012
	title.modulate = Color(0.75, 0.65, 0.3)
	title.position = Vector3(0, PANEL.y * 0.5 - 0.06, 0.03)
	add_child(title)

	for i in 2:
		var hv := HandVisual.new()
		hv.setup(i == 1, SCALE)
		add_child(hv)
		_hands.append(hv)


func update_hands(client: HubClient) -> void:
	var pts := PackedVector3Array()
	pts.resize(21)
	for i in 2:
		var h := client.hands[i]
		if h.valid:
			var wrist := h.lm_world[0]
			for j in 21:
				pts[j] = ANCHORS[i] + (h.lm_world[j] - wrist) * FLIP * SCALE
		_hands[i].update_pose(pts, h.confidence, h.valid)
