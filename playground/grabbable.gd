# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
class_name Grabbable
extends RigidBody3D
## A small physics object (cube or ball) that GrabManager can pick up, hold and throw.

enum Glow { NONE, NEAR, HELD }

var start_xf := Transform3D.IDENTITY
var holder := -1            # index of the hand holding it, -1 if free
var base_color := Color.WHITE

var _mat := StandardMaterial3D.new()
var _mesh_instance := MeshInstance3D.new()
var _glow := -1


## kind: "cube" (8 cm edge) or "ball" (4 cm radius). xf: where it rests / respawns.
func setup(kind: String, color: Color, xf: Transform3D) -> void:
	base_color = color
	start_xf = xf
	mass = 0.25
	continuous_cd = true
	freeze_mode = RigidBody3D.FREEZE_MODE_KINEMATIC
	var pm := PhysicsMaterial.new()
	pm.friction = 0.8
	pm.bounce = 0.25
	physics_material_override = pm

	var shape := CollisionShape3D.new()
	if kind == "ball":
		var s := SphereShape3D.new()
		s.radius = 0.04
		shape.shape = s
		var m := SphereMesh.new()
		m.radius = 0.04
		m.height = 0.08
		m.radial_segments = 16
		m.rings = 8
		_mesh_instance.mesh = m
	else:
		var s := BoxShape3D.new()
		s.size = Vector3.ONE * 0.08
		shape.shape = s
		var m := BoxMesh.new()
		m.size = Vector3.ONE * 0.08
		_mesh_instance.mesh = m
	_mat.albedo_color = color
	_mat.roughness = 0.7
	_mesh_instance.material_override = _mat
	_mesh_instance.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	add_child(shape)
	add_child(_mesh_instance)
	transform = xf
	set_glow(Glow.NONE)


func set_glow(g: int) -> void:
	if g == _glow:
		return
	_glow = g
	_mat.emission_enabled = g != Glow.NONE
	_mat.emission = base_color.lightened(0.55)
	_mat.emission_energy_multiplier = 0.9 if g == Glow.NEAR else 0.3
	_mesh_instance.scale = Vector3.ONE * (1.12 if g == Glow.NEAR else 1.0)


## Put the object back on the table at rest. Works whether it is frozen (held) or free.
func reset_to_start() -> void:
	holder = -1
	var rid := get_rid()
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_TRANSFORM, start_xf)
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_LINEAR_VELOCITY, Vector3.ZERO)
	PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_ANGULAR_VELOCITY, Vector3.ZERO)
	global_transform = start_xf
	freeze = false
	sleeping = false
