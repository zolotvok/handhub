# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
import json

from hub.config import DEFAULTS, Config


def test_defaults_when_file_missing(tmp_path):
    cfg = Config(tmp_path / "missing.json")
    assert cfg.get() == DEFAULTS


def test_update_persists_and_reloads(tmp_path):
    path = tmp_path / "c.json"
    Config(path).update({"smoothing": {"beta": 0.05}})
    assert Config(path).get()["smoothing"]["beta"] == 0.05


def test_unknown_keys_are_dropped(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"nope": 3, "smoothing": {"bogus": 1, "beta": 0.1}}))
    data = Config(path).get()
    assert "nope" not in data
    assert "bogus" not in data["smoothing"]
    assert data["smoothing"]["beta"] == 0.1


def test_wrong_type_is_ignored(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"smoothing": {"beta": "fast"}}))
    assert Config(path).get()["smoothing"]["beta"] == DEFAULTS["smoothing"]["beta"]


def test_list_of_wrong_length_is_ignored(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"camera": {"pos": [1.0, 2.0]}}))
    assert Config(path).get()["camera"]["pos"] == DEFAULTS["camera"]["pos"]


def test_list_with_non_numeric_entries_is_ignored(tmp_path):
    # Extra check: right length, wrong element type.
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"camera": {"pos": ["a", "b", "c"]}}))
    assert Config(path).get()["camera"]["pos"] == DEFAULTS["camera"]["pos"]


def test_invalid_json_falls_back_to_defaults(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json")
    assert Config(path).get() == DEFAULTS


def test_json_that_is_not_an_object_falls_back_to_defaults(tmp_path):
    # Extra check: valid JSON with the wrong top-level type.
    path = tmp_path / "c.json"
    path.write_text("[1, 2, 3]")
    assert Config(path).get() == DEFAULTS
