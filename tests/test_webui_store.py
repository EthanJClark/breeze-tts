import base64
import json
from pathlib import Path

import pytest

import webui_store
from webui_store import (
    APP_DATA_DIR,
    DEFAULT_PRESETS,
    DEFAULT_SETTINGS,
    VOICE_CLONE,
    VOICE_DESCRIBE,
    Store,
    migrate_data_dir,
    normalize_settings,
)


def _preset_payload(**overrides):
    payload = {
        "voice_source": VOICE_DESCRIBE,
        "description": "A calm test voice.",
        "delivery": "",
        "reference": None,
        "settings": {
            **DEFAULT_SETTINGS,
            "cfg_scale": 4.0,
            "temperature": 1.1,
            "seed": 7,
        },
    }
    payload.update(overrides)
    return payload


def _clone_payload(audio=b"RIFF-reference", text="Hello there."):
    return _preset_payload(
        voice_source=VOICE_CLONE,
        description="",
        reference={
            "filename": "clip.wav",
            "text": text,
            "audio_b64": base64.b64encode(audio).decode("ascii"),
        },
    )


def test_settings_defaults_and_clamping(tmp_path):
    store = Store(tmp_path)

    settings, warning = store.load_settings()

    assert warning is None
    assert settings == normalize_settings(DEFAULT_SETTINGS)

    saved = store.save_settings(
        {
            "cfg_scale": 99,
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 999,
            "repetition_penalty": 0.1,
            "max_duration": 999,
            "takes": 99,
            "seed_mode": "nonsense",
            "seed": -5,
            "timeout": 1,
            "autoplay": False,
        }
    )

    assert saved["cfg_scale"] == 6.0
    assert saved["temperature"] == 0.1
    assert saved["top_p"] == 0.05
    assert saved["top_k"] == 100
    assert saved["repetition_penalty"] == 1.0
    assert saved["max_duration"] == 120.0
    assert saved["takes"] == 4
    assert saved["seed_mode"] == "fixed"
    assert saved["seed"] == 0
    assert saved["timeout"] == 30.0
    assert saved["autoplay"] is False

    loaded, warning = store.load_settings()
    assert warning is None
    assert loaded == saved


def test_missing_presets_seed_defaults(tmp_path):
    store = Store(tmp_path)

    presets, warning = store.list_presets()

    assert warning is None
    assert sorted(presets) == sorted(DEFAULT_PRESETS)
    assert len(list(store.presets_dir.glob("*.json"))) == len(DEFAULT_PRESETS)


def test_preset_round_trip(tmp_path):
    store = Store(tmp_path)
    assert store.save_preset("Test voice", _preset_payload()) == "Test voice"

    presets, warning = store.list_presets()
    preset = presets["Test voice"]

    assert warning is None
    assert preset["description"] == "A calm test voice."
    assert preset["settings"]["temperature"] == 1.1
    assert preset["settings"]["seed"] == 7

    path = next(store.presets_dir.glob("Test-voice*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["name"] == "Test voice"
    assert not list(tmp_path.rglob("*.tmp"))
    assert store.preset_path("Test voice") == path
    assert store.preset_path("missing") is None


def test_app_data_dir_lives_in_home():
    assert APP_DATA_DIR.name == ".breeze2"
    assert APP_DATA_DIR.parent == Path.home()


def test_store_migrates_repo_presets(tmp_path, monkeypatch):
    repo_data = tmp_path / "repo" / ".breeze2"
    (repo_data / "presets").mkdir(parents=True)
    (repo_data / "presets" / "Repo.json").write_text('{"name": "Repo"}', encoding="utf-8")
    destination = tmp_path / "home" / ".breeze2"

    monkeypatch.setattr(webui_store, "APP_DATA_DIR", destination)
    monkeypatch.setattr(webui_store, "LEGACY_DATA_DIR", tmp_path / "missing")
    monkeypatch.setattr(webui_store, "REPO_DATA_DIR", repo_data)

    Store(destination)

    assert (destination / "presets" / "Repo.json").is_file()


def test_migrate_data_dir_copies_missing_files_without_overwriting(tmp_path):
    source = tmp_path / "old"
    destination = tmp_path / "new"
    (source / "presets").mkdir(parents=True)
    (source / "presets" / "old.json").write_text('{"name": "old"}', encoding="utf-8")
    (source / "settings.json").write_text('{"settings": {}}', encoding="utf-8")

    copied = migrate_data_dir(source, destination)

    assert len(copied) == 2
    assert (destination / "presets" / "old.json").read_text(encoding="utf-8") == (
        '{"name": "old"}'
    )
    assert (destination / "settings.json").is_file()

    (source / "presets" / "old.json").write_text("changed", encoding="utf-8")
    migrate_data_dir(source, destination)
    assert (destination / "presets" / "old.json").read_text(encoding="utf-8") == (
        '{"name": "old"}'
    )


def test_preset_settings_are_clamped(tmp_path):
    store = Store(tmp_path)
    store.save_preset(
        "Loud",
        _preset_payload(
            settings={
                "cfg_scale": 99,
                "temperature": 0.0,
                "top_p": 0.0,
                "top_k": 999,
                "repetition_penalty": 0.1,
                "max_duration": 999,
                "takes": 99,
                "seed_mode": "nonsense",
                "seed": -5,
                "timeout": 1,
                "autoplay": False,
            }
        ),
    )

    preset = store.load_preset("Loud")

    assert preset is not None
    assert preset["settings"]["cfg_scale"] == 6.0
    assert preset["settings"]["top_k"] == 100
    assert preset["settings"]["takes"] == 4
    assert preset["settings"]["seed_mode"] == "fixed"
    assert preset["settings"]["seed"] == 0
    assert preset["settings"]["autoplay"] is False


def test_legacy_presets_are_migrated(tmp_path):
    (tmp_path / "instruction_presets.json").write_text(
        json.dumps(
            {
                "version": 2,
                "presets": {
                    "Old": {
                        "voice": "A calm voice.",
                        "performance": "Speak slowly.",
                        "cfg_scale": 5.0,
                        "temperature": 1.1,
                        "seed": 7,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    presets, warning = Store(tmp_path).list_presets()

    assert warning is None
    assert presets["Old"]["description"] == "A calm voice.\n\nSpeak slowly."
    assert presets["Old"]["voice_source"] == VOICE_DESCRIBE
    assert presets["Old"]["settings"]["cfg_scale"] == 5.0
    assert presets["Old"]["settings"]["temperature"] == 1.1
    assert presets["Old"]["settings"]["seed"] == 7
    assert presets["Old"]["settings"]["top_k"] == DEFAULT_SETTINGS["top_k"]
    assert "Santa" in presets


def test_malformed_preset_file_is_skipped(tmp_path):
    store = Store(tmp_path)
    store.list_presets()
    (store.presets_dir / "broken.json").write_text("{not json", encoding="utf-8")

    presets, warning = store.list_presets()

    assert sorted(presets) == sorted(DEFAULT_PRESETS)
    assert warning is not None and "invalid" in warning


def test_delete_preset(tmp_path):
    store = Store(tmp_path)
    store.save_preset("Temporary", _preset_payload())

    assert store.delete_preset("Temporary") is True
    assert store.delete_preset("Temporary") is False
    assert "Temporary" not in store.list_presets()[0]


def test_preset_filename_cannot_escape_data_dir(tmp_path):
    store = Store(tmp_path)
    store.save_preset("../../evil name", _preset_payload())

    path = next(store.presets_dir.glob("*.json"))
    assert path.resolve().parent == store.presets_dir.resolve()
    assert ".." not in path.name
    assert store.load_preset("../../evil name") is not None


def test_export_import_round_trip_with_audio(tmp_path):
    store = Store(tmp_path)
    store.save_preset("Clone", _clone_payload())
    exported = store.export_preset("Clone")

    other = Store(tmp_path / "other")
    name = other.import_preset(exported.read_bytes(), "Clone.json")

    assert name == "Clone"
    preset = other.load_preset(name)
    assert preset is not None
    assert preset["reference"]["text"] == "Hello there."
    path = other.materialize_reference(name, preset)
    assert path is not None
    assert path.read_bytes() == b"RIFF-reference"


def test_import_dedupes_existing_names(tmp_path):
    store = Store(tmp_path)
    store.save_preset("Clone", _clone_payload())
    exported = store.export_preset("Clone").read_bytes()

    first = store.import_preset(exported, "Clone.json")
    second = store.import_preset(exported, "Clone.json")

    assert first == "Clone (2)"
    assert second == "Clone (3)"


def test_import_rejects_invalid_json(tmp_path):
    with pytest.raises(ValueError):
        Store(tmp_path).import_preset(b"{not json", "preset.json")


def test_materialize_reference_is_stable(tmp_path):
    store = Store(tmp_path)
    store.save_preset("Clone", _clone_payload())
    preset = store.load_preset("Clone")

    first = store.materialize_reference("Clone", preset)
    second = store.materialize_reference("Clone", preset)

    assert first == second
    assert first.parent == store.preset_refs_dir


def test_voice_profiles_round_trip(tmp_path):
    store = Store(tmp_path)
    source = tmp_path / "reference.wav"
    source.write_bytes(b"RIFF-reference")

    relative = store.import_reference("Old man", source)
    store.save_voices({"Old man": {"audio": relative, "ref_text": "Hello."}})

    voices, warning = store.load_voices()
    profile = voices["Old man"]

    assert warning is None
    assert profile["ref_text"] == "Hello."
    assert store.reference_path(profile).read_bytes() == b"RIFF-reference"


def test_reference_path_rejects_escape(tmp_path):
    store = Store(tmp_path)

    with pytest.raises(ValueError):
        store.reference_path({"audio": "../outside.wav", "ref_text": "x"})

    with pytest.raises(ValueError):
        store.reference_path({"audio": "/tmp/outside.wav", "ref_text": "x"})


def test_remove_reference_deletes_file(tmp_path):
    store = Store(tmp_path)
    source = tmp_path / "reference.wav"
    source.write_bytes(b"RIFF-reference")
    relative = store.import_reference("Voice", source)
    profile = {"audio": relative, "ref_text": "Hello."}

    store.remove_reference(profile)

    assert not store.reference_path(profile).exists()
