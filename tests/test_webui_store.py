import json

import pytest

from webui_store import (
    DEFAULT_PRESETS,
    DEFAULT_SETTINGS,
    Store,
    normalize_settings,
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


def test_missing_preset_file_uses_defaults(tmp_path):
    presets, warning = Store(tmp_path).load_presets()

    assert warning is None
    assert presets == DEFAULT_PRESETS


def test_presets_round_trip(tmp_path):
    store = Store(tmp_path)
    presets, _ = store.load_presets()
    presets["Test voice"] = {
        "voice": "A calm test voice.",
        "performance": "Speak slowly.",
        "cfg_scale": 4.0,
        "temperature": 1.1,
        "seed": 7,
    }

    store.save_presets(presets)
    loaded, warning = store.load_presets()

    assert warning is None
    assert loaded["Test voice"]["voice"] == "A calm test voice."
    assert loaded["Test voice"]["temperature"] == 1.1
    payload = json.loads(store.presets_path.read_text(encoding="utf-8"))
    assert payload["version"] == 2


def test_v1_presets_are_migrated(tmp_path):
    store = Store(tmp_path)
    store.presets_path.write_text(
        json.dumps({"version": 1, "presets": {"Old": "A plain instruction."}}),
        encoding="utf-8",
    )

    presets, warning = store.load_presets()

    assert warning is None
    assert presets["Old"]["voice"] == "A plain instruction."
    assert presets["Old"]["performance"] == ""


def test_malformed_preset_file_falls_back(tmp_path):
    store = Store(tmp_path)
    store.presets_path.write_text("{not json", encoding="utf-8")

    presets, warning = store.load_presets()

    assert presets == DEFAULT_PRESETS
    assert warning is not None and "invalid" in warning


def test_scripts_round_trip(tmp_path):
    store = Store(tmp_path)
    store.save_scripts({"Intro": "Hello there.", "Outro": "Goodbye."})

    scripts, warning = store.load_scripts()

    assert warning is None
    assert scripts == {"Intro": "Hello there.", "Outro": "Goodbye."}


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
    assert not list(tmp_path.rglob("*.tmp"))


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
