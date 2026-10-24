import base64

import pytest

from webui import (
    GenerationError,
    _resolve_voice_inputs,
    apply_preset_handler,
    apply_preset_state,
    collect_preset_state,
    delete_preset_handler,
    export_preset_handler,
    import_preset_handler,
    on_voice_change,
    run_generation,
    save_preset_handler,
    select_take,
)
from webui_store import VOICE_CLONE, VOICE_DESCRIBE, Store


def _run_args(**overrides):
    values = {
        "text": "hello",
        "voice": VOICE_DESCRIBE,
        "description": "A calm narrator.",
        "reference_audio": None,
        "reference_text": "",
        "delivery": "",
        "cfg_scale": 4.0,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_duration": 30.0,
        "takes": 1,
        "seed_mode": "fixed",
        "seed": 42,
        "timeout": 30.0,
        "api_url": "http://127.0.0.1:1",
        "frame_rate": 12.5,
    }
    values.update(overrides)
    return values


def _collect_args(**overrides):
    values = {
        "name": "Test preset",
        "voice": VOICE_DESCRIBE,
        "description": "A calm narrator.",
        "reference_audio": None,
        "reference_text": "",
        "delivery": "",
        "cfg_scale": 4.0,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_duration": 30.0,
        "takes": 1,
        "seed_mode": "fixed",
        "seed": 42,
        "timeout": 30.0,
        "autoplay": True,
    }
    values.update(overrides)
    return values


def test_resolve_describe_uses_description():
    instruction, audio, filename, text = _resolve_voice_inputs(
        VOICE_DESCRIBE, "A baritone.", None, "", ""
    )

    assert instruction == "A baritone."
    assert audio is None
    assert filename is None
    assert text == ""


def test_resolve_clone_requires_transcript(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF-reference")

    with pytest.raises(GenerationError, match="exact transcript"):
        _resolve_voice_inputs(VOICE_CLONE, "", str(clip), "   ", "")


def test_resolve_clone_reads_clip(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF-reference")

    instruction, audio, filename, text = _resolve_voice_inputs(
        VOICE_CLONE, "ignored", str(clip), "Hello there.", "Speak slowly."
    )

    assert instruction == "Speak slowly."
    assert audio == b"RIFF-reference"
    assert filename == "clip.wav"
    assert text == "Hello there."


def test_collect_preset_state_excludes_script():
    payload = collect_preset_state(**_collect_args())

    assert "script" not in payload
    assert payload["description"] == "A calm narrator."
    assert payload["voice_source"] == VOICE_DESCRIBE
    assert payload["reference"] is None
    assert payload["settings"]["top_k"] == 50
    assert payload["settings"]["autoplay"] is True


def test_collect_preset_state_embeds_clone_audio(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF-reference")

    payload = collect_preset_state(
        **_collect_args(
            voice=VOICE_CLONE,
            description="",
            reference_audio=str(clip),
            reference_text="Hello there.",
            delivery="Speak slowly.",
        )
    )

    reference = payload["reference"]
    assert reference["filename"] == "clip.wav"
    assert reference["text"] == "Hello there."
    assert base64.b64decode(reference["audio_b64"]) == b"RIFF-reference"


def test_collect_preset_state_requires_transcript(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF-reference")

    with pytest.raises(GenerationError, match="exact transcript"):
        collect_preset_state(
            **_collect_args(
                voice=VOICE_CLONE,
                description="",
                reference_audio=str(clip),
                reference_text="",
                delivery="Speak slowly.",
            )
        )


def test_collect_preset_state_requires_content():
    with pytest.raises(GenerationError, match="before saving"):
        collect_preset_state(**_collect_args(description="", reference_audio=None))


def test_apply_preset_state_restores_fields(tmp_path):
    store = Store(tmp_path)
    payload = collect_preset_state(
        **_collect_args(description="A weathered baritone.", delivery="Slow.")
    )
    store.save_preset("Test preset", payload)

    updates, message = apply_preset_state(
        "Test preset", store.load_preset("Test preset"), store=store
    )

    assert updates[0]["value"] == "Test preset"
    assert updates[1]["value"] == VOICE_DESCRIBE
    assert updates[2] == "A weathered baritone."
    assert updates[5] == "Slow."
    assert updates[6] == 4.0
    assert updates[13] == "fixed"
    assert updates[16] is True
    assert "Loaded preset" in message


def test_apply_preset_falls_back_to_bundled_clip(tmp_path):
    store = Store(tmp_path)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF-reference")
    payload = collect_preset_state(
        **_collect_args(
            voice=VOICE_CLONE,
            description="",
            reference_audio=str(clip),
            reference_text="Hello there.",
            delivery="Speak slowly.",
        )
    )
    payload["voice_source"] = "An old saved voice"
    store.save_preset("Old preset", payload)

    updates, message = apply_preset_state(
        "Old preset", store.load_preset("Old preset"), store=store
    )

    assert updates[1]["value"] == VOICE_CLONE
    assert updates[3] is not None
    assert updates[4] == "Hello there."
    assert "bundled clip" in message


def test_select_take():
    audio = (24000, [0.0])
    results = [
        {"label": "Take 1 · seed 42", "audio": audio, "seed": 42, "duration": 1.5}
    ]

    assert select_take("Take 1 · seed 42", results)[0] is audio
    assert select_take("missing", results)[0] is None


def test_run_generation_requires_text():
    outputs = list(run_generation(**_run_args(text="   ")))

    assert len(outputs) == 1
    assert "Enter some text" in outputs[0][3]


def test_run_generation_requires_reference_transcript(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF-reference")

    outputs = list(
        run_generation(**_run_args(voice=VOICE_CLONE, reference_audio=str(clip)))
    )

    assert len(outputs) == 1
    assert "needs its exact transcript" in outputs[0][3]


def test_run_generation_reports_unreachable_api():
    outputs = list(run_generation(**_run_args()))

    assert len(outputs) >= 2
    assert "Cannot reach" in outputs[-1][3]


def test_handler_output_arities(tmp_path):
    store = Store(tmp_path)

    assert len(apply_preset_handler(None, store=store)) == 21
    assert len(save_preset_handler("", *([""] * 16), store=store)) == 3
    assert len(delete_preset_handler(None, store=store)) == 3
    assert len(export_preset_handler(None, store=store)) == 2
    assert len(import_preset_handler(None, store=store)) == 23
    assert len(on_voice_change(VOICE_DESCRIBE)) == 3
