from webui import (
    _instruction,
    append_event,
    apply_delivery,
    run_generation,
    select_take,
)
from webui_store import Store


def _run_args(tmp_path, **overrides):
    values = {
        "text": "hello",
        "voice": "A calm narrator.",
        "performance": "",
        "reference_profile": None,
        "reference_audio": None,
        "reference_text": "",
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
        "store": Store(tmp_path),
        "api_url": "http://127.0.0.1:1",
        "frame_rate": 12.5,
    }
    values.update(overrides)
    return values


def test_instruction_combines_voice_and_performance():
    assert _instruction("A baritone.", "Speak slowly.") == (
        "Voice: A baritone.\n\nPerformance: Speak slowly."
    )
    assert _instruction("", "Speak slowly.") == "Performance: Speak slowly."
    assert _instruction("", "") == ""


def test_append_event():
    assert append_event("Hello", "(laugh)") == "Hello (laugh)"
    assert append_event("", "(sigh)") == "(sigh)"


def test_apply_delivery():
    assert "Warm" in apply_delivery("Warm")
    assert apply_delivery("None") == ""
    assert apply_delivery(None) == ""


def test_select_take():
    audio = (24000, [0.0])
    results = [
        {"label": "Take 1 · seed 42", "audio": audio, "seed": 42, "duration": 1.5}
    ]

    assert select_take("Take 1 · seed 42", results)[0] is audio
    assert select_take("missing", results)[0] is None


def test_run_generation_requires_text(tmp_path):
    outputs = list(run_generation(**_run_args(tmp_path, text="   ")))

    assert len(outputs) == 1
    assert "Enter some text" in outputs[0][3]


def test_run_generation_requires_reference_transcript(tmp_path):
    outputs = list(
        run_generation(**_run_args(tmp_path, reference_audio="/tmp/reference.wav"))
    )

    assert len(outputs) == 1
    assert "needs its exact transcript" in outputs[0][3]


def test_run_generation_reports_unreachable_api(tmp_path):
    outputs = list(run_generation(**_run_args(tmp_path)))

    assert len(outputs) >= 2
    assert "Cannot reach" in outputs[-1][3]
