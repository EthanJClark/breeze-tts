import base64
import http.server
import threading
import time
from functools import partial
from typing import ClassVar

import gradio as gr
import pytest

import webui
from webui import (
    GenerationCancelled,
    GenerationCancelToken,
    GenerationControl,
    GenerationError,
    _request_once,
    _resolve_voice_inputs,
    apply_preset_state,
    clear_handler,
    collect_preset_state,
    delete_preset_handler,
    import_preset_handler,
    on_seed_mode_change,
    on_voice_change,
    preset_snapshot,
    refresh_presets_handler,
    resolve_preset_switch,
    run_generation,
    save_preset_handler,
    select_preset_handler,
    select_take,
    stop_generation,
)
from webui_store import VOICE_CLONE, VOICE_DESCRIBE, Store


def _fields(**overrides):
    values = {
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


def _run_args(**overrides):
    fields = _fields()
    fields.pop("autoplay", None)
    values = {
        "text": "hello",
        **fields,
        "api_url": "http://127.0.0.1:1",
        "frame_rate": 12.5,
    }
    values.update(overrides)
    return values


def _collect_args(**overrides):
    values = {"name": "Test preset", **_fields()}
    values.update(overrides)
    return values


def _switch_fields(**overrides):
    values = {"name": "Test preset", **_fields()}
    values.update(overrides)
    return values


def _snapshot(**overrides):
    return preset_snapshot(**_fields(**overrides))


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


def test_collect_preset_state_uses_last_random_seed():
    payload = collect_preset_state(
        **_collect_args(seed_mode="random", seed=42, last_seed=987)
    )

    assert payload["settings"]["seed_mode"] == "random"
    assert payload["settings"]["seed"] == 987


def test_collect_preset_state_keeps_fixed_seed():
    payload = collect_preset_state(
        **_collect_args(seed_mode="fixed", seed=42, last_seed=987)
    )

    assert payload["settings"]["seed"] == 42


def test_apply_preset_state_restores_fields(tmp_path):
    store = Store(tmp_path)
    payload = collect_preset_state(
        **_collect_args(description="A weathered baritone.", delivery="Slow.")
    )
    store.save_preset("Test preset", payload)

    updates, message = apply_preset_state(
        "Test preset", store.load_preset("Test preset"), store=store
    )

    assert len(updates) == 22
    assert updates[0]["value"] == "Test preset"
    assert updates[1]["value"] == VOICE_DESCRIBE
    assert updates[2] == "A weathered baritone."
    assert updates[5] == "Slow."
    assert updates[6] == 4.0
    assert updates[13] == "fixed"
    assert updates[14]["value"] == 42
    assert updates[14]["visible"] is True
    assert updates[16] is True
    assert updates[20] is None
    assert updates[21]["visible"] is False
    assert "Loaded preset" in message


def test_apply_preset_state_shows_saved_random_seed(tmp_path):
    store = Store(tmp_path)
    payload = collect_preset_state(
        **_collect_args(seed_mode="random", seed=42, last_seed=765)
    )
    store.save_preset("Random preset", payload)

    updates, _ = apply_preset_state(
        "Random preset", store.load_preset("Random preset"), store=store
    )

    assert updates[13] == "random"
    assert updates[14]["visible"] is False
    assert updates[20] == 765
    assert updates[21]["value"] == "765"
    assert updates[21]["visible"] is True


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


def test_select_take_resets_playback_and_tracks_seed():
    audio = (24000, [0.0])
    results = [
        {"label": "Take 1 · seed 42", "audio": audio, "seed": 42, "duration": 1.5}
    ]

    selected = select_take("Take 1 · seed 42", results)

    assert selected[0]["value"] is audio
    assert selected[0]["playback_position"] == 0
    assert selected[2] == 42
    assert selected[3]["value"] == "42"
    assert select_take("missing", results)[0] == gr.update()


def test_clear_handler_resets_generation_state():
    outputs = clear_handler()

    assert len(outputs) == 9
    assert outputs[0] == ""
    assert outputs[1]["value"] is None
    assert outputs[1]["playback_position"] == 0
    assert outputs[2]["choices"] == []
    assert outputs[3] == []
    assert outputs[4] is None
    assert "Cleared" in outputs[6]
    assert outputs[7]["interactive"] is True
    assert outputs[8]["interactive"] is False


def test_seed_mode_toggles_seed_fields():
    fixed = on_seed_mode_change("fixed")
    random_mode = on_seed_mode_change("random")

    assert fixed[0]["visible"] is True
    assert fixed[1]["visible"] is False
    assert random_mode[0]["visible"] is False
    assert random_mode[1]["visible"] is True


def test_run_generation_requires_text():
    outputs = list(run_generation(**_run_args(text="   ")))

    assert len(outputs) == 1
    assert "Enter some text" in outputs[0][3]
    assert outputs[0][0]["value"] is None
    assert outputs[0][0]["playback_position"] == 0


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


def test_run_generation_tracks_random_seeds(monkeypatch):
    calls = []

    def fake_request(**kwargs):
        calls.append(kwargs["seed"])
        return (24000, [0.0]), 0.1

    monkeypatch.setattr("webui._request_once", fake_request)

    outputs = list(run_generation(**_run_args(seed_mode="random", takes=2)))
    final = outputs[-1]
    results = final[2]

    assert len(results) == 2
    assert calls == [take["seed"] for take in results]
    assert final[7] == results[-1]["seed"]
    assert final[8]["value"] == str(results[-1]["seed"])

    for output in outputs:
        audio_update = output[0]
        if audio_update.get("value") is not None:
            assert audio_update.get("playback_position") == 0


def test_clean_preset_selection_applies_immediately(tmp_path):
    store = Store(tmp_path)
    store.save_preset(
        "Saved", collect_preset_state(**_collect_args(description="Saved voice."))
    )

    outputs = select_preset_handler(
        "Saved",
        _snapshot(description="Saved voice."),
        None,
        **_switch_fields(description="Saved voice."),
        store=store,
    )

    assert len(outputs) == 29
    assert outputs[0]["value"] == "Saved"
    assert outputs[1]["value"] == "Saved"
    assert outputs[24]["visible"] is False
    assert outputs[25] is None
    assert outputs[27] == "Saved"
    assert outputs[28]["interactive"] is True


def test_dirty_preset_selection_prompts_before_switching(tmp_path):
    store = Store(tmp_path)
    store.save_preset(
        "Saved", collect_preset_state(**_collect_args(description="Saved voice."))
    )

    outputs = select_preset_handler(
        "Saved",
        _snapshot(description="Saved voice."),
        None,
        **_switch_fields(description="Unsaved edit."),
        store=store,
    )

    assert outputs[0]["value"] is None
    assert outputs[1] == gr.update()
    assert "Unsaved changes" in outputs[23]
    assert outputs[24]["visible"] is True
    assert outputs[25] == "Saved"


def test_cancel_preset_switch_keeps_current_settings(tmp_path):
    store = Store(tmp_path)

    outputs = resolve_preset_switch(
        "Saved",
        _snapshot(),
        None,
        None,
        action="cancel",
        **_switch_fields(),
        store=store,
    )

    assert outputs[0]["value"] is None
    assert outputs[24]["visible"] is False
    assert outputs[25] is None
    assert "Kept current settings" in outputs[23]


def test_discard_preset_switch_loads_pending_preset(tmp_path):
    store = Store(tmp_path)
    store.save_preset(
        "Saved", collect_preset_state(**_collect_args(description="Saved voice."))
    )

    outputs = resolve_preset_switch(
        "Saved",
        _snapshot(),
        None,
        None,
        action="discard",
        **_switch_fields(description="Unsaved edit."),
        store=store,
    )

    assert outputs[0]["value"] == "Saved"
    assert outputs[1]["value"] == "Saved"
    assert outputs[3] == "Saved voice."
    assert outputs[24]["visible"] is False
    assert outputs[27] == "Saved"


def test_save_and_switch_preset_saves_current_state(tmp_path):
    store = Store(tmp_path)
    store.save_preset(
        "Saved", collect_preset_state(**_collect_args(description="Saved voice."))
    )

    outputs = resolve_preset_switch(
        "Saved",
        _snapshot(),
        None,
        None,
        action="save",
        name="Fresh",
        **_fields(description="Fresh voice."),
        store=store,
    )

    assert store.load_preset("Fresh") is not None
    assert outputs[0]["value"] == "Saved"
    assert "Saved **Fresh**" in outputs[23]
    assert outputs[24]["visible"] is False
    assert outputs[27] == "Saved"


def test_save_and_switch_requires_a_name(tmp_path):
    store = Store(tmp_path)

    outputs = resolve_preset_switch(
        "Saved",
        _snapshot(),
        None,
        None,
        action="save",
        name="",
        **_fields(),
        store=store,
    )

    assert "Enter a preset name" in outputs[23]
    assert outputs[24]["visible"] is True


def test_save_preset_handler_returns_download_path(tmp_path):
    store = Store(tmp_path)

    outputs = save_preset_handler(
        **_collect_args(name="Handler"), last_seed=456, store=store
    )

    assert len(outputs) == 6
    assert outputs[0]["value"] == "Handler"
    assert outputs[4] == "Handler"
    assert outputs[3]["description"] == "A calm narrator."
    assert outputs[5]["interactive"] is True
    assert outputs[5]["value"].endswith(".json")
    assert store.preset_path("Handler").is_file()


def test_refresh_presets_handler_loads_from_disk(tmp_path):
    store = Store(tmp_path)
    store.save_preset(
        "From disk", collect_preset_state(**_collect_args(name="From disk"))
    )

    outputs = refresh_presets_handler(None, store=store, announce=True)

    assert "From disk" in outputs[0]["choices"]
    assert "refreshed" in outputs[1]
    assert outputs[2]["interactive"] is False

    selected = refresh_presets_handler("From disk", store=store)
    assert selected[0]["value"] == "From disk"
    assert selected[2]["interactive"] is True


def test_switch_partials_accept_positional_inputs(tmp_path):
    """Gradio passes event inputs positionally; bound keywords must not clash."""

    store = Store(tmp_path)
    for action in ("save", "discard", "cancel"):
        handler = partial(resolve_preset_switch, action=action, store=store)
        outputs = handler("Saved", _snapshot(), None, None, **_switch_fields())
        assert len(outputs) == 29


def test_handler_output_arities(tmp_path):
    store = Store(tmp_path)

    assert (
        len(select_preset_handler(None, {}, None, **_switch_fields(), store=store))
        == 29
    )
    assert (
        len(
            resolve_preset_switch(
                None,
                {},
                None,
                None,
                action="cancel",
                **_switch_fields(),
                store=store,
            )
        )
        == 29
    )
    assert (
        len(
            save_preset_handler(
                **_collect_args(name="Arity"), last_seed=None, store=store
            )
        )
        == 6
    )
    assert len(delete_preset_handler(None, None, store=store)) == 5
    assert len(import_preset_handler(None, store=store)) == 30
    assert len(clear_handler()) == 9
    assert len(on_voice_change(VOICE_DESCRIBE)) == 3
    assert len(on_seed_mode_change("fixed")) == 2
    assert len(refresh_presets_handler(None, store=store)) == 3


def test_cancel_token_calls_closers_once():
    token = GenerationCancelToken()
    calls = []
    token.add_closer(lambda: calls.append("first"))

    token.cancel()
    token.cancel()

    assert token.cancelled
    assert calls == ["first"]


def test_cancel_token_closes_late_registrations_immediately():
    token = GenerationCancelToken()
    token.cancel()

    calls = []
    token.add_closer(lambda: calls.append("closed"))

    assert calls == ["closed"]


def test_stop_generation_cancels_the_active_request(tmp_path):
    control = GenerationControl()
    token = control.begin()
    token.add_closer(lambda: None)

    outputs = stop_generation(control=control)

    assert token.cancelled
    assert control.cancel() is False
    assert outputs[2] == "⏹ Stopped."


class _SlowStreamHandler(http.server.BaseHTTPRequestHandler):
    """Stream PCM slowly until the client asks the API to cancel."""

    cancelled = threading.Event()

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/v1/cancel":
            _SlowStreamHandler.cancelled.set()
            body = b'{"cancelled": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Sample-Rate", "24000")
        self.end_headers()
        try:
            for _ in range(500):
                if _SlowStreamHandler.cancelled.is_set():
                    return
                self.wfile.write(b"\x00\x00" * 2400)
                self.wfile.flush()
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _request_kwargs(api_url):
    return {
        "text": "hello",
        "instruction": "A calm narrator.",
        "reference_bytes": None,
        "reference_filename": None,
        "reference_text": "",
        "cfg_scale": 4.0,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_new_tokens": 100,
        "seed": 42,
        "api_url": api_url,
        "timeout": 30.0,
    }


def test_request_once_aborts_when_cancelled():
    _SlowStreamHandler.cancelled.clear()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowStreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = GenerationCancelToken()
    try:
        timer = threading.Timer(0.3, token.cancel)
        timer.start()
        started = time.monotonic()
        with pytest.raises(GenerationCancelled):
            _request_once(
                **_request_kwargs(f"http://127.0.0.1:{server.server_port}"),
                cancel=token,
            )
        assert time.monotonic() - started < 5
    finally:
        token.cancel()
        server.shutdown()
        server.server_close()

    # The client must tell the API to stop, or the model stays busy.
    assert _SlowStreamHandler.cancelled.is_set()


def test_request_once_reports_busy(monkeypatch):
    class _Busy:
        status_code = 409
        headers: ClassVar[dict] = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, *args, **kwargs):
            return _Busy()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(webui.httpx, "Client", _Client)
    monkeypatch.setattr(webui, "BUSY_RETRY_SECONDS", 0.0)

    with pytest.raises(GenerationError, match="busy"):
        _request_once(**_request_kwargs("http://127.0.0.1:1"))


def test_request_once_retries_while_the_model_is_busy(monkeypatch):
    class _Busy:
        status_code = 409

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Ok:
        status_code = 200
        headers: ClassVar[dict] = {"X-Sample-Rate": "24000"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_bytes(self):
            yield b"\x00\x00" * 240

    responses = [_Busy(), _Ok()]

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, *args, **kwargs):
            return responses.pop(0)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(webui.httpx, "Client", _Client)
    monkeypatch.setattr(webui, "BUSY_RETRY_INTERVAL", 0.0)

    (rate, audio), duration = _request_once(**_request_kwargs("http://127.0.0.1:1"))

    assert rate == 24000
    assert audio.shape == (240,)
    assert duration == pytest.approx(0.01)
