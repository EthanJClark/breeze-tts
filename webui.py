"""Gradio web UI for Breeze TTS 2.

The UI does not load the model. It sends requests to a running Breeze TTS
HTTP API, which can be local or remote:

    python -m breeze_infer.api /path/to/breeze-tts-2 --port 7860
    python webui.py --api-url http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import base64
import os
import secrets
import time
from functools import partial
from pathlib import Path

import gradio as gr
import httpx
import numpy as np

from webui_store import (
    DEFAULT_DATA_DIR,
    DEFAULT_SETTINGS,
    VOICE_CLONE,
    VOICE_DESCRIBE,
    Store,
)

DEFAULT_API_URL = "http://127.0.0.1:7860"
DEFAULT_SAMPLE_RATE = 24000
FALLBACK_FRAME_RATE = 12.5
MAX_NEW_TOKENS = 1500
DEFAULT_TIMEOUT = 600.0

EVENT_TAGS = [
    "(laugh)",
    "(sigh)",
    "(cough)",
    "(clears throat)",
]


class GenerationError(Exception):
    """A user-facing synthesis error."""


def _mode(has_reference: bool, has_instruction: bool) -> str:
    if has_reference:
        return "Voice direction" if has_instruction else "Voice clone"
    return "Voice design" if has_instruction else "Plain TTS"


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail or response.text).strip() or f"HTTP {response.status_code}"


def _resolve_voice_inputs(
    voice: str,
    description: str,
    reference_audio: str | None,
    reference_text: str,
    delivery: str,
) -> tuple[str, bytes | None, str | None, str]:
    """Return (instruction, reference audio, filename, transcript).

    Raises GenerationError with a user-facing message.
    """
    voice = (voice or VOICE_DESCRIBE).strip()

    if voice != VOICE_CLONE:
        return (description or "").strip(), None, None, ""

    instruction = (delivery or "").strip()
    path = Path(reference_audio) if reference_audio else None
    transcript = (reference_text or "").strip()
    if path is None:
        raise GenerationError(
            "Upload or record a reference clip, or choose Describe a voice."
        )
    if not transcript:
        raise GenerationError("The reference clip needs its exact transcript.")
    try:
        return instruction, path.read_bytes(), path.name, transcript
    except OSError as exc:
        raise GenerationError(f"Cannot read reference audio: {exc}") from exc


def _summary(
    mode: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_duration: float,
    takes: int,
    seed_mode: str,
    seed: int,
) -> str:
    seed_text = f"seed {seed}" if seed_mode == "fixed" else "random seed"
    return (
        f"Current settings · {mode} · CFG {cfg_scale:g} · temp {temperature:.2f} · "
        f"top-p {top_p:.2f} · top-k {top_k} · rep {repetition_penalty:.2f} · "
        f"max {max_duration:g}s · {seed_text} · "
        f"{takes} take{'s' if takes != 1 else ''}"
    )


def _fetch_frame_rate(api_url: str) -> float:
    try:
        response = httpx.get(f"{api_url}/v1/model", timeout=3.0)
        response.raise_for_status()
        return float(response.json()["frame_rate"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return FALLBACK_FRAME_RATE


def _request_once(
    *,
    text: str,
    instruction: str,
    reference_bytes: bytes | None,
    reference_filename: str | None,
    reference_text: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    seed: int,
    api_url: str,
    timeout: float,
) -> tuple[tuple[int, np.ndarray], float]:
    data = {
        "text": text,
        "cfg_scale": str(cfg_scale),
        "temperature": str(temperature),
        "top_p": str(top_p),
        "top_k": str(top_k),
        "repetition_penalty": str(repetition_penalty),
        "max_new_tokens": str(max_new_tokens),
        "seed": str(seed),
    }
    if instruction:
        data["instruction"] = instruction

    files = None
    if reference_bytes is not None:
        data["ref_text"] = reference_text
        files = {
            "ref_audio": (
                reference_filename or "reference.wav",
                reference_bytes,
                "application/octet-stream",
            )
        }

    timeout_config = httpx.Timeout(connect=5.0, read=timeout, write=60.0, pool=5.0)
    try:
        with httpx.Client(timeout=timeout_config) as client:
            response = client.post(f"{api_url}/v1/audio/speech", data=data, files=files)
    except httpx.ConnectError as exc:
        raise GenerationError(
            f"Cannot reach the Breeze TTS API at `{api_url}`. Start it first."
        ) from exc
    except httpx.TimeoutException as exc:
        raise GenerationError(f"The API timed out after {timeout:g} seconds.") from exc
    except httpx.HTTPError as exc:
        raise GenerationError(f"API request failed: {exc}") from exc

    if response.status_code == 409:
        raise GenerationError("The model is busy with another request. Try again.")
    if response.status_code >= 400:
        raise GenerationError(_error_detail(response))

    pcm = np.frombuffer(response.content, dtype="<i2")
    if pcm.size == 0:
        raise GenerationError("The API returned no audio.")

    sample_rate = int(response.headers.get("X-Sample-Rate", DEFAULT_SAMPLE_RATE))
    audio = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return (sample_rate, audio), pcm.size / sample_rate


def _audio_output(audio, *, reset: bool = False):
    """Update the audio output, optionally forcing playback back to 0:00."""
    if reset:
        return gr.update(value=audio, playback_position=0)
    return gr.update()


def _generation_outputs(
    audio,
    labels: list[str],
    results: list[dict],
    status: str,
    summary: str,
    *,
    busy: bool,
    reset_audio: bool = False,
    last_seed: int | None = None,
):
    selected = labels[-1] if labels else None
    return (
        _audio_output(audio, reset=reset_audio),
        gr.update(choices=labels, value=selected, visible=len(labels) > 1),
        results,
        status,
        summary,
        gr.update(interactive=not busy),
        gr.update(interactive=busy),
        last_seed,
        gr.update(value=str(last_seed) if last_seed is not None else ""),
    )


def run_generation(
    text: str,
    voice: str,
    description: str,
    reference_audio: str | None,
    reference_text: str,
    delivery: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_duration: float,
    takes: float,
    seed_mode: str,
    seed: float,
    timeout: float,
    *,
    api_url: str,
    frame_rate: float,
):
    text = (text or "").strip()

    if not text:
        yield _generation_outputs(
            None,
            [],
            [],
            "❌ Enter some text to speak.",
            "",
            busy=False,
            reset_audio=True,
        )
        return

    try:
        instruction, reference_bytes, reference_filename, resolved_ref_text = (
            _resolve_voice_inputs(
                voice,
                description,
                reference_audio,
                reference_text,
                delivery,
            )
        )
    except GenerationError as exc:
        yield _generation_outputs(
            None, [], [], f"❌ {exc}", "", busy=False, reset_audio=True
        )
        return

    has_instruction = bool(instruction)
    effective_cfg = float(cfg_scale) if has_instruction else 1.0
    mode = _mode(reference_bytes is not None, has_instruction)

    try:
        take_count = max(1, min(4, int(takes)))
        max_duration_value = float(max_duration)
        timeout_value = float(timeout)
        seed_value = int(seed)
    except (TypeError, ValueError):
        yield _generation_outputs(
            None,
            [],
            [],
            "❌ Invalid generation settings.",
            "",
            busy=False,
            reset_audio=True,
        )
        return

    max_new_tokens = max(1, min(MAX_NEW_TOKENS, round(max_duration_value * frame_rate)))
    if seed_mode == "random":
        seeds = [secrets.randbelow(2**31) for _ in range(take_count)]
    else:
        seeds = [(seed_value + index) % (2**31) for index in range(take_count)]

    summary = _summary(
        mode,
        effective_cfg,
        float(temperature),
        float(top_p),
        int(top_k),
        float(repetition_penalty),
        max_duration_value,
        take_count,
        seed_mode,
        seed_value,
    )
    if not has_instruction and float(cfg_scale) != 1.0:
        summary += " · CFG forced to 1 (no instruction)"

    results: list[dict] = []
    started = time.perf_counter()
    yield _generation_outputs(
        None,
        [],
        results,
        f"⏳ Generating {take_count} take{'s' if take_count != 1 else ''}…",
        summary,
        busy=True,
        reset_audio=True,
    )

    for index, take_seed in enumerate(seeds, start=1):
        yield _generation_outputs(
            results[-1]["audio"] if results else None,
            [item["label"] for item in results],
            results,
            f"⏳ Take {index}/{take_count} · seed {take_seed}…",
            summary,
            busy=True,
            reset_audio=not results,
            last_seed=results[-1]["seed"] if results else None,
        )
        try:
            audio, duration = _request_once(
                text=text,
                instruction=instruction,
                reference_bytes=reference_bytes,
                reference_filename=reference_filename,
                reference_text=resolved_ref_text,
                cfg_scale=effective_cfg,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                max_new_tokens=max_new_tokens,
                seed=take_seed,
                api_url=api_url,
                timeout=timeout_value,
            )
        except GenerationError as exc:
            yield _generation_outputs(
                results[-1]["audio"] if results else None,
                [item["label"] for item in results],
                results,
                f"❌ {exc}",
                summary,
                busy=False,
                reset_audio=not results,
                last_seed=results[-1]["seed"] if results else None,
            )
            return

        label = f"Take {index} · seed {take_seed}"
        results.append(
            {
                "label": label,
                "audio": audio,
                "seed": take_seed,
                "duration": duration,
            }
        )
        yield _generation_outputs(
            audio,
            [item["label"] for item in results],
            results,
            f"✅ {label} · {duration:.2f} s",
            summary,
            busy=True,
            reset_audio=True,
            last_seed=take_seed,
        )

    elapsed = time.perf_counter() - started
    yield _generation_outputs(
        results[-1]["audio"],
        [item["label"] for item in results],
        results,
        f"✅ {take_count} take{'s' if take_count != 1 else ''} in {elapsed:.1f} s",
        summary,
        busy=False,
        last_seed=results[-1]["seed"],
    )


def stop_generation():
    return (
        gr.update(interactive=True),
        gr.update(interactive=False),
        "⏹ Stopped.",
    )


def select_take(label: str | None, results: list[dict] | None):
    if not label or not results:
        return gr.update(), "Select a take.", gr.update(), gr.update()
    for take in results:
        if take["label"] == label:
            return (
                _audio_output(take["audio"], reset=True),
                f"Take {take['seed']} · {take['duration']:.2f} s",
                take["seed"],
                gr.update(value=str(take["seed"])),
            )
    return (
        gr.update(),
        "❌ That take is no longer available.",
        gr.update(),
        gr.update(),
    )


def clear_handler():
    return (
        "",
        _audio_output(None, reset=True),
        gr.update(choices=[], value=None, visible=False),
        [],
        None,
        gr.update(value=""),
        "Cleared. Ready.",
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def on_seed_mode_change(mode: str | None):
    random_mode = (mode or "fixed") == "random"
    return gr.update(visible=not random_mode), gr.update(visible=random_mode)


def _voice_ui(selection: str | None):
    """Visibility updates for the voice-specific fields."""
    if (selection or VOICE_DESCRIBE).strip() == VOICE_CLONE:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def on_voice_change(selection: str | None):
    return _voice_ui(selection)


def collect_preset_state(
    name: str,
    voice: str,
    description: str,
    reference_audio: str | None,
    reference_text: str,
    delivery: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_duration: float,
    takes: float,
    seed_mode: str,
    seed: float,
    timeout: float,
    autoplay: bool,
    last_seed: float | None = None,
) -> dict:
    name = (name or "").strip()
    if not name:
        raise GenerationError("Enter a preset name.")

    voice = (voice or VOICE_DESCRIBE).strip()
    description = (description or "").strip()
    delivery = (delivery or "").strip()
    reference = None

    if voice == VOICE_CLONE:
        audio = reference_audio or None
        ref_text = (reference_text or "").strip()
        if audio and not ref_text:
            raise GenerationError(
                "The reference clip needs its exact transcript before saving."
            )
        if ref_text and not audio:
            raise GenerationError("The reference transcript needs an audio clip.")
        if audio:
            try:
                data = Path(audio).read_bytes()
            except OSError as exc:
                raise GenerationError(f"Cannot read reference audio: {exc}") from exc
            reference = {
                "filename": Path(audio).name,
                "text": ref_text,
                "audio_b64": base64.b64encode(data).decode("ascii"),
            }
    else:
        voice = VOICE_DESCRIBE

    if not description and not delivery and reference is None:
        raise GenerationError(
            "Add a voice description or a reference clip before saving."
        )

    effective_seed = int(seed)
    if seed_mode == "random" and last_seed is not None:
        try:
            effective_seed = int(last_seed)
        except (TypeError, ValueError):
            pass

    return {
        "voice_source": voice,
        "description": description,
        "delivery": delivery,
        "reference": reference,
        "settings": {
            "cfg_scale": cfg_scale,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_duration": max_duration,
            "takes": takes,
            "seed_mode": seed_mode,
            "seed": effective_seed,
            "timeout": timeout,
            "autoplay": autoplay,
        },
    }


def apply_preset_state(name: str, preset: dict, *, store: Store):
    """Return component updates in the order used by the preset wiring."""
    voice = str(preset.get("voice_source") or VOICE_DESCRIBE).strip()
    reference = preset.get("reference") or {}
    audio_path = None
    ref_text = ""
    message = f"✅ Loaded preset **{name}**."

    if voice not in (VOICE_DESCRIBE, VOICE_CLONE):
        # Older presets may name a saved voice. The bundled clip replaces it.
        if reference.get("audio_b64"):
            voice = VOICE_CLONE
            message = (
                f"⚠️ Saved voice **{preset.get('voice_source')}** was replaced "
                "by the preset's bundled clip."
            )
        else:
            voice = VOICE_DESCRIBE
            message = (
                f"⚠️ Saved voice **{preset.get('voice_source')}** is unavailable "
                "and the preset has no bundled clip."
            )

    if voice == VOICE_CLONE and reference.get("audio_b64"):
        path = store.materialize_reference(name, preset)
        if path is not None:
            audio_path = str(path)
            ref_text = str(reference.get("text") or "")

    settings = preset.get("settings") or {}
    defaults = DEFAULT_SETTINGS

    def value(key):
        return settings.get(key, defaults[key])

    seed_mode_value = str(value("seed_mode"))
    saved_seed = value("seed")
    random_mode = seed_mode_value == "random"
    last_seed = saved_seed if random_mode else None

    updates = (
        gr.update(value=name),
        gr.update(value=voice),
        str(preset.get("description") or ""),
        audio_path,
        ref_text,
        str(preset.get("delivery") or ""),
        value("cfg_scale"),
        value("temperature"),
        value("top_p"),
        value("top_k"),
        value("repetition_penalty"),
        value("max_duration"),
        value("takes"),
        seed_mode_value,
        gr.update(value=saved_seed, visible=not random_mode),
        value("timeout"),
        bool(value("autoplay")),
        *_voice_ui(voice),
        last_seed,
        gr.update(
            value=str(saved_seed) if random_mode else "",
            visible=random_mode,
        ),
    )
    return updates, message


PRESET_STATE_UPDATE_COUNT = 22


def preset_snapshot(
    voice: str,
    description: str,
    reference_audio: str | None,
    reference_text: str,
    delivery: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_duration: float,
    takes: float,
    seed_mode: str,
    seed: float,
    timeout: float,
    autoplay: bool,
) -> dict:
    """Return a comparable snapshot of the preset-relevant UI fields."""

    def number(value, default: float) -> float:
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return default

    return {
        "voice_source": (voice or VOICE_DESCRIBE).strip(),
        "description": (description or "").strip(),
        "reference_audio": str(reference_audio) if reference_audio else None,
        "reference_text": (reference_text or "").strip(),
        "delivery": (delivery or "").strip(),
        "cfg_scale": number(cfg_scale, DEFAULT_SETTINGS["cfg_scale"]),
        "temperature": number(temperature, DEFAULT_SETTINGS["temperature"]),
        "top_p": number(top_p, DEFAULT_SETTINGS["top_p"]),
        "top_k": int(number(top_k, DEFAULT_SETTINGS["top_k"])),
        "repetition_penalty": number(
            repetition_penalty, DEFAULT_SETTINGS["repetition_penalty"]
        ),
        "max_duration": number(max_duration, DEFAULT_SETTINGS["max_duration"]),
        "takes": int(number(takes, DEFAULT_SETTINGS["takes"])),
        "seed_mode": (seed_mode or "fixed").strip(),
        "seed": int(number(seed, DEFAULT_SETTINGS["seed"])),
        "timeout": number(timeout, DEFAULT_SETTINGS["timeout"]),
        "autoplay": bool(autoplay),
    }


def snapshot_from_updates(updates: tuple) -> dict:
    """Build a baseline snapshot from apply_preset_state's field updates."""
    return preset_snapshot(
        updates[1]["value"],
        updates[2],
        updates[3],
        updates[4],
        updates[5],
        updates[6],
        updates[7],
        updates[8],
        updates[9],
        updates[10],
        updates[11],
        updates[12],
        updates[13],
        updates[14]["value"],
        updates[15],
        updates[16],
    )


def _preset_no_change(count: int = PRESET_STATE_UPDATE_COUNT):
    return tuple(gr.update() for _ in range(count))


def _download_update(store: Store, name: str | None):
    path = store.preset_path(name)
    if path is None:
        return gr.update(value=None, interactive=False)
    return gr.update(value=str(path), interactive=True)


def _apply_preset_outputs(name: str, preset: dict, *, store: Store):
    updates, message = apply_preset_state(name, preset, store=store)
    return updates, message, _download_update(store, name)


def _preset_switch_outputs(
    dropdown_name: str | None,
    updates: tuple,
    message: str,
    *,
    pending: str | None,
    baseline: dict,
    loaded: str | None,
    download,
):
    return (
        gr.update(value=dropdown_name),
        *updates,
        message,
        gr.update(visible=False),
        pending,
        baseline,
        loaded,
        download,
    )


def select_preset_handler(
    selected: str | None,
    baseline: dict,
    loaded: str | None,
    name: str,
    voice: str,
    description: str,
    reference_audio: str | None,
    reference_text: str,
    delivery: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_duration: float,
    takes: float,
    seed_mode: str,
    seed: float,
    timeout: float,
    autoplay: bool,
    *,
    store: Store,
):
    if not selected:
        return (
            gr.update(),
            *_preset_no_change(),
            "❌ Select a preset to load.",
            gr.update(),
            None,
            baseline,
            loaded,
            gr.update(),
        )

    preset = store.load_preset(selected)
    if preset is None:
        return (
            gr.update(value=loaded),
            *_preset_no_change(),
            "❌ Preset not found.",
            gr.update(visible=False),
            None,
            baseline,
            loaded,
            gr.update(),
        )

    current = preset_snapshot(
        voice,
        description,
        reference_audio,
        reference_text,
        delivery,
        cfg_scale,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_duration,
        takes,
        seed_mode,
        seed,
        timeout,
        autoplay,
    )
    if current == baseline:
        updates, message, download = _apply_preset_outputs(
            selected, preset, store=store
        )
        return _preset_switch_outputs(
            selected,
            updates,
            message,
            pending=None,
            baseline=snapshot_from_updates(updates),
            loaded=selected,
            download=download,
        )

    return (
        gr.update(value=loaded),
        *_preset_no_change(),
        f"⚠️ Unsaved changes. Save before loading **{selected}**?",
        gr.update(visible=True),
        selected,
        baseline,
        loaded,
        gr.update(),
    )


def resolve_preset_switch(
    pending: str | None,
    baseline: dict,
    loaded: str | None,
    last_seed: float | None,
    name: str,
    voice: str,
    description: str,
    reference_audio: str | None,
    reference_text: str,
    delivery: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_duration: float,
    takes: float,
    seed_mode: str,
    seed: float,
    timeout: float,
    autoplay: bool,
    *,
    action: str,
    store: Store,
):
    if not pending:
        return (
            gr.update(value=loaded),
            *_preset_no_change(),
            "Select a preset to load.",
            gr.update(visible=False),
            None,
            baseline,
            loaded,
            gr.update(),
        )

    if action == "cancel":
        return (
            gr.update(value=loaded),
            *_preset_no_change(),
            "Kept current settings.",
            gr.update(visible=False),
            None,
            baseline,
            loaded,
            gr.update(),
        )

    saved_name = None
    if action == "save":
        try:
            payload = collect_preset_state(
                name,
                voice,
                description,
                reference_audio,
                reference_text,
                delivery,
                cfg_scale,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_duration,
                takes,
                seed_mode,
                seed,
                timeout,
                autoplay,
                last_seed=last_seed,
            )
            saved_name = store.save_preset(name, payload)
        except (GenerationError, ValueError, OSError) as exc:
            return (
                gr.update(value=loaded),
                *_preset_no_change(),
                f"❌ {exc}",
                gr.update(visible=True),
                pending,
                baseline,
                loaded,
                gr.update(),
            )

    preset = store.load_preset(pending)
    if preset is None:
        return (
            gr.update(value=loaded),
            *_preset_no_change(),
            "❌ Preset not found.",
            gr.update(visible=True),
            pending,
            baseline,
            loaded,
            gr.update(),
        )

    updates, message, download = _apply_preset_outputs(pending, preset, store=store)
    if saved_name:
        message = f"✅ Saved **{saved_name}** and loaded **{pending}**."
    return _preset_switch_outputs(
        pending,
        updates,
        message,
        pending=None,
        baseline=snapshot_from_updates(updates),
        loaded=pending,
        download=download,
    )


def save_preset_handler(
    name: str,
    voice: str,
    description: str,
    reference_audio: str | None,
    reference_text: str,
    delivery: str,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_duration: float,
    takes: float,
    seed_mode: str,
    seed: float,
    timeout: float,
    autoplay: bool,
    last_seed: float | None,
    *,
    store: Store,
):
    try:
        payload = collect_preset_state(
            name,
            voice,
            description,
            reference_audio,
            reference_text,
            delivery,
            cfg_scale,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_duration,
            takes,
            seed_mode,
            seed,
            timeout,
            autoplay,
            last_seed=last_seed,
        )
        stored = store.save_preset(name, payload)
    except (GenerationError, ValueError, OSError) as exc:
        return (
            gr.update(),
            gr.update(),
            f"❌ {exc}",
            gr.update(),
            gr.update(),
            gr.update(),
        )

    presets, _ = store.list_presets()
    baseline = preset_snapshot(
        voice,
        description,
        reference_audio,
        reference_text,
        delivery,
        cfg_scale,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_duration,
        takes,
        seed_mode,
        seed,
        timeout,
        autoplay,
    )
    return (
        gr.update(choices=sorted(presets, key=str.casefold), value=stored),
        gr.update(value=stored),
        f"✅ Saved preset **{stored}**.",
        baseline,
        stored,
        _download_update(store, stored),
    )


def delete_preset_handler(
    name: str | None, loaded: str | None, *, store: Store
):
    if not name:
        return (
            gr.update(),
            gr.update(),
            "❌ Select a preset to delete.",
            loaded,
            gr.update(),
        )
    try:
        deleted = store.delete_preset(name)
    except OSError as exc:
        return (
            gr.update(),
            gr.update(),
            f"❌ Could not delete: {exc}",
            loaded,
            gr.update(),
        )
    if not deleted:
        return (
            gr.update(),
            gr.update(),
            "❌ Preset not found.",
            loaded,
            gr.update(),
        )
    presets, _ = store.list_presets()
    deleting_loaded = loaded == name
    next_loaded = None if deleting_loaded else loaded
    return (
        gr.update(
            choices=sorted(presets, key=str.casefold), value=next_loaded
        ),
        gr.update(value="") if deleting_loaded else gr.update(),
        f"✅ Deleted preset **{name}**.",
        next_loaded,
        _download_update(store, next_loaded),
    )


def import_preset_handler(file_path: str | None, *, store: Store):
    if not file_path:
        return (*(gr.update() for _ in range(29)), "❌ Choose a JSON file.")
    try:
        raw = Path(file_path).read_bytes()
        name = store.import_preset(raw, Path(file_path).name)
    except (OSError, ValueError) as exc:
        return (*(gr.update() for _ in range(29)), f"❌ {exc}")

    presets, _ = store.list_presets()
    preset = presets.get(name)
    if preset is None:
        return (
            *(gr.update() for _ in range(29)),
            "❌ Imported preset could not be read.",
        )
    updates, message, download = _apply_preset_outputs(name, preset, store=store)
    return (
        gr.update(choices=sorted(presets, key=str.casefold), value=name),
        gr.update(value=None),
        *updates,
        message,
        gr.update(visible=False),
        None,
        snapshot_from_updates(updates),
        name,
        download,
    )


def refresh_presets_handler(
    current: str | None, *, store: Store, announce: bool = False
):
    presets, warning = store.list_presets()
    choices = sorted(presets, key=str.casefold)
    value = current if current in presets else None
    if warning:
        message = warning
    elif announce:
        message = "✅ Presets refreshed from disk."
    else:
        message = gr.update()
    return (
        gr.update(choices=choices, value=value),
        message,
        _download_update(store, value),
    )


def save_settings_handler(
    cfg_scale,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    max_duration,
    takes,
    seed_mode,
    seed,
    timeout,
    autoplay,
    *,
    store: Store,
):
    try:
        store.save_settings(
            {
                "cfg_scale": cfg_scale,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "max_duration": max_duration,
                "takes": takes,
                "seed_mode": seed_mode,
                "seed": seed,
                "timeout": timeout,
                "autoplay": autoplay,
            }
        )
    except OSError as exc:
        return f"❌ Could not save settings: {exc}"
    return "✅ Settings saved."


def reset_settings_handler(*, store: Store):
    values = dict(DEFAULT_SETTINGS)
    return (
        values["cfg_scale"],
        values["temperature"],
        values["top_p"],
        values["top_k"],
        values["repetition_penalty"],
        values["max_duration"],
        values["takes"],
        values["seed_mode"],
        gr.update(value=values["seed"], visible=True),
        values["timeout"],
        values["autoplay"],
        "Reset to defaults. Click Save settings to persist.",
        None,
        gr.update(value="", visible=False),
    )


def build_demo(api_url: str, store: Store, frame_rate: float) -> gr.Blocks:
    settings, settings_warning = store.load_settings()
    presets, presets_warning = store.list_presets()

    preset_choices = sorted(presets, key=str.casefold)
    initial_baseline = preset_snapshot(
        VOICE_DESCRIBE,
        "",
        None,
        "",
        "",
        settings["cfg_scale"],
        settings["temperature"],
        settings["top_p"],
        settings["top_k"],
        settings["repetition_penalty"],
        settings["max_duration"],
        settings["takes"],
        settings["seed_mode"],
        settings["seed"],
        settings["timeout"],
        settings["autoplay"],
    )
    initial_last_seed = (
        settings["seed"] if settings["seed_mode"] == "random" else None
    )
    initial_summary = _summary(
        "Voice design",
        settings["cfg_scale"],
        settings["temperature"],
        settings["top_p"],
        settings["top_k"],
        settings["repetition_penalty"],
        settings["max_duration"],
        settings["takes"],
        settings["seed_mode"],
        settings["seed"],
    )
    event_guide = ", ".join(f"`{tag}`" for tag in EVENT_TAGS[:-1])
    event_guide = f"{event_guide}, or `{EVENT_TAGS[-1]}`"

    with gr.Blocks(title="Breeze TTS 2") as demo:
        gr.Markdown("# Breeze TTS 2\nType a script, choose a voice, then generate.")

        with gr.Tabs():
            with gr.Tab("Generate"):  # noqa: SIM117
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### 1 · Script")
                        text = gr.Textbox(
                            label="Script",
                            show_label=False,
                            placeholder="Type the text to synthesize...",
                            lines=7,
                            autofocus=True,
                        )

                        gr.Markdown("### 2 · Voice")
                        voice_source = gr.Dropdown(
                            label="Voice",
                            show_label=False,
                            choices=[VOICE_DESCRIBE, VOICE_CLONE],
                            value=VOICE_DESCRIBE,
                            filterable=False,
                        )
                        gr.Markdown(
                            "Describe a voice in words, or clone one from a clip."
                        )

                        with gr.Column(visible=True) as describe_group:
                            description = gr.Textbox(
                                label="Voice & delivery",
                                placeholder=(
                                    "Age, timbre, accent, and delivery. E.g. a weathered "
                                    "baritone in his early sixties, warm and a little "
                                    "mischievous. Speaks slowly, leaning in for the secret."
                                ),
                                lines=5,
                            )

                        with gr.Column(visible=False) as clone_group:
                            reference_audio = gr.Audio(
                                label="Reference clip",
                                sources=["upload", "microphone"],
                                type="filepath",
                            )
                            reference_text = gr.Textbox(
                                label="Exact transcript of the clip",
                                placeholder="Every word spoken in the clip.",
                                lines=3,
                            )

                        with gr.Column(visible=False) as delivery_group:
                            delivery = gr.Textbox(
                                label="Delivery direction (optional)",
                                placeholder=(
                                    "Emotion, pace, dynamics. E.g. delighted and a little "
                                    "mischievous, leaning in for the secret."
                                ),
                                lines=3,
                            )

                        gr.Markdown("### Presets")
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label="Preset",
                                choices=preset_choices,
                                value=None,
                                filterable=True,
                                scale=2,
                            )
                            preset_name = gr.Textbox(
                                label="Preset name",
                                placeholder="Name this preset...",
                                scale=1,
                            )
                            refresh_presets_button = gr.Button(
                                "Refresh", size="sm", scale=0, min_width=80
                            )
                        with gr.Row():
                            save_preset_button = gr.Button(
                                "Save preset",
                                variant="primary",
                                size="sm",
                                scale=0,
                                min_width=110,
                            )
                            delete_preset_button = gr.Button(
                                "Delete", size="sm", scale=0, min_width=80
                            )
                            import_preset_button = gr.UploadButton(
                                "Import JSON",
                                file_types=[".json"],
                                size="sm",
                                scale=0,
                                min_width=110,
                            )
                            download_preset = gr.DownloadButton(
                                "Download JSON",
                                size="sm",
                                scale=0,
                                min_width=130,
                                interactive=False,
                            )
                        with gr.Group(visible=False) as preset_confirm_group:
                            gr.Markdown(
                                "You have unsaved changes. Save them before switching presets?"
                            )
                            with gr.Row():
                                save_switch_button = gr.Button(
                                    "Save & switch",
                                    variant="primary",
                                    size="sm",
                                    scale=0,
                                    min_width=110,
                                )
                                discard_switch_button = gr.Button(
                                    "Discard & switch",
                                    size="sm",
                                    scale=0,
                                    min_width=130,
                                )
                                cancel_switch_button = gr.Button(
                                    "Cancel", size="sm", scale=0, min_width=80
                                )
                        preset_status = gr.Markdown(
                            presets_warning
                            or (
                                f"Presets are stored in `{store.presets_dir}`. "
                                "The script is not included."
                            )
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 3 · Generate")
                        with gr.Row():
                            generate_button = gr.Button(
                                "Generate",
                                variant="primary",
                                size="md",
                                scale=0,
                                min_width=160,
                            )
                            stop_button = gr.Button(
                                "Stop",
                                variant="stop",
                                size="sm",
                                scale=0,
                                min_width=90,
                                interactive=False,
                            )
                            clear_button = gr.Button(
                                "Clear", size="sm", scale=0, min_width=80
                            )
                        output_audio = gr.Audio(
                            label="Generated audio",
                            type="numpy",
                            format="wav",
                            autoplay=settings["autoplay"],
                            interactive=False,
                        )
                        take_selector = gr.Dropdown(
                            label="Take", choices=[], value=None, visible=False
                        )
                        status = gr.Markdown("Ready.")
                        summary = gr.Markdown(initial_summary)
                        takes_state = gr.State([])
                        last_seed_state = gr.State(initial_last_seed)
                        baseline_state = gr.State(initial_baseline)
                        pending_preset_state = gr.State(None)
                        loaded_preset_state = gr.State(None)
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                f"**Vocal events** — type {event_guide} anywhere "
                                "in the script.\n\n"
                                "**Voice** — Describe types a voice in words; Clone "
                                "uses a clip plus its exact transcript.\n\n"
                                "**Presets** — Save preset stores the voice, reference "
                                "clip, and sampling settings, but not the script. "
                                "Download JSON grabs the saved file in one click.\n\n"
                                "**Seeds** — Random mode shows the last seed used. "
                                "Saving a preset in random mode stores that seed.\n\n"
                                f"API: `{api_url}`"
                            )

            with gr.Tab("Settings"):  # noqa: SIM117
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Sampling")
                        cfg_scale = gr.Slider(
                            1.0,
                            6.0,
                            settings["cfg_scale"],
                            step=0.5,
                            label="CFG scale",
                            info="Instruction strength. Forced to 1 when there is no instruction.",
                        )
                        temperature = gr.Slider(
                            0.5,
                            1.5,
                            settings["temperature"],
                            step=0.05,
                            label="Temperature",
                            info="Lower is steadier; higher is more varied. Above ~1.2 can be unstable.",
                        )
                        top_p = gr.Slider(
                            0.05,
                            1.0,
                            settings["top_p"],
                            step=0.05,
                            label="Top-p",
                        )
                        top_k = gr.Slider(
                            0,
                            100,
                            settings["top_k"],
                            step=1,
                            label="Top-k",
                            info="0 disables top-k filtering.",
                        )
                        repetition_penalty = gr.Slider(
                            1.0,
                            1.5,
                            settings["repetition_penalty"],
                            step=0.05,
                            label="Repetition penalty",
                        )
                        max_duration = gr.Slider(
                            5,
                            120,
                            settings["max_duration"],
                            step=5,
                            label="Max duration (seconds)",
                            info="Caps generation length. The model still stops at its natural end.",
                        )
                    with gr.Column():
                        gr.Markdown("### Generation")
                        takes = gr.Slider(
                            1,
                            4,
                            settings["takes"],
                            step=1,
                            label="Takes",
                            info="Sequential samples. Fixed seeds use seed, seed+1, ...",
                        )
                        seed_mode = gr.Radio(
                            ["fixed", "random"],
                            value=settings["seed_mode"],
                            label="Seed mode",
                        )
                        seed = gr.Number(
                            value=settings["seed"],
                            precision=0,
                            label="Fixed seed",
                            visible=settings["seed_mode"] != "random",
                        )
                        last_seed_display = gr.Textbox(
                            value=(
                                str(initial_last_seed)
                                if initial_last_seed is not None
                                else ""
                            ),
                            label="Last seed",
                            interactive=False,
                            visible=settings["seed_mode"] == "random",
                            info="The actual seed used by the most recent take.",
                        )
                        timeout = gr.Number(
                            value=settings["timeout"],
                            precision=0,
                            label="Request timeout (seconds)",
                        )
                        autoplay = gr.Checkbox(
                            value=settings["autoplay"], label="Autoplay result"
                        )
                        with gr.Row():
                            save_settings_button = gr.Button(
                                "Save settings",
                                variant="primary",
                                size="sm",
                                scale=0,
                                min_width=110,
                            )
                            reset_settings_button = gr.Button(
                                "Reset defaults", size="sm", scale=0, min_width=110
                            )
                        settings_status = gr.Markdown(
                            settings_warning
                            or "Values apply immediately. Save to persist across restarts."
                        )

        preset_field_components = [
            preset_name,
            voice_source,
            description,
            reference_audio,
            reference_text,
            delivery,
            cfg_scale,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_duration,
            takes,
            seed_mode,
            seed,
            timeout,
            autoplay,
        ]
        preset_state_outputs = [
            *preset_field_components,
            describe_group,
            clone_group,
            delivery_group,
            last_seed_state,
            last_seed_display,
        ]
        preset_switch_outputs = [
            preset_dropdown,
            *preset_state_outputs,
            preset_status,
            preset_confirm_group,
            pending_preset_state,
            baseline_state,
            loaded_preset_state,
            download_preset,
        ]

        preset_dropdown.select(
            partial(select_preset_handler, store=store),
            inputs=[
                preset_dropdown,
                baseline_state,
                loaded_preset_state,
                *preset_field_components,
            ],
            outputs=preset_switch_outputs,
            api_name="apply_preset",
        )
        save_preset_button.click(
            partial(save_preset_handler, store=store),
            inputs=[*preset_field_components, last_seed_state],
            outputs=[
                preset_dropdown,
                preset_name,
                preset_status,
                baseline_state,
                loaded_preset_state,
                download_preset,
            ],
            api_name="save_preset",
        )
        delete_preset_button.click(
            partial(delete_preset_handler, store=store),
            inputs=[preset_dropdown, loaded_preset_state],
            outputs=[
                preset_dropdown,
                preset_name,
                preset_status,
                loaded_preset_state,
                download_preset,
            ],
            api_name="delete_preset",
        )
        import_preset_button.upload(
            partial(import_preset_handler, store=store),
            inputs=import_preset_button,
            outputs=[
                preset_dropdown,
                import_preset_button,
                *preset_state_outputs,
                preset_status,
                preset_confirm_group,
                pending_preset_state,
                baseline_state,
                loaded_preset_state,
                download_preset,
            ],
            api_name="import_preset",
        )
        refresh_presets_button.click(
            partial(refresh_presets_handler, store=store, announce=True),
            inputs=preset_dropdown,
            outputs=[preset_dropdown, preset_status, download_preset],
            api_name="refresh_presets",
        )
        demo.load(
            partial(refresh_presets_handler, store=store),
            inputs=preset_dropdown,
            outputs=[preset_dropdown, preset_status, download_preset],
            api_visibility="private",
        )

        save_switch_button.click(
            partial(resolve_preset_switch, action="save", store=store),
            inputs=[
                pending_preset_state,
                baseline_state,
                loaded_preset_state,
                last_seed_state,
                *preset_field_components,
            ],
            outputs=preset_switch_outputs,
            api_name="save_and_switch_preset",
        )
        discard_switch_button.click(
            partial(resolve_preset_switch, action="discard", store=store),
            inputs=[
                pending_preset_state,
                baseline_state,
                loaded_preset_state,
                last_seed_state,
                *preset_field_components,
            ],
            outputs=preset_switch_outputs,
            api_name="discard_and_switch_preset",
        )
        cancel_switch_button.click(
            partial(resolve_preset_switch, action="cancel", store=store),
            inputs=[
                pending_preset_state,
                baseline_state,
                loaded_preset_state,
                last_seed_state,
                *preset_field_components,
            ],
            outputs=preset_switch_outputs,
            api_name="cancel_preset_switch",
        )

        voice_source.input(
            on_voice_change,
            inputs=voice_source,
            outputs=[describe_group, clone_group, delivery_group],
            api_name="change_voice_source",
        )

        event_inputs = [
            text,
            voice_source,
            description,
            reference_audio,
            reference_text,
            delivery,
            cfg_scale,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_duration,
            takes,
            seed_mode,
            seed,
            timeout,
        ]

        generate_event = generate_button.click(
            partial(
                run_generation,
                api_url=api_url,
                frame_rate=frame_rate,
            ),
            inputs=event_inputs,
            outputs=[
                output_audio,
                take_selector,
                takes_state,
                status,
                summary,
                generate_button,
                stop_button,
                last_seed_state,
                last_seed_display,
            ],
            api_name="generate",
        )
        stop_button.click(
            stop_generation,
            outputs=[generate_button, stop_button, status],
            cancels=[generate_event],
            queue=False,
            api_name="stop_generation",
        )
        clear_button.click(
            clear_handler,
            outputs=[
                text,
                output_audio,
                take_selector,
                takes_state,
                last_seed_state,
                last_seed_display,
                status,
                generate_button,
                stop_button,
            ],
            cancels=[generate_event],
            queue=False,
            api_name="clear",
        )
        take_selector.input(
            select_take,
            inputs=[take_selector, takes_state],
            outputs=[output_audio, status, last_seed_state, last_seed_display],
            api_name="select_take",
        )
        seed_mode.change(
            on_seed_mode_change,
            inputs=seed_mode,
            outputs=[seed, last_seed_display],
            api_name="change_seed_mode",
        )

        autoplay.change(
            lambda enabled: gr.update(autoplay=bool(enabled)),
            inputs=autoplay,
            outputs=output_audio,
            api_name="set_autoplay",
        )
        save_settings_button.click(
            partial(save_settings_handler, store=store),
            inputs=[
                cfg_scale,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_duration,
                takes,
                seed_mode,
                seed,
                timeout,
                autoplay,
            ],
            outputs=settings_status,
            api_name="save_settings",
        )
        reset_settings_button.click(
            partial(reset_settings_handler, store=store),
            outputs=[
                cfg_scale,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_duration,
                takes,
                seed_mode,
                seed,
                timeout,
                autoplay,
                settings_status,
                last_seed_state,
                last_seed_display,
            ],
            api_name="reset_settings",
        )

    return demo.queue(default_concurrency_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Web UI for Breeze TTS 2")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("BREEZE_API_URL", DEFAULT_API_URL),
        help="Base URL of the Breeze TTS API (default: %(default)s)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="UI bind address")
    parser.add_argument("--port", type=int, default=7861, help="UI port")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory for settings, presets, and saved voices. Defaults to "
            ".breeze2 next to webui.py (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link"
    )
    parser.add_argument(
        "--inbrowser", action="store_true", help="Open the UI in a browser"
    )
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    try:
        response = httpx.get(f"{api_url}/health", timeout=3.0)
        if response.status_code == 200:
            print(f"Breeze TTS API: {api_url} ({response.json().get('status')})")
        else:
            print(f"Warning: {api_url}/health returned HTTP {response.status_code}")
    except httpx.HTTPError as exc:
        print(f"Warning: cannot reach {api_url}: {exc}")
        print("Start the API first, or pass --api-url.")

    frame_rate = _fetch_frame_rate(api_url)
    store = Store(args.data_dir)
    build_demo(api_url, store, frame_rate).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
        allowed_paths=[str(store.data_dir)],
    )


if __name__ == "__main__":
    main()
