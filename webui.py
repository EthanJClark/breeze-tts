"""Gradio web UI for Breeze TTS 2.

The UI does not load the model. It sends requests to a running Breeze TTS
HTTP API, which can be local or remote:

    python -m breeze_infer.api /path/to/breeze-tts-2 --port 7860
    python webui.py --api-url http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import os
import secrets
import time
from functools import partial
from pathlib import Path

import gradio as gr
import httpx
import numpy as np

from webui_store import DEFAULT_DATA_DIR, DEFAULT_SETTINGS, Store

DEFAULT_API_URL = "http://127.0.0.1:7860"
DEFAULT_SAMPLE_RATE = 24000
FALLBACK_FRAME_RATE = 12.5
MAX_NEW_TOKENS = 1500
DEFAULT_TIMEOUT = 600.0

DELIVERY_PRESETS = {
    "Warm": (
        "Warm, friendly, and reassuring. A gentle smile is audible, the pace is "
        "relaxed, and the delivery feels personal and kind."
    ),
    "Amused": (
        "Amused and playful. The voice lifts with a chuckle, the timing is light, "
        "and a smile runs through the lines."
    ),
    "Tender": (
        "Tender and intimate. Soft volume, slow pace, gentle breath, and long "
        "pauses before the most important words."
    ),
    "Excited": (
        "Excited and energetic. Brighter pitch, quicker pace, rising momentum, "
        "and genuine enthusiasm."
    ),
    "Melancholy": (
        "Melancholy and reflective. Lower volume, slower pace, downward phrasing, "
        "and quiet pauses that carry the weight of the words."
    ),
    "Menacing": (
        "Cold and menacing. Low, controlled, deliberate delivery with long pauses "
        "and a threat held just beneath the surface."
    ),
    "Conspiratorial": (
        "Conspiratorial and close. A near-whisper, leaning toward the listener, "
        "with amused pauses and a sense of shared secrets."
    ),
    "Urgent": (
        "Urgent and focused. Faster pace, clipped phrasing, tight breath, and "
        "forward momentum."
    ),
}

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


def _instruction(voice: str, performance: str) -> str:
    parts = []
    if voice.strip():
        parts.append(f"Voice: {voice.strip()}")
    if performance.strip():
        parts.append(f"Performance: {performance.strip()}")
    return "\n\n".join(parts)


def _summary(
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
        f"CFG {cfg_scale:g} · temp {temperature:.2f} · top-p {top_p:.2f} · "
        f"top-k {top_k} · rep {repetition_penalty:.2f} · max {max_duration:g}s · "
        f"{seed_text} · {takes} take{'s' if takes != 1 else ''}"
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


def _generation_outputs(
    audio,
    labels: list[str],
    results: list[dict],
    status: str,
    summary: str,
    *,
    busy: bool,
):
    selected = labels[-1] if labels else None
    return (
        audio,
        gr.update(choices=labels, value=selected, visible=len(labels) > 1),
        results,
        status,
        summary,
        gr.update(interactive=not busy),
        gr.update(interactive=busy),
    )


def run_generation(
    text: str,
    voice: str,
    performance: str,
    reference_profile: str | None,
    reference_audio: str | None,
    reference_text: str,
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
    store: Store,
    api_url: str,
    frame_rate: float,
):
    text = (text or "").strip()
    voice = (voice or "").strip()
    performance = (performance or "").strip()
    one_off_audio = reference_audio or None
    one_off_text = (reference_text or "").strip()

    if not text:
        yield _generation_outputs(
            None, [], [], "❌ Enter some text to speak.", "", busy=False
        )
        return
    if one_off_audio and not one_off_text:
        yield _generation_outputs(
            None,
            [],
            [],
            "❌ The one-off reference audio needs its exact transcript.",
            "",
            busy=False,
        )
        return
    if one_off_text and not one_off_audio and not reference_profile:
        yield _generation_outputs(
            None,
            [],
            [],
            "❌ The one-off reference transcript needs an audio clip.",
            "",
            busy=False,
        )
        return

    instruction = _instruction(voice, performance)
    has_instruction = bool(instruction)
    effective_cfg = float(cfg_scale) if has_instruction else 1.0

    reference_bytes = None
    reference_filename = None
    resolved_ref_text = ""
    if one_off_audio:
        try:
            reference_bytes = Path(one_off_audio).read_bytes()
        except OSError as exc:
            yield _generation_outputs(
                None, [], [], f"❌ Cannot read reference audio: {exc}", "", busy=False
            )
            return
        reference_filename = Path(one_off_audio).name
        resolved_ref_text = one_off_text
    elif reference_profile:
        voices, _ = store.load_voices()
        profile = voices.get(reference_profile)
        if profile is None:
            yield _generation_outputs(
                None,
                [],
                [],
                f"❌ Saved voice **{reference_profile}** was not found.",
                "",
                busy=False,
            )
            return
        try:
            path = store.reference_path(profile)
            reference_bytes = path.read_bytes()
        except (OSError, ValueError) as exc:
            yield _generation_outputs(
                None, [], [], f"❌ Cannot read saved voice: {exc}", "", busy=False
            )
            return
        reference_filename = path.name
        resolved_ref_text = profile["ref_text"]

    try:
        take_count = max(1, min(4, int(takes)))
        max_duration_value = float(max_duration)
        timeout_value = float(timeout)
        seed_value = int(seed)
    except (TypeError, ValueError):
        yield _generation_outputs(
            None, [], [], "❌ Invalid generation settings.", "", busy=False
        )
        return

    max_new_tokens = max(1, min(MAX_NEW_TOKENS, round(max_duration_value * frame_rate)))
    if seed_mode == "random":
        seeds = [secrets.randbelow(2**31) for _ in range(take_count)]
    else:
        seeds = [(seed_value + index) % (2**31) for index in range(take_count)]

    summary = _summary(
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
    )

    for index, take_seed in enumerate(seeds, start=1):
        yield _generation_outputs(
            results[-1]["audio"] if results else None,
            [item["label"] for item in results],
            results,
            f"⏳ Take {index}/{take_count} · seed {take_seed}…",
            summary,
            busy=True,
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
        )

    elapsed = time.perf_counter() - started
    yield _generation_outputs(
        results[-1]["audio"],
        [item["label"] for item in results],
        results,
        f"✅ {take_count} take{'s' if take_count != 1 else ''} in {elapsed:.1f} s",
        summary,
        busy=False,
    )


def stop_generation():
    return (
        gr.update(interactive=True),
        gr.update(interactive=False),
        "⏹ Stopped.",
    )


def select_take(label: str | None, results: list[dict] | None):
    if not label or not results:
        return None, "Select a take."
    for take in results:
        if take["label"] == label:
            return (
                take["audio"],
                f"Take {take['seed']} · {take['duration']:.2f} s",
            )
    return None, "❌ That take is no longer available."


def append_event(text: str, tag: str) -> str:
    text = (text or "").rstrip()
    return f"{text} {tag}".strip()


def apply_delivery(name: str | None) -> str:
    if not name or name == "None":
        return ""
    return DELIVERY_PRESETS.get(name, "")


def apply_script(name: str | None, *, store: Store) -> str:
    if not name:
        return ""
    scripts, _ = store.load_scripts()
    return scripts.get(name, "")


def apply_instruction_preset(name: str | None, *, store: Store):
    if not name:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    presets, _ = store.load_presets()
    preset = presets.get(name)
    if preset is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    return (
        preset["voice"],
        preset["performance"],
        preset["cfg_scale"],
        preset["temperature"],
        preset["seed"],
    )


def load_preset_editor(name: str | None, *, store: Store):
    if not name:
        return "", "", "", 4.0, 0.9, 42
    presets, _ = store.load_presets()
    preset = presets.get(name)
    if preset is None:
        return name, "", "", 4.0, 0.9, 42
    return (
        name,
        preset["voice"],
        preset["performance"],
        preset["cfg_scale"],
        preset["temperature"],
        preset["seed"],
    )


def save_instruction_preset(
    name: str,
    voice: str,
    performance: str,
    cfg_scale: float,
    temperature: float,
    seed: float,
    *,
    store: Store,
):
    name = (name or "").strip()
    voice = (voice or "").strip()
    performance = (performance or "").strip()
    if not name:
        return gr.update(), gr.update(), "❌ Enter a preset name."
    if not voice and not performance:
        return gr.update(), gr.update(), "❌ Enter a voice or performance description."
    presets, _ = store.load_presets()
    existed = name in presets
    presets[name] = {
        "voice": voice,
        "performance": performance,
        "cfg_scale": float(cfg_scale),
        "temperature": float(temperature),
        "seed": int(seed),
    }
    try:
        store.save_presets(presets)
    except OSError as exc:
        return gr.update(), gr.update(), f"❌ Could not save: {exc}"
    choices = sorted(presets)
    update = gr.update(choices=choices, value=name)
    action = "Updated" if existed else "Saved"
    return update, update, f"✅ {action} **{name}**."


def delete_instruction_preset(name: str | None, *, store: Store):
    if not name:
        return gr.update(), gr.update(), "❌ Select a preset to delete."
    presets, _ = store.load_presets()
    if name not in presets:
        return gr.update(choices=sorted(presets)), gr.update(), "❌ Preset not found."
    del presets[name]
    try:
        store.save_presets(presets)
    except OSError as exc:
        return gr.update(), gr.update(), f"❌ Could not delete: {exc}"
    choices = sorted(presets)
    update = gr.update(choices=choices, value=None)
    return update, update, f"✅ Deleted **{name}**."


def load_script_editor(name: str | None, *, store: Store):
    if not name:
        return "", ""
    scripts, _ = store.load_scripts()
    return name, scripts.get(name, "")


def save_script(name: str, text: str, *, store: Store):
    name = (name or "").strip()
    text = (text or "").strip()
    if not name:
        return gr.update(), gr.update(), "❌ Enter a script name."
    if not text:
        return gr.update(), gr.update(), "❌ Enter script text."
    scripts, _ = store.load_scripts()
    existed = name in scripts
    scripts[name] = text
    try:
        store.save_scripts(scripts)
    except OSError as exc:
        return gr.update(), gr.update(), f"❌ Could not save: {exc}"
    choices = sorted(scripts)
    update = gr.update(choices=choices, value=name)
    action = "Updated" if existed else "Saved"
    return update, update, f"✅ {action} **{name}**."


def delete_script(name: str | None, *, store: Store):
    if not name:
        return gr.update(), gr.update(), "❌ Select a script to delete."
    scripts, _ = store.load_scripts()
    if name not in scripts:
        return gr.update(choices=sorted(scripts)), gr.update(), "❌ Script not found."
    del scripts[name]
    try:
        store.save_scripts(scripts)
    except OSError as exc:
        return gr.update(), gr.update(), f"❌ Could not delete: {exc}"
    choices = sorted(scripts)
    update = gr.update(choices=choices, value=None)
    return update, update, f"✅ Deleted **{name}**."


def load_voice_editor(name: str | None, *, store: Store):
    if not name:
        return "", None, ""
    voices, _ = store.load_voices()
    profile = voices.get(name)
    if profile is None:
        return name, None, ""
    try:
        path = store.reference_path(profile)
    except ValueError:
        return name, None, profile["ref_text"]
    return name, str(path), profile["ref_text"]


def save_voice_profile(
    name: str,
    audio: str | None,
    ref_text: str,
    *,
    store: Store,
):
    name = (name or "").strip()
    ref_text = (ref_text or "").strip()
    if not name:
        return gr.update(), gr.update(), "❌ Enter a voice name."
    if not audio:
        return gr.update(), gr.update(), "❌ Upload reference audio."
    if not ref_text:
        return gr.update(), gr.update(), "❌ Enter the exact transcript."
    voices, _ = store.load_voices()
    old = voices.get(name)
    try:
        relative = store.import_reference(name, audio)
    except OSError as exc:
        return gr.update(), gr.update(), f"❌ Could not copy reference audio: {exc}"
    existed = name in voices
    voices[name] = {"audio": relative, "ref_text": ref_text}
    try:
        store.save_voices(voices)
    except OSError as exc:
        return gr.update(), gr.update(), f"❌ Could not save voice: {exc}"
    if old and old.get("audio") != relative:
        store.remove_reference(old)
    choices = sorted(voices)
    update = gr.update(choices=choices, value=name)
    action = "Updated" if existed else "Saved"
    return update, update, f"✅ {action} voice **{name}**."


def delete_voice_profile(name: str | None, *, store: Store):
    if not name:
        return gr.update(), gr.update(), "❌ Select a voice to delete."
    voices, _ = store.load_voices()
    if name not in voices:
        return gr.update(choices=sorted(voices)), gr.update(), "❌ Voice not found."
    profile = voices.pop(name)
    try:
        store.save_voices(voices)
    except OSError as exc:
        return gr.update(), gr.update(), f"❌ Could not delete: {exc}"
    store.remove_reference(profile)
    choices = sorted(voices)
    update = gr.update(choices=choices, value=None)
    return update, update, f"✅ Deleted voice **{name}**."


def reference_summary(name: str | None) -> str:
    return f"Using saved voice **{name}**." if name else ""


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
        values["seed"],
        values["timeout"],
        values["autoplay"],
        "Reset to defaults. Click Save settings to persist.",
    )


def build_demo(api_url: str, store: Store, frame_rate: float) -> gr.Blocks:
    settings, settings_warning = store.load_settings()
    presets, presets_warning = store.load_presets()
    scripts, scripts_warning = store.load_scripts()
    voices, voices_warning = store.load_voices()

    preset_choices = sorted(presets)
    script_choices = sorted(scripts)
    voice_choices = sorted(voices)
    initial_summary = _summary(
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

    with gr.Blocks(title="Breeze TTS 2") as demo:
        gr.Markdown(
            "# Breeze TTS 2\n"
            "Generate speech with voice design, cloning, or direction. "
            "The API runs separately; this page only sends requests."
        )

        with gr.Tabs():
            with gr.Tab("Generate"):  # noqa: SIM117
                with gr.Row():
                    with gr.Column(scale=3):
                        text = gr.Textbox(
                            label="Script",
                            placeholder="Type the text to synthesize...",
                            lines=7,
                            autofocus=True,
                        )
                        with gr.Row():
                            event_buttons = [
                                gr.Button(tag, size="sm", variant="secondary")
                                for tag in EVENT_TAGS
                            ]
                        script_preset = gr.Dropdown(
                            label="Script preset",
                            choices=script_choices,
                            value=None,
                            filterable=True,
                        )
                        voice = gr.Textbox(
                            label="Voice",
                            placeholder=(
                                "Age, timbre, accent, texture. E.g. a weathered "
                                "baritone in his early sixties."
                            ),
                            lines=3,
                        )
                        performance = gr.Textbox(
                            label="Performance",
                            placeholder=(
                                "Emotion, pace, dynamics. E.g. delighted and a little "
                                "mischievous, leaning in for the secret."
                            ),
                            lines=3,
                        )
                        with gr.Row():
                            instruction_preset = gr.Dropdown(
                                label="Instruction preset",
                                choices=preset_choices,
                                value=None,
                                filterable=True,
                            )
                            delivery_preset = gr.Dropdown(
                                label="Delivery preset",
                                choices=["None"] + list(DELIVERY_PRESETS),
                                value="None",
                            )
                        with gr.Accordion("Reference voice (optional)", open=False):
                            gr.Markdown(
                                "A one-off clip takes precedence over a saved voice. "
                                "Provide the exact transcript for whichever clip you use."
                            )
                            reference_profile = gr.Dropdown(
                                label="Saved voice",
                                choices=voice_choices,
                                value=None,
                                filterable=True,
                            )
                            reference_info = gr.Markdown("")
                            reference_audio = gr.Audio(
                                label="One-off reference audio",
                                sources=["upload", "microphone"],
                                type="filepath",
                            )
                            reference_text = gr.Textbox(
                                label="Exact transcript of the one-off reference",
                                placeholder="Every word spoken in the clip.",
                                lines=3,
                            )

                    with gr.Column(scale=2):
                        with gr.Row():
                            generate_button = gr.Button(
                                "Generate", variant="primary", scale=2
                            )
                            stop_button = gr.Button(
                                "Stop", variant="stop", interactive=False
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
                        with gr.Accordion("Tips", open=False):
                            gr.Markdown(
                                "**Voice design** — fill Voice and/or Performance, "
                                "leave the reference section empty.\n\n"
                                "**Voice clone** — add a reference clip and transcript, "
                                "leave Performance empty.\n\n"
                                "**Voice direction** — add a reference plus Performance.\n\n"
                                "**Vocal events** — use `(laugh)`, `(sigh)`, `(cough)`, "
                                "or `(clears throat)` in the script.\n\n"
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
                            value=settings["seed"], precision=0, label="Fixed seed"
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
                                "Save settings", variant="primary"
                            )
                            reset_settings_button = gr.Button("Reset defaults")
                        settings_status = gr.Markdown(
                            settings_warning
                            or "Values apply immediately. Save to persist across restarts."
                        )

            with gr.Tab("Library"):  # noqa: SIM117
                with gr.Tabs():
                    with gr.Tab("Instructions"):
                        with gr.Row():
                            preset_selector = gr.Dropdown(
                                label="Saved instructions",
                                choices=preset_choices,
                                value=None,
                                filterable=True,
                            )
                            preset_name = gr.Textbox(
                                label="Name", placeholder="Name this instruction..."
                            )
                        preset_voice = gr.Textbox(label="Voice", lines=3)
                        preset_performance = gr.Textbox(label="Performance", lines=3)
                        with gr.Row():
                            preset_cfg = gr.Slider(
                                1.0, 6.0, 4.0, step=0.5, label="CFG scale"
                            )
                            preset_temperature = gr.Slider(
                                0.5, 1.5, 0.9, step=0.05, label="Temperature"
                            )
                            preset_seed = gr.Number(value=42, precision=0, label="Seed")
                        with gr.Row():
                            save_preset_button = gr.Button(
                                "Save instruction", variant="primary"
                            )
                            delete_preset_button = gr.Button("Delete")
                        preset_status = gr.Markdown(
                            presets_warning
                            or "Save voice/performance combinations with their sampling settings."
                        )

                    with gr.Tab("Voices"):
                        with gr.Row():
                            voice_selector = gr.Dropdown(
                                label="Saved voices",
                                choices=voice_choices,
                                value=None,
                                filterable=True,
                            )
                            voice_name = gr.Textbox(
                                label="Name", placeholder="Name this voice..."
                            )
                        voice_audio = gr.Audio(
                            label="Reference audio",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                        voice_text = gr.Textbox(
                            label="Exact transcript",
                            placeholder="Every word spoken in the reference clip.",
                            lines=3,
                        )
                        with gr.Row():
                            save_voice_button = gr.Button(
                                "Save voice", variant="primary"
                            )
                            delete_voice_button = gr.Button("Delete")
                        voice_status = gr.Markdown(
                            voices_warning
                            or "Saved voices keep the audio and transcript for reuse."
                        )

                    with gr.Tab("Scripts"):
                        with gr.Row():
                            script_selector = gr.Dropdown(
                                label="Saved scripts",
                                choices=script_choices,
                                value=None,
                                filterable=True,
                            )
                            script_name = gr.Textbox(
                                label="Name", placeholder="Name this script..."
                            )
                        script_text = gr.Textbox(label="Script", lines=8)
                        with gr.Row():
                            save_script_button = gr.Button(
                                "Save script", variant="primary"
                            )
                            delete_script_button = gr.Button("Delete")
                        script_status = gr.Markdown(
                            scripts_warning
                            or "Saved scripts can be loaded on the Generate tab."
                        )

        event_inputs = [
            text,
            voice,
            performance,
            reference_profile,
            reference_audio,
            reference_text,
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
                store=store,
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
        take_selector.input(
            select_take,
            inputs=[take_selector, takes_state],
            outputs=[output_audio, status],
            api_name="select_take",
        )
        script_preset.input(
            partial(apply_script, store=store),
            inputs=script_preset,
            outputs=text,
            api_name="apply_script",
        )
        instruction_preset.input(
            partial(apply_instruction_preset, store=store),
            inputs=instruction_preset,
            outputs=[voice, performance, cfg_scale, temperature, seed],
            api_name="apply_instruction_preset",
        )
        delivery_preset.input(
            apply_delivery,
            inputs=delivery_preset,
            outputs=performance,
            api_name="apply_delivery",
        )
        for index, (button, tag) in enumerate(zip(event_buttons, EVENT_TAGS)):
            button.click(
                partial(append_event, tag=tag),
                inputs=text,
                outputs=text,
                api_name=f"insert_event_{index}",
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
            ],
            api_name="reset_settings",
        )

        preset_selector.input(
            partial(load_preset_editor, store=store),
            inputs=preset_selector,
            outputs=[
                preset_name,
                preset_voice,
                preset_performance,
                preset_cfg,
                preset_temperature,
                preset_seed,
            ],
            api_name="load_preset",
        )
        save_preset_button.click(
            partial(save_instruction_preset, store=store),
            inputs=[
                preset_name,
                preset_voice,
                preset_performance,
                preset_cfg,
                preset_temperature,
                preset_seed,
            ],
            outputs=[preset_selector, instruction_preset, preset_status],
            api_name="save_preset",
        )
        delete_preset_button.click(
            partial(delete_instruction_preset, store=store),
            inputs=preset_selector,
            outputs=[preset_selector, instruction_preset, preset_status],
            api_name="delete_preset",
        )

        voice_selector.input(
            partial(load_voice_editor, store=store),
            inputs=voice_selector,
            outputs=[voice_name, voice_audio, voice_text],
            api_name="load_voice",
        )
        save_voice_button.click(
            partial(save_voice_profile, store=store),
            inputs=[voice_name, voice_audio, voice_text],
            outputs=[voice_selector, reference_profile, voice_status],
            api_name="save_voice",
        )
        delete_voice_button.click(
            partial(delete_voice_profile, store=store),
            inputs=voice_selector,
            outputs=[voice_selector, reference_profile, voice_status],
            api_name="delete_voice",
        )
        reference_profile.input(
            reference_summary,
            inputs=reference_profile,
            outputs=reference_info,
            api_name="reference_summary",
        )

        script_selector.input(
            partial(load_script_editor, store=store),
            inputs=script_selector,
            outputs=[script_name, script_text],
            api_name="load_script",
        )
        save_script_button.click(
            partial(save_script, store=store),
            inputs=[script_name, script_text],
            outputs=[script_selector, script_preset, script_status],
            api_name="save_script",
        )
        delete_script_button.click(
            partial(delete_script, store=store),
            inputs=script_selector,
            outputs=[script_selector, script_preset, script_status],
            api_name="delete_script",
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
        help="Directory for settings, presets, scripts, and voices (default: %(default)s)",
    )
    parser.add_argument(
        "--presets-file",
        type=Path,
        default=(
            Path(os.environ["BREEZE_PRESETS_FILE"]).expanduser()
            if os.environ.get("BREEZE_PRESETS_FILE")
            else None
        ),
        help="Override just the instruction presets file",
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
    store = Store(args.data_dir, presets_path=args.presets_file)
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
