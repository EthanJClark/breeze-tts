"""Persistent storage for the Breeze TTS web UI.

Keeps settings, instruction presets, scripts, and saved reference voices in a
single data directory. All JSON writes are atomic so a crash cannot leave a
half-written file behind.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path(
    os.environ.get("BREEZE_DATA_DIR") or Path.home() / ".local" / "share" / "breeze-tts"
).expanduser()

DEFAULT_SETTINGS: dict[str, Any] = {
    "cfg_scale": 4.0,
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "max_duration": 60.0,
    "takes": 1,
    "seed_mode": "fixed",
    "seed": 42,
    "timeout": 600.0,
    "autoplay": True,
}

DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "Santa": {
        "voice": (
            "A warm, deep, jolly old man's voice with a rich, booming chest "
            "resonance and a gentle rasp. Speak slowly and cheerfully in a "
            "classic grandfatherly Santa Claus delivery, with a merry chuckle "
            "just beneath the words and a reassuring, friendly tone. Crisp "
            "articulation, hearty low notes, and a slight smile in the voice. "
            "Never rushed."
        ),
        "performance": "",
        "cfg_scale": 4.0,
        "temperature": 0.9,
        "seed": 42,
    },
    "Dark wizard": {
        "voice": (
            "An ancient male wizard with a deep, raspy, gravelly voice. Speak "
            "slowly and ominously with a theatrical, menacing delivery. Drag "
            "out the vowels, hiss the sibilants, and let a dry, cruel laugh "
            "rumble low in his chest. His tone is cold, patient, and utterly "
            "certain of his power, as if he has all the time in the world. "
            "Close and intimate, like he is leaning toward the listener in a "
            "dark stone hall."
        ),
        "performance": "",
        "cfg_scale": 4.0,
        "temperature": 0.9,
        "seed": 42,
    },
}


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: Any, default: float, low: float, high: float) -> float:
    return max(low, min(high, _finite(value, default)))


def normalize_settings(values: dict[str, Any] | None) -> dict[str, Any]:
    raw = values if isinstance(values, dict) else {}
    seed_mode = raw.get("seed_mode", DEFAULT_SETTINGS["seed_mode"])
    return {
        "cfg_scale": _clamp(raw.get("cfg_scale"), 4.0, 1.0, 6.0),
        "temperature": _clamp(raw.get("temperature"), 0.9, 0.1, 2.0),
        "top_p": _clamp(raw.get("top_p"), 1.0, 0.05, 1.0),
        "top_k": int(_clamp(raw.get("top_k"), 50, 0, 100)),
        "repetition_penalty": _clamp(raw.get("repetition_penalty"), 1.1, 1.0, 2.0),
        "max_duration": _clamp(raw.get("max_duration"), 60.0, 5.0, 120.0),
        "takes": int(_clamp(raw.get("takes"), 1, 1, 4)),
        "seed_mode": seed_mode if seed_mode in ("fixed", "random") else "fixed",
        "seed": int(_clamp(raw.get("seed"), 42, 0, 2**31 - 1)),
        "timeout": _clamp(raw.get("timeout"), 600.0, 30.0, 3600.0),
        "autoplay": bool(raw.get("autoplay", True)),
    }


def normalize_preset(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        voice, performance = value.strip(), ""
    elif isinstance(value, dict):
        voice = str(value.get("voice") or value.get("instruction") or "").strip()
        performance = str(value.get("performance") or "").strip()
    else:
        return None
    if not voice and not performance:
        return None
    return {
        "voice": voice,
        "performance": performance,
        "cfg_scale": _clamp(
            value.get("cfg_scale") if isinstance(value, dict) else 4.0, 4.0, 1.0, 6.0
        ),
        "temperature": _clamp(
            value.get("temperature") if isinstance(value, dict) else 0.9, 0.9, 0.1, 2.0
        ),
        "seed": int(
            _clamp(
                value.get("seed") if isinstance(value, dict) else 42, 42, 0, 2**31 - 1
            )
        ),
    }


class Store:
    """Read and write the web UI's persistent state."""

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        *,
        presets_path: Path | str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.settings_path = self.data_dir / "settings.json"
        self.presets_path = (
            Path(presets_path).expanduser()
            if presets_path is not None
            else self.data_dir / "instruction_presets.json"
        )
        self.scripts_path = self.data_dir / "scripts.json"
        self.voices_path = self.data_dir / "voice_profiles.json"
        self.references_dir = self.data_dir / "references"
        self._lock = threading.RLock()

    # -- generic JSON helpers -------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> tuple[Any | None, str | None]:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, None
        except OSError as exc:
            return None, f"Could not read {path.name}: {exc}"
        try:
            return json.loads(raw), None
        except ValueError as exc:
            return None, f"{path.name} is invalid ({exc}); using defaults."

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                json.dump(payload, temp_file, ensure_ascii=False, indent=2)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_name, path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    # -- settings -------------------------------------------------------------

    def load_settings(self) -> tuple[dict[str, Any], str | None]:
        with self._lock:
            data, warning = self._read_json(self.settings_path)
        raw = data.get("settings", data) if isinstance(data, dict) else None
        return normalize_settings(raw), warning

    def save_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_settings(values)
        with self._lock:
            self._write_json(self.settings_path, {"version": 1, "settings": normalized})
        return normalized

    # -- instruction presets --------------------------------------------------

    def load_presets(self) -> tuple[dict[str, dict[str, Any]], str | None]:
        with self._lock:
            data, warning = self._read_json(self.presets_path)
        if data is None:
            return {
                name: dict(value) for name, value in DEFAULT_PRESETS.items()
            }, warning
        raw = data.get("presets", data) if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return (
                {name: dict(value) for name, value in DEFAULT_PRESETS.items()},
                warning or "Instruction presets are invalid; using built-in examples.",
            )
        presets = {}
        for name, value in raw.items():
            normalized = normalize_preset(value)
            if normalized is not None and str(name).strip():
                presets[str(name).strip()] = normalized
        return presets, warning

    def save_presets(self, presets: dict[str, dict[str, Any]]) -> None:
        cleaned = {}
        for name, value in presets.items():
            normalized = normalize_preset(value)
            if normalized is not None and str(name).strip():
                cleaned[str(name).strip()] = normalized
        with self._lock:
            self._write_json(
                self.presets_path,
                {"version": 2, "presets": dict(sorted(cleaned.items()))},
            )

    # -- scripts --------------------------------------------------------------

    def load_scripts(self) -> tuple[dict[str, str], str | None]:
        with self._lock:
            data, warning = self._read_json(self.scripts_path)
        raw = data.get("scripts", data) if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return {}, warning
        return {
            str(name).strip(): str(text).strip()
            for name, text in raw.items()
            if str(name).strip() and isinstance(text, str) and text.strip()
        }, warning

    def save_scripts(self, scripts: dict[str, str]) -> None:
        cleaned = {
            str(name).strip(): str(text).strip()
            for name, text in scripts.items()
            if str(name).strip() and str(text).strip()
        }
        with self._lock:
            self._write_json(
                self.scripts_path,
                {"version": 1, "scripts": dict(sorted(cleaned.items()))},
            )

    # -- reference voices -----------------------------------------------------

    def load_voices(self) -> tuple[dict[str, dict[str, str]], str | None]:
        with self._lock:
            data, warning = self._read_json(self.voices_path)
        raw = data.get("profiles", data) if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return {}, warning
        voices = {}
        for name, value in raw.items():
            if not str(name).strip() or not isinstance(value, dict):
                continue
            audio = str(value.get("audio") or "").strip()
            ref_text = str(value.get("ref_text") or "").strip()
            if audio and ref_text:
                voices[str(name).strip()] = {"audio": audio, "ref_text": ref_text}
        return voices, warning

    def save_voices(self, voices: dict[str, dict[str, str]]) -> None:
        cleaned = {
            str(name).strip(): {
                "audio": str(value["audio"]).strip(),
                "ref_text": str(value["ref_text"]).strip(),
            }
            for name, value in voices.items()
            if str(name).strip()
            and isinstance(value, dict)
            and str(value.get("audio") or "").strip()
            and str(value.get("ref_text") or "").strip()
        }
        with self._lock:
            self._write_json(
                self.voices_path,
                {"version": 1, "profiles": dict(sorted(cleaned.items()))},
            )

    def import_reference(self, name: str, source: str | Path) -> str:
        source_path = Path(source).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {source_path}")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "voice"
        suffix = source_path.suffix or ".wav"
        filename = f"{slug}-{uuid.uuid4().hex[:8]}{suffix}"
        self.references_dir.mkdir(parents=True, exist_ok=True)
        destination = self.references_dir / filename
        shutil.copy2(source_path, destination)
        return destination.relative_to(self.data_dir).as_posix()

    def reference_path(self, profile: dict[str, str]) -> Path:
        relative = Path(profile["audio"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Reference audio path is invalid")
        return self.data_dir / relative

    def remove_reference(self, profile: dict[str, str]) -> None:
        try:
            path = self.reference_path(profile)
        except ValueError:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
