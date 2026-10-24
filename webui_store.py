"""Persistent storage for the Breeze TTS web UI.

Keeps settings, presets, and saved reference voices in a single data directory.
Presets are self-contained JSON files (reference audio is embedded) so one file
can be exported and imported on another machine. All writes are atomic so a
crash cannot leave a half-written file behind.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent
APP_DATA_DIR = APP_ROOT / ".breeze2"
LEGACY_DATA_DIR = Path.home() / ".local" / "share" / "breeze-tts"
DEFAULT_DATA_DIR = Path(
    os.environ.get("BREEZE_DATA_DIR") or APP_DATA_DIR
).expanduser()

PRESET_VERSION = 1
VOICE_DESCRIBE = "Describe a voice"
VOICE_CLONE = "Clone from a clip"

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


def _default_preset(description: str) -> dict[str, Any]:
    return {
        "voice_source": VOICE_DESCRIBE,
        "description": description,
        "delivery": "",
        "reference": None,
        "settings": {
            **DEFAULT_SETTINGS,
            "cfg_scale": 4.0,
            "temperature": 0.9,
            "seed": 42,
        },
    }


DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "Santa": _default_preset(
        "A warm, deep, jolly old man's voice with a rich, booming chest "
        "resonance and a gentle rasp. Speak slowly and cheerfully in a "
        "classic grandfatherly Santa Claus delivery, with a merry chuckle "
        "just beneath the words and a reassuring, friendly tone. Crisp "
        "articulation, hearty low notes, and a slight smile in the voice. "
        "Never rushed."
    ),
    "Dark wizard": _default_preset(
        "An ancient male wizard with a deep, raspy, gravelly voice. Speak "
        "slowly and ominously with a theatrical, menacing delivery. Drag "
        "out the vowels, hiss the sibilants, and let a dry, cruel laugh "
        "rumble low in his chest. His tone is cold, patient, and utterly "
        "certain of his power, as if he has all the time in the world. "
        "Close and intimate, like he is leaning toward the listener in a "
        "dark stone hall."
    ),
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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned[:80] or "preset"


def _safe_filename(value: str) -> str:
    name = Path(str(value)).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    return name or "reference.wav"


def _copy_missing(source: Path, destination: Path) -> list[Path]:
    """Copy files from source into destination without overwriting anything."""
    copied: list[Path] = []
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            copied.extend(_copy_missing(item, target))
        elif item.is_file() and not target.exists():
            shutil.copy2(item, target)
            copied.append(target)
    return copied


def migrate_data_dir(source: Path, destination: Path) -> list[Path]:
    """One-time copy of known user data from an older data directory.

    Only missing files are copied, so running this repeatedly is safe and it
    never overwrites data that already lives in the new directory.
    """
    source = Path(source).expanduser()
    destination = Path(destination).expanduser()
    try:
        if source.resolve() == destination.resolve():
            return []
    except OSError:
        pass
    if not source.is_dir():
        return []

    copied: list[Path] = []
    for name in ("presets", "preset_references", "references"):
        src = source / name
        if src.is_dir():
            copied.extend(_copy_missing(src, destination / name))
    for name in ("settings.json", "voice_profiles.json", "instruction_presets.json"):
        src = source / name
        target = destination / name
        if src.is_file() and not target.exists():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied.append(target)
    return copied


def _decode_audio(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return ""
    return value.strip()


def normalize_preset(payload: Any, fallback_name: str = "") -> dict[str, Any] | None:
    """Validate a preset payload and clamp every user-controlled value."""
    if not isinstance(payload, dict):
        return None

    name = str(payload.get("name") or fallback_name).strip() or "Preset"
    description = str(payload.get("description") or "").strip()
    delivery = str(payload.get("delivery") or "").strip()
    voice_source = str(payload.get("voice_source") or VOICE_DESCRIBE).strip()
    if not voice_source:
        voice_source = VOICE_DESCRIBE

    reference = None
    raw_reference = payload.get("reference")
    if isinstance(raw_reference, dict):
        audio_b64 = _decode_audio(raw_reference.get("audio_b64"))
        if audio_b64:
            reference = {
                "filename": _safe_filename(
                    raw_reference.get("filename") or "reference.wav"
                ),
                "text": str(raw_reference.get("text") or "").strip(),
                "audio_b64": audio_b64,
            }

    if not description and not delivery and reference is None:
        return None

    created = str(payload.get("created") or "").strip() or _utc_now()
    return {
        "version": PRESET_VERSION,
        "name": name,
        "created": created,
        "voice_source": voice_source,
        "description": description,
        "delivery": delivery,
        "reference": reference,
        "settings": normalize_settings(payload.get("settings")),
    }


class Store:
    """Read and write the web UI's persistent state."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self._migrate_legacy_data()
        self.settings_path = self.data_dir / "settings.json"
        self.presets_dir = self.data_dir / "presets"
        self.preset_refs_dir = self.data_dir / "preset_references"
        self.exports_dir = self.data_dir / "exports"
        self.legacy_presets_path = self.data_dir / "instruction_presets.json"
        self.voices_path = self.data_dir / "voice_profiles.json"
        self.references_dir = self.data_dir / "references"
        self._lock = threading.RLock()
        self._presets_ready = False

    def _migrate_legacy_data(self) -> None:
        """Bring data from the old home-directory location into .breeze2."""
        try:
            is_app_data = self.data_dir.resolve() == APP_DATA_DIR.resolve()
        except OSError:
            is_app_data = False
        if not is_app_data:
            return
        try:
            migrate_data_dir(LEGACY_DATA_DIR, self.data_dir)
        except OSError:
            pass

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

    # -- presets --------------------------------------------------------------

    def _ensure_presets(self) -> None:
        if self._presets_ready:
            return
        with self._lock:
            if self._presets_ready:
                return
            if not self.presets_dir.exists():
                self.presets_dir.mkdir(parents=True, exist_ok=True)
                self._presets_ready = True
                self._import_legacy_presets()
                self._seed_default_presets()
            else:
                self._presets_ready = True

    def _import_legacy_presets(self) -> None:
        data, _ = self._read_json(self.legacy_presets_path)
        raw = data.get("presets", data) if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return
        for name, value in raw.items():
            legacy = self._legacy_preset(name, value)
            if legacy is not None:
                self._write_preset(str(name).strip(), legacy)

    @staticmethod
    def _legacy_preset(name: Any, value: Any) -> dict[str, Any] | None:
        if not str(name).strip():
            return None
        if isinstance(value, str):
            voice, performance = value.strip(), ""
            cfg_scale, temperature, seed = 4.0, 0.9, 42
        elif isinstance(value, dict):
            voice = str(value.get("voice") or value.get("instruction") or "").strip()
            performance = str(value.get("performance") or "").strip()
            cfg_scale = _clamp(value.get("cfg_scale"), 4.0, 1.0, 6.0)
            temperature = _clamp(value.get("temperature"), 0.9, 0.1, 2.0)
            seed = int(_clamp(value.get("seed"), 42, 0, 2**31 - 1))
        else:
            return None
        description = "\n\n".join(part for part in (voice, performance) if part)
        if not description:
            return None
        return {
            "voice_source": VOICE_DESCRIBE,
            "description": description,
            "delivery": "",
            "reference": None,
            "settings": {
                **DEFAULT_SETTINGS,
                "cfg_scale": cfg_scale,
                "temperature": temperature,
                "seed": seed,
            },
        }

    def _seed_default_presets(self) -> None:
        for name, preset in DEFAULT_PRESETS.items():
            if self._find_preset_path(name) is None:
                self._write_preset(name, preset)

    def _preset_paths(self) -> list[Path]:
        return sorted(self.presets_dir.glob("*.json"))

    def _find_preset_path(self, name: str) -> Path | None:
        for path in self._preset_paths():
            data, _ = self._read_json(path)
            if isinstance(data, dict) and str(data.get("name") or "").strip() == name:
                return path
        return None

    def _allocate_preset_path(self, name: str) -> Path:
        base = _slug(name)
        candidate = self.presets_dir / f"{base}.json"
        counter = 2
        while candidate.exists():
            candidate = self.presets_dir / f"{base}-{counter}.json"
            counter += 1
        return candidate

    def _write_preset(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_preset({**payload, "name": name})
        if normalized is None:
            raise ValueError("A preset needs a description, delivery, or reference.")
        path = self._find_preset_path(name)
        if path is not None:
            existing, _ = self._read_json(path)
            if isinstance(existing, dict) and existing.get("created"):
                normalized["created"] = str(existing["created"])
        else:
            path = self._allocate_preset_path(name)
        self._write_json(path, normalized)
        return normalized

    def list_presets(self) -> tuple[dict[str, dict[str, Any]], str | None]:
        with self._lock:
            self._ensure_presets()
            warning = None
            presets: dict[str, dict[str, Any]] = {}
            for path in self._preset_paths():
                data, read_warning = self._read_json(path)
                if read_warning:
                    warning = warning or read_warning
                    continue
                normalized = normalize_preset(data)
                if normalized is None:
                    warning = warning or f"{path.name} is not a valid preset; skipped."
                    continue
                presets[normalized["name"]] = normalized
        return (
            dict(sorted(presets.items(), key=lambda item: item[0].lower())),
            warning,
        )

    def load_preset(self, name: str | None) -> dict[str, Any] | None:
        if not name:
            return None
        presets, _ = self.list_presets()
        return presets.get(name)

    def preset_path(self, name: str | None) -> Path | None:
        """Return the canonical on-disk JSON path for a preset."""
        if not name:
            return None
        with self._lock:
            self._ensure_presets()
            return self._find_preset_path(name)

    def save_preset(self, name: str, payload: dict[str, Any]) -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("Enter a preset name.")
        with self._lock:
            self._ensure_presets()
            normalized = self._write_preset(name, payload)
        return normalized["name"]

    def delete_preset(self, name: str | None) -> bool:
        if not name:
            return False
        with self._lock:
            self._ensure_presets()
            path = self._find_preset_path(name)
            if path is None:
                return False
            path.unlink(missing_ok=True)
            return True

    def export_preset(self, name: str | None) -> Path:
        preset = self.load_preset(name)
        if preset is None:
            raise ValueError(f"Preset not found: {name}")
        with self._lock:
            self.exports_dir.mkdir(parents=True, exist_ok=True)
            path = self.exports_dir / f"{_slug(preset['name'])}.json"
            self._write_json(path, preset)
        return path

    def import_preset(self, raw: str | bytes, filename: str | None = None) -> str:
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("That file is not valid JSON.") from exc
        fallback = Path(filename).stem if filename else ""
        preset = normalize_preset(data, fallback_name=fallback)
        if preset is None:
            raise ValueError(
                "That JSON has no voice description, delivery, or reference clip."
            )
        name = preset["name"]
        with self._lock:
            presets, _ = self.list_presets()
            candidate = name
            counter = 2
            while candidate in presets:
                candidate = f"{name} ({counter})"
                counter += 1
            preset["name"] = candidate
            self.save_preset(candidate, preset)
        return candidate

    def materialize_reference(self, name: str, preset: dict[str, Any]) -> Path | None:
        """Write an embedded reference clip to disk and return its path."""
        reference = preset.get("reference")
        if not isinstance(reference, dict):
            return None
        audio_b64 = reference.get("audio_b64")
        if not audio_b64:
            return None
        try:
            data = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError):
            return None
        filename = _safe_filename(reference.get("filename") or "reference.wav")
        suffix = Path(filename).suffix or ".wav"
        digest = hashlib.sha1(audio_b64.encode("ascii")).hexdigest()[:10]
        destination = self.preset_refs_dir / f"{_slug(name)}-{digest}{suffix}"
        if not destination.exists():
            with self._lock:
                if not destination.exists():
                    self.preset_refs_dir.mkdir(parents=True, exist_ok=True)
                    fd, temp_name = tempfile.mkstemp(
                        prefix=f"{destination.name}.",
                        suffix=".tmp",
                        dir=destination.parent,
                    )
                    try:
                        with os.fdopen(fd, "wb") as temp_file:
                            temp_file.write(data)
                            temp_file.flush()
                            os.fsync(temp_file.fileno())
                        os.replace(temp_name, destination)
                    except BaseException:
                        try:
                            os.unlink(temp_name)
                        except OSError:
                            pass
                        raise
        return destination

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
