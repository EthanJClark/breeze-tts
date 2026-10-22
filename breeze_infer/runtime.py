from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from models.breeze import BreezeForConditionalGeneration

DEFAULT_TEMPERATURE = 0.9
MAX_TEMPERATURE = 2.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.1


def validate_temperature(temperature: float) -> float:
    if not math.isfinite(temperature) or not 0.0 < temperature <= MAX_TEMPERATURE:
        raise ValueError(
            f"temperature must be greater than 0 and at most {MAX_TEMPERATURE}."
        )
    return float(temperature)


def validate_top_p(top_p: float) -> float:
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1.0.")
    return float(top_p)


def validate_top_k(top_k: int) -> int:
    if top_k < 0:
        raise ValueError("top_k must be zero or greater.")
    return int(top_k)


def validate_repetition_penalty(repetition_penalty: float) -> float:
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than 0.")
    return float(repetition_penalty)


def set_generation_sampling(
    model: torch.nn.Module,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> None:
    updates: dict[str, float | int] = {}
    if temperature is not None:
        updates["temperature"] = validate_temperature(temperature)
    if top_p is not None:
        updates["top_p"] = validate_top_p(top_p)
    if top_k is not None:
        updates["top_k"] = validate_top_k(top_k)
    if not updates:
        return
    vars(model.generation_config).update(updates)
    vars(model.depth_decoder.generation_config).update(updates)


def set_generation_temperature(model: torch.nn.Module, temperature: float) -> None:
    set_generation_sampling(model, temperature=temperature)


def get_dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def resolve_device(explicit_device: str | None = None) -> str:
    if explicit_device:
        return explicit_device

    _, _, local_rank = get_dist_info()
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_generation_config_for_breeze(
    model: torch.nn.Module,
    generation_config: dict[str, Any] | None = None,
) -> None:
    generation_config = generation_config or {
        "depth_decoder_do_sample": True,
        "depth_decoder_temperature": DEFAULT_TEMPERATURE,
        "depth_decoder_top_p": DEFAULT_TOP_P,
        "depth_decoder_top_k": DEFAULT_TOP_K,
        "do_sample": True,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "max_new_tokens": 750,
        "temperature": DEFAULT_TEMPERATURE,
    }

    prefix = "depth_decoder_"
    depth_decoder_attrs = {
        attr[len(prefix) :]: value
        for attr, value in generation_config.items()
        if attr.startswith(prefix)
    }
    vars(model.depth_decoder.generation_config).update(
        {"_from_model_config": False, **depth_decoder_attrs}
    )
    vars(model.generation_config).update(generation_config)


def load_runtime(
    ckpt_dir: Path,
    *,
    device: str,
    attn_implementation: str,
) -> tuple[AutoTokenizer, BreezeForConditionalGeneration, Any]:

    if device.startswith("cuda"):
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            rank, world_size, local_rank = get_dist_info()
            raise RuntimeError(
                "Failed to set CUDA device "
                f"device={device} rank={rank} world_size={world_size} local_rank={local_rank} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"device_count={torch.cuda.device_count()}"
            ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        fix_mistral_regex=False,
    )
    model = BreezeForConditionalGeneration.from_pretrained(
        ckpt_dir,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    model.to(device).eval()

    from qwen_tts import Qwen3TTSTokenizer

    bundled_audio_tokenizer = ckpt_dir / "audio_tokenizer"
    if not bundled_audio_tokenizer.is_dir():
        raise FileNotFoundError(
            "Bundled audio tokenizer not found at "
            f"{bundled_audio_tokenizer}. The Breeze model package must include "
            "the audio_tokenizer directory."
        )
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(bundled_audio_tokenizer), device_map=device
    )
    return tokenizer, model, audio_tokenizer
