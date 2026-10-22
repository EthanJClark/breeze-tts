from types import SimpleNamespace

import pytest

from breeze_infer.runtime import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MAX_TEMPERATURE,
    set_generation_sampling,
    set_generation_temperature,
    validate_repetition_penalty,
    validate_temperature,
    validate_top_k,
    validate_top_p,
)


def _model() -> SimpleNamespace:
    config = SimpleNamespace(
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        top_k=DEFAULT_TOP_K,
    )
    return SimpleNamespace(
        generation_config=config,
        depth_decoder=SimpleNamespace(generation_config=config),
    )


def test_set_generation_sampling_updates_both_configs() -> None:
    model = _model()

    set_generation_sampling(model, temperature=1.2, top_p=0.9, top_k=40)

    for config in (model.generation_config, model.depth_decoder.generation_config):
        assert config.temperature == 1.2
        assert config.top_p == 0.9
        assert config.top_k == 40


def test_set_generation_temperature_updates_both_configs() -> None:
    model = _model()

    set_generation_temperature(model, 1.2)

    assert model.generation_config.temperature == 1.2
    assert model.depth_decoder.generation_config.temperature == 1.2


def test_validate_temperature_accepts_bounds() -> None:
    assert validate_temperature(0.05) == 0.05
    assert validate_temperature(MAX_TEMPERATURE) == MAX_TEMPERATURE


@pytest.mark.parametrize(
    "temperature",
    [0.0, -0.1, float("nan"), float("inf"), MAX_TEMPERATURE + 0.01],
)
def test_validate_temperature_rejects_invalid_values(temperature) -> None:
    with pytest.raises(ValueError, match="temperature"):
        validate_temperature(temperature)


@pytest.mark.parametrize("top_p", [0.0, -0.1, 1.01, float("nan")])
def test_validate_top_p_rejects_invalid_values(top_p) -> None:
    with pytest.raises(ValueError, match="top_p"):
        validate_top_p(top_p)


def test_validate_top_p_accepts_bounds() -> None:
    assert validate_top_p(0.05) == 0.05
    assert validate_top_p(1.0) == 1.0


def test_validate_top_k_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="top_k"):
        validate_top_k(-1)


@pytest.mark.parametrize("penalty", [0.0, -0.1, float("nan")])
def test_validate_repetition_penalty_rejects_invalid_values(penalty) -> None:
    with pytest.raises(ValueError, match="repetition_penalty"):
        validate_repetition_penalty(penalty)


def test_set_generation_temperature_rejects_invalid_value() -> None:
    model = _model()

    with pytest.raises(ValueError, match="temperature"):
        set_generation_temperature(model, 0.0)

    assert model.generation_config.temperature == DEFAULT_TEMPERATURE
    assert model.depth_decoder.generation_config.temperature == DEFAULT_TEMPERATURE
