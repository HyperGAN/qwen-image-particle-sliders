"""LR admission and exact shared recipe wiring; no pretrained/GPU work."""

import pytest

from qwen_image_sliders.learning_rates import recipe_overrides


@pytest.mark.parametrize("key", ["lr", "d_lr_mult", "prior_lr_mult"])
@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), "0.001"])
def test_invalid_rates(key, value):
    with pytest.raises(ValueError):
        recipe_overrides(**{key: value})


def test_optional_rates_keep_preset_defaults():
    assert recipe_overrides() == {}
    assert recipe_overrides(lr=2e-5, d_lr_mult=3) == {"lr": 2e-5, "d_lr_mult": 3.0}


def test_alias_validation():
    from qwen_image_sliders.learning_rates import resolve_alias

    assert resolve_alias(g_lr=2e-5) == resolve_alias(lr=2e-5)
    assert resolve_alias(lr=2e-5, g_lr=2e-5) == {"lr": 2e-5}
    with pytest.raises(ValueError, match="conflict"):
        resolve_alias(lr=2e-5, g_lr=3e-5)
    with pytest.raises(ValueError):
        resolve_alias(lr=2e-5, g_lr=True)


def test_rates_bind_current_cli_game_resume(tmp_path):
    import torch

    from qwen_image_sliders.teachers import load_prompts
    from qwen_image_sliders.tiny import TinyRuntime
    from qwen_image_sliders.train import train

    rates = {"lr": 2e-5, "d_lr_mult": 3, "prior_lr_mult": 4}
    train(TinyRuntime(), load_prompts(), tmp_path, steps=4, stop_after=2, **rates)
    packet = torch.load(tmp_path / "training.pt", weights_only=True)
    assert packet["run_config"]["recipe_overrides"] == rates
    recipe = packet["game"]["config"]["recipe"]
    assert all(recipe[key] == value for key, value in rates.items())
    train(
        TinyRuntime(),
        load_prompts(),
        tmp_path,
        steps=4,
        resume=tmp_path / "training.pt",
        g_lr=2e-5,
        d_lr_mult=3,
        prior_lr_mult=4,
    )
    with pytest.raises(ValueError, match="resume"):
        train(
            TinyRuntime(),
            load_prompts(),
            tmp_path / "bad",
            steps=4,
            resume=tmp_path / "training.pt",
            lr=3e-5,
            d_lr_mult=3,
            prior_lr_mult=4,
        )
