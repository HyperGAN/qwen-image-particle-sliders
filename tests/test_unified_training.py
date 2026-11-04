"""The merged training dispatcher preserves Qwen fixture identity and continuation."""

import json
import subprocess
import sys

import torch


def test_dispatcher_plan_check_periodic_checkpoint_resume_and_v2(tmp_path):
    profile = {
        "schema": "particle-slider-profile-v1",
        "name": "qwen-pg094-fixture",
        "product": "qwen-image-particle-sliders",
        "config_kind": "session",
        "backend": "qwen",
        "interface": "image",
        "status": "fixture",
        "runtime": {"python": sys.executable},
        "capabilities": {"resume": True, "periodic_checkpoint": True, "check": True},
        "default_checkpoint_every": 1,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    config = {
        "backend": "qwen",
        "cohort": "qwen-pg094-dispatcher-fixture-v1",
        "options": {
            "dummy": True,
            "steps": 2,
            "height": 32,
            "width": 32,
            "trajectory_steps": 4,
            "records_per_pair": 2,
            "ema_decay": 0.9,
            "rows": [
                {"neutral": "a red cup", "positive": "a shiny red cup"},
                {"neutral": "a blue bowl", "positive": "a shiny blue bowl"},
            ],
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    def run(operation, *arguments):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "particle_sliders.execution.cli",
                operation,
                "--config",
                str(path),
                "--profile",
                str(profile_path),
                "--no-reporting",
                *map(str, arguments),
            ],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    plan = json.loads(run("plan"))
    assert plan["resolved"]["profile"]["status"] == "fixture"
    assert plan["training"]["options"]["dummy"] is True
    run("train", "--check")
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    run("train", "--out", first, "--stop-after", 1, "--selection", "ema", "--export-format", "both")
    checkpoint = first / "artifacts/checkpoints/step-000001/training.pt"
    assert checkpoint.is_file()
    run("train", "--out", resumed, "--resume", checkpoint, "--selection", "ema", "--export-format", "both")
    assert json.loads((resumed / "result.json").read_text())["completed_steps"] == 2
    packet = torch.load(resumed / "artifacts/adapter-v2.pt", weights_only=True)
    assert packet["specification"]["configuration"]["dummy"] is True
