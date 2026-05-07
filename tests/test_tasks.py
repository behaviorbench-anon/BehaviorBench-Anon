"""Tests for BehaviorBench task loading and smoke evaluation."""

from pathlib import Path

import pytest

from behaviorbench.eval.tasks import IEOEconomicsTask, PersScorePredTask
from behaviorbench.eval.tasks.game_behavior import GameBehaviorTask

DATA_DIR = Path(__file__).parent.parent / "data"


def require_data(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"BehaviorBench data not found: {path}")
    return str(path)


def test_personality_score_task_smoke():
    data_path = require_data(DATA_DIR / "big_five" / "pers_score_pred" / "test.jsonl")
    task = PersScorePredTask(data_path=data_path, num_samples=10)
    result = task.run_evaluation(lambda prompt, **kwargs: "[37]")

    assert result.task_name == "pers_score_pred"
    assert "MAE_averaged" in result.metrics
    assert len(result.predictions) == 10


def test_ieo_task_smoke():
    data_path = require_data(DATA_DIR / "economics_contests" / "example.jsonl")
    task = IEOEconomicsTask(data_path=data_path)
    result = task.run_evaluation(lambda prompt, **kwargs: "[A]")

    assert result.task_name == "ieo_economics"
    assert "accuracy_multi_choice" in result.metrics
    assert 0 <= result.metrics["accuracy_multi_choice"] <= 1


def test_game_behavior_task_smoke():
    data_path = require_data(DATA_DIR / "moblab" / "game_behavior" / "dictator_test.jsonl")
    task = GameBehaviorTask(
        data_path=data_path,
        name="game_behavior_dictator",
        metrics=["Wasserstein_with_ks"],
        clip_range=(0, 100),
        normalize_range=(0, 100),
        num_samples_per_game=5,
    )
    result = task.run_evaluation(lambda prompt, **kwargs: "[50]")

    assert result.task_name == "game_behavior_dictator"
    assert "Wasserstein_with_ks_distance" in result.metrics
    assert result.metrics["Wasserstein_with_ks_distance"] >= 0
