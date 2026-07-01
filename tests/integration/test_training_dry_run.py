from __future__ import annotations

import json
from pathlib import Path

import pytest

from aimo_training.config import AIMOTrainingConfig
from aimo_training.entrypoints import train as train_entrypoint
from conftest import grpo_group


class FakeStepSummary:

    def as_dict(self) -> dict[str, int | float | str]:

        return {
            "step_index": 0,
            "group_index": 0,
            "problem_id": "p1",
            "loss": 0.0,
            "mean_reward": 0.5,
            "reward_std": 0.5,
            "sample_count": 2,
        }


class FakeTrainer:

    def __init__(self, config: AIMOTrainingConfig) -> None:

        self.config = config

    def train(self, groups: list[object]) -> tuple[list[FakeStepSummary], Path, Path]:

        self.config.output_path.mkdir(parents=True, exist_ok=True)
        adapter_path = self.config.output_path / "adapter_model.safetensors"
        adapter_config_path = self.config.output_path / "adapter_config.json"
        adapter_path.write_text("adapter\n", encoding="utf-8")
        adapter_config_path.write_text("{\"rank\": 64}\n", encoding="utf-8")

        return [
            FakeStepSummary(),
        ], adapter_path, adapter_config_path


def test_training_dry_run_with_mocked_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(train_entrypoint, "AIMOGRPOTrainer", FakeTrainer)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    dataset_path = tmp_path / "groups.jsonl"
    dataset_path.write_text(
        json.dumps(grpo_group(sample_count=2).as_dict()) + "\n",
        encoding="utf-8",
    )
    config = AIMOTrainingConfig(
        model_path=model_path,
        dataset_path=dataset_path,
        output_path=tmp_path / "adapter",
        logdir=tmp_path / "logs",
        group_size=2,
        page_count_method="word_count",
    )

    train_entrypoint.run(config=config)

    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "logs" / "run_metadata.json").exists()
    assert (tmp_path / "logs" / "training_arguments.json").exists()
    assert (tmp_path / "logs" / "source_dataset_manifest.json").exists()
    assert (tmp_path / "logs" / "training_step_summaries.jsonl").exists()
    assert json.loads((tmp_path / "logs" / "final_evaluation_summary.json").read_text(
        encoding="utf-8",
    ))["status"] == "training_complete"


def test_training_entrypoint_failure_writes_diagnostics(tmp_path: Path) -> None:

    with pytest.raises(FileNotFoundError):
        train_entrypoint.main([
            "--model_path",
            str(tmp_path / "model"),
            "--dataset_path",
            str(tmp_path / "missing_groups.jsonl"),
            "--output_path",
            str(tmp_path / "adapter"),
            "--logdir",
            str(tmp_path / "logs"),
        ])

    assert (tmp_path / "logs" / "failure_report.json").exists()
    assert (tmp_path / "logs" / "failure_report.txt").exists()
    assert (tmp_path / "logs" / "failure_traceback.txt").exists()
    assert (tmp_path / "logs" / "phase_events.jsonl").exists()
    assert (tmp_path / "adapter" / "failure_artifacts" / "failure_report.json").exists()
