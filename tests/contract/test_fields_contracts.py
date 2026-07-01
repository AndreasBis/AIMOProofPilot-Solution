from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aimo_inference.config import AIMOConfig
from aimo_inference.entrypoints import run as run_entrypoint
from aimo_training.config import AIMOTrainingConfig
from aimo_training.entrypoints import train as train_entrypoint
from conftest import read_csv_rows
from conftest import write_csv


def test_fields_run_contract_accepts_command_shape_with_mocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    input_path = write_csv(
        tmp_path / "input.csv",
        [
            {
                "id": "p1",
                "problem": "Problem.",
            },
        ],
    )
    output_path = tmp_path / "nested" / "output.csv"
    logdir = tmp_path / "logs"
    captured_configs: list[AIMOConfig] = []

    def fake_run(config: AIMOConfig) -> None:

        captured_configs.append(config)
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        config.logdir.mkdir(parents=True, exist_ok=True)

        with config.output_csv.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "id",
                    "prediction",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "id": "p1",
                "prediction": "Mock proof.",
            })

        (config.logdir / "run_metadata.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(run_entrypoint, "run", fake_run)

    exit_code = run_entrypoint.main([
        "--model_path",
        str(tmp_path / "model"),
        "--input_csv",
        str(input_path),
        "--output_csv",
        str(output_path),
        "--logdir",
        str(logdir),
        "--launch_server",
        "false",
        "--reuse_server",
        "false",
    ])

    assert exit_code == 0
    assert captured_configs[0].model_path == tmp_path / "model"
    assert output_path.parent.exists()
    assert read_csv_rows(output_path) == [
        {
            "id": "p1",
            "prediction": "Mock proof.",
        },
    ]
    assert (logdir / "run_metadata.json").exists()
    assert captured_configs[0].api_base == ""


def test_fields_run_contract_requires_standard_path_arguments() -> None:

    with pytest.raises(SystemExit):
        run_entrypoint.main([])


def test_fields_train_contract_accepts_command_shape_and_distributed_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setenv("GLOBAL_RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29501")
    output_path = tmp_path / "adapter"
    logdir = tmp_path / "logs"
    captured_configs: list[AIMOTrainingConfig] = []

    def fake_run(config: AIMOTrainingConfig) -> None:

        captured_configs.append(config)
        config.output_path.mkdir(parents=True, exist_ok=True)
        config.logdir.mkdir(parents=True, exist_ok=True)
        (config.output_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (config.logdir / "run_metadata.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(train_entrypoint, "run", fake_run)

    exit_code = train_entrypoint.main([
        "--model_path",
        str(tmp_path / "model"),
        "--dataset_path",
        str(tmp_path / "dataset.jsonl"),
        "--output_path",
        str(output_path),
        "--logdir",
        str(logdir),
    ])

    assert exit_code == 0
    assert captured_configs[0].model_path == tmp_path / "model"
    assert captured_configs[0].dataset_path == tmp_path / "dataset.jsonl"
    assert captured_configs[0].group_size == 16
    assert captured_configs[0].global_rank == 1
    assert captured_configs[0].world_size == 2
    assert output_path.exists()
    assert (logdir / "run_metadata.json").exists()
