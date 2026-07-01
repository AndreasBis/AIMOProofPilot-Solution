from __future__ import annotations

import json
from pathlib import Path

import pytest

from aimo_inference.config import AIMOConfig
from aimo_inference.io import AIMOInferenceIO
from aimo_inference.io import AIMOProblemResult
from conftest import read_csv_rows
from conftest import write_csv


def test_csv_input_with_id_and_problem(tmp_path: Path) -> None:

    input_path = write_csv(
        tmp_path / "input.csv",
        [
            {
                "id": "p1",
                "problem": "Prove A.",
            },
            {
                "id": "p2",
                "problem": "Prove B.",
            },
        ],
    )
    config = AIMOConfig(
        input_csv=input_path,
        sample_eval_problems=False,
    )

    records = AIMOInferenceIO(config=config).read_records()

    assert [record.id for record in records] == [
        "p1",
        "p2",
    ]
    assert records[0].problem == "Prove A."
    assert records[0].metadata == {}


def test_csv_input_with_alternative_problem_columns(tmp_path: Path) -> None:

    input_path = write_csv(
        tmp_path / "input.csv",
        [
            {
                "problem_id": "q1",
                "statement": "Show B.",
                "difficulty": "easy",
            },
        ],
    )
    config = AIMOConfig(
        input_csv=input_path,
        sample_eval_problems=False,
    )

    record = AIMOInferenceIO(config=config).read_records()[0]

    assert record.id == "q1"
    assert record.problem == "Show B."
    assert record.metadata == {
        "difficulty": "easy",
    }


def test_parquet_input_for_eval_sampling(tmp_path: Path) -> None:

    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    input_path = tmp_path / "eval.parquet"
    table = pyarrow.Table.from_pylist([
        {
            "id": "p1",
            "problem": "Prove A.",
        },
        {
            "id": "p2",
            "problem": "Prove B.",
        },
    ])
    parquet.write_table(table, input_path)
    config = AIMOConfig(
        eval_dataset_path=input_path,
        sample_eval_problems=True,
        eval_sample_size=2,
        eval_sample_seed=42,
    )

    records = AIMOInferenceIO(config=config).read_records()

    assert [record.id for record in records] == [
        "p1",
        "p2",
    ]


def test_directory_input_with_multiple_parquets(tmp_path: Path) -> None:

    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    input_dir = tmp_path / "eval"
    input_dir.mkdir()
    parquet.write_table(
        pyarrow.Table.from_pylist([
            {
                "id": "p1",
                "problem": "Prove A.",
            },
        ]),
        input_dir / "part-1.parquet",
    )
    parquet.write_table(
        pyarrow.Table.from_pylist([
            {
                "id": "p2",
                "problem": "Prove B.",
            },
        ]),
        input_dir / "part-2.parquet",
    )
    config = AIMOConfig(
        eval_dataset_path=input_dir,
        sample_eval_problems=True,
        eval_sample_size=2,
        eval_sample_seed=42,
    )

    records = AIMOInferenceIO(config=config).read_records()

    assert [record.id for record in records] == [
        "p1",
        "p2",
    ]


def test_deterministic_eval_sampling(tmp_path: Path) -> None:

    eval_path = write_csv(
        tmp_path / "eval.csv",
        [
            {
                "id": f"p{index}",
                "problem": f"Problem {index}",
            }
            for index in range(8)
        ],
    )
    config = AIMOConfig(
        eval_dataset_path=eval_path,
        sample_eval_problems=True,
        eval_sample_size=3,
        eval_sample_seed=17,
    )

    first_ids = [
        record.id
        for record in AIMOInferenceIO(config=config).read_records()
    ]
    second_ids = [
        record.id
        for record in AIMOInferenceIO(config=config).read_records()
    ]

    assert first_ids == second_ids
    assert len(first_ids) == 3


def test_output_csv_log_json_and_atomic_final_path(tmp_path: Path) -> None:

    output_path = tmp_path / "nested" / "predictions.csv"
    logdir = tmp_path / "logs"
    config = AIMOConfig(
        output_csv=output_path,
        logdir=logdir,
        write_intermediate_outputs=True,
    )
    io_manager = AIMOInferenceIO(config=config)
    results = [
        AIMOProblemResult(
            order_index=1,
            id="p2",
            prediction="Proof B.",
            success=True,
            error="",
            metadata={
                "rank": 0,
            },
        ),
        AIMOProblemResult(
            order_index=0,
            id="p1",
            prediction="Proof A.",
            success=True,
            error="",
            metadata={},
        ),
    ]

    written_path = io_manager.write_predictions(results)
    log_path = io_manager.write_problem_log(results[0])
    metadata_path = io_manager.write_run_metadata({
        "summary": {
            "succeeded": 2,
        },
    })

    assert written_path == output_path
    assert read_csv_rows(output_path) == [
        {
            "id": "p1",
            "prediction": "Proof A.",
        },
        {
            "id": "p2",
            "prediction": "Proof B.",
        },
    ]
    assert json.loads(log_path.read_text(encoding="utf-8"))["id"] == "p2"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["summary"] == {
        "succeeded": 2,
    }
    assert not list(output_path.parent.glob(".predictions.csv.*.tmp"))


def test_ranked_output_path_is_under_logdir(tmp_path: Path) -> None:

    config = AIMOConfig(
        output_csv=tmp_path / "predictions.csv",
        logdir=tmp_path / "logs",
        global_rank=3,
        world_size=8,
    )

    output_path = AIMOInferenceIO(config=config).output_csv_path()

    assert output_path == tmp_path / "logs" / "rank_outputs" / "predictions.rank_00003.csv"
