from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Callable

import pytest

from aimo_inference.client import AIMOGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.entrypoints.run import run
from aimo_inference.io import AIMOProblemRecord
from aimo_inference.judge import AIMOProofJudge
from aimo_inference.refinement import AIMORefinementEngine
from aimo_inference.template import AIMOChatMessage
from conftest import FakeHTTPServer
from conftest import FakeSandbox
from conftest import fake_generation
from conftest import read_csv_rows
from conftest import write_csv


class StaticGenerationClient:

    def __init__(self, text: str) -> None:

        self.text = text

    def generate(
        self,
        messages: Sequence[AIMOChatMessage],
        max_tokens: int,
    ) -> AIMOGeneration:

        return fake_generation(self.text)


def chat_response(text: str) -> dict[str, Any]:

    return {
        "choices": [
            {
                "message": {
                    "content": text,
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
        },
    }


def test_fake_proof_inference_against_fake_openai_server(
    tmp_path: Path,
    http_server_factory: Callable[[list[Any]], FakeHTTPServer],
) -> None:

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

    with http_server_factory([
        chat_response("Proof A."),
        chat_response("Proof B."),
    ]) as server:
        config = AIMOConfig(
            model_path=tmp_path / "missing-model",
            input_csv=input_path,
            output_csv=tmp_path / "predictions.csv",
            logdir=tmp_path / "logs",
            api_base=server.api_base,
            launch_server=False,
            reuse_server=False,
            enable_tools=False,
            page_count_method="word_count",
        )

        run(config=config)

    assert read_csv_rows(tmp_path / "predictions.csv") == [
        {
            "id": "p1",
            "prediction": "Proof A.",
        },
        {
            "id": "p2",
            "prediction": "Proof B.",
        },
    ]
    assert (tmp_path / "logs" / "p1.json").exists()
    assert (tmp_path / "logs" / "p2.json").exists()
    run_metadata = json.loads((tmp_path / "logs" / "run_metadata.json").read_text(
        encoding="utf-8",
    ))
    assert run_metadata["summary"]["succeeded"] == 2
    assert len(server.requests) == 2


def test_data_to_inference_smoke_from_fixture_products(
    tmp_path: Path,
    http_server_factory: Callable[[list[Any]], FakeHTTPServer],
) -> None:

    pytest.importorskip("pyarrow")

    from aimo_data.build_dataset import BuildConfig
    from aimo_data.build_dataset import BuildProducts
    from aimo_data.build_dataset import EVAL_SCHEMA_COLUMNS
    from aimo_data.build_dataset import PROOF_SCHEMA_COLUMNS
    from aimo_data.build_dataset import build_manifest
    from aimo_data.build_dataset import normalize_source_row
    from aimo_data.build_dataset import project_row
    from aimo_data.build_dataset import write_products

    source_rows = [
        normalize_source_row({
            "id": "m1",
            "problem_markdown": "Prove A.",
            "solutions_markdown": [
                "Reference A.",
            ],
            "country": "US",
            "competition": "IMO",
            "topics_flat": [
                "algebra",
            ],
            "language": "English",
            "problem_type": "proof only",
            "final_answer": "1",
            "image_count": 0,
            "source_config": "fixture",
        }),
        normalize_source_row({
            "id": "m2",
            "problem_markdown": "Prove B.",
            "solutions_markdown": [
                "Reference B.",
            ],
            "country": "GR",
            "competition": "BMO",
            "topics_flat": [
                "geometry",
            ],
            "language": "English",
            "problem_type": "proof and answer",
            "final_answer": "2",
            "image_count": 0,
            "source_config": "fixture",
        }),
    ]

    for row in source_rows:
        row["reference_rendered_pages"] = 4
        row["reference_page_count_method"] = "word_count"

    eval_rows = [
        project_row(row=source_rows[0], columns=EVAL_SCHEMA_COLUMNS),
    ]
    eval_reference_rows = [
        project_row(row=source_rows[0], columns=PROOF_SCHEMA_COLUMNS),
    ]
    train_rows = [
        project_row(row=source_rows[1], columns=PROOF_SCHEMA_COLUMNS),
    ]
    build_config = BuildConfig(
        source_dir=tmp_path / "source",
        output_dir=tmp_path / "data",
        eval_size=1,
        train_size=0,
        seed=42,
        exclude_images=True,
        language_filter="any",
        page_count_method="word_count",
        latex_command="pdflatex",
        pdfinfo_command="pdfinfo",
        page_count_cache_path=None,
        page_count_timeout_seconds=1,
        source_dataset_name="MathNet",
        source_snapshot="fixture",
        write_answer_dataset=False,
        answer_min=0,
        answer_max=7,
    )
    products = BuildProducts(
        eval_rows=eval_rows,
        eval_input_rows=[
            {
                "id": "m1",
                "problem": "Prove A.",
            },
        ],
        eval_reference_rows=eval_reference_rows,
        train_rows=train_rows,
        judge_rows=train_rows,
        answer_rows=[],
        manifest=build_manifest(
            config=build_config,
            parquet_paths=[],
            source_rows=source_rows,
            filter_counts={
                "eligible_proof_rows": 2,
            },
            excluded_counts={},
            eval_rows=eval_rows,
            eval_reference_rows=eval_reference_rows,
            train_rows=train_rows,
            judge_rows=train_rows,
            answer_rows=[],
            selected_train_topic_counts={
                "algebra": 1,
            },
        ),
    )
    write_products(
        products=products,
        output_dir=build_config.output_dir,
    )

    with http_server_factory([chat_response("Proof from built input.")]) as server:
        run(config=AIMOConfig(
            model_path=tmp_path / "missing-model",
            input_csv=build_config.output_dir / "aimo_proof_eval_input.csv",
            output_csv=tmp_path / "predictions.csv",
            logdir=tmp_path / "logs",
            api_base=server.api_base,
            launch_server=False,
            reuse_server=False,
            enable_tools=False,
            page_count_method="word_count",
        ))

    assert read_csv_rows(tmp_path / "predictions.csv") == [
        {
            "id": "m1",
            "prediction": "Proof from built input.",
        },
    ]


def test_judge_assisted_proof_smoke() -> None:

    judge = AIMOProofJudge(
        config=AIMOConfig(
            page_count_method="word_count",
        ),
        client=StaticGenerationClient("Complete. \\boxed{7}"),
    )
    engine = AIMORefinementEngine(
        config=AIMOConfig(
            enable_tools=False,
            page_count_method="word_count",
        ),
        client=StaticGenerationClient("word " * 2200),
        sandbox=FakeSandbox(),
        judge=judge,
    )

    result = engine.run_problem(
        record=AIMOProblemRecord(
            order_index=0,
            id="p1",
            problem="Problem.",
            metadata={
                "reference_solution": "Reference.",
            },
        )
    )

    assert result.success is True
    assert result.metadata["judge"]["grade"] == 7
    assert result.metadata["judge"]["solution_page_reward"] == 1
