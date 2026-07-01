from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from aimo_data.build_dataset import BuildConfig
from aimo_data.build_dataset import BuildProducts
from aimo_data.build_dataset import EVAL_SCHEMA_COLUMNS
from aimo_data.build_dataset import PageCountResult
from aimo_data.build_dataset import PROOF_SCHEMA_COLUMNS
from aimo_data.build_dataset import build_dataset
from aimo_data.build_dataset import build_answer_rows
from aimo_data.build_dataset import build_manifest
from aimo_data.build_dataset import count_topic_groups
from aimo_data.build_dataset import filter_proof_rows
from aimo_data.build_dataset import normalize_source_row
from aimo_data.build_dataset import parse_aimo_answer
from aimo_data.build_dataset import project_row
from aimo_data.build_dataset import select_eval_rows
from aimo_data.build_dataset import select_train_rows
from aimo_data.build_dataset import validate_eval_input_rows
from aimo_data.build_dataset import validate_eval_train_disjoint
from aimo_data.build_dataset import validate_page_count_fields
from aimo_data.build_dataset import write_products


def build_config(tmp_path: Path) -> BuildConfig:

    return BuildConfig(
        source_dir=tmp_path / "source",
        output_dir=tmp_path / "output",
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
        write_answer_dataset=True,
        answer_min=0,
        answer_max=7,
    )


def normalized_row(
    row_id: str,
    problem_type: str = "proof only",
    solution: str = "Reference proof.",
    image_count: int = 0,
    topic: str = "algebra",
) -> dict[str, object]:

    return normalize_source_row({
        "id": row_id,
        "problem_markdown": f"Problem {row_id} ![image](x.png)",
        "solutions_markdown": [
            solution,
        ] if solution else [],
        "images": [
            "x.png",
        ] if image_count else [],
        "country": row_id,
        "competition": f"Competition {row_id}",
        "topics_flat": [
            topic,
        ],
        "language": "English",
        "problem_type": problem_type,
        "final_answer": "\\boxed{3}",
        "image_count": image_count,
        "source_config": "fixture",
    })


def proof_row(row_id: str) -> dict[str, object]:

    row = normalized_row(
        row_id=row_id,
        topic="number theory",
    )
    row["reference_rendered_pages"] = 4
    row["reference_page_count_method"] = "word_count"

    return row


def test_mathnet_like_parquet_normalization_and_solution_selection() -> None:

    row = normalized_row(
        row_id="m1",
        solution="First solution.",
        image_count=1,
    )

    assert row["id"] == "m1"
    assert row["problem"] == "Problem m1"
    assert row["solution"] == "First solution."
    assert row["has_images"] is True
    assert row["image_count"] == 1
    assert row["topic_group"] == "algebra"


def test_no_solution_and_image_filtering(tmp_path: Path) -> None:

    rows = [
        normalized_row("keep"),
        normalized_row("drop_solution", solution=""),
        normalized_row("drop_image", image_count=1),
        normalized_row("drop_type", problem_type="answer only"),
    ]

    filtered_rows, filter_counts, excluded_counts = filter_proof_rows(
        rows=rows,
        config=build_config(tmp_path),
    )

    assert [row["id"] for row in filtered_rows] == [
        "keep",
    ]
    assert filter_counts["eligible_proof_rows"] == 1
    assert excluded_counts["reference_solution"] == 1
    assert excluded_counts["text_only"] == 1


def test_deterministic_split_and_train_eval_disjointness(tmp_path: Path) -> None:

    rows = [
        proof_row(f"p{index}")
        for index in range(5)
    ]
    config = build_config(tmp_path)

    first_eval_rows = select_eval_rows(rows=rows, config=config)
    second_eval_rows = select_eval_rows(rows=rows, config=config)
    train_rows = [
        row
        for row in rows
        if row["id"] not in {
            eval_row["id"]
            for eval_row in first_eval_rows
        }
    ]

    assert first_eval_rows == second_eval_rows
    validate_eval_train_disjoint(
        eval_rows=first_eval_rows,
        train_rows=train_rows,
    )


def test_stratified_train_selection_preserves_topic_distribution(tmp_path: Path) -> None:

    rows = [
        normalized_row(
            row_id=f"algebra_{index}",
            topic="algebra",
        )
        for index in range(6)
    ] + [
        normalized_row(
            row_id=f"geometry_{index}",
            topic="geometry",
        )
        for index in range(2)
    ]

    for row in rows:
        row["reference_rendered_pages"] = 4
        row["reference_page_count_method"] = "word_count"

    train_rows = select_train_rows(
        rows=rows,
        excluded_ids=set(),
        config=replace(
            build_config(tmp_path),
            train_size=4,
        ),
    )
    topic_counts = {
        "algebra": 0,
        "geometry": 0,
    }

    for row in train_rows:
        topic_counts[str(row["topic_group"])] += 1

    assert len(train_rows) == 4
    assert topic_counts == {
        "algebra": 3,
        "geometry": 1,
    }


def test_eval_input_rows_have_exact_id_problem_columns() -> None:

    validate_eval_input_rows([
        {
            "id": "p1",
            "problem": "Problem.",
        },
    ])

    with pytest.raises(ValueError, match="exactly id,problem"):
        validate_eval_input_rows([
            {
                "id": "p1",
                "problem": "Problem.",
                "extra": "x",
            },
        ])


def test_reference_page_count_fields_exist() -> None:

    validate_page_count_fields([
        {
            "id": "p1",
            "solution": "Proof.",
            "reference_rendered_pages": 4,
            "reference_page_count_method": "word_count",
        },
    ])

    with pytest.raises(ValueError, match="Missing reference page count"):
        validate_page_count_fields([
            {
                "id": "p1",
                "solution": "Proof.",
                "reference_rendered_pages": 0,
                "reference_page_count_method": "word_count",
            },
        ])


def test_manifest_counts_and_output_hashes(tmp_path: Path) -> None:

    config = build_config(tmp_path)
    eval_source_rows = [
        proof_row("eval"),
    ]
    train_source_rows = [
        proof_row("train"),
    ]
    eval_rows = [
        project_row(row=row, columns=EVAL_SCHEMA_COLUMNS)
        for row in eval_source_rows
    ]
    eval_reference_rows = [
        project_row(row=row, columns=PROOF_SCHEMA_COLUMNS)
        for row in eval_source_rows
    ]
    train_rows = [
        project_row(row=row, columns=PROOF_SCHEMA_COLUMNS)
        for row in train_source_rows
    ]
    answer_rows = build_answer_rows(
        rows=eval_source_rows + train_source_rows,
        excluded_ids={
            "eval",
        },
        config=config,
    )
    manifest = build_manifest(
        config=config,
        parquet_paths=[
            tmp_path / "source.parquet",
        ],
        source_rows=eval_source_rows + train_source_rows,
        filter_counts={
            "eligible_proof_rows": 2,
        },
        excluded_counts={},
        eval_rows=eval_rows,
        eval_reference_rows=eval_reference_rows,
        train_rows=train_rows,
        judge_rows=train_rows,
        answer_rows=answer_rows,
        selected_train_topic_counts=count_topic_groups(train_source_rows),
    )
    products = BuildProducts(
        eval_rows=eval_rows,
        eval_input_rows=[
            {
                "id": "eval",
                "problem": "Problem eval",
            },
        ],
        eval_reference_rows=eval_reference_rows,
        train_rows=train_rows,
        judge_rows=train_rows,
        answer_rows=answer_rows,
        manifest=manifest,
    )

    write_products(
        products=products,
        output_dir=config.output_dir,
    )

    manifest_text = (config.output_dir / "manifest.json").read_text(encoding="utf-8")

    assert "\"aimo_proof_eval_input.csv\": 1" in manifest_text
    assert products.manifest["output_hashes"]["aimo_proof_eval_input.csv"]
    assert (config.output_dir / "aimo_proof_eval_input.csv").read_text(
        encoding="utf-8",
    ).splitlines()[0] == "id,problem"


def test_build_dataset_page_counts_only_selected_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    source_path = tmp_path / "source.parquet"
    rows = [
        {
            "id": f"p{index}",
            "problem_markdown": f"Problem {index}.",
            "solutions_markdown": [
                f"Reference proof {index}.",
            ],
            "images": [],
            "country": f"Country {index}",
            "competition": f"Competition {index}",
            "topics_flat": [
                "algebra",
            ],
            "language": "English",
            "problem_type": "proof only",
            "final_answer": "",
        }
        for index in range(12)
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, source_path)
    counted_solutions: list[str] = []

    def fake_count_reference_pages(solution: str, config: BuildConfig) -> PageCountResult:

        counted_solutions.append(solution)

        return PageCountResult(rendered_pages=4, method="latex")

    monkeypatch.setattr(
        "aimo_data.build_dataset.count_reference_pages",
        fake_count_reference_pages,
    )
    products = build_dataset(
        replace(
            build_config(tmp_path),
            source_dir=source_path,
            eval_size=2,
            train_size=4,
            write_answer_dataset=False,
        ),
    )

    assert len(counted_solutions) == 6
    assert len(products.eval_rows) == 2
    assert len(products.train_rows) == 4
    assert products.manifest["selected_train_topic_counts"] == {
        "algebra": 4,
    }


def test_answer_row_parsing() -> None:

    assert parse_aimo_answer("\\boxed{7}", 0, 7) == 7
    assert parse_aimo_answer("The answer is 3.", 0, 7) == 3
    assert parse_aimo_answer("\\boxed{9}", 0, 7) is None
