from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from aimo_training.schema import AIMOTrainingRecord


PROBLEM_COLUMNS = (
    "problem",
    "problem_markdown",
    "statement",
    "prompt",
    "question",
)

REFERENCE_COLUMNS = (
    "solution",
    "reference_solution",
    "target_solution",
    "rubric",
    "all_solutions",
)


def read_training_records(path: Path) -> list[AIMOTrainingRecord]:

    rows = read_dataset_rows(path)

    return [
        training_record_from_row(row=row, order_index=order_index)
        for order_index, row in enumerate(rows)
    ]


def validate_training_dataset(
    path: Path,
    problems_per_update: int,
    group_size: int,
) -> dict[str, Any]:

    if not path.exists():
        raise FileNotFoundError(f"Training dataset does not exist: {path}")

    if not os_access_readable(path):
        raise PermissionError(f"Training dataset is not readable: {path}")

    rows = read_dataset_rows(path)

    if not rows:
        raise ValueError(f"Training dataset is empty: {path}")

    records = [
        training_record_from_row(row=row, order_index=order_index)
        for order_index, row in enumerate(rows)
    ]

    return {
        "dataset_path": str(path),
        "row_count": len(rows),
        "record_count": len(records),
        "problems_per_update": problems_per_update,
        "group_size": group_size,
        "expected_rollouts_per_update": problems_per_update * group_size,
        "first_record_id": records[0].id,
    }


def read_dataset_rows(path: Path) -> list[dict[str, Any]]:

    if path.is_dir():
        return read_directory_rows(path)

    if path.suffix.lower() == ".csv":
        return read_csv_rows(path)

    if path.suffix.lower() == ".parquet":
        return read_parquet_rows([path])

    raise ValueError(f"Unsupported dataset format: {path}")


def os_access_readable(path: Path) -> bool:

    return os.access(path, os.R_OK)


def read_directory_rows(path: Path) -> list[dict[str, Any]]:

    parquet_paths = sorted(path.rglob("*.parquet"))

    if parquet_paths:
        return read_parquet_rows(parquet_paths)

    csv_paths = sorted(path.rglob("*.csv"))

    if csv_paths:
        rows: list[dict[str, Any]] = []

        for csv_path in csv_paths:
            rows.extend(read_csv_rows(csv_path))

        return rows

    raise FileNotFoundError(f"No CSV or parquet files found under: {path}")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:

    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)

        return [
            {
                str(key): value
                for key, value in row.items()
            }
            for row in reader
        ]


def read_parquet_rows(paths: list[Path]) -> list[dict[str, Any]]:

    import pyarrow.parquet as pq

    table = pq.read_table(paths)

    return [
        {
            str(key): value
            for key, value in row.items()
        }
        for row in table.to_pylist()
    ]


def training_record_from_row(row: dict[str, Any], order_index: int) -> AIMOTrainingRecord:

    problem_column = first_present_column(row=row, columns=PROBLEM_COLUMNS)
    reference_column = first_present_optional_column(row=row, columns=REFERENCE_COLUMNS)
    problem_id = str(row.get("id") or row.get("problem_id") or order_index)
    problem = str(row.get(problem_column, "")).strip()
    reference_solution = normalize_reference_value(
        row.get(reference_column, "") if reference_column else ""
    )
    metadata = {
        str(key): value
        for key, value in row.items()
        if key not in {"id", "problem_id", problem_column, reference_column}
    }

    if not problem:
        raise ValueError(f"Training row {order_index} has no problem text.")

    return AIMOTrainingRecord(
        order_index=order_index,
        id=problem_id,
        problem=problem,
        reference_solution=reference_solution,
        metadata=metadata,
    )


def first_present_column(row: dict[str, Any], columns: tuple[str, ...]) -> str:

    for column in columns:
        if column in row and row[column] is not None and str(row[column]).strip():
            return column

    for key, value in row.items():
        if key not in {"id", "problem_id"} and value is not None and str(value).strip():
            return str(key)

    raise ValueError("Dataset row does not include a usable text column.")


def first_present_optional_column(row: dict[str, Any], columns: tuple[str, ...]) -> str:

    for column in columns:
        if column in row and row[column] is not None and str(row[column]).strip():
            return column

    return ""


def normalize_reference_value(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, list):
        return "\n\n".join(
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        )

    text = str(value).strip()

    if text.startswith("["):
        try:
            parsed_value = json.loads(text)
        except json.JSONDecodeError:
            return text

        if isinstance(parsed_value, list):
            return "\n\n".join(
                str(item).strip()
                for item in parsed_value
                if item is not None and str(item).strip()
            )

    return text


def build_source_dataset_manifest(path: Path, records: list[AIMOTrainingRecord]) -> dict[str, Any]:

    file_paths = discover_source_files(path)

    return {
        "dataset_path": str(path),
        "record_count": len(records),
        "source_files": [
            str(file_path)
            for file_path in file_paths
        ],
        "source_file_hashes": {
            str(file_path): hash_file(file_path)
            for file_path in file_paths
        },
        "record_ids": [
            record.id
            for record in records
        ],
    }


def discover_source_files(path: Path) -> list[Path]:

    if path.is_file():
        return [path]

    if not path.exists():
        return []

    return sorted([
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in {".csv", ".parquet", ".jsonl"}
    ])


def hash_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()
