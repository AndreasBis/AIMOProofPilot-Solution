from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimo_inference.config import AIMOConfig


@dataclass(frozen=True)
class AIMOProblemRecord:

    order_index: int
    id: str
    problem: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class AIMOProblemResult:

    order_index: int
    id: str
    prediction: str
    success: bool
    error: str
    metadata: dict[str, Any]


class AIMOInferenceIO:

    problem_columns = (
        "problem",
        "problem_markdown",
        "statement",
        "prompt",
        "question",
    )

    def __init__(self, config: AIMOConfig) -> None:

        self.config = config

    def ensure_logdir(self) -> None:

        self.config.logdir.mkdir(parents=True, exist_ok=True)

    def output_csv_path(self) -> Path:

        if self.config.world_size <= 1:
            return self.config.output_csv

        suffix = self.config.output_csv.suffix or ".csv"
        rank_output_name = (
            f"{self.config.output_csv.stem}.rank_{self.config.global_rank:05d}{suffix}"
        )

        return self.config.logdir / "rank_outputs" / rank_output_name

    def read_records(self) -> list[AIMOProblemRecord]:

        rows = self._read_input_rows()

        return [
            self._record_from_row(row=row, order_index=order_index)
            for order_index, row in enumerate(rows)
        ]

    def write_predictions(self, results: list[AIMOProblemResult]) -> Path:

        output_path = self.output_csv_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        ordered_results = sorted(results, key=lambda result: result.order_index)

        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "id",
                    "prediction",
                ],
            )
            writer.writeheader()

            for result in ordered_results:
                writer.writerow({
                    "id": result.id,
                    "prediction": result.prediction,
                })

        os.replace(temporary_path, output_path)

        return output_path

    def write_answers(self, results: list[AIMOProblemResult]) -> Path:

        output_path = self.output_csv_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        ordered_results = sorted(results, key=lambda result: result.order_index)

        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "id",
                    "answer",
                ],
            )
            writer.writeheader()

            for result in ordered_results:
                writer.writerow({
                    "id": result.id,
                    "answer": result.prediction,
                })

        os.replace(temporary_path, output_path)

        return output_path

    def write_judge_results(self, results: list[AIMOProblemResult]) -> Path:

        output_path = self.output_csv_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        ordered_results = sorted(results, key=lambda result: result.order_index)

        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "id",
                    "grade",
                ],
            )
            writer.writeheader()

            for result in ordered_results:
                writer.writerow({
                    "id": result.id,
                    "grade": result.prediction,
                })

        os.replace(temporary_path, output_path)

        return output_path

    def write_problem_log(self, result: AIMOProblemResult) -> Path:

        self.ensure_logdir()
        log_path = self.config.logdir / f"{self._safe_id(result.id)}.json"
        self._write_json(
            path=log_path,
            payload={
                "id": result.id,
                "order_index": result.order_index,
                "success": result.success,
                "error": result.error,
                "prediction": result.prediction,
                "metadata": result.metadata,
            },
        )

        return log_path

    def write_run_metadata(self, payload: dict[str, Any]) -> Path:

        self.ensure_logdir()
        metadata_path = self.config.logdir / "run_metadata.json"
        self._write_json(
            path=metadata_path,
            payload={
                "created_at_unix": time.time(),
                **payload,
            },
        )

        return metadata_path

    def write_problem_logs(self, results: list[AIMOProblemResult]) -> list[Path]:

        if not self.config.write_intermediate_outputs:
            return []

        return [
            self.write_problem_log(result)
            for result in results
        ]

    def _record_from_row(self, row: dict[str, str], order_index: int) -> AIMOProblemRecord:

        problem_column = self._problem_column(row)
        problem_text = row.get(problem_column, "")
        record_id = row.get("id") or row.get("problem_id") or str(order_index)
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"id", "problem_id", problem_column}
        }

        return AIMOProblemRecord(
            order_index=order_index,
            id=str(record_id),
            problem=str(problem_text),
            metadata=metadata,
        )

    def _problem_column(self, row: dict[str, str]) -> str:

        for column in self.problem_columns:
            if column in row:
                return column

        non_id_columns = [
            column
            for column in row
            if column not in {"id", "problem_id"}
        ]

        if non_id_columns:
            return non_id_columns[0]

        raise ValueError("Input CSV must include a problem text column.")

    def _read_input_rows(self) -> list[dict[str, str]]:

        if self.config.sample_eval_problems:
            rows = self._read_dataset_rows(self.config.eval_dataset_path)

            return self._sample_rows(rows=rows)

        return self._read_csv_rows(self.config.input_csv)

    def _sample_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:

        if len(rows) < self.config.eval_sample_size:
            raise ValueError(
                f"Eval dataset has {len(rows)} rows, "
                f"but {self.config.eval_sample_size} are required."
            )

        selected_indices = sorted(
            random.Random(self.config.eval_sample_seed).sample(
                range(len(rows)),
                self.config.eval_sample_size,
            )
        )

        return [
            rows[index]
            for index in selected_indices
        ]

    def _read_dataset_rows(self, path: Path) -> list[dict[str, str]]:

        if path.is_dir():
            return self._read_directory_rows(path)

        if path.suffix.lower() == ".csv":
            return self._read_csv_rows(path)

        if path.suffix.lower() == ".parquet":
            return self._read_parquet_rows([path])

        raise ValueError(f"Unsupported input dataset format: {path}")

    def _read_directory_rows(self, path: Path) -> list[dict[str, str]]:

        parquet_paths = sorted(path.rglob("*.parquet"))

        if parquet_paths:
            return self._read_parquet_rows(parquet_paths)

        csv_paths = sorted(path.rglob("*.csv"))

        if csv_paths:
            rows: list[dict[str, str]] = []

            for csv_path in csv_paths:
                rows.extend(self._read_csv_rows(csv_path))

            return rows

        raise FileNotFoundError(f"No CSV or parquet files found under: {path}")

    def _read_csv_rows(self, path: Path) -> list[dict[str, str]]:

        with path.open("r", encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)

            return [
                {
                    key: value
                    for key, value in row.items()
                }
                for row in reader
            ]

    def _read_parquet_rows(self, paths: list[Path]) -> list[dict[str, str]]:

        import pyarrow.parquet as pq

        table = pq.read_table(paths)

        return [
            {
                str(key): str(value)
                for key, value in row.items()
            }
            for row in table.to_pylist()
        ]

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:

        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")

        os.replace(temporary_path, path)

    def _safe_id(self, value: str) -> str:

        safe_value = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "_"
            for character in value
        )

        return safe_value or "problem"


def file_sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()
