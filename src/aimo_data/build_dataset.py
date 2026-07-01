from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SOURCE_COLUMNS = (
    "id",
    "problem_markdown",
    "solutions_markdown",
    "images",
    "country",
    "competition",
    "topics_flat",
    "language",
    "problem_type",
    "final_answer",
)

PROOF_TYPES = {
    "proof only",
    "proof and answer",
}

TOPIC_GROUP_ORDER = (
    "algebra",
    "combinatorics",
    "geometry",
    "number_theory",
    "mixed",
)

PAGE_COUNT_METHODS = {
    "latex",
    "sanitized_latex",
    "line_count",
    "word_count",
}
PAGE_COUNT_PROGRESS_INTERVAL = 25

PROOF_SCHEMA_COLUMNS = (
    "id",
    "problem",
    "solution",
    "all_solutions",
    "country",
    "competition",
    "topics_flat",
    "language",
    "problem_type",
    "has_images",
    "image_count",
    "source_config",
    "reference_rendered_pages",
    "reference_page_count_method",
)

EVAL_SCHEMA_COLUMNS = (
    "id",
    "problem",
    "country",
    "competition",
    "topics_flat",
    "language",
    "problem_type",
    "has_images",
    "image_count",
    "source_config",
)

PROOF_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("problem", pa.string()),
    pa.field("solution", pa.string()),
    pa.field("all_solutions", pa.string()),
    pa.field("country", pa.string()),
    pa.field("competition", pa.string()),
    pa.field("topics_flat", pa.string()),
    pa.field("language", pa.string()),
    pa.field("problem_type", pa.string()),
    pa.field("has_images", pa.bool_()),
    pa.field("image_count", pa.int64()),
    pa.field("source_config", pa.string()),
    pa.field("reference_rendered_pages", pa.int64()),
    pa.field("reference_page_count_method", pa.string()),
])

EVAL_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("problem", pa.string()),
    pa.field("country", pa.string()),
    pa.field("competition", pa.string()),
    pa.field("topics_flat", pa.string()),
    pa.field("language", pa.string()),
    pa.field("problem_type", pa.string()),
    pa.field("has_images", pa.bool_()),
    pa.field("image_count", pa.int64()),
    pa.field("source_config", pa.string()),
])

ANSWER_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("problem", pa.string()),
    pa.field("answer", pa.int64()),
])

IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")
HTML_IMAGE_PATTERN = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
MULTIPLE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
MATH_SEGMENT_PATTERN = re.compile(
    r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])",
    re.DOTALL,
)
PDF_PAGE_PATTERN = re.compile(rb"/Type\s*/Page\b")
PDFINFO_PAGE_PATTERN = re.compile(r"(?im)^Pages:\s*(\d+)\s*$")
INTEGER_PATTERN = re.compile(r"(?<!\d)-?\d+(?!\d)")
BOXED_INTEGER_PATTERN = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")


@dataclass(frozen=True)
class BuildConfig:

    source_dir: Path
    output_dir: Path
    eval_size: int
    train_size: int
    seed: int
    exclude_images: bool
    language_filter: str
    page_count_method: str
    latex_command: str
    pdfinfo_command: str
    page_count_cache_path: Path | None
    page_count_timeout_seconds: int
    source_dataset_name: str
    source_snapshot: str
    write_answer_dataset: bool
    answer_min: int
    answer_max: int


@dataclass(frozen=True)
class PageCountResult:

    rendered_pages: int
    method: str


@dataclass(frozen=True)
class BuildProducts:

    eval_rows: list[dict[str, Any]]
    eval_input_rows: list[dict[str, str]]
    eval_reference_rows: list[dict[str, Any]]
    train_rows: list[dict[str, Any]]
    judge_rows: list[dict[str, Any]]
    answer_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def parse_bool(value: str | bool) -> bool:

    if isinstance(value, bool):
        return value

    normalized_value = value.strip().casefold()

    if normalized_value in {"true", "1", "yes", "y"}:
        return True

    if normalized_value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=Path("data/data/all"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/data"))
    parser.add_argument("--eval_size", type=int, default=16)
    parser.add_argument("--train_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_images", type=parse_bool, default=True)
    parser.add_argument("--language_filter", type=str, default="any")
    parser.add_argument(
        "--page_count_method",
        choices=sorted(PAGE_COUNT_METHODS),
        default="latex",
    )
    parser.add_argument("--latex_command", type=str, default="pdflatex")
    parser.add_argument("--pdfinfo_command", type=str, default="pdfinfo")
    parser.add_argument("--page_count_cache_path", type=Path, default=None)
    parser.add_argument("--page_count_timeout_seconds", type=int, default=20)
    parser.add_argument("--source_dataset_name", type=str, default="MathNet")
    parser.add_argument("--source_snapshot", type=str, default="mathnet-v0-local")
    parser.add_argument("--write_answer_dataset", type=parse_bool, default=False)
    parser.add_argument("--answer_min", type=int, default=0)
    parser.add_argument("--answer_max", type=int, default=999)

    return parser


def build_config(args: argparse.Namespace) -> BuildConfig:

    if args.eval_size < 0:
        raise ValueError("eval_size must be non-negative.")

    if args.train_size < 0:
        raise ValueError("train_size must be non-negative.")

    if args.answer_min > args.answer_max:
        raise ValueError("answer_min must be less than or equal to answer_max.")

    return BuildConfig(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        eval_size=args.eval_size,
        train_size=args.train_size,
        seed=args.seed,
        exclude_images=args.exclude_images,
        language_filter=args.language_filter,
        page_count_method=args.page_count_method,
        latex_command=args.latex_command,
        pdfinfo_command=args.pdfinfo_command,
        page_count_cache_path=args.page_count_cache_path,
        page_count_timeout_seconds=args.page_count_timeout_seconds,
        source_dataset_name=args.source_dataset_name,
        source_snapshot=args.source_snapshot,
        write_answer_dataset=args.write_answer_dataset,
        answer_min=args.answer_min,
        answer_max=args.answer_max,
    )


def build_dataset(config: BuildConfig) -> BuildProducts:

    parquet_paths = discover_parquet_paths(config.source_dir)
    source_rows = read_source_rows(
        source_dir=config.source_dir,
        parquet_paths=parquet_paths,
    )
    normalized_rows = [
        normalize_source_row(row)
        for row in source_rows
    ]
    proof_rows, filter_counts, excluded_counts = filter_proof_rows(
        rows=normalized_rows,
        config=config,
    )
    eval_source_rows = select_eval_rows(rows=proof_rows, config=config)
    eval_ids = {
        row["id"]
        for row in eval_source_rows
    }
    train_source_rows = select_train_rows(
        rows=proof_rows,
        excluded_ids=eval_ids,
        config=config,
    )
    selected_proof_rows = eval_source_rows + train_source_rows
    add_reference_page_counts(rows=selected_proof_rows, config=config)
    selected_train_topic_counts = count_topic_groups(train_source_rows)
    answer_rows = build_answer_rows(
        rows=normalized_rows,
        excluded_ids=eval_ids,
        config=config,
    )

    eval_rows = [
        project_row(row=row, columns=EVAL_SCHEMA_COLUMNS)
        for row in eval_source_rows
    ]
    eval_input_rows = [
        {
            "id": row["id"],
            "problem": row["problem"],
        }
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
    judge_rows = [
        project_row(row=row, columns=PROOF_SCHEMA_COLUMNS)
        for row in train_source_rows
    ]

    manifest = build_manifest(
        config=config,
        parquet_paths=parquet_paths,
        source_rows=source_rows,
        filter_counts=filter_counts,
        excluded_counts=excluded_counts,
        eval_rows=eval_rows,
        eval_reference_rows=eval_reference_rows,
        train_rows=train_rows,
        judge_rows=judge_rows,
        answer_rows=answer_rows,
        selected_train_topic_counts=selected_train_topic_counts,
    )
    products = BuildProducts(
        eval_rows=eval_rows,
        eval_input_rows=eval_input_rows,
        eval_reference_rows=eval_reference_rows,
        train_rows=train_rows,
        judge_rows=judge_rows,
        answer_rows=answer_rows,
        manifest=manifest,
    )
    validate_products(products=products, config=config)

    return products


def discover_parquet_paths(source_dir: Path) -> list[Path]:

    if source_dir.is_file():
        if source_dir.suffix.lower() != ".parquet":
            raise ValueError(f"Source file is not a parquet file: {source_dir}")

        return [source_dir]

    if not source_dir.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_dir}")

    parquet_paths = sorted(source_dir.rglob("*.parquet"))

    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under: {source_dir}")

    return parquet_paths


def read_source_rows(source_dir: Path, parquet_paths: list[Path]) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    for parquet_path in parquet_paths:
        schema_names = set(pq.read_schema(parquet_path).names)
        selected_columns = [
            column
            for column in SOURCE_COLUMNS
            if column in schema_names
        ]
        table = pq.read_table(parquet_path, columns=selected_columns)
        image_counts = extract_image_counts(table)
        metadata_columns = [
            column
            for column in table.column_names
            if column != "images"
        ]
        metadata_table = table.select(metadata_columns)
        source_config = infer_source_config(
            source_dir=source_dir,
            parquet_path=parquet_path,
        )

        for row_index, row in enumerate(metadata_table.to_pylist()):
            row["image_count"] = image_counts[row_index]
            row["source_config"] = source_config
            rows.append(row)

    return rows


def extract_image_counts(table: pa.Table) -> list[int]:

    if "images" not in table.column_names:
        return [0 for _ in range(table.num_rows)]

    lengths = pc.list_value_length(table["images"]).to_pylist()

    return [
        int(length or 0)
        for length in lengths
    ]


def infer_source_config(source_dir: Path, parquet_path: Path) -> str:

    if source_dir.is_file():
        return parquet_path.parent.name

    if parquet_path.parent == source_dir:
        return source_dir.name

    if source_dir.name == "all":
        return "all"

    return parquet_path.parent.name


def normalize_source_row(row: dict[str, Any]) -> dict[str, Any]:

    original_solutions = normalize_string_list(row.get("solutions_markdown"))
    normalized_solutions = [
        normalize_markdown_text(solution)
        for solution in original_solutions
    ]
    non_empty_solutions = [
        solution
        for solution in normalized_solutions
        if solution
    ]
    topics = normalize_string_list(row.get("topics_flat"))
    image_count = int(row.get("image_count") or 0)
    final_answer = normalize_optional_text(row.get("final_answer"))

    return {
        "id": normalize_required_text(row.get("id")),
        "problem": normalize_markdown_text(row.get("problem_markdown")),
        "solution": non_empty_solutions[0] if non_empty_solutions else "",
        "all_solutions": serialize_json(original_solutions),
        "country": normalize_optional_text(row.get("country")),
        "competition": normalize_optional_text(row.get("competition")),
        "topics": topics,
        "topics_flat": serialize_json(topics),
        "language": normalize_optional_text(row.get("language")),
        "problem_type": normalize_optional_text(row.get("problem_type")),
        "has_images": image_count > 0,
        "image_count": image_count,
        "source_config": normalize_optional_text(row.get("source_config")),
        "final_answer": final_answer,
        "topic_group": classify_topic_group(topics),
        "reference_rendered_pages": 0,
        "reference_page_count_method": "",
    }


def normalize_string_list(value: Any) -> list[str]:

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item)
            for item in value
            if item is not None and str(item).strip()
        ]

    if isinstance(value, tuple):
        return [
            str(item)
            for item in value
            if item is not None and str(item).strip()
        ]

    text_value = str(value).strip()

    if not text_value:
        return []

    return [text_value]


def normalize_required_text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


def normalize_optional_text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


def normalize_markdown_text(value: Any) -> str:

    text = normalize_optional_text(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_image_markup(text)
    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]
    text = "\n".join(lines).strip()
    text = MULTIPLE_BLANK_LINES_PATTERN.sub("\n\n", text)

    return text


def strip_image_markup(text: str) -> str:

    text = IMAGE_MARKDOWN_PATTERN.sub("", text)
    text = HTML_IMAGE_PATTERN.sub("", text)

    return text


def serialize_json(value: Any) -> str:

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def classify_topic_group(topics: list[str]) -> str:

    topic_text = " ".join(topics).casefold()
    matched_groups: list[str] = []

    if "algebra" in topic_text:
        matched_groups.append("algebra")

    if "combinatorics" in topic_text or "discrete mathematics" in topic_text:
        matched_groups.append("combinatorics")

    if "geometry" in topic_text:
        matched_groups.append("geometry")

    if "number theory" in topic_text:
        matched_groups.append("number_theory")

    unique_groups = list(dict.fromkeys(matched_groups))

    if len(unique_groups) == 1:
        return unique_groups[0]

    return "mixed"


def filter_proof_rows(
    rows: list[dict[str, Any]],
    config: BuildConfig,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:

    filter_counts: dict[str, int] = {
        "source_rows": len(rows),
    }
    excluded_counts: dict[str, int] = {}
    current_rows = rows
    current_rows = apply_filter_stage(
        rows=current_rows,
        name="non_empty_id",
        predicate=lambda row: bool(row["id"]),
        filter_counts=filter_counts,
        excluded_counts=excluded_counts,
    )
    current_rows = apply_filter_stage(
        rows=current_rows,
        name="non_empty_problem",
        predicate=lambda row: bool(row["problem"]),
        filter_counts=filter_counts,
        excluded_counts=excluded_counts,
    )
    current_rows = apply_filter_stage(
        rows=current_rows,
        name="proof_problem_type",
        predicate=lambda row: row["problem_type"].casefold() in PROOF_TYPES,
        filter_counts=filter_counts,
        excluded_counts=excluded_counts,
    )
    current_rows = apply_filter_stage(
        rows=current_rows,
        name="reference_solution",
        predicate=lambda row: bool(row["solution"]),
        filter_counts=filter_counts,
        excluded_counts=excluded_counts,
    )

    if config.language_filter.casefold() != "any":
        expected_language = config.language_filter.casefold()
        current_rows = apply_filter_stage(
            rows=current_rows,
            name="language_filter",
            predicate=lambda row: row["language"].casefold() == expected_language,
            filter_counts=filter_counts,
            excluded_counts=excluded_counts,
        )
    else:
        filter_counts["after_language_filter"] = len(current_rows)
        excluded_counts["language_filter"] = 0

    if config.exclude_images:
        current_rows = apply_filter_stage(
            rows=current_rows,
            name="text_only",
            predicate=lambda row: not row["has_images"],
            filter_counts=filter_counts,
            excluded_counts=excluded_counts,
        )
    else:
        filter_counts["after_text_only"] = len(current_rows)
        excluded_counts["text_only"] = 0

    filter_counts["eligible_proof_rows"] = len(current_rows)

    return current_rows, filter_counts, excluded_counts


def apply_filter_stage(
    rows: list[dict[str, Any]],
    name: str,
    predicate: Callable[[dict[str, Any]], bool],
    filter_counts: dict[str, int],
    excluded_counts: dict[str, int],
) -> list[dict[str, Any]]:

    kept_rows = [
        row
        for row in rows
        if predicate(row)
    ]
    filter_counts[f"after_{name}"] = len(kept_rows)
    excluded_counts[name] = len(rows) - len(kept_rows)

    return kept_rows


def add_reference_page_counts(rows: list[dict[str, Any]], config: BuildConfig) -> None:

    cache_path = resolve_page_count_cache_path(config)
    page_count_cache = read_page_count_cache(cache_path)
    row_cache_keys = [
        (
            row,
            build_page_count_cache_key(solution=row["solution"], config=config),
        )
        for row in rows
    ]
    unique_cache_keys = {
        cache_key
        for _, cache_key in row_cache_keys
    }
    completed_cache_keys: set[str] = set()
    computed_since_cache_write = 0

    for row, cache_key in row_cache_keys:
        if cache_key not in page_count_cache:
            page_count_cache[cache_key] = count_reference_pages(
                solution=row["solution"],
                config=config,
            )
            computed_since_cache_write += 1

            if computed_since_cache_write >= PAGE_COUNT_PROGRESS_INTERVAL:
                write_page_count_cache(path=cache_path, cache=page_count_cache)
                computed_since_cache_write = 0

        page_count = page_count_cache[cache_key]
        row["reference_rendered_pages"] = page_count.rendered_pages
        row["reference_page_count_method"] = page_count.method
        completed_cache_keys.add(cache_key)

        if should_report_page_count_progress(
            completed_count=len(completed_cache_keys),
            total_count=len(unique_cache_keys),
        ):
            report_page_count_progress(
                completed_count=len(completed_cache_keys),
                total_count=len(unique_cache_keys),
                cache_path=cache_path,
            )

    write_page_count_cache(path=cache_path, cache=page_count_cache)


def resolve_page_count_cache_path(config: BuildConfig) -> Path:

    if config.page_count_cache_path is not None:
        return config.page_count_cache_path

    return config.output_dir.parent / f".{config.output_dir.name}_page_count_cache.json"


def build_page_count_cache_key(solution: str, config: BuildConfig) -> str:

    payload = {
        "solution": solution,
        "page_count_method": config.page_count_method,
        "latex_command": config.latex_command,
        "pdfinfo_command": config.pdfinfo_command,
    }

    return hashlib.sha256(serialize_json(payload).encode("utf-8")).hexdigest()


def read_page_count_cache(path: Path) -> dict[str, PageCountResult]:

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    cache: dict[str, PageCountResult] = {}

    for cache_key, value in payload.items():
        cache[str(cache_key)] = PageCountResult(
            rendered_pages=int(value["rendered_pages"]),
            method=str(value["method"]),
        )

    return cache


def write_page_count_cache(path: Path, cache: dict[str, PageCountResult]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        cache_key: {
            "rendered_pages": page_count.rendered_pages,
            "method": page_count.method,
        }
        for cache_key, page_count in sorted(cache.items())
    }
    write_json(path=path, payload=payload)


def should_report_page_count_progress(completed_count: int, total_count: int) -> bool:

    return (
        completed_count == total_count
        or completed_count % PAGE_COUNT_PROGRESS_INTERVAL == 0
    )


def report_page_count_progress(completed_count: int, total_count: int, cache_path: Path) -> None:

    print(
        (
            f"reference page counts: {completed_count}/{total_count} unique solutions "
            f"cached at {cache_path}"
        ),
        file=sys.stderr,
        flush=True,
    )


def count_reference_pages(solution: str, config: BuildConfig) -> PageCountResult:

    for method in page_count_fallback_order(config.page_count_method):
        try:
            if method == "latex":
                rendered_pages = render_solution_to_pdf_page_count(
                    solution=solution,
                    sanitize=False,
                    config=config,
                )

                return PageCountResult(
                    rendered_pages=rendered_pages,
                    method=method,
                )

            if method == "sanitized_latex":
                rendered_pages = render_solution_to_pdf_page_count(
                    solution=solution,
                    sanitize=True,
                    config=config,
                )

                return PageCountResult(
                    rendered_pages=rendered_pages,
                    method=method,
                )

            if method == "line_count":
                return PageCountResult(
                    rendered_pages=estimate_pages_by_lines(solution),
                    method=method,
                )

            if method == "word_count":
                return PageCountResult(
                    rendered_pages=estimate_pages_by_words(solution),
                    method=method,
                )
        except Exception:
            continue

    return PageCountResult(
        rendered_pages=estimate_pages_by_words(solution),
        method="word_count",
    )


def page_count_fallback_order(method: str) -> list[str]:

    if method == "latex":
        return [
            "latex",
            "sanitized_latex",
            "line_count",
            "word_count",
        ]

    if method == "sanitized_latex":
        return [
            "sanitized_latex",
            "line_count",
            "word_count",
        ]

    if method == "line_count":
        return [
            "line_count",
            "word_count",
        ]

    return [
        "word_count",
    ]


def render_solution_to_pdf_page_count(
    solution: str,
    sanitize: bool,
    config: BuildConfig,
) -> int:

    latex_command_path = shutil.which(config.latex_command)

    if latex_command_path is None:
        raise FileNotFoundError(f"LaTeX command not found: {config.latex_command}")

    document = build_latex_document(
        solution=solution,
        sanitize=sanitize,
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        tex_path = temporary_path / "solution.tex"
        pdf_path = temporary_path / "solution.pdf"
        tex_path.write_text(document, encoding="utf-8")
        completed_process = subprocess.run(
            [
                latex_command_path,
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=temporary_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=config.page_count_timeout_seconds,
            check=False,
        )

        if completed_process.returncode != 0 or not pdf_path.exists():
            raise RuntimeError("LaTeX render failed.")

        return count_pdf_pages(pdf_path=pdf_path, config=config)


def build_latex_document(solution: str, sanitize: bool) -> str:

    body = sanitize_latex_body(solution) if sanitize else solution

    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{amsmath,amssymb,amsthm}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\setlength{\\parindent}{0pt}\n"
        "\\setlength{\\parskip}{0.65em}\n"
        "\\begin{document}\n"
        f"{body}\n"
        "\\end{document}\n"
    )


def sanitize_latex_body(value: str) -> str:

    text = strip_image_markup(value)
    segments = MATH_SEGMENT_PATTERN.split(text)
    sanitized_segments = [
        segment if is_math_segment(segment) else escape_latex_text(segment)
        for segment in segments
    ]

    return "".join(sanitized_segments)


def is_math_segment(value: str) -> bool:

    return (
        value.startswith("$")
        or value.startswith("\\(")
        or value.startswith("\\[")
    )


def escape_latex_text(value: str) -> str:

    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }

    return "".join(
        replacements.get(character, character)
        for character in value
    )


def count_pdf_pages(pdf_path: Path, config: BuildConfig) -> int:

    pdfinfo_command_path = shutil.which(config.pdfinfo_command)

    if pdfinfo_command_path is not None:
        completed_process = subprocess.run(
            [
                pdfinfo_command_path,
                str(pdf_path),
            ],
            capture_output=True,
            text=True,
            timeout=config.page_count_timeout_seconds,
            check=False,
        )
        match = PDFINFO_PAGE_PATTERN.search(completed_process.stdout)

        if completed_process.returncode == 0 and match:
            return int(match.group(1))

    page_count = len(PDF_PAGE_PATTERN.findall(pdf_path.read_bytes()))

    if page_count < 1:
        raise ValueError(f"Unable to count PDF pages: {pdf_path}")

    return page_count


def estimate_pages_by_lines(solution: str) -> int:

    non_empty_lines = [
        line
        for line in solution.splitlines()
        if line.strip()
    ]

    return max(1, math.ceil(len(non_empty_lines) / 45))


def estimate_pages_by_words(solution: str) -> int:

    words = solution.split()

    return max(1, math.ceil(len(words) / 550))


def select_eval_rows(rows: list[dict[str, Any]], config: BuildConfig) -> list[dict[str, Any]]:

    if config.eval_size > len(rows):
        raise ValueError(
            f"Requested {config.eval_size} eval rows, but only {len(rows)} rows are eligible."
        )

    if config.eval_size == 0:
        return []

    ordered_rows = sorted(
        rows,
        key=lambda row: deterministic_row_key(row=row, seed=config.seed),
    )
    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_countries: set[str] = set()
    used_competitions: set[str] = set()
    topic_groups = list(TOPIC_GROUP_ORDER)
    random.Random(config.seed).shuffle(topic_groups)

    while len(selected_rows) < config.eval_size:
        progress_made = False

        for topic_group in topic_groups:
            if len(selected_rows) >= config.eval_size:
                break

            candidates = [
                row
                for row in ordered_rows
                if row["topic_group"] == topic_group and row["id"] not in selected_ids
            ]
            candidate = choose_diverse_candidate(
                candidates=candidates,
                used_countries=used_countries,
                used_competitions=used_competitions,
            )

            if candidate is None:
                continue

            append_eval_row(
                row=candidate,
                selected_rows=selected_rows,
                selected_ids=selected_ids,
                used_countries=used_countries,
                used_competitions=used_competitions,
            )
            progress_made = True

        if not progress_made:
            remaining_rows = [
                row
                for row in ordered_rows
                if row["id"] not in selected_ids
            ]
            candidate = choose_diverse_candidate(
                candidates=remaining_rows,
                used_countries=used_countries,
                used_competitions=used_competitions,
            )

            if candidate is None:
                raise ValueError("Unable to select enough eval rows.")

            append_eval_row(
                row=candidate,
                selected_rows=selected_rows,
                selected_ids=selected_ids,
                used_countries=used_countries,
                used_competitions=used_competitions,
            )

    return selected_rows


def deterministic_row_key(row: dict[str, Any], seed: int) -> tuple[str, str]:

    row_id = row["id"]
    digest = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()

    return digest, row_id


def choose_diverse_candidate(
    candidates: list[dict[str, Any]],
    used_countries: set[str],
    used_competitions: set[str],
) -> dict[str, Any] | None:

    for candidate in candidates:
        country = candidate["country"]
        competition = candidate["competition"]

        if (
            country
            and competition
            and country not in used_countries
            and competition not in used_competitions
        ):
            return candidate

    for candidate in candidates:
        country = candidate["country"]

        if country and country not in used_countries:
            return candidate

    if candidates:
        return candidates[0]

    return None


def append_eval_row(
    row: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    selected_ids: set[str],
    used_countries: set[str],
    used_competitions: set[str],
) -> None:

    selected_rows.append(row)
    selected_ids.add(row["id"])

    if row["country"]:
        used_countries.add(row["country"])

    if row["competition"]:
        used_competitions.add(row["competition"])


def select_train_rows(
    rows: list[dict[str, Any]],
    excluded_ids: set[str],
    config: BuildConfig,
) -> list[dict[str, Any]]:

    rows_by_topic: dict[str, list[dict[str, Any]]] = {}
    available_count = 0

    for row in rows:
        row_id = row["id"]

        if row_id in excluded_ids:
            continue

        topic_group = str(row["topic_group"])
        rows_by_topic.setdefault(topic_group, []).append(row)
        available_count += 1

    if config.train_size == 0:
        available_rows = [
            row
            for topic_rows in rows_by_topic.values()
            for row in topic_rows
        ]

        return sorted(
            available_rows,
            key=lambda row: deterministic_row_key(row=row, seed=config.seed),
        )

    if config.train_size > available_count:
        raise ValueError(
            f"Requested {config.train_size} train rows, "
            f"but only {available_count} rows are eligible after eval holdout."
        )

    topic_counts = {
        topic_group: len(topic_rows)
        for topic_group, topic_rows in rows_by_topic.items()
    }
    topic_quotas = proportional_topic_quotas(
        topic_counts=topic_counts,
        target_size=config.train_size,
    )
    selected_rows: list[dict[str, Any]] = []

    for topic_group in sorted(topic_counts):
        topic_quota = topic_quotas[topic_group]

        if topic_quota == 0:
            continue

        topic_rows = rows_by_topic[topic_group]
        topic_rows.sort(key=lambda row: deterministic_row_key(row=row, seed=config.seed))
        selected_rows.extend(topic_rows[:topic_quota])

    return sorted(
        selected_rows,
        key=lambda row: deterministic_row_key(row=row, seed=config.seed),
    )


def count_topic_groups(rows: list[dict[str, Any]]) -> dict[str, int]:

    counts: dict[str, int] = {}

    for row in rows:
        topic_group = str(row["topic_group"])
        counts[topic_group] = counts.get(topic_group, 0) + 1

    return counts


def proportional_topic_quotas(
    topic_counts: dict[str, int],
    target_size: int,
) -> dict[str, int]:

    total_count = sum(topic_counts.values())

    if target_size == 0 or total_count == 0:
        return {
            topic_group: 0
            for topic_group in topic_counts
        }

    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []

    for topic_group, topic_count in topic_counts.items():
        exact_quota = target_size * topic_count / total_count
        quota = min(topic_count, int(math.floor(exact_quota)))
        quotas[topic_group] = quota
        remainders.append((exact_quota - quota, topic_group))

    remaining = target_size - sum(quotas.values())

    for _, topic_group in sorted(remainders, reverse=True):
        if remaining <= 0:
            break

        if quotas[topic_group] >= topic_counts[topic_group]:
            continue

        quotas[topic_group] += 1
        remaining -= 1

    while remaining > 0:
        progress_made = False

        for topic_group in sorted(topic_counts):
            if remaining <= 0:
                break

            if quotas[topic_group] >= topic_counts[topic_group]:
                continue

            quotas[topic_group] += 1
            remaining -= 1
            progress_made = True

        if not progress_made:
            break

    return quotas


def build_answer_rows(
    rows: list[dict[str, Any]],
    excluded_ids: set[str],
    config: BuildConfig,
) -> list[dict[str, Any]]:

    if not config.write_answer_dataset:
        return []

    answer_rows: list[dict[str, Any]] = []

    for row in rows:
        if not row["id"] or not row["problem"] or row["id"] in excluded_ids:
            continue

        answer = parse_aimo_answer(
            value=row["final_answer"],
            answer_min=config.answer_min,
            answer_max=config.answer_max,
        )

        if answer is None:
            continue

        if config.exclude_images and row["has_images"]:
            continue

        answer_rows.append({
            "id": row["id"],
            "problem": row["problem"],
            "answer": answer,
        })

    return answer_rows


def parse_aimo_answer(value: str, answer_min: int, answer_max: int) -> int | None:

    text = value.strip()

    if not text:
        return None

    boxed_matches = BOXED_INTEGER_PATTERN.findall(text)

    if boxed_matches:
        answer = int(boxed_matches[-1])

        if answer_min <= answer <= answer_max:
            return answer

        return None

    if text.lstrip("-").isdigit():
        answer = int(text)

        if answer_min <= answer <= answer_max:
            return answer

        return None

    matches = INTEGER_PATTERN.findall(text)

    if len(matches) == 1:
        answer = int(matches[0])

        if answer_min <= answer <= answer_max:
            return answer

    return None


def project_row(row: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:

    return {
        column: row[column]
        for column in columns
    }


def build_manifest(
    config: BuildConfig,
    parquet_paths: list[Path],
    source_rows: list[dict[str, Any]],
    filter_counts: dict[str, int],
    excluded_counts: dict[str, int],
    eval_rows: list[dict[str, Any]],
    eval_reference_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
    selected_train_topic_counts: dict[str, int],
) -> dict[str, Any]:

    row_counts = {
        "aimo_proof_eval.parquet": len(eval_rows),
        "aimo_proof_eval_input.csv": len(eval_rows),
        "aimo_proof_eval_reference.parquet": len(eval_reference_rows),
        "aimo_proof_train.parquet": len(train_rows),
        "aimo_judge_train.parquet": len(judge_rows),
    }

    if config.write_answer_dataset:
        row_counts["aimo_answer_train.parquet"] = len(answer_rows)

    return {
        "source_dataset_name": config.source_dataset_name,
        "source_snapshot": config.source_snapshot,
        "source_local_paths": [
            str(path)
            for path in parquet_paths
        ],
        "source_row_count": len(source_rows),
        "filter_counts": filter_counts,
        "excluded_counts": excluded_counts,
        "selected_eval_ids": [
            row["id"]
            for row in eval_rows
        ],
        "selected_train_size": len(train_rows),
        "selected_train_topic_counts": selected_train_topic_counts,
        "seed": config.seed,
        "builder_arguments": serialize_build_arguments(config),
        "row_counts": row_counts,
        "output_hashes": {},
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def serialize_build_arguments(config: BuildConfig) -> dict[str, Any]:

    return {
        "source_dir": str(config.source_dir),
        "output_dir": str(config.output_dir),
        "eval_size": config.eval_size,
        "train_size": config.train_size,
        "seed": config.seed,
        "exclude_images": config.exclude_images,
        "language_filter": config.language_filter,
        "page_count_method": config.page_count_method,
        "latex_command": config.latex_command,
        "pdfinfo_command": config.pdfinfo_command,
        "page_count_cache_path": (
            str(config.page_count_cache_path)
            if config.page_count_cache_path is not None
            else ""
        ),
        "page_count_timeout_seconds": config.page_count_timeout_seconds,
        "source_dataset_name": config.source_dataset_name,
        "source_snapshot": config.source_snapshot,
        "write_answer_dataset": config.write_answer_dataset,
        "answer_min": config.answer_min,
        "answer_max": config.answer_max,
    }


def validate_products(products: BuildProducts, config: BuildConfig) -> None:

    validate_non_empty_ids(
        split_name="eval",
        rows=products.eval_rows,
    )
    validate_non_empty_ids(
        split_name="eval_reference",
        rows=products.eval_reference_rows,
    )
    validate_non_empty_ids(
        split_name="train",
        rows=products.train_rows,
    )
    validate_non_empty_ids(
        split_name="judge",
        rows=products.judge_rows,
    )
    validate_non_empty_ids(
        split_name="answer",
        rows=products.answer_rows,
    )
    validate_non_empty_problem_text(
        split_name="eval",
        rows=products.eval_rows,
    )
    validate_non_empty_problem_text(
        split_name="eval_reference",
        rows=products.eval_reference_rows,
    )
    validate_non_empty_problem_text(
        split_name="train",
        rows=products.train_rows,
    )
    validate_non_empty_problem_text(
        split_name="judge",
        rows=products.judge_rows,
    )
    validate_training_solutions(rows=products.train_rows)
    validate_training_solutions(rows=products.judge_rows)
    validate_unique_ids(split_name="eval", rows=products.eval_rows)
    validate_unique_ids(split_name="eval_reference", rows=products.eval_reference_rows)
    validate_unique_ids(split_name="train", rows=products.train_rows)
    validate_unique_ids(split_name="judge", rows=products.judge_rows)
    validate_unique_ids(split_name="answer", rows=products.answer_rows)
    validate_eval_train_disjoint(
        eval_rows=products.eval_rows,
        train_rows=products.train_rows,
    )
    validate_eval_train_disjoint(
        eval_rows=products.eval_rows,
        train_rows=products.judge_rows,
    )
    validate_eval_input_rows(rows=products.eval_input_rows)
    validate_problem_image_markup(rows=products.eval_rows, config=config)
    validate_problem_image_markup(rows=products.eval_reference_rows, config=config)
    validate_problem_image_markup(rows=products.train_rows, config=config)
    validate_problem_image_markup(rows=products.judge_rows, config=config)
    validate_page_count_fields(rows=products.eval_reference_rows)
    validate_page_count_fields(rows=products.train_rows)
    validate_page_count_fields(rows=products.judge_rows)
    validate_manifest_row_counts(products=products)


def validate_non_empty_ids(split_name: str, rows: list[dict[str, Any]]) -> None:

    for row in rows:
        if not row["id"]:
            raise ValueError(f"{split_name} contains an empty id.")


def validate_non_empty_problem_text(split_name: str, rows: list[dict[str, Any]]) -> None:

    for row in rows:
        if not row["problem"]:
            raise ValueError(f"{split_name} contains an empty problem.")


def validate_training_solutions(rows: list[dict[str, Any]]) -> None:

    for row in rows:
        if not row["solution"]:
            raise ValueError("A training row contains an empty solution.")


def validate_unique_ids(split_name: str, rows: list[dict[str, Any]]) -> None:

    seen_ids: set[str] = set()

    for row in rows:
        if row["id"] in seen_ids:
            row_id = row["id"]

            raise ValueError(f"{split_name} contains a duplicate id: {row_id}")

        seen_ids.add(row["id"])


def validate_eval_train_disjoint(
    eval_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> None:

    eval_ids = {
        row["id"]
        for row in eval_rows
    }
    train_ids = {
        row["id"]
        for row in train_rows
    }
    overlapping_ids = sorted(eval_ids & train_ids)

    if overlapping_ids:
        raise ValueError(f"Eval ids appear in train ids: {overlapping_ids[:10]}")


def validate_eval_input_rows(rows: list[dict[str, str]]) -> None:

    for row in rows:
        if tuple(row.keys()) != ("id", "problem"):
            raise ValueError("Eval input CSV rows must contain exactly id,problem.")


def validate_problem_image_markup(rows: list[dict[str, Any]], config: BuildConfig) -> None:

    if not config.exclude_images:
        return

    for row in rows:
        has_markdown_image = IMAGE_MARKDOWN_PATTERN.search(row["problem"]) is not None
        has_html_image = HTML_IMAGE_PATTERN.search(row["problem"]) is not None

        if has_markdown_image or has_html_image:
            row_id = row["id"]

            raise ValueError(f"Image markup remains in problem field for id {row_id}.")


def validate_page_count_fields(rows: list[dict[str, Any]]) -> None:

    for row in rows:
        if row["solution"] and row["reference_rendered_pages"] < 1:
            row_id = row["id"]

            raise ValueError(f"Missing reference page count for id {row_id}.")

        if row["solution"] and row["reference_page_count_method"] not in PAGE_COUNT_METHODS:
            row_id = row["id"]

            raise ValueError(f"Invalid reference page count method for id {row_id}.")


def validate_manifest_row_counts(products: BuildProducts) -> None:

    expected_counts: dict[str, int] = {
        "aimo_proof_eval.parquet": len(products.eval_rows),
        "aimo_proof_eval_input.csv": len(products.eval_input_rows),
        "aimo_proof_eval_reference.parquet": len(products.eval_reference_rows),
        "aimo_proof_train.parquet": len(products.train_rows),
        "aimo_judge_train.parquet": len(products.judge_rows),
    }
    manifest_counts = products.manifest["row_counts"]

    if "aimo_answer_train.parquet" in manifest_counts:
        expected_counts["aimo_answer_train.parquet"] = len(products.answer_rows)

    for name, expected_count in expected_counts.items():
        if manifest_counts.get(name) != expected_count:
            raise ValueError(f"Manifest row count mismatch for {name}.")


def write_products(products: BuildProducts, output_dir: Path) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "aimo_proof_eval.parquet": output_dir / "aimo_proof_eval.parquet",
        "aimo_proof_eval_input.csv": output_dir / "aimo_proof_eval_input.csv",
        "aimo_proof_eval_reference.parquet": output_dir / "aimo_proof_eval_reference.parquet",
        "aimo_proof_train.parquet": output_dir / "aimo_proof_train.parquet",
        "aimo_judge_train.parquet": output_dir / "aimo_judge_train.parquet",
    }
    write_parquet_rows(
        path=output_paths["aimo_proof_eval.parquet"],
        rows=products.eval_rows,
        schema=EVAL_SCHEMA,
    )
    write_eval_input_csv(
        path=output_paths["aimo_proof_eval_input.csv"],
        rows=products.eval_input_rows,
    )
    write_parquet_rows(
        path=output_paths["aimo_proof_eval_reference.parquet"],
        rows=products.eval_reference_rows,
        schema=PROOF_SCHEMA,
    )
    write_parquet_rows(
        path=output_paths["aimo_proof_train.parquet"],
        rows=products.train_rows,
        schema=PROOF_SCHEMA,
    )
    write_parquet_rows(
        path=output_paths["aimo_judge_train.parquet"],
        rows=products.judge_rows,
        schema=PROOF_SCHEMA,
    )

    if "aimo_answer_train.parquet" in products.manifest["row_counts"]:
        answer_path = output_dir / "aimo_answer_train.parquet"
        output_paths["aimo_answer_train.parquet"] = answer_path
        write_parquet_rows(
            path=answer_path,
            rows=products.answer_rows,
            schema=ANSWER_SCHEMA,
        )

    products.manifest["output_hashes"] = {
        name: sha256_file(path)
        for name, path in output_paths.items()
    }
    write_json(
        path=output_dir / "manifest.json",
        payload=products.manifest,
    )


def write_parquet_rows(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:

    temporary_path = temporary_output_path(path)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, temporary_path)
    os.replace(temporary_path, path)


def write_eval_input_csv(path: Path, rows: list[dict[str, str]]) -> None:

    temporary_path = temporary_output_path(path)

    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "id",
                "problem",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary_path, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:

    temporary_path = temporary_output_path(path)

    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    os.replace(temporary_path, path)


def temporary_output_path(path: Path) -> Path:

    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    products = build_dataset(config)
    write_products(products=products, output_dir=config.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
