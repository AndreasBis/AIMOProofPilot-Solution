from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any


DEFAULT_S3_URL_ENVIRONMENT_NAMES = (
    "AIMO_UPLOAD_S3_URL",
    "AIMO_S3_URL",
    "FIELDS_UPLOAD_S3_URL",
    "FIELDS_S3_URL",
    "PRESIGNED_S3_URL",
    "S3_URL",
)
DEFAULT_TIMEOUT_SECONDS = 7200


def main(argv: list[str] | None = None) -> int:

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    source_dir = args.source_dir.resolve()
    validate_source_dir(source_dir)
    s3_url = args.s3_url.strip()

    if not s3_url:
        parser.error(
            "Provide --s3_url or set one of the supported S3 URL environment variables."
        )

    archive_path = resolve_archive_path(
        source_dir=source_dir,
        archive_path=args.archive_path,
    )
    manifest_path = write_upload_manifest(
        source_dir=source_dir,
        archive_path=archive_path,
    )
    create_archive(
        source_dir=source_dir,
        archive_path=archive_path,
    )
    upload_result = upload_archive(
        archive_path=archive_path,
        s3_url=s3_url,
        timeout_seconds=args.timeout_seconds,
    )
    write_upload_receipt(
        source_dir=source_dir,
        archive_path=archive_path,
        manifest_path=manifest_path,
        s3_url=s3_url,
        upload_result=upload_result,
    )

    return 0


def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_url", default=default_s3_url())
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--archive_path", type=Path, default=None)
    parser.add_argument("--timeout_seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)

    return parser


def default_s3_url() -> str:

    for name in DEFAULT_S3_URL_ENVIRONMENT_NAMES:
        value = os.environ.get(name, "").strip()

        if value:
            return value

    return ""


def validate_source_dir(source_dir: Path) -> None:

    if not source_dir.exists():
        raise FileNotFoundError(f"Upload source directory does not exist: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Upload source is not a directory: {source_dir}")


def resolve_archive_path(source_dir: Path, archive_path: Path | None) -> Path:

    if archive_path is not None:
        return archive_path.resolve()

    return source_dir.parent / f"{source_dir.name}.tar.gz"


def write_upload_manifest(source_dir: Path, archive_path: Path) -> Path:

    manifest_path = source_dir / "upload_manifest.json"
    excluded_paths = {
        archive_path.resolve(),
        manifest_path.resolve(),
    }
    file_entries = manifest_file_entries(
        source_dir=source_dir,
        excluded_paths=excluded_paths,
    )
    payload = {
        "created_at_unix": time.time(),
        "source_dir": str(source_dir),
        "file_count": len(file_entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in file_entries),
        "inventory": result_inventory(file_entries),
        "files": file_entries,
    }
    write_json_atomic(path=manifest_path, payload=payload)

    return manifest_path


def manifest_file_entries(
    source_dir: Path,
    excluded_paths: set[Path],
) -> list[dict[str, Any]]:

    entries = []

    for path in sorted(source_dir.rglob("*")):
        resolved_path = path.resolve()

        if resolved_path in excluded_paths or not path.is_file():
            continue

        entries.append({
            "path": path.relative_to(source_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hash_file(path),
        })

    return entries


def result_inventory(file_entries: list[dict[str, Any]]) -> dict[str, Any]:

    relative_paths = [
        str(entry["path"])
        for entry in file_entries
    ]

    return {
        "has_output_csv": any(path.endswith(".csv") for path in relative_paths),
        "has_logs": any(
            any("log" in part for part in Path(path).parts)
            for path in relative_paths
        ),
        "has_run_metadata": any(
            Path(path).name == "run_metadata.json"
            for path in relative_paths
        ),
        "has_lora_adapter": any(
            Path(path).name == "adapter_model.safetensors"
            for path in relative_paths
        ),
        "has_training_score_table": any(
            Path(path).name in {"training_table.jsonl", "training_score_table.jsonl"}
            for path in relative_paths
        ),
        "has_failure_report": any(
            Path(path).name in {
                "failure_report.json",
                "failure_report.txt",
                "failure_traceback.txt",
            }
            for path in relative_paths
        ),
        "has_vllm_diagnostics": any(
            Path(path).name in {
                "vllm_stdout.log",
                "vllm_stderr.log",
                "vllm_command.json",
            }
            for path in relative_paths
        ),
        "has_service_preflight_diagnostics": any(
            Path(path).name == "service_preflight.json"
            for path in relative_paths
        ),
        "failure_reports": [
            path
            for path in relative_paths
            if Path(path).name in {
                "failure_report.json",
                "failure_report.txt",
                "failure_traceback.txt",
            }
        ],
        "compressed_failure_reports": [
            path
            for path in relative_paths
            if "failure" in path.lower()
            and Path(path).suffix in {".gz", ".zip", ".xz", ".bz2"}
        ],
    }


def create_archive(source_dir: Path, archive_path: Path) -> None:

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_path = archive_path.resolve()

    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.resolve() == excluded_path or not path.is_file():
                continue

            archive.add(
                path,
                arcname=Path(source_dir.name) / path.relative_to(source_dir),
                recursive=False,
            )


def upload_archive(
    archive_path: Path,
    s3_url: str,
    timeout_seconds: int,
) -> dict[str, Any]:

    command = [
        "curl",
        "--fail",
        "--show-error",
        "--silent",
        "--location",
        "--request",
        "PUT",
        "--upload-file",
        str(archive_path),
        s3_url,
    ]

    try:
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("curl is required to upload to the presigned S3 URL.") from error

    if completed_process.returncode != 0:
        raise RuntimeError(
            "Upload failed for "
            f"{redacted_url(s3_url)} with exit code {completed_process.returncode}: "
            f"{completed_process.stderr.strip()}"
        )

    return {
        "returncode": completed_process.returncode,
        "stdout": completed_process.stdout.strip(),
        "stderr": completed_process.stderr.strip(),
    }


def write_upload_receipt(
    source_dir: Path,
    archive_path: Path,
    manifest_path: Path,
    s3_url: str,
    upload_result: dict[str, Any],
) -> Path:

    receipt_path = source_dir / "upload_receipt.json"
    payload = {
        "created_at_unix": time.time(),
        "source_dir": str(source_dir),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": hash_file(archive_path),
        "manifest_path": str(manifest_path),
        "s3_url": redacted_url(s3_url),
        "upload": upload_result,
    }
    write_json_atomic(path=receipt_path, payload=payload)

    return receipt_path


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:

    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    os.replace(temporary_path, path)


def hash_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def redacted_url(s3_url: str) -> str:

    base_url = s3_url.split("?", 1)[0]

    if base_url == s3_url:
        return s3_url

    return f"{base_url}?..."


if __name__ == "__main__":
    raise SystemExit(main())
