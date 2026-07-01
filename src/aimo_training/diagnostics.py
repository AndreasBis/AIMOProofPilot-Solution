from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from aimo_training.config import AIMOTrainingConfig


REDACTED_ENVIRONMENT_FRAGMENTS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "S3_URL",
    "AWS",
)

FAILURE_ARTIFACT_PATTERNS = (
    "online_servers/**/vllm_stdout.log",
    "online_servers/**/vllm_stderr.log",
    "online_servers/**/vllm_command.json",
    "online_servers/**/service_preflight.json",
    "online_events.jsonl",
    "online_control/*.json",
    "online_control/stop",
    "failure_report.json",
    "failure_report.txt",
    "failure_traceback.txt",
    "phase_events.jsonl",
)


def write_phase_event(
    logdir: Path,
    event: str,
    payload: dict[str, Any],
) -> None:

    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / "phase_events.jsonl"
    event_payload = {
        "event": event,
        "created_at_unix": time.time(),
        **payload,
    }

    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(event_payload, ensure_ascii=False))
        output_file.write("\n")


def write_failure_diagnostics(
    config: AIMOTrainingConfig,
    phase: str,
    error: Exception,
    started_at_monotonic: float,
    command_arguments: list[str] | None = None,
    last_successful_checkpoint: str = "",
) -> None:

    try:
        traceback_text = "".join(traceback.format_exception(error))
        report = build_failure_report(
            config=config,
            phase=phase,
            error=error,
            traceback_text=traceback_text,
            elapsed_seconds=time.monotonic() - started_at_monotonic,
            command_arguments=command_arguments or [],
            last_successful_checkpoint=last_successful_checkpoint,
        )
        config.logdir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(config.logdir / "failure_report.json", report)
        (config.logdir / "failure_traceback.txt").write_text(
            traceback_text,
            encoding="utf-8",
        )
        (config.logdir / "failure_report.txt").write_text(
            render_failure_report_text(report),
            encoding="utf-8",
        )
        write_phase_event(
            logdir=config.logdir,
            event="phase_failed",
            payload={
                "phase": phase,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "elapsed_seconds": report["elapsed_seconds"],
            },
        )
        collect_failure_artifacts(
            config=config,
            report=report,
        )
    except Exception as diagnostic_error:
        print(
            "Failed to write failure diagnostics: "
            f"{type(diagnostic_error).__name__}: {diagnostic_error}",
            file=sys.stderr,
        )


def build_failure_report(
    config: AIMOTrainingConfig,
    phase: str,
    error: Exception,
    traceback_text: str,
    elapsed_seconds: float,
    command_arguments: list[str],
    last_successful_checkpoint: str,
) -> dict[str, Any]:

    return {
        "phase": phase,
        "role": config.role,
        "rank": config.global_rank,
        "local_rank": config.local_rank,
        "world_size": config.world_size,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback_text,
        "elapsed_seconds": elapsed_seconds,
        "last_successful_checkpoint": last_successful_checkpoint,
        "command_arguments": redact_arguments(command_arguments),
        "environment": redacted_environment_summary(),
        "resolved_paths": resolved_path_summary(config),
        "path_status": path_status_summary(config),
        "recent_log_tails": recent_log_tails(config.logdir),
        "recommended_inspection_paths": recommended_inspection_paths(config),
    }


def render_failure_report_text(report: dict[str, Any]) -> str:

    lines = [
        f"phase: {report['phase']}",
        f"role: {report['role']}",
        f"rank: {report['rank']}",
        f"hostname: {report['hostname']}",
        f"pid: {report['pid']}",
        f"exception: {report['exception_type']}: {report['exception_message']}",
        f"elapsed_seconds: {report['elapsed_seconds']}",
        f"last_successful_checkpoint: {report['last_successful_checkpoint'] or '<none>'}",
        "recommended_inspection_paths:",
        *[
            f"- {path}"
            for path in report["recommended_inspection_paths"]
        ],
        "traceback:",
        str(report["traceback"]),
    ]

    return "\n".join(lines) + "\n"


def redacted_environment_summary() -> dict[str, str]:

    selected_names = sorted([
        name
        for name in os.environ
        if (
            name.startswith("AIMO_")
            or name.startswith("NCCL_")
            or name.startswith("CUDA")
            or name.startswith("HF_")
            or name in {
                "GLOBAL_RANK",
                "LOCAL_RANK",
                "RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "PYTHONPATH",
                "PATH",
                "TMP",
                "TMPDIR",
                "XDG_CACHE_HOME",
                "TRANSFORMERS_CACHE",
                "HUGGINGFACE_HUB_CACHE",
                "TORCH_HOME",
                "TRITON_CACHE_DIR",
                "VLLM_CACHE_ROOT",
            }
        )
    ])

    return {
        name: redact_value(name=name, value=os.environ[name])
        for name in selected_names
    }


def redact_arguments(arguments: list[str]) -> list[str]:

    redacted_arguments: list[str] = []
    redact_next = False

    for argument in arguments:
        if redact_next:
            redacted_arguments.append("<redacted>")
            redact_next = False
            continue

        if is_sensitive_name(argument):
            redacted_arguments.append(argument)
            redact_next = True
            continue

        redacted_arguments.append(redact_value(name=argument, value=argument))

    return redacted_arguments


def redact_value(name: str, value: str) -> str:

    if is_sensitive_name(name):
        return "<redacted>"

    return value


def is_sensitive_name(name: str) -> bool:

    normalized_name = name.upper()

    return any(
        fragment in normalized_name
        for fragment in REDACTED_ENVIRONMENT_FRAGMENTS
    )


def resolved_path_summary(config: AIMOTrainingConfig) -> dict[str, str]:

    return {
        "model_path": str(config.model_path),
        "judge_model_path": str(config.judge_model_path),
        "dataset_path": str(config.dataset_path),
        "output_path": str(config.output_path),
        "logdir": str(config.logdir),
        "group_queue_path": str(config.resolved_group_queue_path),
        "online_control_dir": str(config.online_control_dir or config.logdir / "online_control"),
    }


def path_status_summary(config: AIMOTrainingConfig) -> dict[str, dict[str, Any]]:

    return {
        name: path_status(Path(path))
        for name, path in resolved_path_summary(config).items()
    }


def path_status(path: Path) -> dict[str, Any]:

    payload: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
    }

    try:
        path_stat = path.stat()
    except OSError as error:
        payload["stat_error"] = str(error)
        return payload

    payload["size_bytes"] = path_stat.st_size
    payload["mode"] = stat.filemode(path_stat.st_mode)

    return payload


def recent_log_tails(logdir: Path, max_bytes: int = 12000) -> dict[str, str]:

    if not logdir.exists():
        return {}

    tails: dict[str, str] = {}

    for path in sorted(logdir.rglob("*")):
        if not path.is_file():
            continue

        if path.name not in {
            "vllm_stdout.log",
            "vllm_stderr.log",
            "failure_traceback.txt",
            "phase_events.jsonl",
            "online_events.jsonl",
        }:
            continue

        tails[str(path)] = read_file_tail(path=path, max_bytes=max_bytes)

    return tails


def read_file_tail(path: Path, max_bytes: int) -> str:

    try:
        with path.open("rb") as input_file:
            input_file.seek(0, os.SEEK_END)
            size = input_file.tell()
            input_file.seek(max(0, size - max_bytes), os.SEEK_SET)
            payload = input_file.read()
    except OSError as error:
        return f"<failed to read {path}: {error}>"

    if not payload:
        return "<empty>"

    return payload.decode("utf-8", errors="replace")


def recommended_inspection_paths(config: AIMOTrainingConfig) -> list[str]:

    return [
        str(config.logdir / "failure_report.json"),
        str(config.logdir / "failure_report.txt"),
        str(config.logdir / "failure_traceback.txt"),
        str(config.logdir / "phase_events.jsonl"),
        str(config.logdir / "online_events.jsonl"),
        str(config.logdir / "online_control"),
        str(config.logdir / "online_servers"),
        str(config.output_path / "failure_artifacts"),
    ]


def collect_failure_artifacts(
    config: AIMOTrainingConfig,
    report: dict[str, Any],
) -> Path:

    artifact_dir = config.output_path / "failure_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[dict[str, Any]] = []

    write_json_atomic(artifact_dir / "environment_summary.json", report["environment"])
    write_json_atomic(artifact_dir / "resolved_path_summary.json", report["resolved_paths"])
    write_text(artifact_dir / "nvidia_smi.txt", collect_nvidia_smi())
    write_text(artifact_dir / "process_summary.txt", collect_process_summary())

    for source_path in selected_failure_artifact_paths(config):
        destination_path = artifact_dir / source_path.relative_to(config.logdir)

        if is_relative_to(source_path, artifact_dir):
            continue

        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            copied_files.append({
                "source": str(source_path),
                "destination": str(destination_path),
                "size_bytes": destination_path.stat().st_size,
            })
        except OSError as error:
            copied_files.append({
                "source": str(source_path),
                "destination": str(destination_path),
                "error": str(error),
            })

    write_json_atomic(
        artifact_dir / "failure_artifacts_manifest.json",
        {
            "created_at_unix": time.time(),
            "files": copied_files,
        },
    )

    return artifact_dir


def selected_failure_artifact_paths(config: AIMOTrainingConfig) -> list[Path]:

    paths: list[Path] = []

    for pattern in FAILURE_ARTIFACT_PATTERNS:
        paths.extend([
            path
            for path in config.logdir.glob(pattern)
            if path.is_file()
        ])

    return sorted(set(paths))


def collect_nvidia_smi() -> str:

    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found\n"

    return run_text_command([
        "nvidia-smi",
    ])


def collect_process_summary() -> str:

    process_text = run_text_command([
        "ps",
        "-eo",
        "pid,ppid,stat,comm,args",
    ])
    selected_lines = [
        line
        for line in process_text.splitlines()
        if any(
            fragment in line.lower()
            for fragment in {
                "python",
                "vllm",
                "torch",
                "ray",
                "singularity",
                "apptainer",
            }
        )
    ]

    return "\n".join(selected_lines) + "\n"


def run_text_command(command: list[str]) -> str:

    try:
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"{command}: {type(error).__name__}: {error}\n"

    return (
        f"command: {command}\n"
        f"returncode: {completed_process.returncode}\n"
        f"stdout:\n{completed_process.stdout}\n"
        f"stderr:\n{completed_process.stderr}\n"
    )


def write_json_atomic(path: Path, payload: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    os.replace(temporary_path, path)


def write_text(path: Path, text: str) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def is_relative_to(path: Path, parent: Path) -> bool:

    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
