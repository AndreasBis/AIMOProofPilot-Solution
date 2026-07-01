from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO

from aimo_inference.config import AIMOConfig
from aimo_inference.profiles import resolve_model_profile


VLLM_ENVIRONMENT_OVERRIDES = {
    "VLLM_MLA_DISABLE": "1",
    "VLLM_DO_NOT_TRACK": "1",
    "TRANSFORMERS_NO_TF": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "TRANSFORMERS_NO_FLAX": "1",
    "VLLM_LOG_STATS_INTERVAL": "60",
    "VLLM_ENABLE_CUDAGRAPH_GC": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
}

CACHE_ENVIRONMENT_DEFAULTS = {
    "XDG_CACHE_HOME": "/tmp/aimo-cache",
    "HF_HOME": "/tmp/aimo-cache/huggingface",
    "TRANSFORMERS_CACHE": "/tmp/aimo-cache/huggingface/transformers",
    "HUGGINGFACE_HUB_CACHE": "/tmp/aimo-cache/huggingface/hub",
    "TORCH_HOME": "/tmp/aimo-cache/torch",
    "TRITON_CACHE_DIR": "/tmp/aimo-cache/triton",
    "VLLM_CACHE_ROOT": "/tmp/aimo-cache/vllm",
}

MODEL_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
)

TOKENIZER_FILENAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
}

TOKENIZER_REQUIRED_FILE_GROUPS = (
    (
        "tokenizer.json",
    ),
    (
        "tokenizer_config.json",
    ),
    (
        "tokenizer.model",
    ),
    (
        "vocab.json",
        "merges.txt",
    ),
)


def resolved_enable_expert_parallel(config: AIMOConfig) -> bool:

    served_model_name = config.served_model_name.upper().replace("_", "-")

    return (
        config.enable_expert_parallel
        or (
            config.model_profile == "judge"
            and "GPT-OSS" in served_model_name
        )
    )


class AIMOServicePreflight:

    def __init__(
        self,
        config: AIMOConfig,
        role: str | None = None,
        rank: int | None = None,
        hostname: str | None = None,
        detected_gpu_count: int | None = None,
    ) -> None:

        self.config = config
        self.role = role or config.model_profile
        self.rank = rank
        self.hostname = hostname or socket.gethostname()
        self.injected_gpu_count = detected_gpu_count

    def run(self) -> dict[str, object]:

        failures: list[str] = []
        warnings: list[str] = []
        model_summary = self.model_path_summary(failures=failures)
        logdir_summary = self.logdir_summary(failures=failures)
        port_summary = self.port_summary(failures=failures)
        gpu_summary = self.gpu_summary(failures=failures)
        temporary_summary = self.temporary_path_summary(failures=failures)
        payload = {
            "passed": not failures,
            "failures": failures,
            "warnings": warnings,
            "role": self.role,
            "rank": self.rank,
            "hostname": self.hostname,
            "pid": os.getpid(),
            "model_profile": self.config.model_profile,
            "served_model_name": self.config.served_model_name,
            "model_path": str(self.config.model_path),
            "host": self.config.host,
            "port": self.config.port,
            "health_url": self.config.health_url,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "data_parallel_size": int(
                self.config.extra_server_arguments.get("data_parallel_size", 1)
            ),
            "moe_backend": self.config.moe_backend,
            "all2all_backend": str(self.config.extra_server_arguments.get("all2all_backend", "")),
            "enable_expert_parallel": resolved_enable_expert_parallel(self.config),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "model": model_summary,
            "logdir": logdir_summary,
            "port_check": port_summary,
            "gpu": gpu_summary,
            "temporary_paths": temporary_summary,
            "preflight_scope": "filesystem_ports_gpus_temporary_paths",
        }
        self.write_json(path=self.config.logdir / "service_preflight.json", payload=payload)

        if failures:
            raise RuntimeError(
                "Service preflight failed for "
                f"{self.role} rank {self.rank}: "
                + "; ".join(failures)
            )

        return payload

    def model_path_summary(self, failures: list[str]) -> dict[str, object]:

        model_path = self.config.model_path
        config_path = model_path / "config.json"
        tokenizer_paths = [
            model_path / filename
            for filename in TOKENIZER_FILENAMES
            if (model_path / filename).exists()
        ]
        weight_paths = self.find_weight_paths(model_path)
        summary = {
            "exists": model_path.exists(),
            "is_dir": model_path.is_dir(),
            "readable": os.access(model_path, os.R_OK),
            "config_json": str(config_path),
            "config_json_exists": config_path.exists(),
            "tokenizer_files": [
                str(path)
                for path in tokenizer_paths
            ],
            "weight_files": [
                str(path)
                for path in weight_paths[:32]
            ],
            "weight_file_count": len(weight_paths),
        }

        if not model_path.exists():
            failures.append(f"model path does not exist: {model_path}")
            return summary

        if not model_path.is_dir():
            failures.append(f"model path is not a directory: {model_path}")

        if not os.access(model_path, os.R_OK):
            failures.append(f"model path is not readable: {model_path}")

        if not config_path.exists():
            failures.append(f"model config.json is missing: {config_path}")

        if not self.tokenizer_metadata_exists(model_path):
            failures.append(f"tokenizer metadata is missing under: {model_path}")

        if not weight_paths:
            failures.append(f"model weight files are missing under: {model_path}")

        unreadable_paths = [
            path
            for path in [
                config_path,
                *tokenizer_paths,
                *weight_paths,
            ]
            if path.exists() and not os.access(path, os.R_OK)
        ]

        if unreadable_paths:
            failures.append(
                "model files are not readable: "
                + ", ".join(str(path) for path in unreadable_paths[:32])
            )

        if (
            self.config.model_profile == "judge"
            and "GPT-OSS" in self.config.served_model_name.upper().replace("_", "-")
            and not resolved_enable_expert_parallel(self.config)
        ):
            failures.append("GPT-OSS judge requires --enable-expert-parallel.")

        return summary

    def tokenizer_metadata_exists(self, model_path: Path) -> bool:

        return any(
            all(
                (model_path / filename).exists()
                for filename in filenames
            )
            for filenames in TOKENIZER_REQUIRED_FILE_GROUPS
        )

    def find_weight_paths(self, model_path: Path) -> list[Path]:

        if not model_path.exists() or not model_path.is_dir():
            return []

        weight_paths: list[Path] = []

        for suffix in MODEL_WEIGHT_SUFFIXES:
            weight_paths.extend(sorted(model_path.rglob(f"*{suffix}")))

        return weight_paths

    def logdir_summary(self, failures: list[str]) -> dict[str, object]:

        logdir = self.config.logdir
        probe_path = logdir / f".service_preflight_write_probe.{os.getpid()}"
        summary = {
            "path": str(logdir),
            "exists_before": logdir.exists(),
            "writable": False,
        }

        try:
            logdir.mkdir(parents=True, exist_ok=True)
            probe_path.write_text("ok\n", encoding="utf-8")
            probe_path.unlink()
            summary["writable"] = True
        except OSError as error:
            failures.append(f"logdir is not writable: {logdir}: {error}")

        summary["exists_after"] = logdir.exists()

        return summary

    def port_summary(self, failures: list[str]) -> dict[str, object]:

        available = self.port_is_available(host=self.config.host, port=self.config.port)
        summary = {
            "host": self.config.host,
            "port": self.config.port,
            "available": available,
        }

        if not available:
            failures.append(f"port is already bound: {self.config.host}:{self.config.port}")

        return summary

    def port_is_available(self, host: str, port: int) -> bool:

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((host, port))

            return True
        except OSError:
            return False

    def gpu_summary(self, failures: list[str]) -> dict[str, object]:

        detected_gpu_count = (
            self.injected_gpu_count
            if self.injected_gpu_count is not None
            else self.detect_gpu_count()
        )
        summary = {
            "detected_gpu_count": detected_gpu_count,
            "required_gpu_count": self.config.tensor_parallel_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

        if detected_gpu_count < self.config.tensor_parallel_size:
            failures.append(
                "detected GPU count is below tensor_parallel_size: "
                f"{detected_gpu_count} < {self.config.tensor_parallel_size}"
            )

        return summary

    def detect_gpu_count(self) -> int:

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

        if cuda_visible_devices and cuda_visible_devices not in {"all", "void", "none", "-1"}:
            return len([
                item
                for item in cuda_visible_devices.split(",")
                if item.strip()
            ])

        if shutil.which("nvidia-smi") is None:
            return 0

        try:
            completed_process = subprocess.run(
                [
                    "nvidia-smi",
                    "-L",
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return 0

        if completed_process.returncode != 0:
            return 0

        return len([
            line
            for line in completed_process.stdout.splitlines()
            if line.strip().startswith("GPU ")
        ])

    def temporary_path_summary(self, failures: list[str]) -> dict[str, object]:

        return {
            "tmp": self.path_writable_summary(path=Path("/tmp"), failures=failures),
            "dev_shm": self.shared_memory_summary(failures=failures),
        }

    def path_writable_summary(self, path: Path, failures: list[str]) -> dict[str, object]:

        probe_path = path / f".aimo_preflight_probe.{os.getpid()}"
        summary = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "writable": False,
        }

        if not path.exists() or not path.is_dir():
            failures.append(f"required directory is missing: {path}")
            return summary

        try:
            probe_path.write_text("ok\n", encoding="utf-8")
            probe_path.unlink()
            summary["writable"] = True
        except OSError as error:
            failures.append(f"required directory is not writable: {path}: {error}")

        return summary

    def shared_memory_summary(self, failures: list[str]) -> dict[str, object]:

        path = Path("/dev/shm")
        summary = self.path_writable_summary(path=path, failures=failures)

        if path.exists():
            with contextlib.suppress(OSError):
                stat = os.statvfs(path)
                summary["free_bytes"] = stat.f_bavail * stat.f_frsize
                summary["total_bytes"] = stat.f_blocks * stat.f_frsize

        return summary

    def write_json(self, path: Path, payload: dict[str, object]) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")

        os.replace(temporary_path, path)


class AIMOInferenceServer:

    def __init__(self, config: AIMOConfig) -> None:

        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self.stdout_file: IO[str] | None = None
        self.stderr_file: IO[str] | None = None
        self.stdout_path = self.config.logdir / "vllm_stdout.log"
        self.stderr_path = self.config.logdir / "vllm_stderr.log"
        self.command_path = self.config.logdir / "vllm_command.json"
        self.launch_stage = "not_started"

    def __enter__(self) -> AIMOInferenceServer:

        self.start()

        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:

        self.stop()

    def build_command(self) -> list[str]:

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--seed",
            str(self.config.seed),
            "--model",
            str(self.config.model_path),
            "--served-model-name",
            self.config.served_model_name,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
            "--dtype",
            self.config.dtype,
            "--kv-cache-dtype",
            self.config.kv_cache_dtype,
            "--load-format",
            self.config.load_format,
            "--max-model-len",
            str(self.config.max_model_len),
            "--max-num-seqs",
            str(self.config.max_num_seqs),
            "--max-logprobs",
            str(max(self.config.max_logprobs, self.config.top_logprobs)),
            "--stream-interval",
            str(self.config.stream_interval),
            "--performance-mode",
            self.config.performance_mode,
            "--attention-config",
            json.dumps(self.config.attention_config),
            "--compilation-config",
            json.dumps(self.config.compilation_config),
        ]

        if self.config.moe_backend:
            command.extend([
                "--moe-backend",
                self.config.moe_backend,
            ])

        if resolved_enable_expert_parallel(self.config):
            command.append("--enable-expert-parallel")

        if self.config.max_num_batched_tokens > 0:
            command.extend([
                "--max-num-batched-tokens",
                str(self.config.max_num_batched_tokens),
            ])

        if self.config.lora_adapter_path is not None:
            command.extend([
                "--enable-lora",
                "--lora-modules",
                f"{self.config.lora_served_model_name}={self.config.lora_adapter_path}",
            ])

        if self.config.enable_prefix_caching:
            command.append("--enable-prefix-caching")

        if self.config.enable_chunked_prefill:
            command.append("--enable-chunked-prefill")

        if self.config.async_scheduling:
            command.append("--async-scheduling")

        if self.config.disable_log_stats:
            command.append("--disable-log-stats")

        for name, value in sorted(self.config.extra_server_arguments.items()):
            argument_name = f"--{str(name).replace('_', '-')}"

            if isinstance(value, bool):
                if value:
                    command.append(argument_name)

                continue

            command.extend([
                argument_name,
                str(value),
            ])

        return command

    def start(self) -> None:

        if self.config.launch_server:
            self._launch_process()
            self.wait_until_ready()
            return

        if self.config.reuse_server:
            self.wait_until_ready()

    def wait_until_ready(self) -> None:

        deadline = time.monotonic() + self.config.server_start_timeout_seconds
        last_error = ""
        health_url = self.local_health_url()

        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(self.build_process_exit_message())

            try:
                with urllib.request.urlopen(health_url, timeout=5.0) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = str(error)

            time.sleep(1.0)

        raise RuntimeError(self.build_readiness_timeout_message(last_error=last_error))

    def stop(self) -> None:

        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

            try:
                self.process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30.0)

        self.process = None
        self._close_logs()

    def _launch_process(self) -> None:

        self.config.logdir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(VLLM_ENVIRONMENT_OVERRIDES)

        for key, value in CACHE_ENVIRONMENT_DEFAULTS.items():
            environment.setdefault(key, value)

        environment.update(resolve_model_profile(self.config.model_profile).environment)

        if self.config.tiktoken_encodings_base:
            environment["TIKTOKEN_ENCODINGS_BASE"] = self.config.tiktoken_encodings_base

        command = self.build_command()
        self._write_json(
            path=self.command_path,
            payload={
                "command": command,
                "health_url": self.config.health_url,
                "probed_health_url": self.local_health_url(),
                "environment": {
                    key: environment[key]
                    for key in sorted(
                        set(VLLM_ENVIRONMENT_OVERRIDES)
                        | set(CACHE_ENVIRONMENT_DEFAULTS)
                        | {"TIKTOKEN_ENCODINGS_BASE"}
                    )
                    if key in environment
                },
                "launch_stage": "before_service_preflight",
            },
        )
        self.launch_stage = "service_preflight"
        AIMOServicePreflight(
            config=self.config,
            role=self.config.model_profile,
            rank=self.config.global_rank,
        ).run()
        self.stdout_file = self._open_log(self.stdout_path)
        self.stderr_file = self._open_log(self.stderr_path)
        self.launch_stage = "starting_vllm_process"
        self.process = subprocess.Popen(
            command,
            stdout=self.stdout_file,
            stderr=self.stderr_file,
            text=True,
            env=environment,
        )
        self.launch_stage = "waiting_for_health"

    def _open_log(self, path: Path) -> IO[str]:

        return path.open("a", encoding="utf-8")

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:

        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")

        os.replace(temporary_path, path)

    def local_health_url(self) -> str:

        host = self.config.host

        if host == "0.0.0.0":
            host = "127.0.0.1"

        return f"http://{host}:{self.config.port}/health"

    def build_process_exit_message(self) -> str:

        self._flush_logs()
        return "\n".join([
            "vLLM server exited before becoming ready.",
            f"return_code={self.process.returncode if self.process is not None else 'unknown'}",
            f"command_path={self.command_path}",
            f"health_url={self.config.health_url}",
            f"probed_health_url={self.local_health_url()}",
            f"stdout_path={self.stdout_path}",
            self._format_log_tail("stdout_tail", self.stdout_path),
            f"stderr_path={self.stderr_path}",
            self._format_log_tail("stderr_tail", self.stderr_path),
        ])

    def build_readiness_timeout_message(self, last_error: str) -> str:

        self._flush_logs()
        return "\n".join([
            f"vLLM server did not become healthy: {last_error}",
            f"return_code={self.process.returncode if self.process is not None else 'unknown'}",
            f"command_path={self.command_path}",
            f"health_url={self.config.health_url}",
            f"probed_health_url={self.local_health_url()}",
            f"stdout_path={self.stdout_path}",
            self._format_log_tail("stdout_tail", self.stdout_path),
            f"stderr_path={self.stderr_path}",
            self._format_log_tail("stderr_tail", self.stderr_path),
        ])

    def _flush_logs(self) -> None:

        for log_file in [
            self.stdout_file,
            self.stderr_file,
        ]:
            if log_file is not None:
                with contextlib.suppress(Exception):
                    log_file.flush()

    def _format_log_tail(self, label: str, path: Path, max_bytes: int = 20000) -> str:

        return f"{label}:\n{self._read_file_tail(path=path, max_bytes=max_bytes)}"

    def _read_file_tail(self, path: Path, max_bytes: int) -> str:

        if not path.exists():
            return f"<missing log file: {path}>"

        if not path.is_file():
            return f"<not a regular log file: {path}>"

        try:
            with path.open("rb") as input_file:
                input_file.seek(0, os.SEEK_END)
                size = input_file.tell()
                input_file.seek(max(0, size - max_bytes), os.SEEK_SET)
                payload = input_file.read()
        except OSError as error:
            return f"<failed to read log file {path}: {error}>"

        if not payload:
            return "<empty>"

        return payload.decode("utf-8", errors="replace")

    def _close_logs(self) -> None:

        for log_file in [self.stdout_file, self.stderr_file]:
            if log_file is not None:
                log_file.close()

        self.stdout_file = None
        self.stderr_file = None


class AIMODualServerOrchestrator:

    def __init__(
        self,
        contestant_config: AIMOConfig,
        judge_config: AIMOConfig,
    ) -> None:

        self.contestant_server = AIMOInferenceServer(contestant_config)
        self.judge_server = AIMOInferenceServer(judge_config)

    def __enter__(self) -> AIMODualServerOrchestrator:

        self.contestant_server.start()
        self.judge_server.start()

        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:

        self.judge_server.stop()
        self.contestant_server.stop()
