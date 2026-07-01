from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimo_inference.config import AIMOConfig


@dataclass(frozen=True)
class AIMOSandboxResult:

    success: bool
    output: str
    error: str
    timed_out: bool

    def to_tool_payload(self) -> str:

        if self.timed_out:
            return "Python execution timed out."

        if self.success:
            return self.output or "Python execution completed with no output."

        return self.error or "Python execution failed."


class AIMOSandbox:

    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    custom_error_rewrites = (
        (
            re.compile(r"ValueError: Exceeds the limit \(4300 digits\) for integer string conversion"),
            (
                "ValueError: Exceeds the limit (4300 digits) for integer string conversion. "
                "Use logarithms to count digits or modular arithmetic to check equality."
            ),
        ),
        (
            re.compile(r"AttributeError: .* has no attribute 'valuation'"),
            (
                "AttributeError: module 'sympy' has no attribute 'valuation'. "
                "Use sympy.multiplicity(p, n) to compute the p-adic valuation of n."
            ),
        ),
        (
            re.compile(r"NameError: name '(.+?)' is not defined"),
            (
                "NameError: name '\\1' is not defined. "
                "Ensure you define every variable and helper function before use."
            ),
        ),
    )

    def __init__(
        self,
        config: AIMOConfig | None = None,
        timeout_seconds: float | None = None,
        max_output_chars: int = 2048,
    ) -> None:

        self.config = config or AIMOConfig()
        self.timeout_seconds = timeout_seconds or self.config.tool_timeout_seconds
        self.max_output_chars = max_output_chars

    def execute(self, code: str) -> AIMOSandboxResult:

        try:
            completed_process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    self._runner_code(),
                ],
                input=code,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return AIMOSandboxResult(
                success=False,
                output="",
                error="Python execution timed out.",
                timed_out=True,
            )

        if completed_process.returncode != 0:
            return AIMOSandboxResult(
                success=False,
                output="",
                error=self._rewrite_error(completed_process.stderr),
                timed_out=False,
            )

        try:
            payload = json.loads(completed_process.stdout)
        except json.JSONDecodeError:
            return AIMOSandboxResult(
                success=False,
                output="",
                error=self._rewrite_error(completed_process.stderr or completed_process.stdout),
                timed_out=False,
            )

        output = self._compact_text(
            "\n".join(
                entry
                for entry in [
                    str(payload.get("stdout", "")),
                    str(payload.get("stderr", "")),
                ]
                if entry
            )
        )
        payload_error = str(payload.get("error", ""))
        error = self._rewrite_error(payload_error) if payload_error else ""

        return AIMOSandboxResult(
            success=bool(payload.get("success", False)),
            output=output,
            error=error,
            timed_out=False,
        )

    def reset(self) -> None:

        return None

    def close(self) -> None:

        return None

    def _runner_code(self) -> str:

        return textwrap.dedent(
            """
            import collections
            import contextlib
            import decimal
            import fractions
            import functools
            import io
            import itertools
            import json
            import math
            import random
            import statistics
            import sys
            import traceback

            try:
                import numpy as np
            except Exception:
                np = None

            try:
                import sympy as sp
            except Exception:
                sp = None

            try:
                import networkx as nx
            except Exception:
                nx = None

            try:
                import mpmath as mp
            except Exception:
                mp = None

            try:
                import z3
            except Exception:
                z3 = None

            code = sys.stdin.read()
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            namespace = {
                "collections": collections,
                "decimal": decimal,
                "fractions": fractions,
                "functools": functools,
                "itertools": itertools,
                "math": math,
                "mp": mp,
                "mpmath": mp,
                "random": random,
                "statistics": statistics,
                "Fraction": fractions.Fraction,
                "Decimal": decimal.Decimal,
                "np": np,
                "numpy": np,
                "sp": sp,
                "sympy": sp,
                "nx": nx,
                "networkx": nx,
                "z3": z3,
            }

            try:
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    exec(compile(code, "<aimo-sandbox>", "exec"), namespace, namespace)
                payload = {
                    "success": True,
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                    "error": "",
                }
            except Exception:
                payload = {
                    "success": False,
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                    "error": traceback.format_exc(limit=4),
                }

            print(json.dumps(payload))
            """
        )

    def _rewrite_error(self, error_text: str) -> str:

        cleaned_error = self._compact_traceback(error_text)

        for pattern, replacement in self.custom_error_rewrites:
            if pattern.search(cleaned_error):
                return pattern.sub(replacement, cleaned_error)

        return cleaned_error or "Python execution failed."

    def _compact_traceback(self, error_text: str) -> str:

        cleaned_text = self._strip_ansi(error_text)
        compact_lines = []

        for line in cleaned_text.splitlines():
            stripped_line = line.strip()

            if not stripped_line:
                continue

            if "Traceback (most recent call last)" in stripped_line:
                continue

            if "----------" in stripped_line:
                continue

            if stripped_line.startswith("File "):
                continue

            if "/dist-packages/" in stripped_line:
                continue

            if "/usr/local/lib" in stripped_line or "/usr/lib" in stripped_line:
                continue

            if stripped_line.startswith("^"):
                continue

            compact_lines.append(stripped_line)

        return self._compact_text("\n".join(compact_lines))

    def _compact_text(self, text: str) -> str:

        cleaned_text = self._strip_ansi(text).strip()

        if len(cleaned_text) <= self.max_output_chars:
            return cleaned_text

        slice_chars = self.max_output_chars // 2
        truncated_chars = len(cleaned_text) - self.max_output_chars

        return (
            f"{cleaned_text[:slice_chars]}\n"
            f"... [Truncated {truncated_chars} characters] ...\n"
            f"{cleaned_text[-slice_chars:]}"
        )

    def _strip_ansi(self, text: str) -> str:

        return self.ansi_pattern.sub("", text)


class AIMOSandboxLease:

    def __init__(
        self,
        sandbox_pool: AIMOSandboxPool,
        sandbox: AIMOSandbox,
    ) -> None:

        self.sandbox_pool = sandbox_pool
        self.sandbox = sandbox

    def __enter__(self) -> AIMOSandbox:

        return self.sandbox

    def __exit__(self, *_: object) -> None:

        self.sandbox_pool.release(self.sandbox)


class AIMOSandboxPool:

    def __init__(
        self,
        config: AIMOConfig,
        sandbox_count: int | None = None,
    ) -> None:

        self.config = config
        self.sandbox_count = max(1, sandbox_count or config.sandbox_count)
        self._available_sandboxes: queue.Queue[AIMOSandbox] = queue.Queue()
        self._created_sandboxes: list[AIMOSandbox] = []
        self._lock = threading.Lock()
        self._closed = False

    def acquire(self) -> AIMOSandboxLease:

        sandbox = self._acquire_sandbox()

        return AIMOSandboxLease(
            sandbox_pool=self,
            sandbox=sandbox,
        )

    def release(self, sandbox: AIMOSandbox) -> None:

        with contextlib.suppress(Exception):
            sandbox.reset()

        with self._lock:
            if self._closed:
                with contextlib.suppress(Exception):
                    sandbox.close()

                return

        self._available_sandboxes.put(sandbox)

    def close(self) -> None:

        with self._lock:
            self._closed = True
            sandboxes = list(self._created_sandboxes)
            self._created_sandboxes = []

        for sandbox in sandboxes:
            with contextlib.suppress(Exception):
                sandbox.close()

    def _acquire_sandbox(self) -> AIMOSandbox:

        try:
            return self._available_sandboxes.get_nowait()
        except queue.Empty:
            pass

        with self._lock:
            if self._closed:
                raise RuntimeError("Sandbox pool is closed.")

            if len(self._created_sandboxes) < self.sandbox_count:
                sandbox = self._build_sandbox()
                self._created_sandboxes.append(sandbox)

                return sandbox

        return self._available_sandboxes.get()

    def _build_sandbox(self) -> AIMOSandbox:

        sandbox_class: Any = AIMOJupyterSandbox if self.config.use_jupyter_sandbox else AIMOSandbox

        return sandbox_class(config=self.config)


def run_sandbox_pool_preflight(
    sandbox_pool: AIMOSandboxPool,
    sandbox_count: int,
    log_path: Path,
    pool_role: str,
) -> dict[str, Any]:

    started_at = time.monotonic()
    leases: list[AIMOSandboxLease] = []
    sandboxes: list[AIMOSandbox] = []
    payload: dict[str, Any] = {
        "pool_role": pool_role,
        "requested_sandbox_count": sandbox_count,
        "passed": False,
    }

    try:
        for _ in range(max(1, sandbox_count)):
            lease = sandbox_pool.acquire()
            sandbox = lease.__enter__()
            leases.append(lease)
            sandboxes.append(sandbox)

        result = sandboxes[0].execute("print(2 + 2)")
        payload.update({
            "acquired_sandbox_count": len(sandboxes),
            "execution_success": result.success,
            "execution_output": result.output,
            "execution_error": result.error,
            "execution_timed_out": result.timed_out,
            "elapsed_seconds": time.monotonic() - started_at,
            "passed": result.success and result.output.strip() == "4",
        })

        if not payload["passed"]:
            raise RuntimeError(
                f"Sandbox preflight failed for {pool_role}: {result.to_tool_payload()}"
            )

        return payload
    except Exception as error:
        payload.update({
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "elapsed_seconds": time.monotonic() - started_at,
            "passed": False,
        })
        raise
    finally:
        for lease in reversed(leases):
            lease.__exit__(None, None, None)

        write_sandbox_preflight(path=log_path, payload=payload)


def write_sandbox_preflight(path: Path, payload: dict[str, Any]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    os.replace(temporary_path, path)


class AIMOJupyterSandbox(AIMOSandbox):

    _port_lock = threading.Lock()
    _blacklisted_ports: set[int] = set()
    _last_blacklist_clear = time.time()
    _blacklist_timeout = 300.0
    _next_port = 50000

    def __init__(
        self,
        config: AIMOConfig | None = None,
        timeout_seconds: float | None = None,
        max_output_chars: int = 2048,
        kernel_attempts: int = 3,
        port_increment: int = 10,
        port_attempts: int = 10,
        port_timeout_seconds: float = 300.0,
        min_port: int = 50000,
        max_port: int = 65535,
        backoff_delay_seconds: float = 0.5,
        iopub_timeout_seconds: float = 1.0,
    ) -> None:

        super().__init__(
            config=config,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        type(self)._blacklist_timeout = port_timeout_seconds
        self.kernel_attempts = kernel_attempts
        self.port_increment = port_increment
        self.port_attempts = port_attempts
        self.min_port = min_port
        self.max_port = max_port
        self.backoff_delay_seconds = backoff_delay_seconds
        self.iopub_timeout_seconds = iopub_timeout_seconds
        self._kernel_manager = None
        self._client = None
        self._start_kernel()
        self._prime_kernel()

    def execute(self, code: str) -> AIMOSandboxResult:

        if self._client is None or self._kernel_manager is None:
            return AIMOSandboxResult(
                success=False,
                output="",
                error="Jupyter kernel is not running.",
                timed_out=False,
            )

        message_id = self._client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        started_at = time.monotonic()

        while True:
            if time.monotonic() - started_at > self.timeout_seconds:
                self._kernel_manager.interrupt_kernel()

                return AIMOSandboxResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout_seconds} seconds.",
                    timed_out=True,
                )

            try:
                message = self._client.get_iopub_msg(timeout=self.iopub_timeout_seconds)
            except queue.Empty:
                continue

            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue

            message_type = message.get("msg_type")
            content = message.get("content", {})

            if message_type == "stream":
                if content.get("name") == "stdout":
                    stdout_parts.append(str(content.get("text", "")))
                else:
                    stderr_parts.append(str(content.get("text", "")))

            elif message_type == "error":
                traceback_items = [
                    str(item)
                    for item in content.get("traceback", [])
                ]
                stderr_parts.append(self._rewrite_error("\n".join(traceback_items)))

            elif message_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")

                if text:
                    stdout_parts.append(str(text))

            elif message_type == "status" and content.get("execution_state") == "idle":
                break

        output = self._compact_text("".join(stdout_parts))
        error = self._rewrite_error("".join(stderr_parts)) if stderr_parts else ""

        return AIMOSandboxResult(
            success=not bool(error),
            output=output,
            error=error,
            timed_out=False,
        )

    def reset(self) -> None:

        with contextlib.suppress(Exception):
            self.execute("%reset -f")

        with contextlib.suppress(Exception):
            self._prime_kernel()

    def close(self) -> None:

        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.stop_channels()

        if self._kernel_manager is not None:
            with contextlib.suppress(Exception):
                self._kernel_manager.shutdown_kernel(now=True)

        self._client = None
        self._kernel_manager = None

    def _start_kernel(self) -> None:

        try:
            from jupyter_client import KernelManager
        except Exception as error:
            raise RuntimeError(
                "jupyter_client is required for persistent Jupyter sandboxes."
            ) from error

        last_error: Exception | None = None
        delay_seconds = self.backoff_delay_seconds

        for attempt_index in range(self.kernel_attempts):
            ports: list[int] | None = None

            try:
                ports = self._get_next_ports(5)
                kernel_manager = KernelManager()
                kernel_manager.shell_port = ports[0]
                kernel_manager.iopub_port = ports[1]
                kernel_manager.stdin_port = ports[2]
                kernel_manager.hb_port = ports[3]
                kernel_manager.control_port = ports[4]
                kernel_manager.start_kernel(
                    env=self._kernel_environment(),
                    extra_arguments=[
                        "--Application.log_level=CRITICAL",
                    ],
                )
                client = kernel_manager.blocking_client()
                client.start_channels()
                client.wait_for_ready(timeout=self.timeout_seconds)
                self._kernel_manager = kernel_manager
                self._client = client

                return
            except Exception as error:
                last_error = error

                if ports and "Address already in use" in str(error):
                    self._blacklist_ports(ports)

                if self._kernel_manager is not None:
                    with contextlib.suppress(Exception):
                        self._kernel_manager.shutdown_kernel(now=True)

                if attempt_index < self.kernel_attempts - 1:
                    time.sleep(delay_seconds)
                    delay_seconds *= 2

        raise RuntimeError(f"Failed to start Jupyter kernel: {last_error}") from last_error

    def _prime_kernel(self) -> None:

        self.execute(
            "import collections\n"
            "import decimal\n"
            "import fractions\n"
            "import functools\n"
            "import itertools\n"
            "import math\n"
            "import random\n"
            "import statistics\n"
            "import numpy as np\n"
            "import sympy as sp\n"
            "from fractions import Fraction\n"
            "from decimal import Decimal, getcontext\n"
            "getcontext().prec = 64\n"
        )

    def _kernel_environment(self) -> dict[str, str]:

        environment = os.environ.copy()
        environment.update({
            "PYDEVD_DISABLE_FILE_VALIDATION": "1",
            "PYDEVD_WARN_EVALUATION_TIMEOUT": "0",
            "JUPYTER_PLATFORM_DIRS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONWARNINGS": "ignore",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        })

        return environment

    @classmethod
    def _is_port_available(cls, port: int) -> bool:

        if port in cls._blacklisted_ports:
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_object:
                socket_object.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                socket_object.bind(("127.0.0.1", port))

                return True
        except OSError:
            return False

    @classmethod
    def _clear_old_blacklist(cls) -> None:

        current_time = time.time()

        if current_time - cls._last_blacklist_clear > cls._blacklist_timeout:
            cls._blacklisted_ports.clear()
            cls._last_blacklist_clear = current_time

    def _get_next_ports(self, count: int) -> list[int]:

        with type(self)._port_lock:
            type(self)._clear_old_blacklist()

            for _ in range(self.port_attempts):
                start_port = type(self)._next_port
                candidate_ports: list[int] = []

                for offset in range(count):
                    port = start_port + offset

                    if port > self.max_port:
                        start_port = self.min_port
                        port = start_port + offset
                        candidate_ports = []

                    if type(self)._is_port_available(port):
                        candidate_ports.append(port)
                    else:
                        type(self)._next_port = port + 1
                        candidate_ports = []
                        break

                if len(candidate_ports) == count:
                    type(self)._next_port = candidate_ports[-1] + 1

                    if type(self)._next_port > self.max_port:
                        type(self)._next_port = self.min_port

                    return candidate_ports

                type(self)._next_port += self.port_increment

                if type(self)._next_port > self.max_port:
                    type(self)._next_port = self.min_port

        raise RuntimeError(f"Unable to find {count} available Jupyter ports.")

    def _blacklist_ports(self, ports: list[int]) -> None:

        with type(self)._port_lock:
            type(self)._blacklisted_ports.update(ports)
