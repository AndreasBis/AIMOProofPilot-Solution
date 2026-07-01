from __future__ import annotations

import csv
import json
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from typing import Callable

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aimo_inference.client import AIMOGeneration
from aimo_inference.sandbox import AIMOSandboxResult
from aimo_training.schema import AIMOGRPOGroup
from aimo_training.schema import AIMORolloutSample
from aimo_training.schema import AIMORewardBreakdown


@dataclass(frozen=True)
class RecordedRequest:

    path: str
    payload: dict[str, Any]


class FakeHTTPServer:

    def __init__(self, responses: list[Any]) -> None:

        self.responses = list(responses)
        self.requests: list[RecordedRequest] = []
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def api_base(self) -> str:

        if self.server is None:
            raise RuntimeError("Fake HTTP server has not been started.")

        host, port = self.server.server_address

        return f"http://{host}:{port}/v1"

    def __enter__(self) -> FakeHTTPServer:

        owner = self

        class Handler(BaseHTTPRequestHandler):

            def do_POST(self) -> None:

                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw_body)
                owner.requests.append(
                    RecordedRequest(
                        path=self.path,
                        payload=payload,
                    )
                )
                status_code, response_payload = owner._next_response()
                response_bytes = json.dumps(response_payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)

            def log_message(self, format: str, *args: object) -> None:

                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:

        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _next_response(self) -> tuple[int, dict[str, Any]]:

        if not self.responses:
            raise RuntimeError("Fake HTTP server has no queued response.")

        response = self.responses.pop(0)

        if isinstance(response, tuple):
            status_code, payload = response
            response_payload = (
                {"error": payload}
                if isinstance(payload, str)
                else dict(payload)
            )

            return int(status_code), response_payload

        return 200, dict(response)


@pytest.fixture
def http_server_factory() -> Callable[[list[Any]], FakeHTTPServer]:

    return FakeHTTPServer


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:

    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def fake_generation(
    text: str,
    input_tokens: int | None = 11,
    output_tokens: int | None = 7,
    finish_reason: str = "stop",
) -> AIMOGeneration:

    return AIMOGeneration(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        latency_seconds=0.25,
        raw={
            "text": text,
        },
    )


def reward_breakdown(
    judge_grade: int = 7,
    context_reward: int = 1,
    solution_page_reward: int = 1,
    scalar_reward: float | None = None,
    judge_parse_failed: bool = False,
) -> AIMORewardBreakdown:

    resolved_scalar_reward = (
        float(judge_grade + context_reward + solution_page_reward)
        if scalar_reward is None
        else scalar_reward
    )

    return AIMORewardBreakdown(
        judge_grade=judge_grade,
        context_reward=context_reward,
        solution_page_reward=solution_page_reward,
        scalar_reward=resolved_scalar_reward,
        rendered_page_count=4,
        page_count_method="word_count",
        latex_compile_status="not_attempted",
        page_count_fallback_reason="",
        judge_response=f"Grade \\boxed{{{judge_grade}}}",
        judge_parse_failed=judge_parse_failed,
        input_tokens=10,
        output_tokens=20,
        finish_reason="stop",
        latency_seconds=0.1,
    )


def rollout_sample(
    problem_id: str = "p1",
    group_index: int = 0,
    rollout_index: int = 0,
    scalar_reward: float | None = None,
    judge_parse_failed: bool = False,
) -> AIMORolloutSample:

    return AIMORolloutSample(
        problem_id=problem_id,
        group_index=group_index,
        rollout_index=rollout_index,
        prompt=f"Problem {problem_id}",
        completion=f"Proof {rollout_index}",
        token_ids=[
            10,
            11,
            12,
        ],
        token_logprobs=[
            -0.3,
            -0.2,
            -0.1,
        ],
        input_tokens=5,
        output_tokens=3,
        finish_reason="stop",
        python_calls=0,
        python_errors=0,
        tool_call_count=0,
        tool_error_count=0,
        reward=reward_breakdown(
            scalar_reward=scalar_reward,
            judge_parse_failed=judge_parse_failed,
        ),
        prompt_ids=[
            1,
            2,
            3,
        ],
        env_mask=[
            1,
            1,
            1,
        ],
    )


def grpo_group(
    problem_id: str = "p1",
    group_index: int = 0,
    sample_count: int = 2,
) -> AIMOGRPOGroup:

    return AIMOGRPOGroup(
        group_index=group_index,
        problem_id=problem_id,
        problem=f"Problem {problem_id}",
        reference_solution=f"Reference {problem_id}",
        samples=[
            rollout_sample(
                problem_id=problem_id,
                group_index=group_index,
                rollout_index=rollout_index,
                scalar_reward=float(rollout_index),
                judge_parse_failed=rollout_index == sample_count - 1,
            )
            for rollout_index in range(sample_count)
        ],
        metadata={
            "source": "fixture",
        },
    )


@pytest.fixture
def fake_openai_harmony_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:

    module = ModuleType("openai_harmony")

    class FakeRole:

        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
        TOOL = "tool"

    class FakeReasoningEffort:

        HIGH = "high"

    class FakeHarmonyEncodingName:

        HARMONY_GPT_OSS = "harmony_gpt_oss"

    class FakeTextContent:

        def __init__(self, text: str) -> None:

            self.text = text

    class FakeAuthor:

        def __init__(self, role: str, name: str | None = None) -> None:

            self.role = role
            self.name = name

    class FakeMessage:

        def __init__(
            self,
            author: FakeAuthor | None = None,
            content: list[object] | None = None,
        ) -> None:

            self.author = author
            self.role = author.role if author is not None else ""
            self.content = content or []
            self.recipient: str | None = None
            self.channel: str | None = None

        @classmethod
        def from_role_and_content(cls, role: str, content: object) -> FakeMessage:

            return cls(
                author=FakeAuthor(role=role),
                content=[
                    FakeTextContent(text=str(content)),
                ],
            )

        def with_recipient(self, recipient: str) -> FakeMessage:

            self.recipient = recipient

            return self

        def with_channel(self, channel: str) -> FakeMessage:

            self.channel = channel

            return self

    class FakeConversation:

        def __init__(self, messages: list[object]) -> None:

            self.messages = messages

        @classmethod
        def from_messages(cls, messages: list[object]) -> FakeConversation:

            return cls(messages=messages)

    class FakeSystemContent:

        def __init__(self) -> None:

            self.model_identity = ""
            self.reasoning_effort = ""
            self.tools: object | None = None

        @classmethod
        def new(cls) -> FakeSystemContent:

            return cls()

        def with_model_identity(self, model_identity: str) -> FakeSystemContent:

            self.model_identity = model_identity

            return self

        def with_reasoning_effort(self, reasoning_effort: str) -> FakeSystemContent:

            self.reasoning_effort = reasoning_effort

            return self

        def with_tools(self, tools: object) -> FakeSystemContent:

            self.tools = tools

            return self

        def __str__(self) -> str:

            return self.model_identity

    class FakeToolNamespaceConfig:

        def __init__(
            self,
            name: str,
            description: str,
            tools: list[object],
        ) -> None:

            self.name = name
            self.description = description
            self.tools = tools

    class FakeEncoding:

        def render_conversation_for_completion(
            self,
            conversation: object,
            role: object,
        ) -> list[int]:

            messages = getattr(conversation, "messages", [])

            return [
                101,
                len(messages),
            ]

        def stop_tokens_for_assistant_actions(self) -> list[int]:

            return [
                200002,
                200003,
            ]

        def parse_messages_from_completion_tokens(
            self,
            token_ids: list[int],
            role: object,
            strict: bool = False,
        ) -> list[object]:

            return []

        def encode(
            self,
            text: str,
            allowed_special: str | None = None,
            disallowed_special: object | None = None,
        ) -> list[int]:

            return list(range(max(1, len(text.encode("utf-8")) // 4)))

    def load_harmony_encoding(name: str) -> FakeEncoding:

        return FakeEncoding()

    module.Role = FakeRole
    module.ReasoningEffort = FakeReasoningEffort
    module.HarmonyEncodingName = FakeHarmonyEncodingName
    module.TextContent = FakeTextContent
    module.Author = FakeAuthor
    module.Message = FakeMessage
    module.Conversation = FakeConversation
    module.SystemContent = FakeSystemContent
    module.ToolNamespaceConfig = FakeToolNamespaceConfig
    module.load_harmony_encoding = load_harmony_encoding
    module.SimpleNamespace = SimpleNamespace
    monkeypatch.setitem(sys.modules, "openai_harmony", module)

    return module


class FakeSandbox:

    def __init__(self) -> None:

        self.codes: list[str] = []
        self.reset_count = 0
        self.close_count = 0

    def execute(self, code: str) -> AIMOSandboxResult:

        self.codes.append(code)

        return AIMOSandboxResult(
            success=True,
            output="4",
            error="",
            timed_out=False,
        )

    def reset(self) -> None:

        self.reset_count += 1

    def close(self) -> None:

        self.close_count += 1
