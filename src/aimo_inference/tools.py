from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol

from aimo_inference.sandbox import AIMOSandboxResult


class AIMOPythonSandbox(Protocol):

    def execute(self, code: str) -> AIMOSandboxResult:

        ...

    def reset(self) -> None:

        ...


@dataclass(frozen=True)
class AIMOToolExecution:

    code: str
    output: str
    success: bool
    timed_out: bool

    def as_metadata(self) -> dict[str, str | bool]:

        return {
            "code": self.code,
            "output": self.output,
            "success": self.success,
            "timed_out": self.timed_out,
        }


@dataclass(frozen=True)
class AIMOToolExecutionSummary:

    executions: list[AIMOToolExecution]

    @property
    def python_calls(self) -> int:

        return len(self.executions)

    @property
    def python_errors(self) -> int:

        return sum(
            1
            for execution in self.executions
            if not execution.success
        )

    @property
    def timeout_events(self) -> int:

        return sum(
            1
            for execution in self.executions
            if execution.timed_out
        )

    @property
    def payload(self) -> str:

        return "\n\n".join(
            execution.output
            for execution in self.executions
        )

    def as_metadata(self) -> dict[str, Any]:

        return {
            "python_calls": self.python_calls,
            "python_errors": self.python_errors,
            "timeout_events": self.timeout_events,
            "executions": [
                execution.as_metadata()
                for execution in self.executions
            ],
        }


class AIMOPythonTool:

    def __init__(
        self,
        sandbox: AIMOPythonSandbox,
        tool_prompt: str,
    ) -> None:

        self.sandbox = sandbox
        self.tool_prompt = tool_prompt

    def execute(self, code: str) -> AIMOToolExecution:

        result = self.sandbox.execute(code)

        return AIMOToolExecution(
            code=code,
            output=result.to_tool_payload(),
            success=result.success,
            timed_out=result.timed_out,
        )

    def harmony_tool_config(self) -> object:

        try:
            from openai_harmony import ToolNamespaceConfig

            return ToolNamespaceConfig(
                name="python",
                description=self.tool_prompt,
                tools=[],
            )
        except Exception as error:
            raise RuntimeError(
                "openai_harmony is required for Harmony tool configuration."
            ) from error

    def harmony_tool_response(self, message: object) -> object:

        try:
            from openai_harmony import Author
            from openai_harmony import Message
            from openai_harmony import Role
            from openai_harmony import TextContent
        except Exception as error:
            raise RuntimeError(
                "openai_harmony is required for Harmony tool responses."
            ) from error

        code = self._message_text(message)
        execution = self.execute(code)
        channel = getattr(message, "channel", None)
        response_message = Message(
            author=Author(
                role=Role.TOOL,
                name="python",
            ),
            content=[
                TextContent(text=execution.output),
            ],
        ).with_recipient("assistant")

        if channel:
            response_message = response_message.with_channel(channel)

        return response_message

    def _message_text(self, message: object) -> str:

        content = getattr(message, "content", [])

        if not content:
            return ""

        first_content = content[0]

        return str(getattr(first_content, "text", ""))
