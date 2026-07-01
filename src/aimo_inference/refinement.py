from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Protocol
from typing import Sequence

from aimo_inference.client import AIMOGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.harmony import AIMOHarmonyToolLoop
from aimo_inference.io import AIMOProblemRecord
from aimo_inference.io import AIMOProblemResult
from aimo_inference.judge import AIMOProofJudge
from aimo_inference.page_count import AIMOPageCounter
from aimo_inference.page_count import strip_code_blocks_and_tool_transcripts
from aimo_inference.prompts import AIMOPromptBuilder
from aimo_inference.sandbox import AIMOSandbox
from aimo_inference.template import AIMOChatMessage
from aimo_inference.tools import AIMOToolExecution
from aimo_inference.tools import AIMOToolExecutionSummary


class AIMOGenerationClient(Protocol):

    def generate(
        self,
        messages: Sequence[AIMOChatMessage],
        max_tokens: int,
    ) -> AIMOGeneration:

        ...


@dataclass(frozen=True)
class AIMORefinementPass:

    pass_index: int
    name: str
    text: str
    tool_output: str
    success: bool
    error: str
    finish_reason: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    python_calls: int
    python_errors: int
    timeout_events: int
    tool_tokens: int
    prompt_hash: str
    model_profile: str
    entropy: float | None
    tool_executions: list[dict[str, str | bool]]

    def as_metadata(
        self,
    ) -> dict[str, str | int | float | bool | None | list[dict[str, str | bool]]]:

        return {
            "pass_index": self.pass_index,
            "name": self.name,
            "text": self.text,
            "tool_output": self.tool_output,
            "success": self.success,
            "error": self.error,
            "finish_reason": self.finish_reason,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "python_calls": self.python_calls,
            "python_errors": self.python_errors,
            "timeout_events": self.timeout_events,
            "tool_tokens": self.tool_tokens,
            "prompt_hash": self.prompt_hash,
            "model_profile": self.model_profile,
            "entropy": self.entropy,
            "tool_executions": self.tool_executions,
        }


class AIMORefinementEngine:

    python_block_pattern = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

    def __init__(
        self,
        config: AIMOConfig,
        client: AIMOGenerationClient,
        prompt_builder: AIMOPromptBuilder | None = None,
        sandbox: AIMOSandbox | None = None,
        judge: AIMOProofJudge | None = None,
    ) -> None:

        self.config = config
        self.client = client
        self.prompt_builder = prompt_builder or AIMOPromptBuilder()
        self.sandbox = sandbox or AIMOSandbox(config=config)
        self.judge = judge
        self.page_counter = AIMOPageCounter(config=config)

    def run_problem(self, record: AIMOProblemRecord) -> AIMOProblemResult:

        return self.run_problem_with_sandbox(
            record=record,
            sandbox=self.sandbox,
        )

    def run_problem_with_sandbox(
        self,
        record: AIMOProblemRecord,
        sandbox: AIMOSandbox,
    ) -> AIMOProblemResult:

        passes: list[AIMORefinementPass] = []
        first_pass_messages = self.prompt_builder.build_first_pass_messages(
            problem_text=record.problem,
            enable_tools=self.config.enable_tools,
        )
        first_pass = self._run_pass(
            pass_index=1,
            name="solve",
            messages=first_pass_messages,
            max_tokens=self.config.max_tokens_for_pass(self.config.first_pass_max_tokens),
            sandbox=sandbox,
        )
        passes.append(first_pass)

        if self.config.sequential_refinement_enabled:
            second_pass_messages = self.prompt_builder.build_second_pass_messages(
                problem_text=record.problem,
                first_solution=first_pass.text,
                first_tool_output=first_pass.tool_output,
                enable_tools=self.config.enable_tools,
            )
            second_pass = self._run_pass(
                pass_index=2,
                name="audit_repair",
                messages=second_pass_messages,
                max_tokens=self.config.max_tokens_for_pass(self.config.second_pass_max_tokens),
                sandbox=sandbox,
            )
            passes.append(second_pass)
            third_pass_messages = self.prompt_builder.build_third_pass_messages(
                problem_text=record.problem,
                first_solution=first_pass.text,
                repaired_solution=second_pass.text,
                second_tool_output=second_pass.tool_output,
                enable_tools=self.config.enable_tools,
            )
            third_pass = self._run_pass(
                pass_index=3,
                name="finalize",
                messages=third_pass_messages,
                max_tokens=self.config.max_tokens_for_pass(self.config.third_pass_max_tokens),
                sandbox=sandbox,
            )
            passes.append(third_pass)

        prediction = self._clean_prediction(self._best_available_prediction(passes))
        success = bool(prediction.strip()) and any(
            refinement_pass.success
            for refinement_pass in passes
        )
        error = self._combined_error(passes)
        page_count_result = self.page_counter.count(prediction)
        judge_metadata = self._judge_metadata(
            record=record,
            prediction=prediction,
        )

        return AIMOProblemResult(
            order_index=record.order_index,
            id=record.id,
            prediction=prediction,
            success=success,
            error=error,
            metadata={
                "passes": [
                    refinement_pass.as_metadata()
                    for refinement_pass in passes
                ],
                "page_count": page_count_result.as_metadata(),
                "judge": judge_metadata,
                "record_metadata": record.metadata,
                "sequential_refinement_enabled": self.config.sequential_refinement_enabled,
            },
        )

    def _run_pass(
        self,
        pass_index: int,
        name: str,
        messages: list[AIMOChatMessage],
        max_tokens: int,
        sandbox: AIMOSandbox,
    ) -> AIMORefinementPass:

        started_at = time.monotonic()
        prompt_hash = self._prompt_hash(messages)

        try:
            if self.config.template_format == "harmony":
                harmony_result = AIMOHarmonyToolLoop(
                    config=self.config,
                    client=self.client,
                    sandbox=sandbox,
                ).run(
                    messages=messages,
                    max_tokens=max_tokens,
                )

                return AIMORefinementPass(
                    pass_index=pass_index,
                    name=name,
                    text=harmony_result.text.strip(),
                    tool_output="",
                    success=True,
                    error="",
                    finish_reason=harmony_result.finish_reason,
                    latency_seconds=harmony_result.latency_seconds,
                    input_tokens=harmony_result.input_tokens,
                    output_tokens=harmony_result.output_tokens,
                    python_calls=harmony_result.python_calls,
                    python_errors=harmony_result.python_errors,
                    timeout_events=harmony_result.timeout_events,
                    tool_tokens=harmony_result.tool_tokens,
                    prompt_hash=prompt_hash,
                    model_profile=self.config.model_profile,
                    entropy=harmony_result.entropy,
                    tool_executions=[],
                )

            generation = self.client.generate(
                messages=messages,
                max_tokens=max_tokens,
            )
            tool_summary = self._run_python_blocks(
                text=generation.text,
                sandbox=sandbox,
            )

            return AIMORefinementPass(
                pass_index=pass_index,
                name=name,
                text=generation.text.strip(),
                tool_output=tool_summary.payload,
                success=True,
                error="",
                finish_reason=generation.finish_reason,
                latency_seconds=generation.latency_seconds,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                python_calls=tool_summary.python_calls,
                python_errors=tool_summary.python_errors,
                timeout_events=tool_summary.timeout_events,
                tool_tokens=0,
                prompt_hash=prompt_hash,
                model_profile=self.config.model_profile,
                entropy=None,
                tool_executions=[
                    execution.as_metadata()
                    for execution in tool_summary.executions
                ],
            )
        except Exception as error:
            return AIMORefinementPass(
                pass_index=pass_index,
                name=name,
                text="",
                tool_output="",
                success=False,
                error=str(error),
                finish_reason="error",
                latency_seconds=time.monotonic() - started_at,
                input_tokens=None,
                output_tokens=None,
                python_calls=0,
                python_errors=0,
                timeout_events=0,
                tool_tokens=0,
                prompt_hash=prompt_hash,
                model_profile=self.config.model_profile,
                entropy=None,
                tool_executions=[],
            )

    def _run_python_blocks(self, text: str, sandbox: AIMOSandbox) -> AIMOToolExecutionSummary:

        if not self.config.enable_tools:
            return AIMOToolExecutionSummary(executions=[])

        code_blocks = [
            match.group(1).strip()
            for match in self.python_block_pattern.finditer(text)
            if match.group(1).strip()
        ]

        if not code_blocks:
            return AIMOToolExecutionSummary(executions=[])

        executions: list[AIMOToolExecution] = []

        for code_block in code_blocks[:self.config.max_python_calls]:
            result = sandbox.execute(code_block)
            executions.append(
                AIMOToolExecution(
                    code=code_block,
                    output=result.to_tool_payload(),
                    success=result.success,
                    timed_out=result.timed_out,
                )
            )

        return AIMOToolExecutionSummary(executions=executions)

    def _best_available_prediction(self, passes: list[AIMORefinementPass]) -> str:

        for refinement_pass in reversed(passes):
            if refinement_pass.text.strip():
                return refinement_pass.text.strip()

        return "No proof was produced."

    def _combined_error(self, passes: list[AIMORefinementPass]) -> str:

        errors = [
            f"{refinement_pass.name}: {refinement_pass.error}"
            for refinement_pass in passes
            if refinement_pass.error
        ]

        return "\n".join(errors)

    def _clean_prediction(self, prediction: str) -> str:

        return strip_code_blocks_and_tool_transcripts(
            prediction,
            preserve_blank_lines=True,
        )

    def _prompt_hash(self, messages: Sequence[AIMOChatMessage]) -> str:

        payload = "\n".join(
            f"{message.role}:{message.content}"
            for message in messages
        )

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _judge_metadata(
        self,
        record: AIMOProblemRecord,
        prediction: str,
    ) -> dict[str, object]:

        if self.judge is None:
            return {}

        reference = (
            record.metadata.get("reference_solution")
            or record.metadata.get("solution")
            or ""
        )
        judge_result = self.judge.grade(
            problem=record.problem,
            proof=prediction,
            reference=reference,
        )

        return judge_result.as_metadata()
