from __future__ import annotations

import re
from dataclasses import dataclass

from aimo_inference.config import AIMOConfig
from aimo_inference.harmony import AIMOCompletionClient
from aimo_inference.harmony import AIMOHarmonyToolLoop
from aimo_inference.io import AIMOProblemRecord
from aimo_inference.io import AIMOProblemResult
from aimo_inference.prompts import AIMOAnswerPromptBuilder
from aimo_inference.sandbox import AIMOJupyterSandbox
from aimo_inference.sandbox import AIMOSandbox


BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\s*\{\s*([0-9,_]+)\s*\}")


@dataclass(frozen=True)
class AIMOAnswerGeneration:

    answer: int | None
    text: str
    input_tokens: int
    output_tokens: int
    python_calls: int
    python_errors: int
    latency_seconds: float
    finish_reason: str

    def as_metadata(self) -> dict[str, str | int | float | None]:

        return {
            "answer": self.answer,
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "python_calls": self.python_calls,
            "python_errors": self.python_errors,
            "latency_seconds": self.latency_seconds,
            "finish_reason": self.finish_reason,
        }


class AIMOAnswerEngine:

    def __init__(
        self,
        config: AIMOConfig,
        client: AIMOCompletionClient,
        prompt_builder: AIMOAnswerPromptBuilder | None = None,
    ) -> None:

        self.config = config
        self.client = client
        self.prompt_builder = prompt_builder or AIMOAnswerPromptBuilder()

    def run_problem(self, record: AIMOProblemRecord) -> AIMOProblemResult:

        generation = self._run_generation(record=record)
        prediction = generation.answer if generation.answer is not None else 0

        return AIMOProblemResult(
            order_index=record.order_index,
            id=record.id,
            prediction=str(prediction),
            success=generation.answer is not None,
            error="" if generation.answer is not None else "No answer was produced.",
            metadata={
                "answer": generation.answer,
                "generation": generation.as_metadata(),
                "record_metadata": record.metadata,
            },
        )

    def _run_generation(self, record: AIMOProblemRecord) -> AIMOAnswerGeneration:

        sandbox = self._build_sandbox()
        loop = AIMOHarmonyToolLoop(
            config=self.config,
            client=self.client,
            sandbox=sandbox,
        )

        try:
            run_result = loop.run(
                messages=self.prompt_builder.build_messages(
                    problem=record.problem,
                    enable_tools=self.config.enable_tools,
                ),
                max_tokens=self.config.max_tokens_for_pass(self.config.max_new_tokens),
                seed=self.config.seed,
            )
            answer = extract_boxed_answer(run_result.text)

            return AIMOAnswerGeneration(
                answer=answer,
                text=run_result.text,
                input_tokens=run_result.input_tokens,
                output_tokens=run_result.output_tokens,
                python_calls=run_result.python_calls,
                python_errors=run_result.python_errors,
                latency_seconds=run_result.latency_seconds,
                finish_reason=run_result.finish_reason,
            )
        finally:
            sandbox.reset()
            sandbox.close()

    def _build_sandbox(self) -> AIMOSandbox:

        if self.config.use_jupyter_sandbox:
            return AIMOJupyterSandbox(config=self.config)

        return AIMOSandbox(config=self.config)


def extract_boxed_answer(text: str) -> int | None:

    matches = BOXED_ANSWER_PATTERN.findall(text)

    if not matches:
        return None

    clean_value = re.sub(r"[,_]", "", matches[-1])

    try:
        value = int(clean_value)
    except ValueError:
        return None

    if not 0 <= value <= 7:
        return None

    return value
