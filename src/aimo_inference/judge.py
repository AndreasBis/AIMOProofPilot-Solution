from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol
from typing import Sequence

from aimo_inference.client import AIMOGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.io import AIMOProblemRecord
from aimo_inference.io import AIMOProblemResult
from aimo_inference.page_count import AIMOPageCounter
from aimo_inference.prompts import AIMOJudgePromptBuilder
from aimo_inference.template import AIMOChatMessage


VALID_PROOF_GRADES = {
    0,
    1,
    6,
    7,
}

BOXED_GRADE_PATTERN = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}")


class AIMOJudgeGenerationClient(Protocol):

    def generate(
        self,
        messages: Sequence[AIMOChatMessage],
        max_tokens: int,
    ) -> AIMOGeneration:

        ...


@dataclass(frozen=True)
class AIMOJudgeResult:

    grade: int
    response: str
    reward: int
    context_reward: int
    solution_page_reward: int
    page_count_metadata: dict[str, str | int]
    finish_reason: str
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float

    def as_metadata(
        self,
    ) -> dict[str, str | int | float | None | dict[str, str | int]]:

        return {
            "grade": self.grade,
            "response": self.response,
            "reward": self.reward,
            "context_reward": self.context_reward,
            "solution_page_reward": self.solution_page_reward,
            "page_count": self.page_count_metadata,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_seconds": self.latency_seconds,
        }


class AIMOProofJudge:

    def __init__(
        self,
        config: AIMOConfig,
        client: AIMOJudgeGenerationClient,
        prompt_builder: AIMOJudgePromptBuilder | None = None,
    ) -> None:

        self.config = config
        self.client = client
        self.page_counter = AIMOPageCounter(config=config)
        self.prompt_builder = prompt_builder or AIMOJudgePromptBuilder()

    def grade(
        self,
        problem: str,
        proof: str,
        reference: str = "",
    ) -> AIMOJudgeResult:

        generation = self.client.generate(
            messages=self.prompt_builder.build_messages(
                problem=problem,
                proof=proof,
                reference=reference,
                enable_tools=self.config.enable_tools,
            ),
            max_tokens=self.config.judge_max_tokens,
        )
        grade = extract_boxed_grade(generation.text)
        page_count_result = self.page_counter.count(proof)
        context_reward = (
            1
            if proof.strip() and proof.strip() != "No proof was produced."
            else -1
        )
        reward = grade + context_reward + page_count_result.reward

        return AIMOJudgeResult(
            grade=grade,
            response=generation.text,
            reward=reward,
            context_reward=context_reward,
            solution_page_reward=page_count_result.reward,
            page_count_metadata=page_count_result.as_metadata(),
            finish_reason=generation.finish_reason,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            latency_seconds=generation.latency_seconds,
        )


class AIMOJudgeEngine:

    def __init__(
        self,
        config: AIMOConfig,
        judge: AIMOProofJudge,
    ) -> None:

        self.config = config
        self.judge = judge

    def run_problem(self, record: AIMOProblemRecord) -> AIMOProblemResult:

        proof = self._proof_from_record(record)
        reference = self._reference_from_record(record)
        judge_result = self.judge.grade(
            problem=record.problem,
            proof=proof,
            reference=reference,
        )

        return AIMOProblemResult(
            order_index=record.order_index,
            id=record.id,
            prediction=str(judge_result.grade),
            success=True,
            error="",
            metadata={
                "judge": judge_result.as_metadata(),
                "record_metadata": record.metadata,
            },
        )

    def _proof_from_record(self, record: AIMOProblemRecord) -> str:

        for key in ["proof", "prediction", "generated_proof", "solution"]:
            value = record.metadata.get(key)

            if value:
                return value

        return record.problem

    def _reference_from_record(self, record: AIMOProblemRecord) -> str:

        for key in ["reference", "reference_solution", "rubric", "target_solution"]:
            value = record.metadata.get(key)

            if value:
                return value

        return ""


def extract_boxed_grade(text: str) -> int:

    matches = BOXED_GRADE_PATTERN.findall(text)

    if not matches:
        return 0

    try:
        grade = int(matches[-1])
    except ValueError:
        return 0

    if grade not in VALID_PROOF_GRADES:
        return 0

    return grade
