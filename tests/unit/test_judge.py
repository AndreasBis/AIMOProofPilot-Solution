from __future__ import annotations

from collections.abc import Sequence

import pytest

from aimo_inference.client import AIMOGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.judge import AIMOProofJudge
from aimo_inference.judge import extract_boxed_grade
from aimo_inference.template import AIMOChatMessage
from conftest import fake_generation


class JudgeClient:

    def __init__(self, response: str) -> None:

        self.response = response
        self.calls: list[list[AIMOChatMessage]] = []

    def generate(
        self,
        messages: Sequence[AIMOChatMessage],
        max_tokens: int,
    ) -> AIMOGeneration:

        self.calls.append(list(messages))

        return fake_generation(self.response)


@pytest.mark.parametrize("grade", [0, 1, 6, 7])
def test_valid_boxed_grades(grade: int) -> None:

    assert extract_boxed_grade(f"Reasoning. \\boxed{{{grade}}}") == grade


def test_boxed_grade_extraction_returns_last_valid_grade() -> None:

    assert extract_boxed_grade("First \\boxed{1}, then final \\boxed{7}.") == 7


def test_missing_or_invalid_boxed_grade_becomes_zero() -> None:

    assert extract_boxed_grade("No final grade.") == 0
    assert extract_boxed_grade("Invalid \\boxed{5}") == 0
    assert extract_boxed_grade("Negative \\boxed{-1}") == 0


def test_optional_judge_prose_does_not_break_extraction() -> None:

    assert extract_boxed_grade("The proof is almost complete.\nFinal grade: \\boxed{6}") == 6


def test_reward_scalar_equals_grade_plus_context_and_page_reward() -> None:

    proof = "word " * 2200
    judge = AIMOProofJudge(
        config=AIMOConfig(
            page_count_method="word_count",
        ),
        client=JudgeClient("Complete proof. \\boxed{6}"),
    )

    result = judge.grade(
        problem="Problem.",
        proof=proof,
        reference="Reference.",
    )

    assert result.grade == 6
    assert result.context_reward == 1
    assert result.solution_page_reward == 1
    assert result.reward == 8
    assert result.page_count_metadata["page_count_method"] == "word_count"


def test_no_final_solution_output_gets_negative_context_reward() -> None:

    judge = AIMOProofJudge(
        config=AIMOConfig(
            page_count_method="word_count",
        ),
        client=JudgeClient("No grade."),
    )

    result = judge.grade(
        problem="Problem.",
        proof="No proof was produced.",
        reference="Reference.",
    )

    assert result.grade == 0
    assert result.context_reward == -1
    assert result.solution_page_reward == -1
    assert result.reward == -2
