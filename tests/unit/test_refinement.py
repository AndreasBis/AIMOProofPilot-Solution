from __future__ import annotations

from collections.abc import Sequence

from aimo_inference.client import AIMOGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.io import AIMOProblemRecord
from aimo_inference.refinement import AIMORefinementEngine
from aimo_inference.template import AIMOChatMessage
from conftest import FakeSandbox
from conftest import fake_generation


class SequenceClient:

    def __init__(self, responses: list[AIMOGeneration | Exception]) -> None:

        self.responses = list(responses)
        self.calls: list[tuple[list[AIMOChatMessage], int]] = []

    def generate(
        self,
        messages: Sequence[AIMOChatMessage],
        max_tokens: int,
    ) -> AIMOGeneration:

        self.calls.append((list(messages), max_tokens))
        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def problem_record() -> AIMOProblemRecord:

    return AIMOProblemRecord(
        order_index=0,
        id="p1",
        problem="Prove that 1=1.",
        metadata={
            "source": "fixture",
        },
    )


def kaggle_refinement_config() -> AIMOConfig:

    return AIMOConfig(
        mode="kaggle",
        inference_mode="proof",
        enable_tools=False,
        enable_judge=False,
        tensor_parallel_size=1,
        num_gpus=1,
        page_count_method="word_count",
    )


def test_three_passes_are_called_in_order() -> None:

    client = SequenceClient([
        fake_generation("First proof."),
        fake_generation("Second proof."),
        fake_generation("Final proof."),
    ])
    engine = AIMORefinementEngine(
        config=kaggle_refinement_config(),
        client=client,
        sandbox=FakeSandbox(),
    )

    result = engine.run_problem(problem_record())
    pass_names = [
        pass_metadata["name"]
        for pass_metadata in result.metadata["passes"]
    ]

    assert pass_names == [
        "solve",
        "audit_repair",
        "finalize",
    ]
    assert result.prediction == "Final proof."
    assert len(client.calls) == 3


def test_previous_solution_is_passed_into_repair_and_finalize_prompts() -> None:

    client = SequenceClient([
        fake_generation("First proof."),
        fake_generation("Second proof."),
        fake_generation("Final proof."),
    ])
    engine = AIMORefinementEngine(
        config=kaggle_refinement_config(),
        client=client,
        sandbox=FakeSandbox(),
    )

    engine.run_problem(problem_record())
    second_user_prompt = client.calls[1][0][1].content
    third_user_prompt = client.calls[2][0][1].content

    assert "First proof." in second_user_prompt
    assert "Second proof." in third_user_prompt


def test_python_blocks_are_executed_when_enabled() -> None:

    sandbox = FakeSandbox()
    client = SequenceClient([
        fake_generation("Before.\n```python\nprint(2 + 2)\n```\nAfter."),
    ])
    engine = AIMORefinementEngine(
        config=AIMOConfig(
            enable_tools=True,
            page_count_method="word_count",
        ),
        client=client,
        sandbox=sandbox,
    )

    result = engine.run_problem(problem_record())
    pass_metadata = result.metadata["passes"][0]

    assert sandbox.codes == [
        "print(2 + 2)",
    ]
    assert result.prediction == "Before.\n\nAfter."
    assert pass_metadata["python_calls"] == 1
    assert pass_metadata["python_errors"] == 0
    assert pass_metadata["tool_output"] == "4"


def test_final_prediction_falls_back_to_last_non_empty_pass() -> None:

    client = SequenceClient([
        fake_generation(""),
        fake_generation("Repair proof."),
        fake_generation(""),
    ])
    engine = AIMORefinementEngine(
        config=kaggle_refinement_config(),
        client=client,
        sandbox=FakeSandbox(),
    )

    result = engine.run_problem(problem_record())

    assert result.prediction == "Repair proof."


def test_pass_metadata_contains_tokens_latency_tool_output_and_errors() -> None:

    client = SequenceClient([
        RuntimeError("generation failed"),
    ])
    engine = AIMORefinementEngine(
        config=AIMOConfig(
            page_count_method="word_count",
        ),
        client=client,
        sandbox=FakeSandbox(),
    )

    result = engine.run_problem(problem_record())
    pass_metadata = result.metadata["passes"][0]

    assert result.success is False
    assert result.prediction == "No proof was produced."
    assert pass_metadata["finish_reason"] == "error"
    assert pass_metadata["input_tokens"] is None
    assert pass_metadata["output_tokens"] is None
    assert pass_metadata["latency_seconds"] >= 0.0
    assert pass_metadata["tool_output"] == ""
    assert "generation failed" in pass_metadata["error"]
