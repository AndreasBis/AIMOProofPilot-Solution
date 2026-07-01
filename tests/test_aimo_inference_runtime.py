from __future__ import annotations

from typing import Sequence

from aimo_inference.client import AIMOGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.config import MAX_GENERATION_CONTEXT_TOKENS
from aimo_inference.io import AIMOProblemRecord
from aimo_inference.prompts import TOOL_PROMPT
from aimo_inference.refinement import AIMORefinementEngine
from aimo_inference.template import AIMOChatMessage


class RecordingClient:

    def __init__(self) -> None:

        self.calls: list[tuple[list[AIMOChatMessage], int]] = []

    def generate(
        self,
        messages: Sequence[AIMOChatMessage],
        max_tokens: int,
    ) -> AIMOGeneration:

        stored_messages = list(messages)
        self.calls.append((stored_messages, max_tokens))

        return AIMOGeneration(
            text=f"Proof {len(self.calls)}",
            input_tokens=100,
            output_tokens=10,
            finish_reason="stop",
            latency_seconds=0.0,
            raw={},
        )


def test_profile_sampling_payloads_use_only_requested_sampling_parameters() -> None:

    contestant_config = AIMOConfig().with_profile_defaults("contestant")
    judge_config = AIMOConfig().with_profile_defaults("judge")

    assert contestant_config.sampling_payload(max_tokens=123) == {
        "max_tokens": 123,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    assert judge_config.sampling_payload(max_tokens=123) == {
        "max_tokens": 123,
        "temperature": 1.0,
        "min_p": 0.02,
    }


def test_generation_budget_is_capped_at_65536_minus_input_tokens() -> None:

    config = AIMOConfig(max_model_len=81920)

    assert config.max_tokens_for_pass(configured_max_tokens=0) == MAX_GENERATION_CONTEXT_TOKENS
    assert config.max_tokens_for_pass(configured_max_tokens=70000) == MAX_GENERATION_CONTEXT_TOKENS
    assert config.available_generation_tokens(input_tokens=4096) == 61440


def test_tool_prompt_is_shared_and_lists_sandbox_libraries() -> None:

    for library_name in [
        "math",
        "statistics",
        "random",
        "collections",
        "itertools",
        "functools",
        "fractions",
        "decimal",
        "sympy",
        "numpy",
        "mpmath",
        "networkx",
        "z3",
    ]:
        assert library_name in TOOL_PROMPT


def test_refinement_runs_once_outside_kaggle_mode() -> None:

    client = RecordingClient()
    config = AIMOConfig(
        mode="colab",
        inference_mode="proof",
        enable_tools=False,
        page_count_method="word_count",
    )
    engine = AIMORefinementEngine(
        config=config,
        client=client,
    )
    record = AIMOProblemRecord(
        order_index=0,
        id="problem-1",
        problem="Find x.",
        metadata={},
    )

    result = engine.run_problem(record)

    assert result.prediction == "Proof 1"
    assert len(result.metadata["passes"]) == 1
    assert result.metadata["sequential_refinement_enabled"] is False
    assert len(client.calls) == 1


def test_refinement_records_kaggle_sequential_metadata() -> None:

    client = RecordingClient()
    config = AIMOConfig(
        mode="kaggle",
        inference_mode="proof",
        enable_tools=False,
        enable_judge=False,
        tensor_parallel_size=1,
        num_gpus=1,
        page_count_method="word_count",
    )
    engine = AIMORefinementEngine(
        config=config,
        client=client,
    )
    record = AIMOProblemRecord(
        order_index=0,
        id="problem-1",
        problem="Find x.",
        metadata={},
    )

    result = engine.run_problem(record)
    pass_metadata = result.metadata["passes"]

    assert result.prediction == "Proof 3"
    assert len(pass_metadata) == 3
    assert result.metadata["sequential_refinement_enabled"] is True
    assert [
        metadata["name"]
        for metadata in pass_metadata
    ] == [
        "solve",
        "audit_repair",
        "finalize",
    ]
    assert len(client.calls) == 3
