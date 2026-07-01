from __future__ import annotations

from pathlib import Path

import pytest

from aimo_training.config import AIMOTrainingConfig
from aimo_training.rollout import AIMORolloutCoordinator
from aimo_training.schema import AIMORolloutSample
from aimo_training.schema import AIMORewardBreakdown
from aimo_training.schema import AIMOTrainingRecord
from aimo_training.tool_rollout import AIMOToolRolloutResult
from conftest import reward_breakdown


class FakePromptBuilder:

    def build_first_pass_messages(
        self,
        problem_text: str,
        enable_tools: bool,
    ) -> list[dict[str, str]]:

        return [
            {
                "role": "user",
                "content": problem_text,
            },
        ]


class FakeChatTemplate:

    def render(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
    ) -> str:

        content = messages[0]["content"]

        return f"Prompt: {content}"


class FakeSandboxLease:

    def __enter__(self) -> object:

        return object()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:

        return None


class FakeSandboxPool:

    def __init__(self) -> None:

        self.closed = False

    def acquire(self) -> FakeSandboxLease:

        return FakeSandboxLease()

    def close(self) -> None:

        self.closed = True


class FakeRewardScorer:

    def __init__(self) -> None:

        self.generated_proofs: list[str] = []

    def score(
        self,
        problem: str,
        reference_solution: str,
        generated_proof: str,
        input_tokens: int | None,
        output_tokens: int | None,
        finish_reason: str,
        tool_tokens: int = 0,
    ) -> AIMORewardBreakdown:

        self.generated_proofs.append(generated_proof)

        return reward_breakdown()


class FakeToolRolloutEngine:

    seeds: list[int | None] = []

    def __init__(
        self,
        config: object,
        client: object,
        prompt_builder: object,
        sandbox: object,
    ) -> None:

        self.config = config
        self.client = client
        self.prompt_builder = prompt_builder
        self.sandbox = sandbox

    def run_problem(
        self,
        problem_text: str,
        seed: int | None = None,
    ) -> AIMOToolRolloutResult:

        self.seeds.append(seed)
        token_start = 100 + int(seed or 0)

        return AIMOToolRolloutResult(
            prompt="Rendered rollout prompt",
            prompt_ids=[
                1,
                2,
                3,
            ],
            completion="Full completion",
            completion_ids=[
                token_start,
                token_start + 1,
                token_start + 2,
            ],
            token_logprobs=[
                -0.3,
                -0.2,
                -0.1,
            ],
            env_mask=[
                1,
                0,
                1,
            ],
            proof_text=f"Proof for {problem_text}",
            input_tokens=3,
            output_tokens=3,
            tool_tokens=1,
            finish_reason="stop",
            python_calls=1,
            python_errors=0,
            timeout_events=0,
            latency_seconds=0.2,
            raw_generations=[],
        )


def training_config(
    tmp_path: Path,
    group_size: int = 2,
) -> AIMOTrainingConfig:

    return AIMOTrainingConfig(
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "dataset.jsonl",
        output_path=tmp_path / "output",
        logdir=tmp_path / "logs",
        group_size=group_size,
        seed=11,
        page_count_method="word_count",
    )


def training_record() -> AIMOTrainingRecord:

    return AIMOTrainingRecord(
        order_index=0,
        id="problem-1",
        problem="Show that 1 + 1 = 2.",
        reference_solution="Use Peano arithmetic.",
        metadata={
            "source": "unit",
        },
    )


def rollout_coordinator(
    tmp_path: Path,
    group_size: int = 2,
) -> AIMORolloutCoordinator:

    coordinator = AIMORolloutCoordinator.__new__(AIMORolloutCoordinator)
    coordinator.config = training_config(
        tmp_path=tmp_path,
        group_size=group_size,
    )
    coordinator.prompt_builder = FakePromptBuilder()
    coordinator.chat_template = FakeChatTemplate()
    coordinator.rollout_config = object()
    coordinator.rollout_client = object()
    coordinator.rollout_sandbox_pool = FakeSandboxPool()
    coordinator.judge_sandbox_pool = FakeSandboxPool()
    coordinator.reward_scorer = FakeRewardScorer()

    return coordinator


def test_build_sample_returns_valid_rollout_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        "aimo_training.rollout.AIMOToolRolloutEngine",
        FakeToolRolloutEngine,
    )
    coordinator = rollout_coordinator(tmp_path)

    sample = coordinator.build_sample(
        record=training_record(),
        group_index=4,
        rollout_index=2,
    )

    assert isinstance(sample, AIMORolloutSample)
    assert sample.problem_id == "problem-1"
    assert sample.group_index == 4
    assert sample.rollout_index == 2
    assert sample.prompt == "Rendered rollout prompt"
    assert sample.completion == "Proof for Show that 1 + 1 = 2."
    assert sample.token_ids == [
        113,
        114,
        115,
    ]
    assert sample.token_logprobs == [
        -0.3,
        -0.2,
        -0.1,
    ]
    assert sample.env_mask == [
        1,
        0,
        1,
    ]
    assert sample.sampling_logprobs == sample.token_logprobs
    assert len(sample.token_ids) == len(sample.token_logprobs) == len(sample.env_mask)


def test_build_group_returns_requested_valid_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        "aimo_training.rollout.AIMOToolRolloutEngine",
        FakeToolRolloutEngine,
    )
    coordinator = rollout_coordinator(
        tmp_path=tmp_path,
        group_size=3,
    )

    group = coordinator.build_group(
        record=training_record(),
        group_index=7,
    )

    assert len(group.samples) == 3
    assert all(isinstance(sample, AIMORolloutSample) for sample in group.samples)
    assert [
        sample.rollout_index
        for sample in group.samples
    ] == [
        0,
        1,
        2,
    ]
    assert all(
        len(sample.token_ids) == len(sample.token_logprobs) == len(sample.env_mask)
        for sample in group.samples
    )
