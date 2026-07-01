from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class AIMOTrainingRecord:

    order_index: int
    id: str
    problem: str
    reference_solution: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:

        return asdict(self)


@dataclass(frozen=True)
class AIMORewardBreakdown:

    judge_grade: int
    context_reward: int
    solution_page_reward: int
    scalar_reward: float
    rendered_page_count: int
    page_count_method: str
    latex_compile_status: str
    page_count_fallback_reason: str
    judge_response: str
    judge_parse_failed: bool
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str
    latency_seconds: float
    tool_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:

        return asdict(self)


@dataclass(frozen=True)
class AIMORolloutSample:

    problem_id: str
    group_index: int
    rollout_index: int
    prompt: str
    completion: str
    token_ids: list[int]
    token_logprobs: list[float]
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str
    python_calls: int
    python_errors: int
    tool_call_count: int
    tool_error_count: int
    reward: AIMORewardBreakdown
    prompt_ids: list[int] = field(default_factory=list)
    env_mask: list[int] = field(default_factory=list)
    endpoint_index: int | None = None
    tool_tokens: int = 0
    sampling_logprobs: list[float] = field(default_factory=list)
    policy_update_index: int = 0
    policy_adapter_hash: str = ""
    policy_adapter_path: str = ""

    def as_dict(self) -> dict[str, Any]:

        payload = asdict(self)
        payload["reward"] = self.reward.as_dict()

        return payload


@dataclass(frozen=True)
class AIMOGRPOGroup:

    group_index: int
    problem_id: str
    problem: str
    reference_solution: str
    samples: list[AIMORolloutSample]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:

        return {
            "group_index": self.group_index,
            "problem_id": self.problem_id,
            "problem": self.problem,
            "reference_solution": self.reference_solution,
            "samples": [
                sample.as_dict()
                for sample in self.samples
            ],
            "metadata": self.metadata,
        }

    @property
    def rewards(self) -> list[float]:

        return [
            sample.reward.scalar_reward
            for sample in self.samples
        ]

    @property
    def is_complete(self) -> bool:

        return bool(self.samples)
