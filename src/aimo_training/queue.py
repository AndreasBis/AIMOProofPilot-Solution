from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimo_training.schema import AIMOGRPOGroup
from aimo_training.schema import AIMORolloutSample
from aimo_training.schema import AIMORewardBreakdown
from aimo_training.schema import AIMOTrainingRecord


@dataclass
class AIMOPartialGroup:

    record: AIMOTrainingRecord
    group_index: int
    samples: list[AIMORolloutSample]

    def append(self, sample: AIMORolloutSample) -> None:

        self.samples.append(sample)

    def is_complete(self, group_size: int) -> bool:

        return len(self.samples) >= group_size

    def to_group(self) -> AIMOGRPOGroup:

        return AIMOGRPOGroup(
            group_index=self.group_index,
            problem_id=self.record.id,
            problem=self.record.problem,
            reference_solution=self.record.reference_solution,
            samples=list(self.samples),
            metadata=self.record.metadata,
        )


class AIMOInterleavedGroupBuilder:

    def __init__(
        self,
        records: list[AIMOTrainingRecord],
        group_size: int,
        active_problem_count: int,
    ) -> None:

        self.pending_records = iter(records)
        self.group_size = group_size
        self.active_problem_count = active_problem_count
        self.next_group_index = 0
        self.active_groups: dict[str, AIMOPartialGroup] = {}
        self._admit_initial_records()

    def add_sample(self, sample: AIMORolloutSample) -> AIMOGRPOGroup | None:

        partial_group = self.active_groups.get(sample.problem_id)

        if partial_group is None:
            raise KeyError(f"Rollout sample references inactive problem: {sample.problem_id}")

        partial_group.append(sample)

        if not partial_group.is_complete(self.group_size):
            return None

        completed_group = partial_group.to_group()
        del self.active_groups[sample.problem_id]
        self._admit_one_record()

        return completed_group

    def active_problem_ids(self) -> list[str]:

        return list(self.active_groups)

    def _admit_initial_records(self) -> None:

        for _ in range(self.active_problem_count):
            self._admit_one_record()

    def _admit_one_record(self) -> None:

        try:
            record = next(self.pending_records)
        except StopIteration:
            return

        self.active_groups[record.id] = AIMOPartialGroup(
            record=record,
            group_index=self.next_group_index,
            samples=[],
        )
        self.next_group_index += 1


class AIMODurableGroupQueue:

    def __init__(self, path: Path) -> None:

        self.path = path

    def append_group(self, group: AIMOGRPOGroup) -> None:

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        existing_payload = ""

        if self.path.exists():
            existing_payload = self.path.read_text(encoding="utf-8")

        with temporary_path.open("w", encoding="utf-8") as output_file:
            output_file.write(existing_payload)
            output_file.write(json.dumps(group.as_dict(), ensure_ascii=False))
            output_file.write("\n")

        os.replace(temporary_path, self.path)

    def read_groups(self) -> list[AIMOGRPOGroup]:

        if not self.path.exists():
            return []

        groups: list[AIMOGRPOGroup] = []

        with self.path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                groups.append(group_from_dict(json.loads(stripped_line)))

        return groups


def group_from_dict(payload: dict[str, Any]) -> AIMOGRPOGroup:

    samples = [
        sample_from_dict(sample_payload)
        for sample_payload in payload.get("samples", [])
    ]

    return AIMOGRPOGroup(
        group_index=int(payload.get("group_index", 0)),
        problem_id=str(payload.get("problem_id", "")),
        problem=str(payload.get("problem", "")),
        reference_solution=str(payload.get("reference_solution", "")),
        samples=samples,
        metadata=dict(payload.get("metadata", {})),
    )


def sample_from_dict(payload: dict[str, Any]) -> AIMORolloutSample:

    reward_payload = dict(payload.get("reward", {}))
    reward = AIMORewardBreakdown(
        judge_grade=int(reward_payload.get("judge_grade", 0)),
        context_reward=int(reward_payload.get("context_reward", -1)),
        solution_page_reward=int(reward_payload.get("solution_page_reward", -1)),
        scalar_reward=float(reward_payload.get("scalar_reward", -2.0)),
        rendered_page_count=int(reward_payload.get("rendered_page_count", 0)),
        page_count_method=str(reward_payload.get("page_count_method", "")),
        latex_compile_status=str(reward_payload.get("latex_compile_status", "")),
        page_count_fallback_reason=str(reward_payload.get("page_count_fallback_reason", "")),
        judge_response=str(reward_payload.get("judge_response", "")),
        judge_parse_failed=bool(reward_payload.get("judge_parse_failed", False)),
        input_tokens=optional_int(reward_payload.get("input_tokens")),
        output_tokens=optional_int(reward_payload.get("output_tokens")),
        finish_reason=str(reward_payload.get("finish_reason", "")),
        latency_seconds=float(reward_payload.get("latency_seconds", 0.0)),
        tool_tokens=int(reward_payload.get("tool_tokens", 0)),
    )

    return AIMORolloutSample(
        problem_id=str(payload.get("problem_id", "")),
        group_index=int(payload.get("group_index", 0)),
        rollout_index=int(payload.get("rollout_index", 0)),
        prompt=str(payload.get("prompt", "")),
        completion=str(payload.get("completion", "")),
        token_ids=[
            int(token_id)
            for token_id in payload.get("token_ids", [])
        ],
        token_logprobs=[
            float(token_logprob)
            for token_logprob in payload.get("token_logprobs", [])
        ],
        input_tokens=optional_int(payload.get("input_tokens")),
        output_tokens=optional_int(payload.get("output_tokens")),
        finish_reason=str(payload.get("finish_reason", "")),
        python_calls=int(payload.get("python_calls", 0)),
        python_errors=int(payload.get("python_errors", 0)),
        tool_call_count=int(payload.get("tool_call_count", 0)),
        tool_error_count=int(payload.get("tool_error_count", 0)),
        reward=reward,
        prompt_ids=[
            int(token_id)
            for token_id in payload.get("prompt_ids", [])
        ],
        env_mask=[
            int(mask_value)
            for mask_value in payload.get("env_mask", [])
        ],
        endpoint_index=optional_int(payload.get("endpoint_index")),
        tool_tokens=int(payload.get("tool_tokens", 0)),
        sampling_logprobs=[
            float(token_logprob)
            for token_logprob in payload.get("sampling_logprobs", payload.get("token_logprobs", []))
        ],
        policy_update_index=int(payload.get("policy_update_index", 0)),
        policy_adapter_hash=str(payload.get("policy_adapter_hash", "")),
        policy_adapter_path=str(payload.get("policy_adapter_path", "")),
    )


def optional_int(value: object) -> int | None:

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None
