from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from aimo_training.schema import AIMOGRPOGroup
from aimo_training.schema import AIMORolloutSample


class AIMOTrainingArtifactWriter:

    def __init__(self, output_path: Path, logdir: Path) -> None:

        self.output_path = output_path
        self.logdir = logdir

    def ensure_directories(self) -> None:

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logdir.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> Path:

        path = self.logdir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")

        os.replace(temporary_path, path)

        return path

    def append_jsonl(self, relative_path: str, payload: dict[str, Any]) -> Path:

        path = self.logdir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload, ensure_ascii=False))
            output_file.write("\n")

        return path

    def write_group_artifacts(self, groups: list[AIMOGRPOGroup]) -> None:

        self.write_gradient_update_reward_artifacts(groups=groups)

        for group in groups:
            self.append_jsonl(
                "per_step_reward_summaries.jsonl",
                reward_summary_for_group(group),
            )

            for sample in group.samples:
                self.append_jsonl(
                    "training_table.jsonl",
                    training_table_row(group=group, sample=sample),
                )

                if sample.reward.judge_parse_failed:
                    self.append_jsonl(
                        "judge_parse_failures.jsonl",
                        judge_parse_failure_row(group=group, sample=sample),
                    )

            sample_proofs = group.samples[: min(2, len(group.samples))]

            for sample in sample_proofs:
                self.append_jsonl(
                    "sample_generated_proofs.jsonl",
                    sample_proof_row(group=group, sample=sample),
                )

    def write_checkpoint_hashes(self, paths: list[Path]) -> Path:

        hashes = {
            str(path): hash_file(path)
            for path in paths
            if path.exists()
        }

        return self.write_json("checkpoint_hashes.json", hashes)

    def write_gradient_update_reward_artifacts(self, groups: list[AIMOGRPOGroup]) -> None:

        self.append_jsonl(
            "gradient_update_reward_summaries.jsonl",
            gradient_update_reward_summary(groups=groups),
        )

        for group in groups:
            for sample in group.samples:
                self.append_jsonl(
                    "gradient_update_reward_samples.jsonl",
                    gradient_update_reward_sample_row(group=group, sample=sample),
                )


def reward_summary_for_group(group: AIMOGRPOGroup) -> dict[str, Any]:

    rewards = group.rewards
    reward_count = len(rewards)
    mean_reward = sum(rewards) / reward_count if reward_count else 0.0

    return {
        "group_index": group.group_index,
        "problem_id": group.problem_id,
        "reward_count": reward_count,
        "mean_reward": mean_reward,
        "mean_judge_grade": mean_value([
            sample.reward.judge_grade
            for sample in group.samples
        ]),
        "mean_context_reward": mean_value([
            sample.reward.context_reward
            for sample in group.samples
        ]),
        "mean_solution_page_reward": mean_value([
            sample.reward.solution_page_reward
            for sample in group.samples
        ]),
        "mean_penalty": mean_value([
            reward_penalty(sample)
            for sample in group.samples
        ]),
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "judge_grade_counts": count_values([
            sample.reward.judge_grade
            for sample in group.samples
        ]),
        "context_reward_counts": count_values([
            sample.reward.context_reward
            for sample in group.samples
        ]),
        "page_reward_counts": count_values([
            sample.reward.solution_page_reward
            for sample in group.samples
        ]),
    }


def training_table_row(group: AIMOGRPOGroup, sample: AIMORolloutSample) -> dict[str, Any]:

    return {
        "problem_id": group.problem_id,
        "group_index": group.group_index,
        "rollout_index": sample.rollout_index,
        "endpoint_index": sample.endpoint_index,
        "boxed_judge_grade": sample.reward.judge_grade,
        "judge_score": sample.reward.judge_grade,
        "context_reward": sample.reward.context_reward,
        "page_reward": sample.reward.solution_page_reward,
        "penalty": reward_penalty(sample),
        "scalar_reward": sample.reward.scalar_reward,
        "total_reward": sample.reward.scalar_reward,
        "input_tokens": sample.input_tokens,
        "output_tokens": sample.output_tokens,
        "tool_tokens": sample.tool_tokens,
        "completion_token_count": len(sample.token_ids),
        "token_logprob_count": len(sample.token_logprobs),
        "model_token_mask_count": sum(sample.env_mask) if sample.env_mask else len(sample.token_ids),
        "rendered_page_count": sample.reward.rendered_page_count,
        "page_count_method": sample.reward.page_count_method,
        "finish_reason": sample.finish_reason,
        "python_calls": sample.python_calls,
        "python_errors": sample.python_errors,
        "tool_call_count": sample.tool_call_count,
        "tool_error_count": sample.tool_error_count,
        "policy_update_index": sample.policy_update_index,
        "policy_adapter_hash": sample.policy_adapter_hash,
        "policy_adapter_path": sample.policy_adapter_path,
    }


def judge_parse_failure_row(group: AIMOGRPOGroup, sample: AIMORolloutSample) -> dict[str, Any]:

    return {
        "problem_id": group.problem_id,
        "group_index": group.group_index,
        "rollout_index": sample.rollout_index,
        "judge_response": sample.reward.judge_response,
    }


def sample_proof_row(group: AIMOGRPOGroup, sample: AIMORolloutSample) -> dict[str, Any]:

    return {
        "problem_id": group.problem_id,
        "group_index": group.group_index,
        "rollout_index": sample.rollout_index,
        "completion": sample.completion,
        "scalar_reward": sample.reward.scalar_reward,
        "boxed_judge_grade": sample.reward.judge_grade,
    }


def count_values(values: list[int]) -> dict[str, int]:

    counts: dict[str, int] = {}

    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1

    return counts


def gradient_update_reward_summary(groups: list[AIMOGRPOGroup]) -> dict[str, Any]:

    samples = [
        sample
        for group in groups
        for sample in group.samples
    ]
    rollout_policy_update_indices = sorted({
        sample.policy_update_index
        for sample in samples
    })
    rollout_adapter_hashes = sorted({
        sample.policy_adapter_hash
        for sample in samples
        if sample.policy_adapter_hash
    })

    return {
        "gradient_update_index": (
            max(rollout_policy_update_indices) + 1
            if rollout_policy_update_indices
            else 0
        ),
        "rollout_policy_update_indices": rollout_policy_update_indices,
        "rollout_adapter_hashes": rollout_adapter_hashes,
        "group_count": len(groups),
        "sample_count": len(samples),
        "mean_judge_score": mean_value([
            sample.reward.judge_grade
            for sample in samples
        ]),
        "mean_context_reward": mean_value([
            sample.reward.context_reward
            for sample in samples
        ]),
        "mean_solution_page_reward": mean_value([
            sample.reward.solution_page_reward
            for sample in samples
        ]),
        "mean_penalty": mean_value([
            reward_penalty(sample)
            for sample in samples
        ]),
        "mean_total_reward": mean_value([
            sample.reward.scalar_reward
            for sample in samples
        ]),
        "min_total_reward": min([
            sample.reward.scalar_reward
            for sample in samples
        ]) if samples else 0.0,
        "max_total_reward": max([
            sample.reward.scalar_reward
            for sample in samples
        ]) if samples else 0.0,
        "judge_parse_failure_count": sum(
            1
            for sample in samples
            if sample.reward.judge_parse_failed
        ),
        "tool_error_count": sum(
            sample.tool_error_count
            for sample in samples
        ),
        "python_error_count": sum(
            sample.python_errors
            for sample in samples
        ),
    }


def gradient_update_reward_sample_row(
    group: AIMOGRPOGroup,
    sample: AIMORolloutSample,
) -> dict[str, Any]:

    return {
        "gradient_update_index": sample.policy_update_index + 1,
        "problem_id": group.problem_id,
        "group_index": group.group_index,
        "rollout_index": sample.rollout_index,
        "judge_score": sample.reward.judge_grade,
        "context_reward": sample.reward.context_reward,
        "solution_page_reward": sample.reward.solution_page_reward,
        "penalty": reward_penalty(sample),
        "total_reward": sample.reward.scalar_reward,
        "judge_parse_failed": sample.reward.judge_parse_failed,
        "policy_update_index": sample.policy_update_index,
        "policy_adapter_hash": sample.policy_adapter_hash,
        "policy_adapter_path": sample.policy_adapter_path,
    }


def mean_value(values: list[int | float]) -> float:

    if not values:
        return 0.0

    return float(sum(values) / len(values))


def reward_penalty(sample: AIMORolloutSample) -> float:

    return float(
        min(0, sample.reward.context_reward)
        + min(0, sample.reward.solution_page_reward)
    )


def hash_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()
