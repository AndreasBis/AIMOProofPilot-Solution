from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from aimo_inference.client import AIMOInferenceClient
from aimo_inference.config import AIMOConfig
from aimo_inference.prompts import AIMOPromptBuilder
from aimo_inference.sandbox import AIMOSandboxPool
from aimo_inference.sandbox import run_sandbox_pool_preflight
from aimo_inference.server import AIMOInferenceServer
from aimo_inference.template import AIMOChatTemplate
from aimo_training.config import AIMOTrainingConfig
from aimo_training.data import read_training_records
from aimo_training.data import validate_training_dataset
from aimo_training.entrypoints.train import run as run_training_step
from aimo_training.queue import AIMODurableGroupQueue
from aimo_training.queue import AIMOPartialGroup
from aimo_training.rewards import AIMORewardConfig
from aimo_training.rewards import AIMOTrainingRewardScorer
from aimo_training.rollout import build_judge_inference_config
from aimo_training.rollout import build_rollout_inference_config
from aimo_training.schema import AIMOGRPOGroup
from aimo_training.schema import AIMORolloutSample
from aimo_training.schema import AIMORewardBreakdown
from aimo_training.schema import AIMOTrainingRecord
from aimo_training.tool_rollout import AIMOToolRolloutEngine


@dataclass(frozen=True)
class AIMORolloutEndpoint:

    endpoint_index: int
    api_base: str
    config: AIMOConfig
    client: AIMOInferenceClient


@dataclass(frozen=True)
class AIMORunningRollout:

    endpoint_index: int
    problem_key: str
    record: AIMOTrainingRecord
    group_index: int
    rollout_index: int
    attempt_index: int = 0


@dataclass(frozen=True)
class AIMORolloutAttemptResult:

    sample: AIMORolloutSample | None
    failure_metadata: dict[str, object]


@dataclass(frozen=True)
class AIMOOnlineRolloutSummary:

    admitted_problem_count: int
    completed_group_count: int
    written_sample_count: int
    skipped_group_count: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, int | float]:

        return {
            "admitted_problem_count": self.admitted_problem_count,
            "completed_group_count": self.completed_group_count,
            "written_sample_count": self.written_sample_count,
            "skipped_group_count": self.skipped_group_count,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class AIMOOnlineUpdateSummary:

    chunk_index: int
    start_order_index: int
    problem_count: int
    rollout_adapter_state: AIMOAdapterState
    trained_adapter_state: AIMOAdapterState
    queue_path: Path
    adapter_path: Path
    adapter_config_path: Path
    rollout_summary: AIMOOnlineRolloutSummary

    def as_dict(self) -> dict[str, object]:

        return {
            "chunk_index": self.chunk_index,
            "start_order_index": self.start_order_index,
            "problem_count": self.problem_count,
            "rollout_adapter_state": self.rollout_adapter_state.as_dict(),
            "trained_adapter_state": self.trained_adapter_state.as_dict(),
            "queue_path": str(self.queue_path),
            "adapter_path": str(self.adapter_path),
            "adapter_config_path": str(self.adapter_config_path),
            "rollout_summary": self.rollout_summary.as_dict(),
        }


@dataclass(frozen=True)
class AIMOAdapterState:

    update_index: int
    adapter_path: Path | None
    adapter_config_path: Path | None
    adapter_hash: str
    created_at_unix: float

    def as_dict(self) -> dict[str, int | float | str]:

        return {
            "update_index": self.update_index,
            "adapter_path": str(self.adapter_path) if self.adapter_path is not None else "",
            "adapter_config_path": (
                str(self.adapter_config_path)
                if self.adapter_config_path is not None
                else ""
            ),
            "adapter_hash": self.adapter_hash,
            "created_at_unix": self.created_at_unix,
        }


@dataclass(frozen=True)
class AIMOOnlineServiceTarget:

    role: str
    model_role: str
    rank: int
    host: str
    port: int
    api_base: str
    health_url: str


@dataclass(frozen=True)
class AIMOOnlineServiceStatus:

    target: AIMOOnlineServiceTarget
    healthy: bool
    last_error: str
    failure_marker_path: Path | None
    failure_marker_payload: dict[str, object] | None

    def as_text(self) -> str:

        marker_text = (
            f" failure_marker={self.failure_marker_path}"
            if self.failure_marker_path is not None
            else ""
        )

        return (
            f"{self.target.role} rank={self.target.rank} "
            f"host={self.target.host} port={self.target.port} "
            f"health_url={self.target.health_url} healthy={self.healthy} "
            f"last_error={self.last_error or '<none>'}{marker_text}"
        )


class AIMOInterleavedRolloutCoordinator:

    def __init__(
        self,
        config: AIMOTrainingConfig,
        rollout_api_bases: list[str],
        judge_api_base: str,
        adapter_state: AIMOAdapterState,
    ) -> None:

        self.config = config
        self.adapter_state = adapter_state
        self.prompt_builder = AIMOPromptBuilder()
        self.chat_template = AIMOChatTemplate()
        self.endpoints = [
            self._build_rollout_endpoint(
                endpoint_index=endpoint_index,
                api_base=api_base,
                adapter_path=adapter_state.adapter_path,
            )
            for endpoint_index, api_base in enumerate(rollout_api_bases)
        ]
        self.judge_config = build_judge_inference_config(config).with_overrides(
            api_base=judge_api_base,
            logdir=config.logdir / "online_judge_client",
        )
        self.rollout_sandbox_pool = AIMOSandboxPool(
            config=self.endpoints[0].config,
            sandbox_count=config.sandbox_count,
        )
        self.judge_sandbox_pool = AIMOSandboxPool(
            config=self.judge_config,
            sandbox_count=config.sandbox_count,
        )
        run_sandbox_pool_preflight(
            sandbox_pool=self.rollout_sandbox_pool,
            sandbox_count=config.sandbox_count,
            log_path=config.logdir / "sandbox_preflight_contestant.json",
            pool_role="contestant_rollout",
        )
        run_sandbox_pool_preflight(
            sandbox_pool=self.judge_sandbox_pool,
            sandbox_count=config.sandbox_count,
            log_path=config.logdir / "sandbox_preflight_judge.json",
            pool_role="judge_scoring",
        )
        self.reward_scorer = AIMOTrainingRewardScorer(
            inference_config=self.judge_config,
            judge_client=AIMOInferenceClient(config=self.judge_config),
            reward_config=AIMORewardConfig(weights=config.reward_weights),
            sandbox_pool=self.judge_sandbox_pool,
        )

    def write_chunk(
        self,
        records: list[AIMOTrainingRecord],
        queue_path: Path,
        starting_group_index: int,
    ) -> AIMOOnlineRolloutSummary:

        started_at = time.monotonic()

        if not records:
            return AIMOOnlineRolloutSummary(
                admitted_problem_count=0,
                completed_group_count=0,
                written_sample_count=0,
                skipped_group_count=0,
                elapsed_seconds=0.0,
            )

        queue = AIMODurableGroupQueue(path=queue_path)
        partial_groups: dict[str, AIMOPartialGroup] = {}
        active_problem_counts = {
            endpoint.endpoint_index: 0
            for endpoint in self.endpoints
        }
        admitted_problem_count = 0
        completed_groups: list[AIMOGRPOGroup] = []
        skipped_group_count = 0
        group_attempt_counts: dict[str, int] = {}
        next_rollout_indices: dict[str, int] = {}
        active_problem_limit = self.active_problem_limit_per_endpoint()
        max_workers = max(
            1,
            min(
                len(records) * self.config.group_size,
                len(self.endpoints) * self.config.sandbox_count,
            ),
        )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                running: dict[
                    concurrent.futures.Future[AIMORolloutAttemptResult],
                    AIMORunningRollout,
                ] = {}

                def submit_rollout(
                    endpoint: AIMORolloutEndpoint,
                    record: AIMOTrainingRecord,
                    problem_key: str,
                    group_index: int,
                    rollout_index: int,
                    attempt_index: int,
                ) -> None:

                    future = executor.submit(
                        self.build_sample,
                        endpoint,
                        record,
                        group_index,
                        rollout_index,
                        attempt_index,
                    )
                    running[future] = AIMORunningRollout(
                        endpoint_index=endpoint.endpoint_index,
                        problem_key=problem_key,
                        record=record,
                        group_index=group_index,
                        rollout_index=rollout_index,
                        attempt_index=attempt_index,
                    )

                def admit_problem(endpoint: AIMORolloutEndpoint) -> bool:

                    nonlocal admitted_problem_count

                    if admitted_problem_count >= len(records):
                        return False

                    record = records[admitted_problem_count]
                    group_index = starting_group_index + admitted_problem_count
                    problem_key = f"{endpoint.endpoint_index}:{group_index}:{record.id}"
                    partial_groups[problem_key] = AIMOPartialGroup(
                        record=record,
                        group_index=group_index,
                        samples=[],
                    )
                    group_attempt_counts[problem_key] = 0
                    next_rollout_indices[problem_key] = self.config.group_size
                    active_problem_counts[endpoint.endpoint_index] += 1

                    for rollout_index in range(self.config.group_size):
                        submit_rollout(
                            endpoint=endpoint,
                            record=record,
                            problem_key=problem_key,
                            group_index=group_index,
                            rollout_index=rollout_index,
                            attempt_index=0,
                        )
                        group_attempt_counts[problem_key] += 1

                    admitted_problem_count += 1

                    return True

                for endpoint in self.endpoints:
                    for _ in range(active_problem_limit):
                        if not admit_problem(endpoint):
                            break

                while running:
                    done_futures, _ = concurrent.futures.wait(
                        running,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for future in done_futures:
                        running_rollout = running.pop(future)
                        attempt_result = self._sample_from_future(
                            future=future,
                            running_rollout=running_rollout,
                        )
                        partial_group = partial_groups.get(running_rollout.problem_key)

                        if partial_group is None:
                            continue

                        if attempt_result.sample is None:
                            replacement_index = next_rollout_indices[running_rollout.problem_key]
                            retry_attempt_index = running_rollout.attempt_index + 1
                            replacement_budget = group_attempt_counts[running_rollout.problem_key]

                            if (
                                retry_attempt_index <= self.config.max_rollout_retries_per_sample
                                and replacement_budget < self.config.max_group_replacement_attempts
                            ):
                                next_rollout_indices[
                                    running_rollout.problem_key
                                ] = replacement_index + 1
                                group_attempt_counts[running_rollout.problem_key] += 1
                                endpoint = self.endpoints[running_rollout.endpoint_index]
                                submit_rollout(
                                    endpoint=endpoint,
                                    record=running_rollout.record,
                                    problem_key=running_rollout.problem_key,
                                    group_index=running_rollout.group_index,
                                    rollout_index=replacement_index,
                                    attempt_index=retry_attempt_index,
                                )

                                continue

                            self.write_failure_report(
                                queue_path=queue_path,
                                filename="rollout_failures.jsonl",
                                payload=attempt_result.failure_metadata,
                            )
                            self.write_failure_report(
                                queue_path=queue_path,
                                filename="skipped_groups.jsonl",
                                payload={
                                    "problem_id": running_rollout.record.id,
                                    "group_index": running_rollout.group_index,
                                    "rollout_index": running_rollout.rollout_index,
                                    "attempt_index": running_rollout.attempt_index,
                                    "failure_metadata": attempt_result.failure_metadata,
                                    "retry_count": group_attempt_counts[running_rollout.problem_key],
                                },
                            )
                            skipped_group_count += 1
                            del partial_groups[running_rollout.problem_key]
                            active_problem_counts[running_rollout.endpoint_index] -= 1
                            endpoint = self.endpoints[running_rollout.endpoint_index]

                            if active_problem_counts[endpoint.endpoint_index] < active_problem_limit:
                                admit_problem(endpoint)

                            continue

                        partial_group.append(attempt_result.sample)

                        if not partial_group.is_complete(self.config.group_size):
                            continue

                        completed_group = partial_group.to_group()
                        validate_complete_rollout_group(
                            group=completed_group,
                            group_size=self.config.group_size,
                            minimum_trainable_tokens=self.config.minimum_trainable_tokens_per_sample,
                        )
                        queue.append_group(completed_group)
                        completed_groups.append(completed_group)
                        del partial_groups[running_rollout.problem_key]
                        active_problem_counts[running_rollout.endpoint_index] -= 1
                        endpoint = self.endpoints[running_rollout.endpoint_index]

                        if active_problem_counts[endpoint.endpoint_index] < active_problem_limit:
                            admit_problem(endpoint)
        finally:
            self.close()

        written_sample_count = sum(
            len(group.samples)
            for group in completed_groups
        )

        return AIMOOnlineRolloutSummary(
            admitted_problem_count=admitted_problem_count,
            completed_group_count=len(completed_groups),
            written_sample_count=written_sample_count,
            skipped_group_count=skipped_group_count,
            elapsed_seconds=time.monotonic() - started_at,
        )

    def active_problem_limit_per_endpoint(self) -> int:

        return max(
            1,
            (
                self.config.active_problem_count
                + len(self.endpoints)
                - 1
            ) // len(self.endpoints),
        )

    def build_sample(
        self,
        endpoint: AIMORolloutEndpoint,
        record: AIMOTrainingRecord,
        group_index: int,
        rollout_index: int,
        attempt_index: int = 0,
    ) -> AIMORolloutAttemptResult:

        try:
            sample = self._build_sample(
                endpoint=endpoint,
                record=record,
                group_index=group_index,
                rollout_index=rollout_index,
            )

            if sum(sample.env_mask) < self.config.minimum_trainable_tokens_per_sample:
                return AIMORolloutAttemptResult(
                    sample=None,
                    failure_metadata={
                        "problem_id": record.id,
                        "group_index": group_index,
                        "rollout_index": rollout_index,
                        "attempt_index": attempt_index,
                        "exception": "minimum_trainable_tokens_not_met",
                        "trainable_token_count": sum(sample.env_mask),
                    },
                )

            return AIMORolloutAttemptResult(
                sample=sample,
                failure_metadata={},
            )
        except Exception as error:
            return AIMORolloutAttemptResult(
                sample=None,
                failure_metadata={
                    "problem_id": record.id,
                    "group_index": group_index,
                    "rollout_index": rollout_index,
                    "attempt_index": attempt_index,
                    "exception": str(error),
                },
            )

    def _build_sample(
        self,
        endpoint: AIMORolloutEndpoint,
        record: AIMOTrainingRecord,
        group_index: int,
        rollout_index: int,
    ) -> AIMORolloutSample:

        messages = self.prompt_builder.build_first_pass_messages(
            problem_text=record.problem,
            enable_tools=True,
        )
        prompt = self.chat_template.render(
            messages=messages,
            add_generation_prompt=True,
        )
        result = self._run_tool_rollout(
            endpoint=endpoint,
            problem_text=record.problem,
            seed=self.config.seed + rollout_index,
        )
        reward = self.reward_scorer.score(
            problem=record.problem,
            reference_solution=record.reference_solution,
            generated_proof=result.proof_text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            finish_reason=result.finish_reason,
            tool_tokens=result.tool_tokens,
        )

        return AIMORolloutSample(
            problem_id=record.id,
            group_index=group_index,
            rollout_index=rollout_index,
            prompt=result.prompt or prompt,
            completion=result.proof_text,
            token_ids=result.completion_ids,
            token_logprobs=result.token_logprobs,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            finish_reason=result.finish_reason,
            python_calls=result.python_calls,
            python_errors=result.python_errors,
            tool_call_count=result.python_calls,
            tool_error_count=result.python_errors,
            reward=reward,
            prompt_ids=result.prompt_ids,
            env_mask=result.env_mask,
            endpoint_index=endpoint.endpoint_index,
            tool_tokens=result.tool_tokens,
            sampling_logprobs=result.token_logprobs,
            policy_update_index=self.adapter_state.update_index,
            policy_adapter_hash=self.adapter_state.adapter_hash,
            policy_adapter_path=str(endpoint.config.lora_adapter_path or ""),
        )

    def _run_tool_rollout(
        self,
        endpoint: AIMORolloutEndpoint,
        problem_text: str,
        seed: int,
    ) -> object:

        if self.config.tool_protocol == "harmony":
            raise ValueError(
                "Structured Harmony training rollouts require token-level "
                "Harmony transcript support and are not enabled in this build."
            )

        if self.config.tool_protocol not in {"markdown_code", "olmo_chatml"}:
            raise ValueError(f"Unsupported tool_protocol: {self.config.tool_protocol}")

        with self.rollout_sandbox_pool.acquire() as sandbox:
            return AIMOToolRolloutEngine(
                config=endpoint.config,
                client=endpoint.client,
                prompt_builder=self.prompt_builder,
                sandbox=sandbox,
            ).run_problem(
                problem_text=problem_text,
                seed=seed,
            )

    def close(self) -> None:

        rollout_sandbox_pool = getattr(self, "rollout_sandbox_pool", None)
        judge_sandbox_pool = getattr(self, "judge_sandbox_pool", None)

        if rollout_sandbox_pool is not None:
            rollout_sandbox_pool.close()

        if judge_sandbox_pool is not None:
            judge_sandbox_pool.close()

    def _sample_from_future(
        self,
        future: concurrent.futures.Future[AIMORolloutAttemptResult],
        running_rollout: AIMORunningRollout,
    ) -> AIMORolloutAttemptResult:

        try:
            result = future.result()

            if isinstance(result, AIMORolloutSample):
                return AIMORolloutAttemptResult(
                    sample=result,
                    failure_metadata={},
                )

            return result
        except Exception as error:
            return AIMORolloutAttemptResult(
                sample=None,
                failure_metadata={
                    "problem_id": running_rollout.record.id,
                    "group_index": running_rollout.group_index,
                    "rollout_index": running_rollout.rollout_index,
                    "attempt_index": running_rollout.attempt_index,
                    "exception": str(error),
                },
            )

    def write_failure_report(
        self,
        queue_path: Path,
        filename: str,
        payload: dict[str, object],
    ) -> None:

        path = queue_path.parent / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload, ensure_ascii=False))
            output_file.write("\n")

    def _failure_sample(
        self,
        record: AIMOTrainingRecord,
        group_index: int,
        rollout_index: int,
        error: str,
    ) -> AIMORolloutSample:

        reward = AIMORewardBreakdown(
            judge_grade=0,
            context_reward=-1,
            solution_page_reward=-1,
            scalar_reward=self.reward_scorer.scalar_reward(
                judge_grade=0,
                context_reward=-1,
                solution_page_reward=-1,
            ),
            rendered_page_count=0,
            page_count_method=self.config.page_count_method,
            latex_compile_status="not_attempted",
            page_count_fallback_reason="rollout_failed",
            judge_response=error,
            judge_parse_failed=True,
            input_tokens=None,
            output_tokens=1,
            finish_reason="error",
            latency_seconds=0.0,
            tool_tokens=0,
        )

        return AIMORolloutSample(
            problem_id=record.id,
            group_index=group_index,
            rollout_index=rollout_index,
            prompt=record.problem,
            completion="No proof was produced.",
            token_ids=[
                0,
            ],
            token_logprobs=[
                0.0,
            ],
            input_tokens=None,
            output_tokens=1,
            finish_reason="error",
            python_calls=0,
            python_errors=0,
            tool_call_count=0,
            tool_error_count=0,
            reward=reward,
            env_mask=[
                0,
            ],
            tool_tokens=0,
        )

    def _build_rollout_endpoint(
        self,
        endpoint_index: int,
        api_base: str,
        adapter_path: Path | None,
    ) -> AIMORolloutEndpoint:

        rollout_config = build_rollout_inference_config(self.config).with_overrides(
            api_base=api_base,
            logdir=self.config.logdir / "online_rollout_clients" / f"endpoint_{endpoint_index}",
            lora_adapter_path=adapter_path,
        )

        return AIMORolloutEndpoint(
            endpoint_index=endpoint_index,
            api_base=api_base,
            config=rollout_config,
            client=AIMOInferenceClient(config=rollout_config),
        )


class AIMOOnlineTrainingPipeline:

    def __init__(self, config: AIMOTrainingConfig) -> None:

        self.config = config
        self.control_dir = resolve_control_dir(config)

    def run(self) -> list[AIMOOnlineUpdateSummary]:

        if self.config.initial_adapter_path is None and not self.config.allow_base_rollouts:
            raise ValueError(
                "Online training requires initial_adapter_path unless allow_base_rollouts is true."
            )

        if (
            self.config.initial_adapter_path is not None
            and not self.config.initial_adapter_path.exists()
        ):
            raise FileNotFoundError(
                f"initial_adapter_path does not exist: {self.config.initial_adapter_path}"
            )

        dataset_preflight = validate_training_dataset(
            path=self.config.dataset_path,
            problems_per_update=self.config.problems_per_update,
            group_size=self.config.group_size,
        )
        self.write_online_event(
            event="dataset_preflight_completed",
            chunk_index=-1,
            payload=dataset_preflight,
        )
        records = read_training_records(self.config.dataset_path)
        summaries: list[AIMOOnlineUpdateSummary] = []
        current_adapter_state = build_adapter_state(
            update_index=0,
            adapter_path=self.config.initial_adapter_path,
        )

        self.control_dir.mkdir(parents=True, exist_ok=True)
        write_adapter_state(
            control_dir=self.control_dir,
            adapter_state=current_adapter_state,
        )

        try:
            for chunk_index, chunk_start in enumerate(
                range(0, len(records), self.config.problems_per_update)
            ):
                chunk_records = records[
                    chunk_start: chunk_start + self.config.problems_per_update
                ]

                if not chunk_records:
                    continue

                summary = self.run_update_chunk(
                    chunk_index=chunk_index,
                    chunk_start=chunk_start,
                    records=chunk_records,
                    current_adapter_state=current_adapter_state,
                )
                summaries.append(summary)
                current_adapter_state = summary.trained_adapter_state
                write_adapter_state(
                    control_dir=self.control_dir,
                    adapter_state=current_adapter_state,
                )

            self.write_online_manifest(summaries=summaries)

            return summaries
        finally:
            write_stop_signal(self.control_dir)

    def run_update_chunk(
        self,
        chunk_index: int,
        chunk_start: int,
        records: list[AIMOTrainingRecord],
        current_adapter_state: AIMOAdapterState,
    ) -> AIMOOnlineUpdateSummary:

        chunk_name = f"chunk_{chunk_index:05d}"
        chunk_root = self.config.logdir / "online_chunks" / chunk_name
        queue_path = chunk_root / "grpo_groups.jsonl"
        adapter_output_path = self.config.output_path / chunk_name
        adapter_temporary_output_path = self.config.output_path / f".{chunk_name}.{os.getpid()}.tmp"
        training_logdir = chunk_root / "training_logs"

        self.write_online_event(
            event="chunk_started",
            chunk_index=chunk_index,
            payload={
                "chunk_start": chunk_start,
                "problem_count": len(records),
                "rollout_adapter_state": current_adapter_state.as_dict(),
                "queue_path": str(queue_path),
            },
        )

        try:
            rollout_summary = self.run_rollout_phase(
                records=records,
                queue_path=queue_path,
                starting_group_index=chunk_start,
                adapter_state=current_adapter_state,
            )
            self.write_online_event(
                event="rollout_phase_completed",
                chunk_index=chunk_index,
                payload=rollout_summary.as_dict(),
            )
        except Exception as error:
            self.write_online_event(
                event="rollout_phase_failed",
                chunk_index=chunk_index,
                payload={
                    "exception": str(error),
                    "rollout_adapter_state": current_adapter_state.as_dict(),
                },
            )

            raise

        training_config = replace(
            self.config,
            online=False,
            dataset_path=queue_path,
            output_path=adapter_temporary_output_path,
            logdir=training_logdir,
            group_queue_path=queue_path,
            initial_adapter_path=current_adapter_state.adapter_path,
        )
        if adapter_temporary_output_path.exists():
            shutil.rmtree(adapter_temporary_output_path)

        try:
            self.write_online_event(
                event="training_phase_started",
                chunk_index=chunk_index,
                payload={
                    "dataset_path": str(queue_path),
                    "output_path": str(adapter_temporary_output_path),
                    "initial_adapter_path": str(current_adapter_state.adapter_path or ""),
                },
            )
            run_gradient_update_training(config=training_config)
            self.write_online_event(
                event="training_phase_completed",
                chunk_index=chunk_index,
                payload={
                    "temporary_adapter_path": str(adapter_temporary_output_path),
                },
            )
        except Exception as error:
            self.write_online_event(
                event="training_phase_failed",
                chunk_index=chunk_index,
                payload={
                    "exception": str(error),
                    "dataset_path": str(queue_path),
                    "output_path": str(adapter_temporary_output_path),
                },
            )

            raise

        try:
            verify_adapter_directory(adapter_temporary_output_path)
            self.write_online_event(
                event="adapter_verification_completed",
                chunk_index=chunk_index,
                payload={
                    "temporary_adapter_path": str(adapter_temporary_output_path),
                },
            )
        except Exception as error:
            self.write_online_event(
                event="adapter_verification_failed",
                chunk_index=chunk_index,
                payload={
                    "exception": str(error),
                    "temporary_adapter_path": str(adapter_temporary_output_path),
                },
            )

            raise

        if adapter_output_path.exists():
            self.write_online_event(
                event="adapter_publish_failed",
                chunk_index=chunk_index,
                payload={
                    "exception": f"Adapter output path already exists: {adapter_output_path}",
                    "adapter_output_path": str(adapter_output_path),
                },
            )

            raise FileExistsError(f"Adapter output path already exists: {adapter_output_path}")

        adapter_output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(adapter_temporary_output_path, adapter_output_path)
        adapter_path = adapter_output_path
        adapter_config_path = adapter_output_path / "adapter_config.json"

        trained_adapter_state = build_adapter_state(
            update_index=chunk_index + 1,
            adapter_path=adapter_path,
            adapter_config_path=adapter_config_path,
        )
        self.write_online_event(
            event="adapter_published",
            chunk_index=chunk_index,
            payload={
                "trained_adapter_state": trained_adapter_state.as_dict(),
            },
        )

        return AIMOOnlineUpdateSummary(
            chunk_index=chunk_index,
            start_order_index=chunk_start,
            problem_count=len(records),
            rollout_adapter_state=current_adapter_state,
            trained_adapter_state=trained_adapter_state,
            queue_path=queue_path,
            adapter_path=adapter_path,
            adapter_config_path=adapter_config_path,
            rollout_summary=rollout_summary,
        )

    def run_rollout_phase(
        self,
        records: list[AIMOTrainingRecord],
        queue_path: Path,
        starting_group_index: int,
        adapter_state: AIMOAdapterState,
    ) -> AIMOOnlineRolloutSummary:

        with contextlib.ExitStack() as server_stack:
            self.enter_local_servers(
                server_stack=server_stack,
                current_adapter_path=adapter_state.adapter_path,
            )
            wait_for_online_services(
                config=self.config,
                adapter_state=adapter_state,
            )

            return AIMOInterleavedRolloutCoordinator(
                config=self.config,
                rollout_api_bases=resolve_rollout_api_bases(self.config),
                judge_api_base=resolve_judge_api_base(self.config),
                adapter_state=adapter_state,
            ).write_chunk(
                records=records,
                queue_path=queue_path,
                starting_group_index=starting_group_index,
            )

    def enter_local_servers(
        self,
        server_stack: contextlib.ExitStack,
        current_adapter_path: Path | None,
    ) -> None:

        if should_launch_local_rollout_server(self.config):
            server_stack.enter_context(AIMOInferenceServer(
                build_local_rollout_server_config(
                    config=self.config,
                    adapter_path=current_adapter_path,
                )
            ))

        if should_launch_local_judge_server(self.config):
            server_stack.enter_context(AIMOInferenceServer(
                build_local_judge_server_config(config=self.config)
            ))

    def write_online_manifest(self, summaries: list[AIMOOnlineUpdateSummary]) -> Path:

        path = self.config.logdir / "online_training_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                {
                    "problems_per_update": self.config.problems_per_update,
                    "group_size": self.config.group_size,
                    "rollout_api_bases": resolve_rollout_api_bases(self.config),
                    "judge_api_base": resolve_judge_api_base(self.config),
                    "updates": [
                        summary.as_dict()
                        for summary in summaries
                    ],
                },
                output_file,
                ensure_ascii=False,
                indent=2,
            )
            output_file.write("\n")

        os.replace(temporary_path, path)

        return path

    def write_online_event(
        self,
        event: str,
        chunk_index: int,
        payload: dict[str, object],
    ) -> None:

        path = self.config.logdir / "online_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        event_payload = {
            "event": event,
            "chunk_index": chunk_index,
            "created_at_unix": time.time(),
            **payload,
        }

        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(event_payload, ensure_ascii=False))
            output_file.write("\n")


def run_online_training(config: AIMOTrainingConfig) -> None:

    if config.role == "rollout_server":
        run_rollout_node_service(config=config)

        return

    if config.role == "judge_server":
        run_judge_node_service(config=config)

        return

    if config.role == "train_update":
        run_training_phase(config=config)

        return

    if config.world_size > 1 and config.global_rank != config.trainer_node_rank:
        run_online_service_node(config=config)

        return

    run_controller(config=config)


def run_controller(config: AIMOTrainingConfig) -> None:

    AIMOOnlineTrainingPipeline(config=config).run()


def run_rollout_node_service(config: AIMOTrainingConfig) -> None:

    control_dir = resolve_control_dir(config)
    control_dir.mkdir(parents=True, exist_ok=True)
    run_rollout_service_node(
        config=config,
        control_dir=control_dir,
    )


def run_judge_node_service(config: AIMOTrainingConfig) -> None:

    control_dir = resolve_control_dir(config)
    control_dir.mkdir(parents=True, exist_ok=True)
    run_judge_service_node(
        config=config,
        control_dir=control_dir,
    )


def run_training_phase(config: AIMOTrainingConfig) -> None:

    run_training_step(config=config)


def run_gradient_update_training(config: AIMOTrainingConfig) -> None:

    if should_launch_distributed_training(config):
        subprocess.run(
            build_distributed_training_command(config),
            check=True,
            env=build_distributed_training_environment(),
        )

        return

    run_training_phase(config=config)


def should_launch_distributed_training(config: AIMOTrainingConfig) -> bool:

    return (
        config.train_processes_per_node > 1
        and not is_torchrun_training_process()
    )


def is_torchrun_training_process() -> bool:

    if os.environ.get("TORCHELASTIC_RUN_ID") is not None:
        return True

    return (
        os.environ.get("LOCAL_RANK") is not None
        and os.environ.get("RANK") is not None
        and os.environ.get("LOCAL_WORLD_SIZE") is not None
    )


def build_distributed_training_environment() -> dict[str, str]:

    environment = dict(os.environ)
    environment["AIMO_TRAINING_ROLE"] = "train_update"

    return environment


def build_distributed_training_command(config: AIMOTrainingConfig) -> list[str]:

    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc-per-node",
        str(config.train_processes_per_node),
        "--module",
        "aimo_training.entrypoints.train",
        *training_phase_cli_args(config),
    ]


def training_phase_cli_args(config: AIMOTrainingConfig) -> list[str]:

    config = config.with_dummy_test_defaults()

    arguments = [
        "--model_path",
        str(config.model_path),
        "--dataset_path",
        str(config.dataset_path),
        "--output_path",
        str(config.output_path),
        "--logdir",
        str(config.logdir),
        "--role",
        "train_update",
        "--online",
        "false",
        "--rollout_mode",
        config.rollout_mode,
        "--tool_protocol",
        config.tool_protocol,
        "--num_gpus",
        str(config.num_gpus),
        "--learning_rate",
        str(config.learning_rate),
        "--num_train_epochs",
        str(config.num_train_epochs),
        "--per_device_batch_size",
        str(config.per_device_batch_size),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation_steps),
        "--lora_rank",
        str(config.lora_rank),
        "--lora_alpha",
        str(config.lora_alpha),
        "--max_model_len",
        str(config.max_model_len),
        "--group_size",
        str(config.group_size),
        "--judge_model_path",
        str(config.judge_model_path),
        "--judge_port",
        str(config.judge_port),
        "--rollout_port",
        str(config.rollout_port),
        "--rollout_temperature",
        str(config.rollout_temperature),
        "--rollout_top_p",
        str(config.rollout_top_p),
        "--rollout_min_p",
        str(config.rollout_min_p),
        "--max_python_calls",
        str(config.max_python_calls),
        "--active_problem_count",
        str(config.active_problem_count),
        "--sandbox_count",
        str(config.sandbox_count),
        "--kv_cache_dtype",
        config.kv_cache_dtype,
        "--page_count_method",
        config.page_count_method,
        "--page_template",
        config.page_template,
        "--importance_sampling_level",
        config.importance_sampling_level,
        "--kl_beta",
        str(config.kl_beta),
        "--weight_decay",
        str(config.weight_decay),
        "--scheduler",
        config.scheduler,
        "--warmup_ratio",
        str(config.warmup_ratio),
        "--max_grad_norm",
        str(config.max_grad_norm),
        "--grpo_epsilon",
        str(config.grpo_epsilon),
        "--seed",
        str(config.seed),
        "--train_processes_per_node",
        str(config.train_processes_per_node),
        "--train_sharding_strategy",
        config.train_sharding_strategy,
        "--train_bf16",
        str(config.train_bf16).lower(),
        "--train_gradient_checkpointing",
        str(config.train_gradient_checkpointing).lower(),
        "--train_fsdp_transformer_layer_cls_to_wrap",
        config.train_fsdp_transformer_layer_cls_to_wrap,
        "--max_new_tokens",
        str(config.max_new_tokens),
        "--problems_per_update",
        str(config.problems_per_update),
        "--node_hostnames",
        config.node_hostnames,
        "--train_node_ranks",
        config.train_node_ranks,
        "--rollout_node_ranks",
        config.rollout_node_ranks,
        "--judge_node_rank",
        str(config.judge_node_rank),
        "--trainer_node_rank",
        str(config.trainer_node_rank),
        "--rollout_tensor_parallel_size",
        str(config.rollout_tensor_parallel_size),
        "--rollout_max_num_seqs",
        str(config.rollout_max_num_seqs),
        "--rollout_max_num_batched_tokens",
        str(config.rollout_max_num_batched_tokens),
        "--judge_tensor_parallel_size",
        str(config.judge_tensor_parallel_size),
        "--judge_max_num_seqs",
        str(config.judge_max_num_seqs),
        "--judge_max_num_batched_tokens",
        str(config.judge_max_num_batched_tokens),
        "--rollout_wave_problem_count",
        str(config.rollout_wave_problem_count),
        "--adapter_reload_timeout_seconds",
        str(config.adapter_reload_timeout_seconds),
        "--max_rollout_retries_per_sample",
        str(config.max_rollout_retries_per_sample),
        "--max_group_replacement_attempts",
        str(config.max_group_replacement_attempts),
        "--minimum_trainable_tokens_per_sample",
        str(config.minimum_trainable_tokens_per_sample),
        "--group_queue_path",
        str(config.resolved_group_queue_path),
    ]

    if config.dummy_test:
        arguments.extend([
            "--dummy_test",
            "true",
            "--dummy_model_path",
            str(config.dummy_model_path),
        ])

    if config.allow_base_rollouts:
        arguments.extend([
            "--allow_base_rollouts",
            "true",
        ])

    if config.reward_weights_json:
        arguments.extend([
            "--reward_weights_json",
            config.reward_weights_json,
        ])

    if config.rollout_api_base:
        arguments.extend([
            "--rollout_api_base",
            config.rollout_api_base,
        ])

    if config.rollout_api_bases:
        arguments.extend([
            "--rollout_api_bases",
            config.rollout_api_bases,
        ])

    if config.judge_api_base:
        arguments.extend([
            "--judge_api_base",
            config.judge_api_base,
        ])

    if config.online_control_dir is not None:
        arguments.extend([
            "--online_control_dir",
            str(config.online_control_dir),
        ])

    if config.initial_adapter_path is not None:
        arguments.extend([
            "--initial_adapter_path",
            str(config.initial_adapter_path),
        ])

    if config.target_module_suffixes:
        arguments.extend([
            "--target_module_suffixes",
            ",".join(config.target_module_suffixes),
        ])

    return arguments


def run_online_service_node(config: AIMOTrainingConfig) -> None:

    control_dir = resolve_control_dir(config)
    control_dir.mkdir(parents=True, exist_ok=True)
    rollout_ranks = parse_int_list(config.rollout_node_ranks)

    if config.global_rank in rollout_ranks:
        run_rollout_node_service(config=config)

        return

    if config.global_rank == config.judge_node_rank:
        run_judge_node_service(config=config)

        return

    wait_for_stop_signal(control_dir)


def run_rollout_service_node(
    config: AIMOTrainingConfig,
    control_dir: Path,
) -> None:

    current_adapter_marker = object()
    server: AIMOInferenceServer | None = None

    try:
        while not stop_signal_path(control_dir).exists():
            requested_adapter_state = read_adapter_state(control_dir) or build_adapter_state(
                update_index=0,
                adapter_path=None,
            )
            requested_adapter_marker = (
                requested_adapter_state.update_index,
                requested_adapter_state.adapter_hash,
                str(requested_adapter_state.adapter_path or ""),
            )

            if requested_adapter_marker != current_adapter_marker:
                if server is not None:
                    server.stop()

                server_config = build_remote_rollout_server_config(
                    config=config,
                    adapter_path=requested_adapter_state.adapter_path,
                )
                server = AIMOInferenceServer(
                    server_config
                )
                write_service_started(
                    control_dir=control_dir,
                    role="contestant",
                    rank=config.global_rank,
                    server=server,
                    adapter_state=requested_adapter_state,
                )
                server.start()
                write_service_ready(
                    control_dir=control_dir,
                    role="contestant",
                    rank=config.global_rank,
                    adapter_state=requested_adapter_state,
                    served_model_name=server.config.resolved_generation_model_name,
                    health_url=server.config.health_url,
                )
                current_adapter_marker = requested_adapter_marker

            time.sleep(3.0)
    except Exception as error:
        write_service_failed(
            control_dir=control_dir,
            role="contestant",
            rank=config.global_rank,
            server=server,
            error=error,
        )

        raise
    finally:
        if server is not None:
            server.stop()


def run_judge_service_node(
    config: AIMOTrainingConfig,
    control_dir: Path,
) -> None:

    server = AIMOInferenceServer(build_remote_judge_server_config(config=config))

    try:
        write_service_started(
            control_dir=control_dir,
            role="judge",
            rank=config.global_rank,
            server=server,
            adapter_state=build_adapter_state(
                update_index=0,
                adapter_path=None,
            ),
        )
        server.start()
        write_service_ready(
            control_dir=control_dir,
            role="judge",
            rank=config.global_rank,
            adapter_state=build_adapter_state(
                update_index=0,
                adapter_path=None,
            ),
            served_model_name=server.config.resolved_generation_model_name,
            health_url=server.config.health_url,
        )
        wait_for_stop_signal(control_dir)
    except Exception as error:
        write_service_failed(
            control_dir=control_dir,
            role="judge",
            rank=config.global_rank,
            server=server,
            error=error,
        )

        raise
    finally:
        server.stop()


def build_local_rollout_server_config(
    config: AIMOTrainingConfig,
    adapter_path: Path | None,
) -> AIMOConfig:

    return build_rollout_inference_config(config).with_overrides(
        host="0.0.0.0" if config.world_size > 1 else "127.0.0.1",
        launch_server=True,
        reuse_server=False,
        logdir=config.logdir / "online_servers" / "local_contestant",
        lora_adapter_path=adapter_path,
    )


def build_remote_rollout_server_config(
    config: AIMOTrainingConfig,
    adapter_path: Path | None,
) -> AIMOConfig:

    return build_rollout_inference_config(config).with_overrides(
        host="0.0.0.0",
        launch_server=True,
        reuse_server=False,
        logdir=config.logdir / "online_servers" / f"rank_{config.global_rank}_contestant",
        lora_adapter_path=adapter_path,
    )


def build_local_judge_server_config(config: AIMOTrainingConfig) -> AIMOConfig:

    return build_judge_inference_config(config).with_overrides(
        host="127.0.0.1",
        launch_server=True,
        reuse_server=False,
        logdir=config.logdir / "online_servers" / "local_judge",
    )


def build_remote_judge_server_config(config: AIMOTrainingConfig) -> AIMOConfig:

    return build_judge_inference_config(config).with_overrides(
        host="0.0.0.0",
        launch_server=True,
        reuse_server=False,
        logdir=config.logdir / "online_servers" / f"rank_{config.global_rank}_judge",
    )


def should_launch_local_rollout_server(config: AIMOTrainingConfig) -> bool:

    if config.world_size > 1:
        return False

    return not bool(config.rollout_api_base or config.rollout_api_bases)


def should_launch_local_judge_server(config: AIMOTrainingConfig) -> bool:

    return config.world_size == 1 and not bool(config.judge_api_base)


def resolve_rollout_api_bases(config: AIMOTrainingConfig) -> list[str]:

    if config.rollout_api_bases.strip():
        return split_csv(config.rollout_api_bases)

    if config.rollout_api_base.strip():
        return [
            config.rollout_api_base.strip().rstrip("/"),
        ]

    if config.world_size > 1:
        hosts = resolve_node_hostnames(config)

        return [
            f"http://{hosts[rank]}:{config.rollout_port}/v1"
            for rank in parse_int_list(config.rollout_node_ranks)
        ]

    return [
        f"http://127.0.0.1:{config.rollout_port}/v1",
    ]


def resolve_judge_api_base(config: AIMOTrainingConfig) -> str:

    if config.judge_api_base.strip():
        return config.judge_api_base.strip().rstrip("/")

    if config.world_size > 1:
        hosts = resolve_node_hostnames(config)

        return f"http://{hosts[config.judge_node_rank]}:{config.judge_port}/v1"

    return f"http://127.0.0.1:{config.judge_port}/v1"


def resolve_node_hostnames(config: AIMOTrainingConfig) -> list[str]:

    configured_hostnames = config.node_hostnames or os.environ.get("AIMO_NODE_HOSTNAMES", "")
    hostnames = split_csv(configured_hostnames)

    if len(hostnames) != config.world_size:
        raise ValueError(
            "node_hostnames or AIMO_NODE_HOSTNAMES must provide exactly one hostname "
            "per WORLD_SIZE rank for online training."
        )

    return hostnames


def wait_for_online_services(
    config: AIMOTrainingConfig,
    adapter_state: AIMOAdapterState | None = None,
) -> None:

    targets = resolve_online_service_targets(config)
    deadline = time.monotonic() + 900.0
    last_statuses: list[AIMOOnlineServiceStatus] = []

    while time.monotonic() < deadline:
        last_statuses = poll_online_service_statuses(
            config=config,
            targets=targets,
        )
        failed_statuses = [
            status
            for status in last_statuses
            if status.failure_marker_path is not None
        ]

        if failed_statuses:
            raise RuntimeError(online_service_status_error(
                heading="Online service failure marker detected.",
                statuses=last_statuses,
            ))

        if all(status.healthy for status in last_statuses):
            break

        time.sleep(1.0)
    else:
        raise RuntimeError(online_service_status_error(
            heading="Online services did not become healthy.",
            statuses=last_statuses,
        ))

    if config.world_size > 1 and adapter_state is not None:
        wait_for_rollout_adapter_readiness(
            config=config,
            adapter_state=adapter_state,
        )


def resolve_online_service_targets(config: AIMOTrainingConfig) -> list[AIMOOnlineServiceTarget]:

    if config.world_size > 1:
        hosts = resolve_node_hostnames(config)
        rollout_targets = [
            AIMOOnlineServiceTarget(
                role="contestant",
                model_role="contestant",
                rank=rank,
                host=hosts[rank],
                port=config.rollout_port,
                api_base=f"http://{hosts[rank]}:{config.rollout_port}/v1",
                health_url=f"http://{hosts[rank]}:{config.rollout_port}/health",
            )
            for rank in parse_int_list(config.rollout_node_ranks)
        ]
        judge_host = hosts[config.judge_node_rank]

        return [
            *rollout_targets,
            AIMOOnlineServiceTarget(
                role="judge",
                model_role="judge",
                rank=config.judge_node_rank,
                host=judge_host,
                port=config.judge_port,
                api_base=f"http://{judge_host}:{config.judge_port}/v1",
                health_url=f"http://{judge_host}:{config.judge_port}/health",
            ),
        ]

    return [
        AIMOOnlineServiceTarget(
            role="contestant",
            model_role="contestant",
            rank=0,
            host="127.0.0.1",
            port=config.rollout_port,
            api_base=f"http://127.0.0.1:{config.rollout_port}/v1",
            health_url=f"http://127.0.0.1:{config.rollout_port}/health",
        ),
        AIMOOnlineServiceTarget(
            role="judge",
            model_role="judge",
            rank=0,
            host="127.0.0.1",
            port=config.judge_port,
            api_base=f"http://127.0.0.1:{config.judge_port}/v1",
            health_url=f"http://127.0.0.1:{config.judge_port}/health",
        ),
    ]


def poll_online_service_statuses(
    config: AIMOTrainingConfig,
    targets: list[AIMOOnlineServiceTarget],
) -> list[AIMOOnlineServiceStatus]:

    return [
        poll_online_service_status(
            config=config,
            target=target,
        )
        for target in targets
    ]


def poll_online_service_status(
    config: AIMOTrainingConfig,
    target: AIMOOnlineServiceTarget,
) -> AIMOOnlineServiceStatus:

    failure_marker_path = find_service_failure_marker(
        control_dir=resolve_control_dir(config),
        role=target.role,
        rank=target.rank,
    )
    failure_marker_payload = read_failure_marker(failure_marker_path)

    if failure_marker_payload is not None:
        return AIMOOnlineServiceStatus(
            target=target,
            healthy=False,
            last_error=str(failure_marker_payload.get("exception_message", "")),
            failure_marker_path=failure_marker_path,
            failure_marker_payload=failure_marker_payload,
        )

    try:
        with urllib.request.urlopen(target.health_url, timeout=5.0) as response:
            return AIMOOnlineServiceStatus(
                target=target,
                healthy=response.status == 200,
                last_error="" if response.status == 200 else f"HTTP {response.status}",
                failure_marker_path=None,
                failure_marker_payload=None,
            )
    except (urllib.error.URLError, TimeoutError) as error:
        return AIMOOnlineServiceStatus(
            target=target,
            healthy=False,
            last_error=str(error),
            failure_marker_path=None,
            failure_marker_payload=None,
        )


def online_service_status_error(
    heading: str,
    statuses: list[AIMOOnlineServiceStatus],
) -> str:

    return "\n".join([
        heading,
        *[
            status.as_text()
            for status in statuses
        ],
    ])


def wait_for_health(health_url: str, timeout_seconds: float) -> None:

    deadline = time.monotonic() + timeout_seconds
    last_error = ""

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = str(error)

        time.sleep(1.0)

    raise RuntimeError(f"Service did not become healthy at {health_url}: {last_error}")


def api_base_to_health_url(api_base: str) -> str:

    base = api_base.rstrip("/")

    if base.endswith("/v1"):
        base = base.removesuffix("/v1")

    return f"{base}/health"


def resolve_control_dir(config: AIMOTrainingConfig) -> Path:

    if config.online_control_dir is not None:
        return config.online_control_dir

    return config.logdir / "online_control"


def parse_int_list(value: str) -> list[int]:

    ranks = [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]

    if not ranks:
        raise ValueError("At least one rank is required.")

    return ranks


def split_csv(value: str) -> list[str]:

    return [
        item.strip().rstrip("/")
        for item in value.split(",")
        if item.strip()
    ]


def validate_complete_rollout_group(
    group: AIMOGRPOGroup,
    group_size: int,
    minimum_trainable_tokens: int,
) -> None:

    if len(group.samples) != group_size:
        raise ValueError(
            f"GRPO group {group.group_index} has {len(group.samples)} samples, "
            f"but group_size is {group_size}."
        )

    rollout_indices: set[int] = set()
    expected_prompt = group.samples[0].prompt if group.samples else ""
    expected_adapter_hash = group.samples[0].policy_adapter_hash if group.samples else ""
    rewards = set()

    for sample in group.samples:
        rewards.add(sample.reward.scalar_reward)

        if sample.problem_id != group.problem_id:
            raise ValueError(f"GRPO group {group.group_index} has mixed problem_id values.")

        if sample.group_index != group.group_index:
            raise ValueError(f"GRPO group {group.group_index} has mixed group_index values.")

        if sample.prompt != expected_prompt:
            raise ValueError(f"GRPO group {group.group_index} has mixed prompts.")

        if sample.rollout_index in rollout_indices:
            raise ValueError(f"GRPO group {group.group_index} has duplicate rollout_index values.")

        rollout_indices.add(sample.rollout_index)

        if sample.policy_adapter_hash != expected_adapter_hash:
            raise ValueError(f"GRPO group {group.group_index} has mixed adapter hashes.")

        if len(sample.token_ids) != len(sample.token_logprobs):
            raise ValueError(
                f"Sample {sample.problem_id}:{sample.rollout_index} "
                "has mismatched token logprobs."
            )

        if len(sample.env_mask) != len(sample.token_ids):
            raise ValueError(
                f"Sample {sample.problem_id}:{sample.rollout_index} "
                "has mismatched env_mask."
            )

        if sum(sample.env_mask) < minimum_trainable_tokens:
            raise ValueError(
                f"Sample {sample.problem_id}:{sample.rollout_index} "
                "has no trainable tokens."
            )

    if len(rewards) < 2:
        group.metadata["zero_variance_reward_warning"] = True


def adapter_state_path(control_dir: Path) -> Path:

    return control_dir / "adapter_state.json"


def stop_signal_path(control_dir: Path) -> Path:

    return control_dir / "stop"


def normalized_service_role(role: str) -> str:

    if role in {
        "rollout",
        "contestant_rollout",
        "rollout_server",
    }:
        return "contestant"

    return role


def service_marker_paths(
    control_dir: Path,
    role: str,
    rank: int,
    state: str,
) -> list[Path]:

    normalized_role = normalized_service_role(role)
    paths = [
        control_dir / f"{normalized_role}_rank_{rank}_{state}.json",
    ]

    if normalized_role == "contestant":
        paths.append(control_dir / f"rollout_rank_{rank}_{state}.json")

    return paths


def write_service_started(
    control_dir: Path,
    role: str,
    rank: int,
    server: AIMOInferenceServer,
    adapter_state: AIMOAdapterState,
) -> None:

    normalized_role = normalized_service_role(role)
    payload = {
        "role": (
            "contestant_rollout"
            if normalized_role == "contestant"
            else normalized_role
        ),
        "model_role": normalized_role,
        "rank": rank,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "host": server.config.host,
        "port": server.config.port,
        "health_url": server.config.health_url,
        "command": server.build_command(),
        "command_path": str(server.command_path),
        "stdout_path": str(server.stdout_path),
        "stderr_path": str(server.stderr_path),
        "service_preflight_path": str(server.config.logdir / "service_preflight.json"),
        "launch_stage": server.launch_stage,
        "update_index": adapter_state.update_index,
        "adapter_hash": adapter_state.adapter_hash,
        "adapter_path": (
            str(adapter_state.adapter_path)
            if adapter_state.adapter_path is not None
            else ""
        ),
        "started_at_unix": time.time(),
    }

    for path in service_marker_paths(
        control_dir=control_dir,
        role=normalized_role,
        rank=rank,
        state="started",
    ):
        write_json(path=path, payload=payload)


def write_service_failed(
    control_dir: Path,
    role: str,
    rank: int,
    server: AIMOInferenceServer | None,
    error: Exception,
) -> None:

    normalized_role = normalized_service_role(role)
    payload = {
        "role": (
            "contestant_rollout"
            if normalized_role == "contestant"
            else normalized_role
        ),
        "model_role": normalized_role,
        "rank": rank,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "failed_at_unix": time.time(),
        "host": server.config.host if server is not None else "",
        "port": server.config.port if server is not None else 0,
        "health_url": server.config.health_url if server is not None else "",
        "command_path": str(server.command_path) if server is not None else "",
        "stdout_path": str(server.stdout_path) if server is not None else "",
        "stderr_path": str(server.stderr_path) if server is not None else "",
        "service_preflight_path": (
            str(server.config.logdir / "service_preflight.json")
            if server is not None
            else ""
        ),
        "launch_stage": server.launch_stage if server is not None else "",
    }

    for path in service_marker_paths(
        control_dir=control_dir,
        role=normalized_role,
        rank=rank,
        state="failed",
    ):
        write_json(path=path, payload=payload)


def find_service_failure_marker(
    control_dir: Path,
    role: str,
    rank: int,
) -> Path | None:

    for path in service_marker_paths(
        control_dir=control_dir,
        role=role,
        rank=rank,
        state="failed",
    ):
        if path.exists():
            return path

    return None


def read_failure_marker(path: Path | None) -> dict[str, object] | None:

    if path is None:
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "path": str(path),
        }


def write_adapter_state(
    control_dir: Path,
    adapter_state: AIMOAdapterState,
) -> None:

    write_json(
        path=adapter_state_path(control_dir),
        payload=adapter_state.as_dict(),
    )


def read_adapter_state(control_dir: Path) -> AIMOAdapterState | None:

    path = adapter_state_path(control_dir)

    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    adapter_path = optional_path(payload.get("adapter_path"))
    adapter_config_path = optional_path(payload.get("adapter_config_path"))

    return AIMOAdapterState(
        update_index=int(payload.get("update_index", 0)),
        adapter_path=adapter_path,
        adapter_config_path=adapter_config_path,
        adapter_hash=str(payload.get("adapter_hash", "")),
        created_at_unix=float(payload.get("created_at_unix", 0.0)),
    )


def write_service_ready(
    control_dir: Path,
    role: str,
    rank: int,
    adapter_state: AIMOAdapterState,
    served_model_name: str,
    health_url: str,
) -> None:

    normalized_role = normalized_service_role(role)
    payload = {
        "role": (
            "contestant_rollout"
            if normalized_role == "contestant"
            else normalized_role
        ),
        "model_role": normalized_role,
        "rank": rank,
        "update_index": adapter_state.update_index,
        "adapter_path": (
            str(adapter_state.adapter_path)
            if adapter_state.adapter_path is not None
            else ""
        ),
        "adapter_hash": adapter_state.adapter_hash,
        "adapter_config_path": (
            str(adapter_state.adapter_config_path)
            if adapter_state.adapter_config_path is not None
            else ""
        ),
        "served_model_name": served_model_name,
        "health_url": health_url,
        "loaded_at_unix": time.time(),
    }

    for path in service_marker_paths(
        control_dir=control_dir,
        role=normalized_role,
        rank=rank,
        state="ready",
    ):
        write_json(path=path, payload=payload)


def wait_for_rollout_adapter_readiness(
    config: AIMOTrainingConfig,
    adapter_state: AIMOAdapterState,
) -> None:

    control_dir = resolve_control_dir(config)
    deadline = time.monotonic() + config.adapter_reload_timeout_seconds
    rollout_ranks = parse_int_list(config.rollout_node_ranks)
    last_error = ""

    while time.monotonic() < deadline:
        ready_ranks = []

        for rank in rollout_ranks:
            if rank == config.global_rank and should_launch_local_rollout_server(config):
                ready_ranks.append(rank)

                continue

            ready_path = next(
                (
                    path
                    for path in service_marker_paths(
                        control_dir=control_dir,
                        role="contestant",
                        rank=rank,
                        state="ready",
                    )
                    if path.exists()
                ),
                None,
            )

            if ready_path is None:
                last_error = f"rollout rank {rank} has not written readiness"
                continue

            payload = json.loads(ready_path.read_text(encoding="utf-8"))

            if int(payload.get("update_index", -1)) != adapter_state.update_index:
                last_error = f"rollout rank {rank} has stale update_index"
                continue

            if str(payload.get("adapter_hash", "")) != adapter_state.adapter_hash:
                last_error = f"rollout rank {rank} has stale adapter_hash"
                continue

            if str(payload.get("adapter_config_path", "")) != (
                str(adapter_state.adapter_config_path)
                if adapter_state.adapter_config_path is not None
                else ""
            ):
                last_error = f"rollout rank {rank} has stale adapter_config_path"
                continue

            if str(payload.get("adapter_path", "")) != (
                str(adapter_state.adapter_path)
                if adapter_state.adapter_path is not None
                else ""
            ):
                last_error = f"rollout rank {rank} has stale adapter_path"
                continue

            if not str(payload.get("served_model_name", "")).strip():
                last_error = f"rollout rank {rank} has no served_model_name"
                continue

            if not str(payload.get("health_url", "")).strip():
                last_error = f"rollout rank {rank} has no health_url"
                continue

            ready_ranks.append(rank)

        if len(ready_ranks) == len(rollout_ranks):
            return

        time.sleep(1.0)

    raise RuntimeError(f"Rollout adapters did not become ready: {last_error}")


def build_adapter_state(
    update_index: int,
    adapter_path: Path | None,
    adapter_config_path: Path | None = None,
) -> AIMOAdapterState:

    resolved_config_path = adapter_config_path

    if resolved_config_path is None and adapter_path is not None:
        candidate_config_path = (
            adapter_path / "adapter_config.json"
            if adapter_path.is_dir()
            else adapter_path.parent / "adapter_config.json"
        )

        if candidate_config_path.exists():
            resolved_config_path = candidate_config_path

    if adapter_path is not None and adapter_path.is_dir():
        adapter_model_path = adapter_path / "adapter_model.safetensors"
        adapter_config_file_path = adapter_path / "adapter_config.json"

        if not adapter_model_path.exists():
            raise FileNotFoundError(f"Missing adapter_model.safetensors under {adapter_path}.")

        if not adapter_config_file_path.exists():
            raise FileNotFoundError(f"Missing adapter_config.json under {adapter_path}.")

        resolved_config_path = adapter_config_file_path

    return AIMOAdapterState(
        update_index=update_index,
        adapter_path=adapter_path,
        adapter_config_path=resolved_config_path,
        adapter_hash=hash_adapter_files(
            adapter_path=adapter_path,
            adapter_config_path=resolved_config_path,
        ),
        created_at_unix=time.time(),
    )


def verify_adapter_directory(adapter_path: Path) -> None:

    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")

    for name in [
        "adapter_model.safetensors",
        "adapter_config.json",
    ]:
        path = adapter_path / name

        if not path.exists():
            raise FileNotFoundError(f"Missing {name} under {adapter_path}.")


def hash_adapter_files(
    adapter_path: Path | None,
    adapter_config_path: Path | None,
) -> str:

    if adapter_path is None or not adapter_path.exists():
        return ""

    digest = hashlib.sha256()
    paths = adapter_hash_paths(
        adapter_path=adapter_path,
        adapter_config_path=adapter_config_path,
    )

    for path in paths:
        digest.update(str(path.name).encode("utf-8"))

        with path.open("rb") as input_file:
            while True:
                chunk = input_file.read(1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

    return digest.hexdigest()


def adapter_hash_paths(
    adapter_path: Path,
    adapter_config_path: Path | None,
) -> list[Path]:

    paths: list[Path] = []

    if adapter_path.is_dir():
        adapter_model_path = adapter_path / "adapter_model.safetensors"
        adapter_config_file_path = adapter_path / "adapter_config.json"

        if not adapter_model_path.exists():
            raise FileNotFoundError(f"Missing adapter_model.safetensors under {adapter_path}.")

        if not adapter_config_file_path.exists():
            raise FileNotFoundError(f"Missing adapter_config.json under {adapter_path}.")

        paths.append(adapter_model_path)
        paths.append(adapter_config_file_path)

    else:
        paths.append(adapter_path)

    if adapter_config_path is not None and adapter_config_path.exists():
        paths.append(adapter_config_path)

    if not paths:
        raise FileNotFoundError(f"No adapter files found under {adapter_path}.")

    return sorted(set(paths))


def optional_path(value: object) -> Path | None:

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return Path(text)


def write_stop_signal(control_dir: Path) -> None:

    stop_signal_path(control_dir).write_text(
        f"{time.time()}\n",
        encoding="utf-8",
    )


def wait_for_stop_signal(control_dir: Path) -> None:

    while not stop_signal_path(control_dir).exists():
        time.sleep(5.0)


def write_json(path: Path, payload: dict[str, object]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    os.replace(temporary_path, path)
