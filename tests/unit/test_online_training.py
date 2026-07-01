from __future__ import annotations

import json
import urllib.error
from dataclasses import replace
from pathlib import Path

from aimo_inference.config import AIMOConfig
from aimo_training.config import AIMOTrainingConfig
from aimo_training.online import AIMORolloutEndpoint
from aimo_training.online import AIMOInterleavedRolloutCoordinator
from aimo_training.online import AIMORolloutAttemptResult
from aimo_training.online import build_adapter_state
from aimo_training.online import build_distributed_training_command
from aimo_training.online import poll_online_service_statuses
from aimo_training.online import parse_int_list
from aimo_training.online import resolve_online_service_targets
from aimo_training.online import resolve_judge_api_base
from aimo_training.online import resolve_rollout_api_bases
from aimo_training.online import should_launch_distributed_training
from aimo_training.online import training_phase_cli_args
from aimo_training.online import wait_for_online_services
from aimo_training.online import wait_for_rollout_adapter_readiness
from aimo_training.online import write_json
from aimo_training.online import write_service_failed
from aimo_training.online import write_service_ready
from aimo_training.queue import AIMODurableGroupQueue
from aimo_training.schema import AIMORolloutSample
from aimo_training.schema import AIMOTrainingRecord
from conftest import rollout_sample


class FakeInterleavedRolloutCoordinator(AIMOInterleavedRolloutCoordinator):

    def __init__(self, config: AIMOTrainingConfig) -> None:

        self.config = config
        self.adapter_state = build_adapter_state(
            update_index=0,
            adapter_path=None,
        )
        self.endpoints = [
            AIMORolloutEndpoint(
                endpoint_index=0,
                api_base="http://rollout-0/v1",
                config=AIMOConfig(),
                client=None,
            ),
            AIMORolloutEndpoint(
                endpoint_index=1,
                api_base="http://rollout-1/v1",
                config=AIMOConfig(),
                client=None,
            ),
        ]

    def build_sample(
        self,
        endpoint: AIMORolloutEndpoint,
        record: AIMOTrainingRecord,
        group_index: int,
        rollout_index: int,
        attempt_index: int = 0,
    ) -> AIMORolloutAttemptResult:

        sample = rollout_sample(
            problem_id=record.id,
            group_index=group_index,
            rollout_index=rollout_index,
        )

        return AIMORolloutAttemptResult(
            sample=sample.__class__(
                **{
                    **sample.as_dict(),
                    "policy_update_index": self.adapter_state.update_index,
                    "policy_adapter_hash": self.adapter_state.adapter_hash,
                    "policy_adapter_path": str(self.adapter_state.adapter_path or ""),
                    "reward": sample.reward,
                }
            ),
            failure_metadata={},
        )


def training_config(tmp_path: Path) -> AIMOTrainingConfig:

    return AIMOTrainingConfig(
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "dataset.parquet",
        output_path=tmp_path / "adapter",
        logdir=tmp_path / "logs",
        online=True,
        group_size=2,
        active_problem_count=2,
        sandbox_count=4,
        problems_per_update=4,
        rollout_node_ranks="1",
        judge_node_rank=0,
        trainer_node_rank=2,
    )


def training_records(count: int) -> list[AIMOTrainingRecord]:

    return [
        AIMOTrainingRecord(
            order_index=index,
            id=f"p{index}",
            problem=f"Problem {index}",
            reference_solution=f"Reference {index}",
            metadata={},
        )
        for index in range(count)
    ]


def test_online_config_cli_accepts_update_and_topology_flags(tmp_path: Path) -> None:

    config = AIMOTrainingConfig.from_cli_args([
        "--model_path",
        str(tmp_path / "model"),
        "--dataset_path",
        str(tmp_path / "dataset.parquet"),
        "--output_path",
        str(tmp_path / "adapter"),
        "--logdir",
        str(tmp_path / "logs"),
        "--online",
        "true",
        "--problems_per_update",
        "64",
        "--node_hostnames",
        "node0,node1,node2",
    ])

    assert config.online is True
    assert config.rollout_mode == "queued"
    assert config.tool_protocol == "olmo_chatml"
    assert config.problems_per_update == 64
    assert config.node_hostnames == "node0,node1,node2"
    assert config.train_processes_per_node == 8
    assert config.train_sharding_strategy == "fsdp_full_shard"
    assert parse_int_list(config.rollout_node_ranks) == [
        1,
    ]
    assert config.judge_node_rank == 0
    assert config.trainer_node_rank == 2
    assert config.judge_port == 8000
    assert config.rollout_port == 8001


def test_online_config_reads_service_topology_env(
    tmp_path: Path,
    monkeypatch: object,
) -> None:

    monkeypatch.setenv("AIMO_SERVICE_RANK", "2")
    monkeypatch.setenv("AIMO_SERVICE_WORLD_SIZE", "3")

    config = AIMOTrainingConfig.from_cli_args([
        "--model_path",
        str(tmp_path / "model"),
        "--dataset_path",
        str(tmp_path / "dataset.parquet"),
        "--output_path",
        str(tmp_path / "adapter"),
        "--logdir",
        str(tmp_path / "logs"),
        "--node_hostnames",
        "node0,node1,node2",
    ])

    assert config.global_rank == 2
    assert config.world_size == 3


def test_online_topology_resolves_olmo_nodes_and_judge_node(tmp_path: Path) -> None:

    config = AIMOTrainingConfig(
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "dataset.parquet",
        output_path=tmp_path / "adapter",
        logdir=tmp_path / "logs",
        world_size=3,
        node_hostnames="node0,node1,node2",
    )

    assert resolve_rollout_api_bases(config) == [
        "http://node1:8001/v1",
    ]
    assert resolve_judge_api_base(config) == "http://node0:8000/v1"


def test_online_training_phase_launches_eight_rank_train_update(
    tmp_path: Path,
) -> None:

    adapter_path = tmp_path / "adapter_state"
    queue_path = tmp_path / "chunk" / "grpo_groups.jsonl"
    config = replace(
        training_config(tmp_path),
        dataset_path=queue_path,
        output_path=tmp_path / "adapter_chunk",
        logdir=tmp_path / "training_logs",
        group_queue_path=queue_path,
        initial_adapter_path=adapter_path,
        train_processes_per_node=8,
    )

    command = build_distributed_training_command(config)
    arguments = training_phase_cli_args(config)

    assert "--nproc-per-node" in command
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert "--module" in command
    assert command[command.index("--module") + 1] == "aimo_training.entrypoints.train"
    assert arguments[arguments.index("--role") + 1] == "train_update"
    assert arguments[arguments.index("--dataset_path") + 1] == str(queue_path)
    assert arguments[arguments.index("--group_queue_path") + 1] == str(queue_path)
    assert arguments[arguments.index("--initial_adapter_path") + 1] == str(adapter_path)
    assert arguments[arguments.index("--judge_node_rank") + 1] == "0"
    assert arguments[arguments.index("--rollout_node_ranks") + 1] == "1"
    assert arguments[arguments.index("--judge_port") + 1] == "8000"
    assert arguments[arguments.index("--rollout_port") + 1] == "8001"
    assert arguments[arguments.index("--train_sharding_strategy") + 1] == "fsdp_full_shard"
    assert (
        arguments[arguments.index("--train_fsdp_transformer_layer_cls_to_wrap") + 1]
        == "Olmo3DecoderLayer"
    )


def test_training_phase_cli_args_preserve_dummy_test_paths(tmp_path: Path) -> None:

    config = AIMOTrainingConfig(
        model_path=tmp_path / "models" / "contestant",
        dataset_path=tmp_path / "dataset.parquet",
        output_path=tmp_path / "adapter",
        logdir=tmp_path / "logs",
        judge_model_path=tmp_path / "models" / "judge",
        dummy_test=True,
        online=True,
    )
    arguments = training_phase_cli_args(config)
    expected_dummy_path = tmp_path / "models" / "dummy"

    assert arguments[arguments.index("--model_path") + 1] == str(expected_dummy_path)
    assert arguments[arguments.index("--judge_model_path") + 1] == str(expected_dummy_path)
    assert arguments[arguments.index("--dummy_test") + 1] == "true"
    assert arguments[arguments.index("--dummy_model_path") + 1] == str(expected_dummy_path)


def test_distributed_training_launch_is_skipped_inside_torchrun(
    tmp_path: Path,
    monkeypatch: object,
) -> None:

    config = training_config(tmp_path)

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

    assert should_launch_distributed_training(config) is True

    monkeypatch.setenv("LOCAL_RANK", "0")

    assert should_launch_distributed_training(config) is True

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")

    assert should_launch_distributed_training(config) is False


def test_interleaved_rollout_only_admits_chunk_problem_count(tmp_path: Path) -> None:

    config = training_config(tmp_path)
    coordinator = FakeInterleavedRolloutCoordinator(config)
    queue_path = tmp_path / "chunk" / "grpo_groups.jsonl"

    summary = coordinator.write_chunk(
        records=training_records(4),
        queue_path=queue_path,
        starting_group_index=0,
    )
    groups = AIMODurableGroupQueue(queue_path).read_groups()

    assert summary.admitted_problem_count == 4
    assert summary.completed_group_count == 4
    assert summary.written_sample_count == 8
    assert sorted([
        group.problem_id
        for group in groups
    ]) == [
        "p0",
        "p1",
        "p2",
        "p3",
    ]
    assert all(
        len(group.samples) == config.group_size
        for group in groups
    )


def test_interleaved_rollout_records_adapter_version(tmp_path: Path) -> None:

    adapter_path = tmp_path / "adapter_state"
    adapter_path.mkdir()
    (adapter_path / "adapter_model.safetensors").write_text("adapter\n", encoding="utf-8")
    (adapter_path / "adapter_config.json").write_text("{\"rank\": 64}\n", encoding="utf-8")
    config = training_config(tmp_path)
    coordinator = FakeInterleavedRolloutCoordinator(config)
    coordinator.adapter_state = build_adapter_state(
        update_index=3,
        adapter_path=adapter_path,
    )
    queue_path = tmp_path / "chunk" / "grpo_groups.jsonl"

    coordinator.write_chunk(
        records=training_records(1),
        queue_path=queue_path,
        starting_group_index=0,
    )
    groups = AIMODurableGroupQueue(queue_path).read_groups()

    assert {
        sample.policy_update_index
        for sample in groups[0].samples
    } == {
        3,
    }
    assert {
        sample.policy_adapter_hash
        for sample in groups[0].samples
    } == {
        coordinator.adapter_state.adapter_hash,
    }


def test_failed_rollout_is_replaced_before_group_write(tmp_path: Path) -> None:

    class ReplacementCoordinator(FakeInterleavedRolloutCoordinator):

        def build_sample(
            self,
            endpoint: AIMORolloutEndpoint,
            record: AIMOTrainingRecord,
            group_index: int,
            rollout_index: int,
            attempt_index: int = 0,
        ) -> AIMORolloutAttemptResult:

            if rollout_index == 0:
                return AIMORolloutAttemptResult(
                    sample=None,
                    failure_metadata={
                        "exception": "failed request",
                    },
                )

            return AIMORolloutAttemptResult(
                sample=rollout_sample(
                    problem_id=record.id,
                    group_index=group_index,
                    rollout_index=rollout_index,
                ),
                failure_metadata={},
            )

    config = training_config(tmp_path)
    coordinator = ReplacementCoordinator(config)
    queue_path = tmp_path / "chunk" / "grpo_groups.jsonl"

    summary = coordinator.write_chunk(
        records=training_records(1),
        queue_path=queue_path,
        starting_group_index=0,
    )
    groups = AIMODurableGroupQueue(queue_path).read_groups()

    assert summary.completed_group_count == 1
    assert summary.skipped_group_count == 0
    assert sorted([
        sample.rollout_index
        for sample in groups[0].samples
    ]) == [
        1,
        2,
    ]
    assert (queue_path.parent / "skipped_groups.jsonl").exists() is False


def test_poisoned_group_is_skipped_after_retry_budget(tmp_path: Path) -> None:

    class FailingCoordinator(FakeInterleavedRolloutCoordinator):

        def build_sample(
            self,
            endpoint: AIMORolloutEndpoint,
            record: AIMOTrainingRecord,
            group_index: int,
            rollout_index: int,
            attempt_index: int = 0,
        ) -> AIMORolloutAttemptResult:

            return AIMORolloutAttemptResult(
                sample=None,
                failure_metadata={
                    "exception": "failed request",
                },
            )

    config = training_config(tmp_path)
    config = replace(
        config,
        max_rollout_retries_per_sample=0,
        max_group_replacement_attempts=2,
    )
    coordinator = FailingCoordinator(config)
    queue_path = tmp_path / "chunk" / "grpo_groups.jsonl"

    summary = coordinator.write_chunk(
        records=training_records(1),
        queue_path=queue_path,
        starting_group_index=0,
    )

    assert summary.completed_group_count == 0
    assert summary.skipped_group_count == 1
    assert queue_path.exists() is False
    assert (queue_path.parent / "rollout_failures.jsonl").exists()
    assert (queue_path.parent / "skipped_groups.jsonl").exists()


def test_adapter_readiness_rejects_stale_adapter_hash(tmp_path: Path) -> None:

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_model.safetensors").write_text("new\n", encoding="utf-8")
    (adapter_path / "adapter_config.json").write_text("{\"rank\": 64}\n", encoding="utf-8")
    config = AIMOTrainingConfig(
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "dataset.parquet",
        output_path=tmp_path / "output",
        logdir=tmp_path / "logs",
        world_size=3,
        global_rank=2,
        node_hostnames="node0,node1,node2",
        online_control_dir=tmp_path / "control",
        adapter_reload_timeout_seconds=0.01,
    )
    adapter_state = build_adapter_state(
        update_index=1,
        adapter_path=adapter_path,
    )
    stale_adapter_state = build_adapter_state(
        update_index=1,
        adapter_path=None,
    )

    write_service_ready(
        control_dir=config.online_control_dir,
        role="contestant",
        rank=1,
        adapter_state=stale_adapter_state,
        served_model_name="OLMo-3.1-32B-Think",
        health_url="http://node1:8001/health",
    )

    try:
        wait_for_rollout_adapter_readiness(
            config=config,
            adapter_state=adapter_state,
        )
    except RuntimeError as error:
        assert "stale adapter_hash" in str(error)
    else:
        raise AssertionError("stale adapter readiness was accepted")


def test_online_service_targets_match_requested_topology(tmp_path: Path) -> None:

    config = AIMOTrainingConfig(
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "dataset.parquet",
        output_path=tmp_path / "adapter",
        logdir=tmp_path / "logs",
        world_size=3,
        node_hostnames="judge-node,contestant-node,trainer-node",
        online=True,
    )

    targets = resolve_online_service_targets(config)

    assert [
        (target.role, target.rank, target.port, target.api_base)
        for target in targets
    ] == [
        ("contestant", 1, 8001, "http://contestant-node:8001/v1"),
        ("judge", 0, 8000, "http://judge-node:8000/v1"),
    ]


def test_controller_fails_fast_on_service_failure_marker(
    tmp_path: Path,
    monkeypatch: object,
) -> None:

    config = AIMOTrainingConfig(
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "dataset.parquet",
        output_path=tmp_path / "adapter",
        logdir=tmp_path / "logs",
        world_size=3,
        node_hostnames="judge-node,contestant-node,trainer-node",
        online=True,
        online_control_dir=tmp_path / "control",
    )

    def fail_urlopen(*_: object, **__: object) -> object:

        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("aimo_training.online.urllib.request.urlopen", fail_urlopen)
    write_json(
        path=tmp_path / "control" / "contestant_rank_1_failed.json",
        payload={
            "exception_message": "vllm stderr tail",
        },
    )

    try:
        wait_for_online_services(config=config)
    except RuntimeError as error:
        message = str(error)
        assert "failure marker" in message
        assert "contestant rank=1" in message
        assert "judge rank=0" in message
        assert "vllm stderr tail" in message
    else:
        raise AssertionError("failure marker did not stop health polling")


def test_aggregate_health_status_reports_every_unhealthy_service(
    tmp_path: Path,
    monkeypatch: object,
) -> None:

    config = AIMOTrainingConfig(
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "dataset.parquet",
        output_path=tmp_path / "adapter",
        logdir=tmp_path / "logs",
        world_size=3,
        node_hostnames="judge-node,contestant-node,trainer-node",
        online=True,
        online_control_dir=tmp_path / "control",
    )

    def fail_urlopen(*_: object, **__: object) -> object:

        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("aimo_training.online.urllib.request.urlopen", fail_urlopen)
    statuses = poll_online_service_statuses(
        config=config,
        targets=resolve_online_service_targets(config),
    )

    assert [
        status.target.role
        for status in statuses
    ] == [
        "contestant",
        "judge",
    ]
    assert all(status.healthy is False for status in statuses)
    assert all("connection refused" in status.last_error for status in statuses)


def test_service_failed_marker_contains_traceback_and_role_names(tmp_path: Path) -> None:

    try:
        raise RuntimeError("service crashed")
    except RuntimeError as error:
        write_service_failed(
            control_dir=tmp_path / "control",
            role="contestant",
            rank=1,
            server=None,
            error=error,
        )

    payload = json.loads(
        (tmp_path / "control" / "contestant_rank_1_failed.json").read_text(
            encoding="utf-8",
        )
    )

    assert payload["role"] == "contestant_rollout"
    assert payload["model_role"] == "contestant"
    assert payload["rank"] == 1
    assert payload["exception_type"] == "RuntimeError"
    assert "service crashed" in payload["traceback"]
    assert (tmp_path / "control" / "rollout_rank_1_failed.json").exists()
