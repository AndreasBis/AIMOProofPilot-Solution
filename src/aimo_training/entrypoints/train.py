from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from aimo_training.artifacts import AIMOTrainingArtifactWriter
from aimo_training.config import AIMOTrainingConfig
from aimo_training.data import build_source_dataset_manifest
from aimo_training.data import read_training_records
from aimo_training.diagnostics import write_failure_diagnostics
from aimo_training.diagnostics import write_phase_event
from aimo_training.queue import AIMODurableGroupQueue
from aimo_training.queue import group_from_dict
from aimo_training.rewards import AIMORewardConfig
from aimo_training.schema import AIMOGRPOGroup
from aimo_training.schema import AIMOTrainingRecord
from aimo_training.trl_trainer import AIMOTRLGRPOTrainer as AIMOGRPOTrainer


def main(argv: list[str] | None = None) -> int:

    config = AIMOTrainingConfig.from_cli_args(argv)
    phase = phase_name_for_config(config)
    started_at = time.monotonic()
    command_arguments = list(argv) if argv is not None else sys.argv[1:]

    try:
        write_phase_event(
            logdir=config.logdir,
            event="phase_started",
            payload={
                "phase": phase,
                "role": config.role,
                "rank": config.global_rank,
            },
        )

        if config.role == "rollout_server":
            from aimo_training.online import run_rollout_node_service

            run_rollout_node_service(config=config)

            return 0

        if config.role == "judge_server":
            from aimo_training.online import run_judge_node_service

            run_judge_node_service(config=config)

            return 0

        if config.role == "train_update":
            run(config=config)

            return 0

        if config.online:
            from aimo_training.online import run_online_training

            run_online_training(config=config)

            return 0

        run(config=config)

        return 0
    except Exception as error:
        write_failure_diagnostics(
            config=config,
            phase=phase,
            error=error,
            started_at_monotonic=started_at,
            command_arguments=command_arguments,
        )

        raise


def phase_name_for_config(config: AIMOTrainingConfig) -> str:

    if config.role == "rollout_server":
        return "vllm_rollout_service"

    if config.role == "judge_server":
        return "vllm_judge_service"

    if config.role == "train_update":
        return "trainer_launch"

    if config.online:
        return "online_controller"

    return "offline_training"


def run(config: AIMOTrainingConfig) -> None:

    artifact_process = is_training_artifact_process()
    artifact_writer = AIMOTrainingArtifactWriter(
        output_path=config.output_path,
        logdir=config.logdir,
    )
    artifact_writer.ensure_directories()
    groups = read_available_groups(
        dataset_path=config.dataset_path,
        queue_path=config.resolved_group_queue_path,
    )
    records = read_records_for_manifest(
        dataset_path=config.dataset_path,
        groups=groups,
    )
    reward_config = AIMORewardConfig(weights=config.reward_weights)

    if artifact_process:
        artifact_writer.write_json(
            "run_metadata.json",
            {
                "created_at_unix": time.time(),
                "config": config.as_dict(),
                "environment": runtime_environment_summary(),
                "resolved_paths": resolved_paths(config=config),
            },
        )
        artifact_writer.write_json("training_arguments.json", config.as_dict())
        artifact_writer.write_json("reward_configuration.json", reward_config.as_dict())
        artifact_writer.write_json(
            "source_dataset_manifest.json",
            build_source_dataset_manifest(
                path=config.dataset_path,
                records=records,
            ),
        )

    validate_complete_groups(groups=groups, group_size=config.group_size)

    if artifact_process:
        artifact_writer.write_group_artifacts(groups)

    trainer = AIMOGRPOTrainer(config=config)
    step_summaries, adapter_path, adapter_config_path = trainer.train(groups)

    if artifact_process:
        for summary in step_summaries:
            artifact_writer.append_jsonl("training_step_summaries.jsonl", summary.as_dict())

        artifact_writer.write_json(
            "tokenizer_manifest.json",
            tokenizer_manifest(config.model_path),
        )
        artifact_writer.write_checkpoint_hashes([
            adapter_path,
            adapter_config_path,
        ])
        artifact_writer.write_json(
            "final_evaluation_summary.json",
            {
                "status": "training_complete",
                "adapter_path": str(adapter_path),
                "adapter_config_path": str(adapter_config_path),
                "trained_group_count": len(groups),
                "trained_sample_count": sum(len(group.samples) for group in groups),
                "local_eval_required": True,
                "mathnet_eval_required": True,
                "base_vs_adapter_comparison_required": True,
            },
        )


def is_training_artifact_process() -> bool:

    return int(os.environ.get("RANK", "0")) == 0


def read_available_groups(dataset_path: Path, queue_path: Path) -> list[AIMOGRPOGroup]:

    if dataset_path.suffix.lower() == ".jsonl":
        return read_groups_from_jsonl(dataset_path)

    if queue_path.exists():
        return AIMODurableGroupQueue(queue_path).read_groups()

    candidate_path = dataset_path / "grpo_groups.jsonl" if dataset_path.is_dir() else None

    if candidate_path is not None and candidate_path.exists():
        return read_groups_from_jsonl(candidate_path)

    raise FileNotFoundError(
        "No complete GRPO group queue was found. "
        f"Expected {queue_path} or a grpo_groups.jsonl dataset."
    )


def read_groups_from_jsonl(path: Path) -> list[AIMOGRPOGroup]:

    groups = []

    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            groups.append(group_from_dict(json.loads(stripped_line)))

    return groups


def read_records_for_manifest(
    dataset_path: Path,
    groups: list[AIMOGRPOGroup],
) -> list[AIMOTrainingRecord]:

    group_dataset_path = dataset_path.suffix.lower() == ".jsonl" or (
        dataset_path.is_dir()
        and (dataset_path / "grpo_groups.jsonl").exists()
    )

    if not group_dataset_path:
        return read_training_records(dataset_path)

    records_by_id: dict[str, AIMOTrainingRecord] = {}

    for group in groups:
        if group.problem_id in records_by_id:
            continue

        records_by_id[group.problem_id] = AIMOTrainingRecord(
            order_index=len(records_by_id),
            id=group.problem_id,
            problem=group.problem,
            reference_solution=group.reference_solution,
            metadata=group.metadata,
        )

    return list(records_by_id.values())


def validate_complete_groups(groups: list[AIMOGRPOGroup], group_size: int) -> None:

    if not groups:
        raise ValueError("At least one complete GRPO group is required.")

    for group in groups:
        if len(group.samples) != group_size:
            raise ValueError(
                f"GRPO group {group.group_index} has {len(group.samples)} samples, "
                f"but group_size is {group_size}."
            )


def tokenizer_manifest(model_path: Path) -> dict[str, object]:

    tokenizer_files = sorted([
        path
        for path in model_path.iterdir()
        if path.is_file() and path.name in {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
        }
    ])

    return {
        "model_path": str(model_path),
        "tokenizer_files": [
            str(path)
            for path in tokenizer_files
        ],
    }


def runtime_environment_summary() -> dict[str, str]:

    environment_names = [
        "GLOBAL_RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "AIMO_NUM_GPUS",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
    ]

    return {
        name: os.environ[name]
        for name in environment_names
        if name in os.environ
    }


def resolved_paths(config: AIMOTrainingConfig) -> dict[str, str]:

    return {
        "model_path": str(config.model_path),
        "dataset_path": str(config.dataset_path),
        "output_path": str(config.output_path),
        "logdir": str(config.logdir),
        "group_queue_path": str(config.resolved_group_queue_path),
        "judge_model_path": str(config.judge_model_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
