from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aimo_inference.defaults import DEFAULT_DUMMY_TEST
from aimo_training.artifacts import AIMOTrainingArtifactWriter
from aimo_training.config import AIMOTrainingConfig
from aimo_training.entrypoints.train import validate_complete_groups
from aimo_training.queue import AIMODurableGroupQueue
from aimo_training.rollout import build_judge_inference_config
from aimo_training.rollout import build_rollout_inference_config
from aimo_training.trainer import batch_groups
from aimo_training.trainer import encode_prompt_and_completion
from aimo_training.trainer import normalize_rewards
from aimo_training.trl_trainer import AIMOTRLQueuedRolloutFunction
from aimo_training.trl_trainer import build_grpo_config
from aimo_training.trl_trainer import validate_grpo_batch_contract
from conftest import grpo_group
from conftest import rollout_sample


class Tokenizer:

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:

        return [
            ord(character) % 17
            for character in text
        ]


class FakeTRLTrainer:

    num_generations = 2


class FakeGRPOConfig:

    def __init__(self, **kwargs: object) -> None:

        self.kwargs = kwargs


def training_config(tmp_path: Path) -> AIMOTrainingConfig:

    return AIMOTrainingConfig(
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "groups.jsonl",
        output_path=tmp_path / "output",
        logdir=tmp_path / "logs",
        group_size=2,
        page_count_method="word_count",
    )


def test_training_cli_required_args_defaults_and_distributed_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setenv("GLOBAL_RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    config = AIMOTrainingConfig.from_cli_args([
        "--model_path",
        str(tmp_path / "model"),
        "--dataset_path",
        str(tmp_path / "dataset.jsonl"),
        "--output_path",
        str(tmp_path / "output"),
        "--logdir",
        str(tmp_path / "logs"),
    ])

    assert config.learning_rate == 5e-6
    assert config.group_size == 16
    assert config.active_problem_count == 6
    assert config.sandbox_count == 96
    assert config.kv_cache_dtype == "auto"
    assert config.rollout_mode == "queued"
    assert config.tool_protocol == "olmo_chatml"
    assert config.global_rank == 1
    assert config.world_size == 2
    assert config.train_processes_per_node == 8
    assert config.train_sharding_strategy == "fsdp_full_shard"
    assert config.dummy_test is DEFAULT_DUMMY_TEST


def test_training_cli_reads_dummy_test_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setenv("AIMO_DUMMY_TEST", "true")
    monkeypatch.setenv("AIMO_DUMMY_MODEL_PATH", str(tmp_path / "models" / "dummy"))

    config = AIMOTrainingConfig.from_cli_args([
        "--model_path",
        str(tmp_path / "models" / "contestant"),
        "--dataset_path",
        str(tmp_path / "dataset.jsonl"),
        "--output_path",
        str(tmp_path / "output"),
        "--logdir",
        str(tmp_path / "logs"),
        "--judge_model_path",
        str(tmp_path / "models" / "judge"),
    ])
    expected_dummy_path = tmp_path / "models" / "dummy"

    assert config.dummy_test is True
    assert config.model_path == expected_dummy_path
    assert config.judge_model_path == expected_dummy_path
    assert config.dummy_model_path == expected_dummy_path


def test_grpo_config_enables_fsdp_full_shard_under_torchrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(GRPOConfig=FakeGRPOConfig))
    monkeypatch.setenv("WORLD_SIZE", "8")

    config = build_grpo_config(training_config(tmp_path))

    assert config.kwargs["fsdp"] == "full_shard auto_wrap"
    assert config.kwargs["fsdp_config"] == {
        "activation_checkpointing": True,
        "sync_module_states": True,
        "transformer_layer_cls_to_wrap": "Olmo3DecoderLayer",
        "use_orig_params": True,
    }
    assert config.kwargs["use_vllm"] is False
    assert config.kwargs["bf16"] is True


def test_grpo_config_omits_fsdp_outside_torchrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(GRPOConfig=FakeGRPOConfig))
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    config = build_grpo_config(training_config(tmp_path))

    assert "fsdp" not in config.kwargs
    assert config.kwargs["use_vllm"] is False


def test_reward_weights_json_merges_defaults(tmp_path: Path) -> None:

    config = AIMOTrainingConfig(
        model_path=tmp_path / "model",
        dataset_path=tmp_path / "dataset.jsonl",
        output_path=tmp_path / "output",
        logdir=tmp_path / "logs",
        reward_weights_json="{\"context_reward\": 2.5}",
    )

    assert config.reward_weights == {
        "judge_grade": 1.0,
        "context_reward": 2.5,
        "solution_page_reward": 1.0,
    }

    with pytest.raises(ValueError, match="Unknown reward weight"):
        AIMOTrainingConfig(
            model_path=tmp_path / "model",
            dataset_path=tmp_path / "dataset.jsonl",
            output_path=tmp_path / "output",
            logdir=tmp_path / "logs",
            reward_weights_json="{\"unknown\": 1}",
        ).reward_weights


def test_normalize_rewards_and_batch_groups() -> None:

    groups = [
        grpo_group(problem_id=f"p{index}")
        for index in range(3)
    ]

    assert normalize_rewards([
        1.0,
        1.0,
    ]) == [
        0.0,
        0.0,
    ]
    assert normalize_rewards([
        0.0,
        2.0,
    ]) == [
        -1.0,
        1.0,
    ]
    assert [
        len(batch)
        for batch in batch_groups(groups, batch_size=2)
    ] == [
        2,
        1,
    ]


def test_encode_prompt_and_completion_uses_token_ids_and_truncates() -> None:

    sample = grpo_group(sample_count=1).samples[0]

    encoded = encode_prompt_and_completion(
        tokenizer=Tokenizer(),
        sample=sample,
        max_model_len=4,
    )

    assert encoded["input_ids"] == [
        15,
        10,
        11,
        12,
    ]
    assert encoded["labels"] == [
        -100,
        10,
        11,
        12,
    ]


def test_encode_prompt_and_completion_masks_tool_output_tokens() -> None:

    sample = rollout_sample()
    masked_sample = sample.__class__(
        **{
            **sample.as_dict(),
            "token_ids": [
                10,
                99,
                11,
            ],
            "token_logprobs": [
                -0.3,
                0.0,
                -0.1,
            ],
            "env_mask": [
                1,
                0,
                1,
            ],
            "reward": sample.reward,
        }
    )

    encoded = encode_prompt_and_completion(
        tokenizer=Tokenizer(),
        sample=masked_sample,
        max_model_len=20,
    )

    assert encoded["input_ids"][-3:] == [
        10,
        99,
        11,
    ]
    assert encoded["labels"][-3:] == [
        10,
        -100,
        11,
    ]


def test_durable_group_queue_round_trip(tmp_path: Path) -> None:

    queue = AIMODurableGroupQueue(path=tmp_path / "groups" / "grpo_groups.jsonl")
    group = grpo_group(sample_count=2)

    queue.append_group(group)
    groups = queue.read_groups()

    assert len(groups) == 1
    assert groups[0].problem_id == group.problem_id
    assert len(groups[0].samples) == 2
    assert groups[0].samples[0].reward.scalar_reward == 0.0


def test_training_artifact_writer_outputs_manifests_and_tables(tmp_path: Path) -> None:

    writer = AIMOTrainingArtifactWriter(
        output_path=tmp_path / "adapter",
        logdir=tmp_path / "logs",
    )
    writer.ensure_directories()
    group = grpo_group(sample_count=2)
    checkpoint_path = tmp_path / "adapter" / "adapter_config.json"
    checkpoint_path.write_text("{\"rank\": 64}\n", encoding="utf-8")

    writer.write_json(
        "run_metadata.json",
        {
            "status": "ok",
        },
    )
    writer.write_group_artifacts([
        group,
    ])
    writer.write_checkpoint_hashes([
        checkpoint_path,
    ])

    assert json.loads((tmp_path / "logs" / "run_metadata.json").read_text(
        encoding="utf-8",
    )) == {
        "status": "ok",
    }
    assert (tmp_path / "logs" / "per_step_reward_summaries.jsonl").exists()
    assert (tmp_path / "logs" / "training_table.jsonl").exists()
    assert (tmp_path / "logs" / "gradient_update_reward_summaries.jsonl").exists()
    assert (tmp_path / "logs" / "gradient_update_reward_samples.jsonl").exists()
    assert (tmp_path / "logs" / "judge_parse_failures.jsonl").exists()
    assert str(checkpoint_path) in json.loads(
        (tmp_path / "logs" / "checkpoint_hashes.json").read_text(encoding="utf-8")
    )


def test_rollout_and_judge_inference_configs(tmp_path: Path) -> None:

    config = training_config(tmp_path)
    rollout_config = build_rollout_inference_config(config)
    judge_config = build_judge_inference_config(config)

    assert rollout_config.inference_mode == "proof"
    assert rollout_config.launch_server is False
    assert rollout_config.reuse_server is True
    assert rollout_config.group_size == 2
    assert rollout_config.port == config.rollout_port
    assert rollout_config.tensor_parallel_size == 8
    assert rollout_config.num_gpus == 8
    assert rollout_config.template_format == "chatml"
    assert rollout_config.tool_protocol == "olmo_chatml"
    assert rollout_config.top_logprobs == 0
    assert rollout_config.max_logprobs == 0
    assert judge_config.inference_mode == "judge"
    assert judge_config.tensor_parallel_size == 8
    assert judge_config.num_gpus == 8
    assert judge_config.template_format == "harmony"
    assert judge_config.tool_protocol == "harmony"
    assert judge_config.port == config.judge_port
    assert judge_config.moe_backend == "marlin"
    assert judge_config.enable_expert_parallel is True


def test_dummy_rollout_and_judge_inference_configs_use_smol_lm_chatml(tmp_path: Path) -> None:

    config = AIMOTrainingConfig(
        model_path=tmp_path / "models" / "contestant",
        dataset_path=tmp_path / "dataset.jsonl",
        output_path=tmp_path / "output",
        logdir=tmp_path / "logs",
        judge_model_path=tmp_path / "models" / "judge",
        dummy_test=True,
        group_size=2,
        page_count_method="word_count",
    )
    expected_dummy_path = tmp_path / "models" / "dummy"
    rollout_config = build_rollout_inference_config(config)
    judge_config = build_judge_inference_config(config)

    assert rollout_config.model_path == expected_dummy_path
    assert rollout_config.served_model_name == "SmolLM-3B"
    assert rollout_config.tensor_parallel_size == 2
    assert rollout_config.num_gpus == 2
    assert rollout_config.template_format == "chatml"
    assert judge_config.model_path == expected_dummy_path
    assert judge_config.judge_model_path == expected_dummy_path
    assert judge_config.served_model_name == "SmolLM-3B"
    assert judge_config.judge_served_model_name == "SmolLM-3B"
    assert judge_config.tensor_parallel_size == 2
    assert judge_config.num_gpus == 2
    assert judge_config.template_format == "chatml"
    assert judge_config.tool_protocol == "olmo_chatml"
    assert judge_config.moe_backend == ""
    assert judge_config.enable_expert_parallel is False


def test_partial_groups_are_rejected_before_training(tmp_path: Path) -> None:

    complete_group = grpo_group(sample_count=2)
    partial_group = grpo_group(sample_count=1)

    validate_complete_groups(
        groups=[
            complete_group,
        ],
        group_size=2,
    )

    with pytest.raises(ValueError, match="has 1 samples"):
        validate_complete_groups(
            groups=[
                partial_group,
            ],
            group_size=2,
        )


def test_rollout_sample_validation_rejects_missing_logprobs(tmp_path: Path) -> None:

    group = grpo_group(sample_count=2)
    broken_sample = group.samples[0]
    group.samples[0] = broken_sample.__class__(
        **{
            **broken_sample.as_dict(),
            "token_logprobs": [],
            "reward": broken_sample.reward,
        }
    )

    with pytest.raises(ValueError, match="selected-token logprobs"):
        validate_grpo_batch_contract(
            config=training_config(tmp_path),
            groups=[
                group,
            ],
        )


def test_rollout_sample_validation_rejects_zero_trainable_tokens(tmp_path: Path) -> None:

    group = grpo_group(sample_count=2)

    for index, sample in enumerate(group.samples):
        group.samples[index] = sample.__class__(
            **{
                **sample.as_dict(),
                "env_mask": [
                    0,
                    0,
                    0,
                ],
                "reward": sample.reward,
            }
        )

    with pytest.raises(ValueError, match="no trainable tokens"):
        validate_grpo_batch_contract(
            config=training_config(tmp_path),
            groups=[
                group,
            ],
        )


def test_group_validation_rejects_duplicate_rollout_indices(tmp_path: Path) -> None:

    group = grpo_group(sample_count=2)
    duplicate_sample = group.samples[1]
    group.samples[1] = duplicate_sample.__class__(
        **{
            **duplicate_sample.as_dict(),
            "rollout_index": group.samples[0].rollout_index,
            "reward": duplicate_sample.reward,
        }
    )

    with pytest.raises(ValueError, match="duplicate rollout_index"):
        validate_grpo_batch_contract(
            config=training_config(tmp_path),
            groups=[
                group,
            ],
        )


def test_group_validation_rejects_mixed_adapter_hashes(tmp_path: Path) -> None:

    group = grpo_group(sample_count=2)

    for index, sample in enumerate(group.samples):
        group.samples[index] = sample.__class__(
            **{
                **sample.as_dict(),
                "policy_adapter_hash": f"hash-{index}",
                "reward": sample.reward,
            }
        )

    with pytest.raises(ValueError, match="mixed adapter hashes"):
        validate_grpo_batch_contract(
            config=training_config(tmp_path),
            groups=[
                group,
            ],
        )


def test_queued_rollout_returns_exact_sample_count_for_unique_prompt_contract(
    tmp_path: Path,
) -> None:

    group = grpo_group(sample_count=2)
    rollout_function = AIMOTRLQueuedRolloutFunction(
        config=training_config(tmp_path),
        groups=[
            group,
        ],
        tokenizer=Tokenizer(),
    )

    payload = rollout_function(
        prompts=[
            group.samples[0].prompt,
        ],
        trainer=FakeTRLTrainer(),
    )

    assert len(payload["completion_ids"]) == 2
    assert payload["sampling_logprobs"] == payload["logprobs"]


def test_queued_rollout_returns_exact_sample_count_for_generation_occurrence_contract(
    tmp_path: Path,
) -> None:

    group = grpo_group(sample_count=2)
    rollout_function = AIMOTRLQueuedRolloutFunction(
        config=training_config(tmp_path),
        groups=[
            group,
        ],
        tokenizer=Tokenizer(),
    )

    payload = rollout_function(
        prompts=[
            group.samples[0].prompt,
            group.samples[0].prompt,
        ],
        trainer=FakeTRLTrainer(),
    )

    assert len(payload["completion_ids"]) == 2
