from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from aimo_inference.config import DEFAULT_PAGE_TEMPLATE
from aimo_inference.config import MAX_GENERATION_CONTEXT_TOKENS
from aimo_inference.defaults import DEFAULT_DUMMY_TEST
from aimo_inference.defaults import DUMMY_MODEL_PATH
from aimo_inference.defaults import DUMMY_TENSOR_PARALLEL_SIZE


DEFAULT_REWARD_WEIGHTS = {
    "judge_grade": 1.0,
    "context_reward": 1.0,
    "solution_page_reward": 1.0,
}

DEFAULT_TARGET_MODULE_SUFFIXES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "query",
    "key",
    "value",
    "dense",
    "fc1",
    "fc2",
]


@dataclass(frozen=True)
class AIMOTrainingConfig:

    model_path: Path
    dataset_path: Path
    output_path: Path
    logdir: Path
    role: str = "controller"
    online: bool = False
    allow_base_rollouts: bool = False
    rollout_mode: str = "queued"
    tool_protocol: str = "olmo_chatml"
    num_gpus: int = 1
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    lora_rank: int = 64
    lora_alpha: int = 128
    max_model_len: int = MAX_GENERATION_CONTEXT_TOKENS
    group_size: int = 16
    judge_model_path: Path = Path("models/judge")
    dummy_test: bool = False
    dummy_model_path: Path = DUMMY_MODEL_PATH
    judge_port: int = 8000
    rollout_port: int = 8001
    rollout_temperature: float = 0.6
    rollout_top_p: float = 0.95
    rollout_min_p: float = 0.0
    max_python_calls: int = 64
    reward_weights_json: str = ""
    active_problem_count: int = 6
    sandbox_count: int = 96
    kv_cache_dtype: str = "auto"
    page_count_method: str = "latex"
    page_template: str = DEFAULT_PAGE_TEMPLATE
    importance_sampling_level: str = "token"
    kl_beta: float = 0.001
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    grpo_epsilon: float = 0.2
    seed: int = 42
    train_processes_per_node: int = 8
    train_sharding_strategy: str = "fsdp_full_shard"
    train_bf16: bool = True
    train_gradient_checkpointing: bool = True
    train_fsdp_transformer_layer_cls_to_wrap: str = "Olmo3DecoderLayer"
    rollout_api_base: str = ""
    rollout_api_bases: str = ""
    judge_api_base: str = ""
    max_new_tokens: int = 0
    problems_per_update: int = 64
    node_hostnames: str = ""
    train_node_ranks: str = "2"
    rollout_node_ranks: str = "1"
    judge_node_rank: int = 0
    trainer_node_rank: int = 2
    rollout_tensor_parallel_size: int = 8
    rollout_max_num_seqs: int = 96
    rollout_max_num_batched_tokens: int = 0
    judge_tensor_parallel_size: int = 8
    judge_max_num_seqs: int = 96
    judge_max_num_batched_tokens: int = 0
    rollout_wave_problem_count: int = 6
    adapter_reload_timeout_seconds: float = 900.0
    max_rollout_retries_per_sample: int = 2
    max_group_replacement_attempts: int = 32
    minimum_trainable_tokens_per_sample: int = 1
    online_control_dir: Path | None = None
    initial_adapter_path: Path | None = None
    group_queue_path: Path | None = None
    target_module_suffixes: list[str] | None = None
    global_rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"
    cuda_visible_devices: str = ""

    @classmethod
    def from_cli_args(cls, argv: list[str] | None = None) -> AIMOTrainingConfig:

        parser = cls.build_argument_parser()
        args = parser.parse_args(argv)
        config = cls(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            logdir=args.logdir,
            role=args.role,
            online=args.online,
            allow_base_rollouts=args.allow_base_rollouts,
            rollout_mode=args.rollout_mode,
            tool_protocol=args.tool_protocol,
            num_gpus=args.num_gpus,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            max_model_len=args.max_model_len,
            group_size=args.group_size,
            judge_model_path=args.judge_model_path,
            dummy_test=args.dummy_test,
            dummy_model_path=args.dummy_model_path,
            judge_port=args.judge_port,
            rollout_port=args.rollout_port,
            rollout_temperature=args.rollout_temperature,
            rollout_top_p=args.rollout_top_p,
            rollout_min_p=args.rollout_min_p,
            max_python_calls=args.max_python_calls,
            reward_weights_json=args.reward_weights_json,
            active_problem_count=args.active_problem_count,
            sandbox_count=args.sandbox_count,
            kv_cache_dtype=args.kv_cache_dtype,
            page_count_method=args.page_count_method,
            page_template=args.page_template,
            importance_sampling_level=args.importance_sampling_level,
            kl_beta=args.kl_beta,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            grpo_epsilon=args.grpo_epsilon,
            seed=args.seed,
            train_processes_per_node=args.train_processes_per_node,
            train_sharding_strategy=args.train_sharding_strategy,
            train_bf16=args.train_bf16,
            train_gradient_checkpointing=args.train_gradient_checkpointing,
            train_fsdp_transformer_layer_cls_to_wrap=args.train_fsdp_transformer_layer_cls_to_wrap,
            rollout_api_base=args.rollout_api_base,
            rollout_api_bases=args.rollout_api_bases,
            judge_api_base=args.judge_api_base,
            max_new_tokens=args.max_new_tokens,
            problems_per_update=args.problems_per_update,
            node_hostnames=args.node_hostnames,
            train_node_ranks=args.train_node_ranks,
            rollout_node_ranks=args.rollout_node_ranks,
            judge_node_rank=args.judge_node_rank,
            trainer_node_rank=args.trainer_node_rank,
            rollout_tensor_parallel_size=args.rollout_tensor_parallel_size,
            rollout_max_num_seqs=args.rollout_max_num_seqs,
            rollout_max_num_batched_tokens=args.rollout_max_num_batched_tokens,
            judge_tensor_parallel_size=args.judge_tensor_parallel_size,
            judge_max_num_seqs=args.judge_max_num_seqs,
            judge_max_num_batched_tokens=args.judge_max_num_batched_tokens,
            rollout_wave_problem_count=args.rollout_wave_problem_count,
            adapter_reload_timeout_seconds=args.adapter_reload_timeout_seconds,
            max_rollout_retries_per_sample=args.max_rollout_retries_per_sample,
            max_group_replacement_attempts=args.max_group_replacement_attempts,
            minimum_trainable_tokens_per_sample=args.minimum_trainable_tokens_per_sample,
            online_control_dir=args.online_control_dir,
            initial_adapter_path=args.initial_adapter_path,
            group_queue_path=args.group_queue_path,
            target_module_suffixes=args.target_module_suffixes,
            global_rank=cls._environment_int(
                "AIMO_SERVICE_RANK",
                cls._environment_int("GLOBAL_RANK", 0),
            ),
            local_rank=cls._environment_int("LOCAL_RANK", 0),
            world_size=cls._environment_int(
                "AIMO_SERVICE_WORLD_SIZE",
                cls._environment_int("WORLD_SIZE", 1),
            ),
            master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
            master_port=os.environ.get("MASTER_PORT", "29500"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
        config = config.with_dummy_test_defaults()
        config.validate()

        return config

    @classmethod
    def build_argument_parser(cls) -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=Path, required=True)
        parser.add_argument("--dataset_path", type=Path, required=True)
        parser.add_argument("--output_path", type=Path, required=True)
        parser.add_argument("--logdir", type=Path, required=True)
        parser.add_argument("--role", default="controller")
        parser.add_argument("--online", type=cls._parse_bool, default=False)
        parser.add_argument("--allow_base_rollouts", type=cls._parse_bool, default=False)
        parser.add_argument("--rollout_mode", default="queued")
        parser.add_argument("--tool_protocol", default="olmo_chatml")
        parser.add_argument("--num_gpus", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=5e-6)
        parser.add_argument("--num_train_epochs", type=float, default=1.0)
        parser.add_argument("--per_device_batch_size", type=int, default=1)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--lora_rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=128)
        parser.add_argument("--max_model_len", type=int, default=MAX_GENERATION_CONTEXT_TOKENS)
        parser.add_argument("--group_size", type=int, default=16)
        parser.add_argument("--judge_model_path", type=Path, default=Path("models/judge"))
        parser.add_argument(
            "--dummy_test",
            type=cls._parse_bool,
            default=cls._environment_bool("AIMO_DUMMY_TEST", DEFAULT_DUMMY_TEST),
        )
        parser.add_argument(
            "--dummy_model_path",
            type=Path,
            default=cls._environment_path("AIMO_DUMMY_MODEL_PATH", DUMMY_MODEL_PATH),
        )
        parser.add_argument("--judge_port", type=int, default=8000)
        parser.add_argument("--rollout_port", type=int, default=8001)
        parser.add_argument("--rollout_temperature", type=float, default=0.6)
        parser.add_argument("--rollout_top_p", type=float, default=0.95)
        parser.add_argument("--rollout_min_p", type=float, default=0.0)
        parser.add_argument("--max_python_calls", type=int, default=64)
        parser.add_argument("--reward_weights_json", default="")
        parser.add_argument("--active_problem_count", type=int, default=6)
        parser.add_argument("--sandbox_count", type=int, default=96)
        parser.add_argument("--kv_cache_dtype", default="auto")
        parser.add_argument("--page_count_method", default="latex")
        parser.add_argument("--page_template", default=DEFAULT_PAGE_TEMPLATE)
        parser.add_argument("--importance_sampling_level", default="token")
        parser.add_argument("--kl_beta", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--scheduler", default="cosine")
        parser.add_argument("--warmup_ratio", type=float, default=0.03)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--grpo_epsilon", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--train_processes_per_node", type=int, default=8)
        parser.add_argument("--train_sharding_strategy", default="fsdp_full_shard")
        parser.add_argument("--train_bf16", type=cls._parse_bool, default=True)
        parser.add_argument("--train_gradient_checkpointing", type=cls._parse_bool, default=True)
        parser.add_argument(
            "--train_fsdp_transformer_layer_cls_to_wrap",
            default="Olmo3DecoderLayer",
        )
        parser.add_argument("--rollout_api_base", default="")
        parser.add_argument("--rollout_api_bases", default="")
        parser.add_argument("--judge_api_base", default="")
        parser.add_argument("--max_new_tokens", type=int, default=0)
        parser.add_argument("--problems_per_update", type=int, default=64)
        parser.add_argument("--node_hostnames", default="")
        parser.add_argument("--train_node_ranks", default="2")
        parser.add_argument("--rollout_node_ranks", default="1")
        parser.add_argument("--judge_node_rank", type=int, default=0)
        parser.add_argument("--trainer_node_rank", type=int, default=2)
        parser.add_argument("--rollout_tensor_parallel_size", type=int, default=8)
        parser.add_argument("--rollout_max_num_seqs", type=int, default=96)
        parser.add_argument("--rollout_max_num_batched_tokens", type=int, default=0)
        parser.add_argument("--judge_tensor_parallel_size", type=int, default=8)
        parser.add_argument("--judge_max_num_seqs", type=int, default=96)
        parser.add_argument("--judge_max_num_batched_tokens", type=int, default=0)
        parser.add_argument("--rollout_wave_problem_count", type=int, default=6)
        parser.add_argument("--adapter_reload_timeout_seconds", type=float, default=900.0)
        parser.add_argument("--max_rollout_retries_per_sample", type=int, default=2)
        parser.add_argument("--max_group_replacement_attempts", type=int, default=32)
        parser.add_argument("--minimum_trainable_tokens_per_sample", type=int, default=1)
        parser.add_argument("--online_control_dir", type=Path, default=None)
        parser.add_argument("--initial_adapter_path", type=Path, default=None)
        parser.add_argument("--group_queue_path", type=Path, default=None)
        parser.add_argument(
            "--target_module_suffixes",
            type=cls._parse_string_list,
            default=None,
        )

        return parser

    @property
    def reward_weights(self) -> dict[str, float]:

        if not self.reward_weights_json.strip():
            return copy.deepcopy(DEFAULT_REWARD_WEIGHTS)

        parsed_value = json.loads(self.reward_weights_json)

        if not isinstance(parsed_value, dict):
            raise ValueError("reward_weights_json must decode to a JSON object.")

        weights = copy.deepcopy(DEFAULT_REWARD_WEIGHTS)

        for key, value in parsed_value.items():
            if key not in weights:
                raise ValueError(f"Unknown reward weight: {key}")

            weights[key] = float(value)

        return weights

    @property
    def resolved_group_queue_path(self) -> Path:

        if self.group_queue_path is not None:
            return self.group_queue_path

        return self.logdir / "grpo_groups.jsonl"

    @property
    def resolved_target_module_suffixes(self) -> list[str]:

        if self.target_module_suffixes:
            return list(self.target_module_suffixes)

        return list(DEFAULT_TARGET_MODULE_SUFFIXES)

    @property
    def resolved_dummy_model_path(self) -> Path:

        if self.dummy_model_path != DUMMY_MODEL_PATH:
            return self.dummy_model_path

        default_model_paths = {
            Path("models/contestant"),
            Path("models/judge"),
        }

        for candidate_path in [
            self.model_path,
            self.judge_model_path,
        ]:
            if candidate_path in default_model_paths and self.model_path not in default_model_paths:
                continue

            if candidate_path.name in {
                "contestant",
                "judge",
            }:
                return candidate_path.parent / "dummy"

        return self.model_path

    def with_dummy_test_defaults(self) -> AIMOTrainingConfig:

        if not self.dummy_test:
            return self

        dummy_model_path = self.resolved_dummy_model_path

        return replace(
            self,
            model_path=dummy_model_path,
            judge_model_path=dummy_model_path,
            dummy_model_path=dummy_model_path,
            rollout_tensor_parallel_size=DUMMY_TENSOR_PARALLEL_SIZE,
            judge_tensor_parallel_size=DUMMY_TENSOR_PARALLEL_SIZE,
        )

    @property
    def generation_batch_size(self) -> int:

        return self.problems_per_update * self.group_size

    @property
    def global_prompt_batch_size(self) -> int:

        return self.generation_batch_size // self.group_size

    def validate(self) -> None:

        if self.num_gpus < 1:
            raise ValueError("num_gpus must be at least 1.")

        if self.role not in {
            "controller",
            "rollout_server",
            "judge_server",
            "train_update",
        }:
            raise ValueError(
                "role must be controller, rollout_server, judge_server, or train_update."
            )

        if self.rollout_mode != "queued":
            raise ValueError("Only queued rollout_mode is supported.")

        if self.tool_protocol not in {
            "harmony",
            "markdown_code",
            "olmo_chatml",
        }:
            raise ValueError("tool_protocol must be harmony, markdown_code, or olmo_chatml.")

        if self.group_size < 1:
            raise ValueError("group_size must be at least 1.")

        if self.judge_port < 1:
            raise ValueError("judge_port must be positive.")

        if self.rollout_port < 1:
            raise ValueError("rollout_port must be positive.")

        if self.per_device_batch_size < 1:
            raise ValueError("per_device_batch_size must be at least 1.")

        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1.")

        if self.active_problem_count < 1:
            raise ValueError("active_problem_count must be at least 1.")

        if self.sandbox_count < self.group_size:
            raise ValueError("sandbox_count must be at least group_size.")

        if self.lora_rank < 1:
            raise ValueError("lora_rank must be at least 1.")

        if self.lora_alpha < 1:
            raise ValueError("lora_alpha must be at least 1.")

        if self.importance_sampling_level != "token":
            raise ValueError("Only token importance sampling is supported.")

        if self.grpo_epsilon <= 0:
            raise ValueError("grpo_epsilon must be positive.")

        if self.train_processes_per_node < 1:
            raise ValueError("train_processes_per_node must be at least 1.")

        if self.train_sharding_strategy not in {
            "none",
            "fsdp_full_shard",
        }:
            raise ValueError("train_sharding_strategy must be none or fsdp_full_shard.")

        if self.problems_per_update < 1:
            raise ValueError("problems_per_update must be at least 1.")

        if self.generation_batch_size % self.group_size != 0:
            raise ValueError("generation_batch_size must be divisible by group_size.")

        if self.global_prompt_batch_size != self.problems_per_update:
            raise ValueError("global_prompt_batch_size must equal problems_per_update.")

        if self.kl_beta < 0:
            raise ValueError("kl_beta must be non-negative.")

        if self.rollout_tensor_parallel_size < 1:
            raise ValueError("rollout_tensor_parallel_size must be at least 1.")

        if self.rollout_max_num_seqs < 1:
            raise ValueError("rollout_max_num_seqs must be at least 1.")

        if self.rollout_max_num_batched_tokens < 0:
            raise ValueError("rollout_max_num_batched_tokens must be non-negative.")

        if self.judge_tensor_parallel_size < 1:
            raise ValueError("judge_tensor_parallel_size must be at least 1.")

        if self.judge_max_num_seqs < 1:
            raise ValueError("judge_max_num_seqs must be at least 1.")

        if self.judge_max_num_batched_tokens < 0:
            raise ValueError("judge_max_num_batched_tokens must be non-negative.")

        if self.rollout_wave_problem_count < 1:
            raise ValueError("rollout_wave_problem_count must be at least 1.")

        if self.adapter_reload_timeout_seconds <= 0:
            raise ValueError("adapter_reload_timeout_seconds must be positive.")

        if self.max_rollout_retries_per_sample < 0:
            raise ValueError("max_rollout_retries_per_sample must be non-negative.")

        if self.max_group_replacement_attempts < self.group_size:
            raise ValueError("max_group_replacement_attempts must be at least group_size.")

        if self.minimum_trainable_tokens_per_sample < 1:
            raise ValueError("minimum_trainable_tokens_per_sample must be at least 1.")

        if self.trainer_node_rank < 0:
            raise ValueError("trainer_node_rank must be non-negative.")

        if not self.train_node_ranks.strip():
            raise ValueError("train_node_ranks must include at least one rank.")

        if not self.rollout_node_ranks.strip():
            raise ValueError("rollout_node_ranks must include at least one rank.")

        if self.judge_node_rank < 0:
            raise ValueError("judge_node_rank must be non-negative.")

        if self.world_size < 1:
            raise ValueError("WORLD_SIZE must be at least 1.")

        if not 0 <= self.global_rank < self.world_size:
            raise ValueError("GLOBAL_RANK must satisfy 0 <= GLOBAL_RANK < WORLD_SIZE.")

        if self.world_size > 1:
            missing_names = [
                name
                for name in [
                    "AIMO_SERVICE_RANK",
                    "AIMO_SERVICE_WORLD_SIZE",
                ]
                if name not in os.environ
            ]

            legacy_missing_names = [
                name
                for name in [
                    "GLOBAL_RANK",
                    "WORLD_SIZE",
                ]
                if name not in os.environ
            ]

            if missing_names and legacy_missing_names:
                missing_text = ", ".join(missing_names)

                raise ValueError(f"Missing service topology environment variables: {missing_text}.")

        if self.world_size > 1 and self._should_validate_online_topology():
            self._validate_online_topology()

    def _should_validate_online_topology(self) -> bool:

        return (
            self.online
            and self.role != "train_update"
        ) or self.role in {
            "rollout_server",
            "judge_server",
        }

    def _validate_online_topology(self) -> None:

        rollout_ranks = self._parse_rank_list(self.rollout_node_ranks)
        train_ranks = self._parse_rank_list(self.train_node_ranks)

        for rank in [
            self.judge_node_rank,
            self.trainer_node_rank,
            *rollout_ranks,
            *train_ranks,
        ]:
            if not 0 <= rank < self.world_size:
                raise ValueError(
                    f"online topology rank {rank} is outside WORLD_SIZE={self.world_size}."
                )

        if len(set(rollout_ranks)) != len(rollout_ranks):
            raise ValueError("rollout_node_ranks must not contain duplicate ranks.")

        if len(set(train_ranks)) != len(train_ranks):
            raise ValueError("train_node_ranks must not contain duplicate ranks.")

        if self.trainer_node_rank not in train_ranks:
            raise ValueError("trainer_node_rank must be included in train_node_ranks.")

        assigned_service_ranks = [
            self.judge_node_rank,
            self.trainer_node_rank,
            *rollout_ranks,
        ]

        if len(set(assigned_service_ranks)) != len(assigned_service_ranks):
            raise ValueError(
                "judge_node_rank, trainer_node_rank, and rollout_node_ranks must be disjoint."
            )

        hostnames = self._split_csv(
            self.node_hostnames or os.environ.get("AIMO_NODE_HOSTNAMES", "")
        )

        if hostnames and len(hostnames) != self.world_size:
            raise ValueError(
                "node_hostnames or AIMO_NODE_HOSTNAMES must provide exactly one hostname "
                f"per WORLD_SIZE rank; got {len(hostnames)} for WORLD_SIZE={self.world_size}."
            )

        if hostnames:
            endpoints = [
                (hostnames[self.judge_node_rank], self.judge_port),
                *[
                    (hostnames[rank], self.rollout_port)
                    for rank in rollout_ranks
                ],
            ]

            if len(set(endpoints)) != len(endpoints):
                raise ValueError("Service port assignments collide on the same host.")

    def as_dict(self) -> dict[str, Any]:

        payload = asdict(self)

        return {
            key: self._json_value(value)
            for key, value in payload.items()
        }

    @staticmethod
    def _json_value(value: Any) -> Any:

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, list):
            return [
                AIMOTrainingConfig._json_value(item)
                for item in value
            ]

        if isinstance(value, dict):
            return {
                str(key): AIMOTrainingConfig._json_value(item)
                for key, item in value.items()
            }

        return value

    @staticmethod
    def _environment_int(name: str, default: int) -> int:

        value = os.environ.get(name)

        if value is None:
            return default

        return int(value)

    @classmethod
    def _environment_bool(cls, name: str, default: bool) -> bool:

        value = os.environ.get(name)

        if value is None:
            return default

        return cls._parse_bool(value)

    @staticmethod
    def _environment_path(name: str, default: Path) -> Path:

        value = os.environ.get(name)

        if value is None:
            return default

        return Path(value)

    @staticmethod
    def _parse_bool(value: str | bool) -> bool:

        if isinstance(value, bool):
            return value

        normalized_value = value.strip().lower()

        if normalized_value in {"1", "true", "yes", "y", "on"}:
            return True

        if normalized_value in {"0", "false", "no", "n", "off"}:
            return False

        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    @staticmethod
    def _parse_string_list(value: str | list[str] | None) -> list[str] | None:

        if value is None:
            return None

        if isinstance(value, list):
            return value

        stripped_value = value.strip()

        if not stripped_value:
            return None

        if stripped_value.startswith("["):
            parsed_value = json.loads(stripped_value)

            if not isinstance(parsed_value, list):
                raise argparse.ArgumentTypeError("Expected a JSON list.")

            return [
                str(item)
                for item in parsed_value
            ]

        return [
            item.strip()
            for item in stripped_value.split(",")
            if item.strip()
        ]

    @staticmethod
    def _parse_rank_list(value: str) -> list[int]:

        ranks = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]

        if not ranks:
            raise ValueError("At least one rank is required.")

        return ranks

    @staticmethod
    def _split_csv(value: str) -> list[str]:

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]
