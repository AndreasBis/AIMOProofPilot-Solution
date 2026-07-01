from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimo_inference.client import AIMOInferenceClient
from aimo_inference.config import AIMOConfig
from aimo_inference.prompts import AIMOPromptBuilder
from aimo_inference.sandbox import AIMOSandbox
from aimo_training.config import AIMOTrainingConfig
from aimo_training.rewards import AIMORewardConfig
from aimo_training.rewards import AIMOTrainingRewardScorer
from aimo_training.rollout import build_judge_inference_config
from aimo_training.rollout import build_rollout_inference_config
from aimo_training.schema import AIMOGRPOGroup
from aimo_training.schema import AIMORolloutSample
from aimo_training.schema import AIMORewardBreakdown
from aimo_training.tool_rollout import AIMOToolRolloutEngine
from aimo_training.trainer import AIMOTrainingStepSummary


@dataclass(frozen=True)
class AIMOTRLRolloutEndpoint:

    endpoint_index: int
    config: AIMOConfig
    client: AIMOInferenceClient


class AIMOTRLRewardFunction:

    def __call__(
        self,
        prompts: list[object],
        completions: list[object],
        completion_ids: list[list[int]],
        scalar_reward: list[float] | None = None,
        log_extra: object | None = None,
        log_metric: object | None = None,
        **metadata: object,
    ) -> list[float]:

        if scalar_reward is None:
            raise ValueError("scalar_reward metadata is required for TRL GRPO rewards.")

        rewards = [
            float(value)
            for value in scalar_reward
        ]

        if callable(log_extra):
            for key in [
                "problem_id",
                "rollout_index",
                "endpoint_index",
                "input_tokens",
                "output_tokens",
                "tool_tokens",
                "finish_reason",
                "judge_grade",
                "context_reward",
                "solution_page_reward",
                "rendered_page_count",
                "judge_parse_failed",
            ]:
                values = metadata.get(key)

                if isinstance(values, list):
                    log_extra(key, values)

        if callable(log_metric) and rewards:
            log_metric("aimo_scalar_reward_mean", sum(rewards) / len(rewards))

        return rewards


class AIMOTRLQueuedRolloutFunction:

    def __init__(
        self,
        config: AIMOTrainingConfig,
        groups: list[AIMOGRPOGroup],
        tokenizer: object,
    ) -> None:

        self.config = config
        if config.rollout_mode != "queued":
            raise ValueError("AIMOTRLGRPOTrainer only supports queued rollout_mode.")

        if config.tool_protocol == "harmony":
            raise ValueError(
                "Structured Harmony training rollouts are not supported by the queued "
                "TRL trainer until rollout records include Harmony token masks."
            )

        self.groups_by_prompt: dict[str, list[AIMOGRPOGroup]] = {}
        self.next_group_index_by_prompt: dict[str, int] = {}

        for group in groups:
            prompt = group_prompt(group)
            self.groups_by_prompt.setdefault(prompt, []).append(group)

    def __call__(self, prompts: list[object], trainer: object) -> dict[str, list[object]]:

        payload = empty_rollout_payload()
        prompt_texts = [
            str(prompt)
            for prompt in prompts
        ]
        num_generations = int(getattr(trainer, "num_generations", self.config.group_size))

        if is_prompt_occurrence_batch(
            prompt_texts=prompt_texts,
            group_size=num_generations,
        ):
            prompt_occurrence_counts: dict[str, int] = {}

            for prompt in prompt_texts:
                occurrence_index = prompt_occurrence_counts.get(prompt, 0)
                prompt_occurrence_counts[prompt] = occurrence_index + 1
                sample = self._next_sample(
                    prompt=prompt,
                    rollout_index=occurrence_index % self.config.group_size,
                )
                sample_payload = self._sample_payload(sample=sample)
                extend_rollout_payload(payload=payload, addition=sample_payload)
        else:
            for prompt in prompt_texts:
                group = self._next_group(prompt)
                group_payload = self._group_payload(group)
                extend_rollout_payload(payload=payload, addition=group_payload)

        validate_grpo_batch_contract(
            config=self.config,
            rollout_batch=payload,
            expected_sample_counts=[
                len(prompt_texts),
                len(prompt_texts) * self.config.group_size,
            ],
        )

        return payload

    def _next_group(self, prompt: str) -> AIMOGRPOGroup:

        groups = self.groups_by_prompt.get(prompt)

        if not groups:
            raise KeyError(f"No queued GRPO group is available for prompt: {prompt[:80]}")

        index = self.next_group_index_by_prompt.get(prompt, 0)
        self.next_group_index_by_prompt[prompt] = index + 1

        return groups[index % len(groups)]

    def _next_sample(
        self,
        prompt: str,
        rollout_index: int,
    ) -> AIMORolloutSample:

        group = self._next_group(prompt)
        samples_by_rollout = {
            sample.rollout_index: sample
            for sample in group.samples
        }

        if rollout_index not in samples_by_rollout:
            raise KeyError(
                f"No queued GRPO sample is available for rollout index {rollout_index}."
            )

        return samples_by_rollout[rollout_index]

    def _group_payload(self, group: AIMOGRPOGroup) -> dict[str, list[object]]:

        payload = empty_rollout_payload()
        samples = sorted(
            group.samples,
            key=lambda sample: sample.rollout_index,
        )

        for sample in samples:
            extend_rollout_payload(
                payload=payload,
                addition=self._sample_payload(sample=sample),
            )

        return payload

    def _sample_payload(self, sample: AIMORolloutSample) -> dict[str, list[object]]:

        payload = empty_rollout_payload()
        add_rollout_sample_payload(
            payload=payload,
            prompt_ids=sample.prompt_ids,
            completion_ids=sample.token_ids,
            token_logprobs=sample.token_logprobs,
            env_mask=sample.env_mask,
            problem_id=sample.problem_id,
            rollout_index=sample.rollout_index,
            endpoint_index=sample.endpoint_index,
            input_tokens=sample.input_tokens,
            output_tokens=sample.output_tokens,
            tool_tokens=sample.tool_tokens,
            finish_reason=sample.finish_reason,
            reward=sample.reward,
        )

        return payload

class AIMOTRLGRPORolloutFunction:

    def __init__(
        self,
        config: AIMOTrainingConfig,
        rollout_api_bases: list[str],
        judge_api_base: str,
        adapter_path: Path | None = None,
    ) -> None:

        self.config = config
        raise ValueError("Live TRL GRPO rollouts are experimental and disabled.")
        self.prompt_builder = AIMOPromptBuilder()
        self.endpoints = [
            self._build_endpoint(
                endpoint_index=endpoint_index,
                api_base=api_base,
                adapter_path=adapter_path,
            )
            for endpoint_index, api_base in enumerate(rollout_api_bases)
        ]
        judge_config = build_judge_inference_config(config).with_overrides(
            api_base=judge_api_base,
            logdir=config.logdir / "trl_judge_client",
        )
        self.reward_scorer = AIMOTrainingRewardScorer(
            inference_config=judge_config,
            judge_client=AIMOInferenceClient(config=judge_config),
            reward_config=AIMORewardConfig(weights=config.reward_weights),
        )

    def __call__(self, prompts: list[object], trainer: object) -> dict[str, list[object]]:

        payload = empty_rollout_payload()
        num_generations = int(getattr(trainer, "num_generations", self.config.group_size))

        for prompt_index, prompt in enumerate(prompts):
            prompt_record = prompt_to_record(prompt=prompt, prompt_index=prompt_index)

            for rollout_index in range(num_generations):
                endpoint = self.endpoints[
                    (prompt_index * num_generations + rollout_index) % len(self.endpoints)
                ]
                sample_payload = self._build_sample_payload(
                    endpoint=endpoint,
                    prompt_record=prompt_record,
                    rollout_index=rollout_index,
                )
                extend_rollout_payload(payload=payload, addition=sample_payload)

        validate_grpo_batch_contract(
            config=self.config,
            rollout_batch=payload,
            expected_sample_counts=[
                len(prompts) * num_generations,
            ],
        )

        return payload

    def _build_sample_payload(
        self,
        endpoint: AIMOTRLRolloutEndpoint,
        prompt_record: dict[str, str],
        rollout_index: int,
    ) -> dict[str, list[object]]:

        try:
            sandbox = AIMOSandbox(config=endpoint.config)
            result = AIMOToolRolloutEngine(
                config=endpoint.config,
                client=endpoint.client,
                sandbox=sandbox,
                prompt_builder=self.prompt_builder,
            ).run_problem(
                problem_text=prompt_record["problem"],
                seed=self.config.seed + rollout_index,
            )
            reward = self.reward_scorer.score(
                problem=prompt_record["problem"],
                reference_solution=prompt_record["reference_solution"],
                generated_proof=result.proof_text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                finish_reason=result.finish_reason,
                tool_tokens=result.tool_tokens,
            )
            payload = empty_rollout_payload()
            add_rollout_sample_payload(
                payload=payload,
                prompt_ids=result.prompt_ids,
                completion_ids=result.completion_ids,
                token_logprobs=result.token_logprobs,
                env_mask=result.env_mask,
                problem_id=prompt_record["problem_id"],
                rollout_index=rollout_index,
                endpoint_index=endpoint.endpoint_index,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                tool_tokens=result.tool_tokens,
                finish_reason=result.finish_reason,
                reward=reward,
            )

            return payload
        except Exception as error:
            return self._failure_payload(
                endpoint=endpoint,
                prompt_record=prompt_record,
                rollout_index=rollout_index,
                error=str(error),
            )

    def _failure_payload(
        self,
        endpoint: AIMOTRLRolloutEndpoint,
        prompt_record: dict[str, str],
        rollout_index: int,
        error: str,
    ) -> dict[str, list[object]]:

        tokenizer = self._load_tokenizer(endpoint.config)
        prompt_ids = tokenizer.encode(prompt_record["problem"], add_special_tokens=False)
        completion_ids = tokenizer.encode("No proof was produced.", add_special_tokens=False)
        reward = AIMORewardBreakdown(
            judge_grade=0,
            context_reward=-1,
            solution_page_reward=-1,
            scalar_reward=-2.0,
            rendered_page_count=0,
            page_count_method=self.config.page_count_method,
            latex_compile_status="not_attempted",
            page_count_fallback_reason="rollout_failed",
            judge_response=error,
            judge_parse_failed=True,
            input_tokens=len(prompt_ids),
            output_tokens=len(completion_ids),
            finish_reason="error",
            latency_seconds=0.0,
            tool_tokens=0,
        )
        payload = empty_rollout_payload()
        add_rollout_sample_payload(
            payload=payload,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            token_logprobs=[
                0.0
                for _ in completion_ids
            ],
            env_mask=[
                0
                for _ in completion_ids
            ],
            problem_id=prompt_record["problem_id"],
            rollout_index=rollout_index,
            endpoint_index=endpoint.endpoint_index,
            input_tokens=len(prompt_ids),
            output_tokens=len(completion_ids),
            tool_tokens=0,
            finish_reason="error",
            reward=reward,
        )

        return payload

    def _build_endpoint(
        self,
        endpoint_index: int,
        api_base: str,
        adapter_path: Path | None,
    ) -> AIMOTRLRolloutEndpoint:

        endpoint_config = build_rollout_inference_config(self.config).with_overrides(
            api_base=api_base,
            logdir=self.config.logdir / "trl_rollout_clients" / f"endpoint_{endpoint_index}",
            lora_adapter_path=adapter_path,
        )

        return AIMOTRLRolloutEndpoint(
            endpoint_index=endpoint_index,
            config=endpoint_config,
            client=AIMOInferenceClient(config=endpoint_config),
        )

    def _load_tokenizer(self, config: AIMOConfig) -> object:

        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )


class AIMOTRLGRPOTrainer:

    def __init__(self, config: AIMOTrainingConfig) -> None:

        self.config = config

    def train(
        self,
        groups: list[AIMOGRPOGroup],
    ) -> tuple[list[AIMOTrainingStepSummary], Path, Path]:

        validate_grpo_batch_contract(
            config=self.config,
            groups=groups,
        )
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer
        from trl import GRPOTrainer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        train_dataset = Dataset.from_list([
            {
                "prompt": group_prompt(group),
                "problem_id": group.problem_id,
                "group_index": group.group_index,
            }
            for group in groups
        ])
        grpo_config = build_grpo_config(self.config)
        peft_config = None
        model: object = str(self.config.model_path)

        if self.config.initial_adapter_path is not None:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype="bfloat16",
                trust_remote_code=True,
                local_files_only=True,
            )
            model = PeftModel.from_pretrained(
                base_model,
                self.config.initial_adapter_path,
                is_trainable=True,
            )
        else:
            peft_config = build_peft_config(self.config)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=AIMOTRLRewardFunction(),
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            rollout_func=AIMOTRLQueuedRolloutFunction(
                config=self.config,
                groups=groups,
                tokenizer=tokenizer,
            ),
        )
        train_result = trainer.train()
        trainer.save_model(str(self.config.output_path))
        adapter_path = self.config.output_path / "adapter_model.safetensors"
        adapter_config_path = self.config.output_path / "adapter_config.json"
        training_loss = float(getattr(train_result, "training_loss", 0.0) or 0.0)

        return self._summaries(
            groups=groups,
            loss=training_loss,
        ), adapter_path, adapter_config_path

    def _summaries(
        self,
        groups: list[AIMOGRPOGroup],
        loss: float,
    ) -> list[AIMOTrainingStepSummary]:

        summaries = []

        for step_index, group in enumerate(groups):
            rewards = group.rewards
            mean_reward = sum(rewards) / len(rewards)
            variance = sum(
                (reward - mean_reward) ** 2
                for reward in rewards
            ) / max(1, len(rewards))
            summaries.append(
                AIMOTrainingStepSummary(
                    step_index=step_index,
                    group_index=group.group_index,
                    problem_id=group.problem_id,
                    loss=loss,
                    mean_reward=mean_reward,
                    reward_std=math.sqrt(variance),
                    sample_count=len(group.samples),
                )
            )

        return summaries


def build_grpo_config(config: AIMOTrainingConfig) -> object:

    from trl import GRPOConfig

    arguments: dict[str, object] = {
        "output_dir": str(config.output_path),
        "logging_dir": str(config.logdir),
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.scheduler,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "seed": config.seed,
        "bf16": config.train_bf16,
        "gradient_checkpointing": config.train_gradient_checkpointing,
        "save_on_each_node": False,
        "ddp_find_unused_parameters": False,
        "model_init_kwargs": {
            "torch_dtype": "bfloat16",
            "trust_remote_code": True,
            "local_files_only": True,
        },
        "remove_unused_columns": False,
        "num_generations": config.group_size,
        "generation_batch_size": config.generation_batch_size,
        "max_completion_length": config.max_model_len,
        "temperature": config.rollout_temperature,
        "top_p": config.rollout_top_p,
        "top_k": 0 if config.rollout_top_p > 0 else config.group_size,
        "min_p": config.rollout_min_p or None,
        "repetition_penalty": 1.0,
        "use_vllm": False,
        "beta": config.kl_beta,
        "epsilon": config.grpo_epsilon,
        "importance_sampling_level": config.importance_sampling_level,
        "scale_rewards": "group",
        "loss_type": "grpo",
        "mask_truncated_completions": True,
        "shuffle_dataset": False,
        "save_strategy": "no",
        "report_to": [],
    }

    if should_enable_fsdp_training(config):
        arguments["fsdp"] = "full_shard auto_wrap"
        arguments["fsdp_config"] = {
            "activation_checkpointing": config.train_gradient_checkpointing,
            "sync_module_states": True,
            "transformer_layer_cls_to_wrap": config.train_fsdp_transformer_layer_cls_to_wrap,
            "use_orig_params": True,
        }

    return GRPOConfig(**arguments)


def should_enable_fsdp_training(config: AIMOTrainingConfig) -> bool:

    return (
        config.train_sharding_strategy == "fsdp_full_shard"
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def build_peft_config(config: AIMOTrainingConfig) -> object:

    from peft import LoraConfig

    return LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        use_rslora=True,
        target_modules=config.resolved_target_module_suffixes,
    )


def validate_grpo_batch_contract(
    config: AIMOTrainingConfig,
    groups: list[AIMOGRPOGroup] | None = None,
    rollout_batch: dict[str, list[object]] | None = None,
    expected_sample_counts: list[int] | None = None,
    require_main_run: bool = False,
) -> None:

    if config.generation_batch_size != config.problems_per_update * config.group_size:
        raise ValueError("generation_batch_size must equal problems_per_update * group_size.")

    if config.generation_batch_size % config.group_size != 0:
        raise ValueError("generation_batch_size must be divisible by group_size.")

    if config.global_prompt_batch_size != config.problems_per_update:
        raise ValueError("global_prompt_batch_size must equal problems_per_update.")

    if require_main_run:
        if config.problems_per_update != 64:
            raise ValueError("Main GRPO run requires problems_per_update == 64.")

        if config.group_size != 16:
            raise ValueError("Main GRPO run requires group_size == 16.")

    if groups is not None:
        validate_group_contract(groups=groups, group_size=config.group_size)

    if rollout_batch is not None:
        validate_rollout_payload_contract(
            payload=rollout_batch,
            group_size=config.group_size,
            expected_sample_counts=expected_sample_counts,
        )


def validate_group_contract(groups: list[AIMOGRPOGroup], group_size: int) -> None:

    if not groups:
        raise ValueError("At least one GRPO group is required.")

    for group in groups:
        if len(group.samples) != group_size:
            raise ValueError(
                f"GRPO group {group.group_index} has {len(group.samples)} samples, "
                f"but group_size is {group_size}."
            )

        rollout_indices: set[int] = set()
        rewards = set()
        expected_prompt = group.samples[0].prompt if group.samples else ""
        expected_adapter_hash = group.samples[0].policy_adapter_hash if group.samples else ""

        for sample in group.samples:
            validate_rollout_sample_contract(sample=sample)
            rewards.add(sample.reward.scalar_reward)

            if sample.problem_id != group.problem_id:
                raise ValueError(f"GRPO group {group.group_index} has mixed problem_id values.")

            if sample.group_index != group.group_index:
                raise ValueError(f"GRPO group {group.group_index} has mixed group_index values.")

            if sample.prompt != expected_prompt:
                raise ValueError(f"GRPO group {group.group_index} has mixed prompts.")

            if sample.rollout_index in rollout_indices:
                raise ValueError(
                    f"GRPO group {group.group_index} has duplicate rollout_index values."
                )

            rollout_indices.add(sample.rollout_index)

            if sample.policy_adapter_hash != expected_adapter_hash:
                raise ValueError(f"GRPO group {group.group_index} has mixed adapter hashes.")

        if not any(sum(sample.env_mask) > 0 for sample in group.samples):
            raise ValueError(f"GRPO group {group.group_index} has no trainable tokens.")

        if len(rewards) < 2:
            group.metadata["zero_variance_reward_warning"] = True


def validate_rollout_sample_contract(sample: AIMORolloutSample) -> None:

    sample_name = f"{sample.problem_id}:{sample.rollout_index}"

    if not sample.prompt_ids:
        raise ValueError(f"Sample {sample_name} has no prompt token IDs.")

    if not sample.token_ids:
        raise ValueError(f"Sample {sample_name} has no completion token IDs.")

    if not sample.token_logprobs:
        raise ValueError(f"Sample {sample_name} has no selected-token logprobs.")

    if len(sample.token_ids) != len(sample.token_logprobs):
        raise ValueError(f"Sample {sample_name} has mismatched token logprobs.")

    if sample.sampling_logprobs and len(sample.sampling_logprobs) != len(sample.token_ids):
        raise ValueError(f"Sample {sample_name} has mismatched sampling logprobs.")

    if not sample.env_mask:
        raise ValueError(f"Sample {sample_name} has no env_mask.")

    if len(sample.env_mask) != len(sample.token_ids):
        raise ValueError(f"Sample {sample_name} has mismatched env_mask.")

    if sum(sample.env_mask) <= 0:
        raise ValueError(f"Sample {sample_name} has no trainable tokens.")


def validate_rollout_payload_contract(
    payload: dict[str, list[object]],
    group_size: int,
    expected_sample_counts: list[int] | None = None,
) -> None:

    required_keys = [
        "prompt_ids",
        "completion_ids",
        "logprobs",
        "env_mask",
    ]
    missing_keys = [
        key
        for key in required_keys
        if key not in payload
    ]

    if missing_keys:
        raise ValueError(f"Rollout payload missing keys: {', '.join(missing_keys)}.")

    sample_count = len(payload["prompt_ids"])

    if expected_sample_counts is not None and sample_count not in expected_sample_counts:
        expected_text = ", ".join(str(count) for count in expected_sample_counts)

        raise ValueError(
            f"Rollout payload sample count {sample_count} did not match expected counts: "
            f"{expected_text}."
        )

    for key in required_keys:
        if len(payload[key]) != sample_count:
            raise ValueError(f"Rollout payload key {key} has inconsistent sample count.")

    if sample_count % group_size != 0:
        raise ValueError("Rollout payload sample count must be divisible by group_size.")

    for index in range(sample_count):
        completion_ids = payload["completion_ids"][index]
        logprobs = payload["logprobs"][index]
        env_mask = payload["env_mask"][index]
        prompt_ids = payload["prompt_ids"][index]

        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError(f"Rollout sample {index} has no prompt token IDs.")

        if not isinstance(completion_ids, list) or not completion_ids:
            raise ValueError(f"Rollout sample {index} has no completion token IDs.")

        if not isinstance(logprobs, list) or not logprobs:
            raise ValueError(f"Rollout sample {index} has no selected-token logprobs.")

        if len(logprobs) != len(completion_ids):
            raise ValueError(f"Rollout sample {index} has mismatched logprobs.")

        if not isinstance(env_mask, list) or len(env_mask) != len(completion_ids):
            raise ValueError(f"Rollout sample {index} has mismatched env_mask.")

        if sum(int(mask_value) for mask_value in env_mask) <= 0:
            raise ValueError(f"Rollout sample {index} has no trainable tokens.")


def is_prompt_occurrence_batch(
    prompt_texts: list[str],
    group_size: int,
) -> bool:

    if not prompt_texts:
        return False

    counts: dict[str, int] = {}

    for prompt in prompt_texts:
        counts[prompt] = counts.get(prompt, 0) + 1

    return all(count == group_size for count in counts.values())


def empty_rollout_payload() -> dict[str, list[object]]:

    return {
        "prompt_ids": [],
        "completion_ids": [],
        "logprobs": [],
        "sampling_logprobs": [],
        "env_mask": [],
        "problem_id": [],
        "rollout_index": [],
        "endpoint_index": [],
        "input_tokens": [],
        "output_tokens": [],
        "tool_tokens": [],
        "finish_reason": [],
        "judge_grade": [],
        "context_reward": [],
        "solution_page_reward": [],
        "scalar_reward": [],
        "rendered_page_count": [],
        "judge_parse_failed": [],
        "judge_response": [],
    }


def add_rollout_sample_payload(
    payload: dict[str, list[object]],
    prompt_ids: list[int],
    completion_ids: list[int],
    token_logprobs: list[float],
    env_mask: list[int],
    problem_id: str,
    rollout_index: int,
    endpoint_index: int | None,
    input_tokens: int | None,
    output_tokens: int | None,
    tool_tokens: int,
    finish_reason: str,
    reward: AIMORewardBreakdown,
) -> None:

    payload["prompt_ids"].append(list(prompt_ids))
    payload["completion_ids"].append(list(completion_ids))
    payload["logprobs"].append(list(token_logprobs))
    payload["sampling_logprobs"].append(list(token_logprobs))
    payload["env_mask"].append(list(env_mask))
    payload["problem_id"].append(problem_id)
    payload["rollout_index"].append(rollout_index)
    payload["endpoint_index"].append(-1 if endpoint_index is None else endpoint_index)
    payload["input_tokens"].append(input_tokens)
    payload["output_tokens"].append(output_tokens)
    payload["tool_tokens"].append(tool_tokens)
    payload["finish_reason"].append(finish_reason)
    payload["judge_grade"].append(reward.judge_grade)
    payload["context_reward"].append(reward.context_reward)
    payload["solution_page_reward"].append(reward.solution_page_reward)
    payload["scalar_reward"].append(reward.scalar_reward)
    payload["rendered_page_count"].append(reward.rendered_page_count)
    payload["judge_parse_failed"].append(reward.judge_parse_failed)
    payload["judge_response"].append(reward.judge_response)


def extend_rollout_payload(
    payload: dict[str, list[object]],
    addition: dict[str, list[object]],
) -> None:

    for key, values in addition.items():
        payload.setdefault(key, []).extend(values)


def group_prompt(group: AIMOGRPOGroup) -> str:

    if group.samples and group.samples[0].prompt.strip():
        return group.samples[0].prompt

    return group.problem


def prompt_to_record(prompt: object, prompt_index: int) -> dict[str, str]:

    if isinstance(prompt, dict):
        problem = str(prompt.get("problem") or prompt.get("prompt") or "")
        problem_id = str(prompt.get("problem_id") or prompt.get("id") or prompt_index)
        reference_solution = str(
            prompt.get("reference_solution")
            or prompt.get("reference")
            or ""
        )

        return {
            "problem": problem,
            "problem_id": problem_id,
            "reference_solution": reference_solution,
        }

    return {
        "problem": str(prompt),
        "problem_id": str(prompt_index),
        "reference_solution": "",
    }
