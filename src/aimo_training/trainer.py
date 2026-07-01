from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimo_training.config import AIMOTrainingConfig
from aimo_training.lora import AIMOLoRAConfig
from aimo_training.lora import inject_lora_adapters
from aimo_training.lora import load_lora_adapter
from aimo_training.lora import mark_only_lora_trainable
from aimo_training.lora import save_lora_adapter
from aimo_training.schema import AIMOGRPOGroup
from aimo_training.schema import AIMORolloutSample


@dataclass(frozen=True)
class AIMOTrainingStepSummary:

    step_index: int
    group_index: int
    problem_id: str
    loss: float
    mean_reward: float
    reward_std: float
    sample_count: int

    def as_dict(self) -> dict[str, int | float | str]:

        return {
            "step_index": self.step_index,
            "group_index": self.group_index,
            "problem_id": self.problem_id,
            "loss": self.loss,
            "mean_reward": self.mean_reward,
            "reward_std": self.reward_std,
            "sample_count": self.sample_count,
        }


class AIMOLegacyGRPOTrainer:

    def __init__(self, config: AIMOTrainingConfig) -> None:

        self.config = config

    def train(self, groups: list[AIMOGRPOGroup]) -> tuple[list[AIMOTrainingStepSummary], Path, Path]:

        if not groups:
            raise ValueError("No complete GRPO groups were available for training.")

        import torch
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        torch.manual_seed(self.config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        lora_config = AIMOLoRAConfig(
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            target_module_suffixes=self.config.resolved_target_module_suffixes,
        )
        replaced_modules = inject_lora_adapters(model=model, config=lora_config)

        if self.config.initial_adapter_path is not None:
            load_lora_adapter(
                model=model,
                adapter_path=self.config.initial_adapter_path,
            )

        mark_only_lora_trainable(model)
        device = self._resolve_device(torch)
        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        summaries: list[AIMOTrainingStepSummary] = []
        step_index = 0
        total_epochs = max(1, math.ceil(self.config.num_train_epochs))

        optimizer.zero_grad(set_to_none=True)

        for _ in range(total_epochs):
            for group_batch in batch_groups(groups, self.config.per_device_batch_size):
                losses = [
                    self._group_loss(
                        model=model,
                        tokenizer=tokenizer,
                        group=group,
                        device=device,
                        torch=torch,
                    )
                    for group in group_batch
                ]
                loss = torch.stack(losses).mean()
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()

                should_step = (
                    (step_index + 1) % self.config.gradient_accumulation_steps == 0
                    or step_index + 1 == total_epochs * math.ceil(
                        len(groups) / max(1, self.config.per_device_batch_size)
                    )
                )

                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        self.config.max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                batch_loss = float(loss.detach().cpu())

                for group in group_batch:
                    summaries.append(
                        self._step_summary(
                            step_index=step_index,
                            group=group,
                            loss=batch_loss,
                        )
                    )

                step_index += 1

        adapter_path = self.config.output_path / "adapter_model.safetensors"
        adapter_config_path = self.config.output_path / "adapter_config.json"
        save_lora_adapter(
            model=model,
            adapter_path=adapter_path,
            config_path=adapter_config_path,
            config=lora_config,
            replaced_modules=replaced_modules,
        )

        return summaries, adapter_path, adapter_config_path

    def _group_loss(
        self,
        model: object,
        tokenizer: object,
        group: AIMOGRPOGroup,
        device: object,
        torch: object,
    ) -> object:

        advantages = normalize_rewards(group.rewards)
        sample_losses = []

        for sample, advantage in zip(group.samples, advantages, strict=True):
            encoded = encode_prompt_and_completion(
                tokenizer=tokenizer,
                sample=sample,
                max_model_len=self.config.max_model_len,
            )
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
            labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                dim=-1,
                index=shifted_labels.clamp_min(0).unsqueeze(-1),
            ).squeeze(-1)
            mask = shifted_labels.ne(-100)

            if mask.any():
                completion_log_probs = selected_log_probs[mask]
                sample_losses.append(
                    self._sample_policy_loss(
                        completion_log_probs=completion_log_probs,
                        sample=sample,
                        advantage=advantage,
                        torch=torch,
                        device=device,
                    )
                )

        if not sample_losses:
            raise ValueError(f"GRPO group {group.group_index} has no trainable completion tokens.")

        return torch.stack(sample_losses).mean()

    def _step_summary(
        self,
        step_index: int,
        group: AIMOGRPOGroup,
        loss: float,
    ) -> AIMOTrainingStepSummary:

        rewards = group.rewards
        mean_reward = sum(rewards) / len(rewards)
        variance = sum(
            (reward - mean_reward) ** 2
            for reward in rewards
        ) / max(1, len(rewards))

        return AIMOTrainingStepSummary(
            step_index=step_index,
            group_index=group.group_index,
            problem_id=group.problem_id,
            loss=loss,
            mean_reward=mean_reward,
            reward_std=math.sqrt(variance),
            sample_count=len(group.samples),
        )

    def _sample_policy_loss(
        self,
        completion_log_probs: object,
        sample: AIMORolloutSample,
        advantage: float,
        torch: object,
        device: object,
    ) -> object:

        old_token_logprobs = trainable_token_logprobs(sample)

        if old_token_logprobs and len(old_token_logprobs) == len(completion_log_probs):
            old_log_probs = torch.tensor(
                old_token_logprobs,
                dtype=completion_log_probs.dtype,
                device=device,
            )
            ratio = torch.exp(completion_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.config.grpo_epsilon,
                1.0 + self.config.grpo_epsilon,
            )
            advantage_tensor = torch.tensor(
                float(advantage),
                dtype=completion_log_probs.dtype,
                device=device,
            )
            policy_loss = -torch.minimum(
                ratio * advantage_tensor,
                clipped_ratio * advantage_tensor,
            ).mean()
            kl_loss = self.config.kl_beta * (completion_log_probs - old_log_probs).pow(2).mean()

            return policy_loss + kl_loss

        return -completion_log_probs.mean() * float(advantage)

    def _resolve_device(self, torch: object) -> object:

        if torch.cuda.is_available():
            if self.config.local_rank < torch.cuda.device_count():
                return torch.device(f"cuda:{self.config.local_rank}")

            return torch.device("cuda")

        return torch.device("cpu")


def normalize_rewards(rewards: list[float]) -> list[float]:

    if not rewards:
        return []

    mean_reward = sum(rewards) / len(rewards)
    variance = sum(
        (reward - mean_reward) ** 2
        for reward in rewards
    ) / max(1, len(rewards))
    std_reward = math.sqrt(variance)

    if std_reward <= 1e-8:
        return [
            0.0
            for _ in rewards
        ]

    return [
        (reward - mean_reward) / std_reward
        for reward in rewards
    ]


def batch_groups(
    groups: list[AIMOGRPOGroup],
    batch_size: int,
) -> list[list[AIMOGRPOGroup]]:

    resolved_batch_size = max(1, batch_size)

    return [
        groups[index: index + resolved_batch_size]
        for index in range(0, len(groups), resolved_batch_size)
    ]


def encode_prompt_and_completion(
    tokenizer: object,
    sample: AIMORolloutSample,
    max_model_len: int,
) -> dict[str, list[int]]:

    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    completion_ids = (
        sample.token_ids
        if sample.token_ids
        else tokenizer.encode(sample.completion, add_special_tokens=False)
    )
    completion_labels = build_completion_labels(
        completion_ids=completion_ids,
        env_mask=sample.env_mask,
    )
    input_ids = prompt_ids + completion_ids
    labels = [-100 for _ in prompt_ids] + completion_labels

    if len(input_ids) > max_model_len:
        overflow = len(input_ids) - max_model_len
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def build_completion_labels(
    completion_ids: list[int],
    env_mask: list[int],
) -> list[int]:

    if not env_mask:
        return list(completion_ids)

    if len(env_mask) != len(completion_ids):
        raise ValueError("env_mask length must match completion_ids length.")

    return [
        token_id if int(mask_value) else -100
        for token_id, mask_value in zip(completion_ids, env_mask, strict=True)
    ]


def trainable_token_logprobs(sample: AIMORolloutSample) -> list[float]:

    if not sample.env_mask:
        return list(sample.token_logprobs)

    if len(sample.env_mask) != len(sample.token_logprobs):
        raise ValueError("env_mask length must match token_logprobs length.")

    return [
        logprob
        for logprob, mask_value in zip(sample.token_logprobs, sample.env_mask, strict=True)
        if int(mask_value)
    ]
