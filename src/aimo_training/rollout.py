from __future__ import annotations

from dataclasses import dataclass

from aimo_inference.client import AIMOInferenceClient
from aimo_inference.config import AIMOConfig
from aimo_inference.config import DUMMY_SERVED_MODEL_NAME
from aimo_inference.config import MAX_GENERATION_CONTEXT_TOKENS
from aimo_inference.prompts import AIMOPromptBuilder
from aimo_inference.sandbox import AIMOSandboxPool
from aimo_inference.sandbox import run_sandbox_pool_preflight
from aimo_inference.template import AIMOChatTemplate
from aimo_training.config import AIMOTrainingConfig
from aimo_training.queue import AIMODurableGroupQueue
from aimo_training.rewards import AIMORewardConfig
from aimo_training.rewards import AIMOTrainingRewardScorer
from aimo_training.schema import AIMOGRPOGroup
from aimo_training.schema import AIMORolloutSample
from aimo_training.schema import AIMOTrainingRecord
from aimo_training.tool_rollout import AIMOToolRolloutEngine


@dataclass(frozen=True)
class AIMORolloutCoordinatorSummary:

    written_group_count: int
    written_sample_count: int

    def as_dict(self) -> dict[str, int]:

        return {
            "written_group_count": self.written_group_count,
            "written_sample_count": self.written_sample_count,
        }


class AIMORolloutCoordinator:

    def __init__(self, config: AIMOTrainingConfig) -> None:

        self.config = config
        self.prompt_builder = AIMOPromptBuilder()
        self.chat_template = AIMOChatTemplate()
        self.rollout_config = build_rollout_inference_config(config)
        self.judge_config = build_judge_inference_config(config)
        self.rollout_client = AIMOInferenceClient(config=self.rollout_config)
        self.judge_client = AIMOInferenceClient(config=self.judge_config)
        self.rollout_sandbox_pool = AIMOSandboxPool(
            config=self.rollout_config,
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
            judge_client=self.judge_client,
            reward_config=AIMORewardConfig(weights=config.reward_weights),
            sandbox_pool=self.judge_sandbox_pool,
        )

    def write_groups(
        self,
        records: list[AIMOTrainingRecord],
        queue: AIMODurableGroupQueue,
    ) -> AIMORolloutCoordinatorSummary:

        written_group_count = 0
        written_sample_count = 0

        try:
            for group_index, record in enumerate(records):
                group = self.build_group(
                    record=record,
                    group_index=group_index,
                )
                queue.append_group(group)
                written_group_count += 1
                written_sample_count += len(group.samples)
        finally:
            self.close()

        return AIMORolloutCoordinatorSummary(
            written_group_count=written_group_count,
            written_sample_count=written_sample_count,
        )

    def build_group(
        self,
        record: AIMOTrainingRecord,
        group_index: int,
    ) -> AIMOGRPOGroup:

        samples = [
            self.build_sample(
                record=record,
                group_index=group_index,
                rollout_index=rollout_index,
            )
            for rollout_index in range(self.config.group_size)
        ]

        return AIMOGRPOGroup(
            group_index=group_index,
            problem_id=record.id,
            problem=record.problem,
            reference_solution=record.reference_solution,
            samples=samples,
            metadata=record.metadata,
        )

    def build_sample(
        self,
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
        with self.rollout_sandbox_pool.acquire() as sandbox:
            result = AIMOToolRolloutEngine(
                config=self.rollout_config,
                client=self.rollout_client,
                prompt_builder=self.prompt_builder,
                sandbox=sandbox,
            ).run_problem(
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
            endpoint_index=0,
            tool_tokens=result.tool_tokens,
            sampling_logprobs=result.token_logprobs,
        )

    def close(self) -> None:

        self.rollout_sandbox_pool.close()
        self.judge_sandbox_pool.close()


def build_rollout_inference_config(config: AIMOTrainingConfig) -> AIMOConfig:

    config = config.with_dummy_test_defaults()

    return AIMOConfig(
        model_path=config.model_path,
        contestant_model_path=config.model_path,
        dummy_test=config.dummy_test,
        dummy_model_path=config.dummy_model_path,
        logdir=config.logdir / "rollout",
        inference_mode="proof",
        model_profile="contestant",
        served_model_name=rollout_served_model_name(config),
        port=config.rollout_port,
        contestant_port=config.rollout_port,
        judge_port=config.judge_port,
        api_base=config.rollout_api_base,
        launch_server=False,
        reuse_server=True,
        tensor_parallel_size=config.rollout_tensor_parallel_size,
        num_gpus=config.rollout_tensor_parallel_size,
        max_num_seqs=config.rollout_max_num_seqs,
        max_num_batched_tokens=config.rollout_max_num_batched_tokens,
        max_model_len=MAX_GENERATION_CONTEXT_TOKENS,
        max_new_tokens=0,
        temperature=config.rollout_temperature,
        top_p=config.rollout_top_p,
        min_p=config.rollout_min_p,
        top_logprobs=0,
        max_logprobs=0,
        group_size=config.group_size,
        active_problem_count=config.active_problem_count,
        sandbox_count=config.sandbox_count,
        kv_cache_dtype=config.kv_cache_dtype,
        page_count_method=config.page_count_method,
        page_template=config.page_template,
        max_python_calls=config.max_python_calls,
        use_jupyter_sandbox=True,
        tool_protocol=config.tool_protocol,
        seed=config.seed,
        global_rank=config.global_rank,
        local_rank=config.local_rank,
        world_size=config.world_size,
        master_addr=config.master_addr,
        master_port=config.master_port,
        cuda_visible_devices=config.cuda_visible_devices,
    )


def build_judge_inference_config(config: AIMOTrainingConfig) -> AIMOConfig:

    config = config.with_dummy_test_defaults()

    return AIMOConfig(
        model_path=config.judge_model_path,
        judge_model_path=config.judge_model_path,
        dummy_test=config.dummy_test,
        dummy_model_path=config.dummy_model_path,
        logdir=config.logdir / "judge",
        inference_mode="judge",
        model_profile="judge",
        template_format=judge_template_format(config),
        served_model_name=judge_served_model_name(config),
        judge_served_model_name=judge_served_model_name(config),
        port=config.judge_port,
        judge_port=config.judge_port,
        api_base=config.judge_api_base,
        launch_server=False,
        reuse_server=True,
        tensor_parallel_size=config.judge_tensor_parallel_size,
        num_gpus=config.judge_tensor_parallel_size,
        max_num_seqs=config.judge_max_num_seqs,
        max_num_batched_tokens=config.judge_max_num_batched_tokens,
        max_model_len=MAX_GENERATION_CONTEXT_TOKENS,
        max_new_tokens=0,
        judge_max_tokens=0,
        moe_backend=judge_moe_backend(config),
        enable_expert_parallel=judge_enable_expert_parallel(config),
        kv_cache_dtype=config.kv_cache_dtype,
        page_count_method=config.page_count_method,
        page_template=config.page_template,
        max_python_calls=config.max_python_calls,
        use_jupyter_sandbox=True,
        tool_protocol=judge_tool_protocol(config),
        seed=config.seed,
        global_rank=config.global_rank,
        local_rank=config.local_rank,
        world_size=config.world_size,
        master_addr=config.master_addr,
        master_port=config.master_port,
        cuda_visible_devices=config.cuda_visible_devices,
    )


def rollout_served_model_name(config: AIMOTrainingConfig) -> str:

    if config.dummy_test:
        return DUMMY_SERVED_MODEL_NAME

    return "OLMo-3.1-32B-Think"


def judge_served_model_name(config: AIMOTrainingConfig) -> str:

    if config.dummy_test:
        return DUMMY_SERVED_MODEL_NAME

    return "GPT-OSS-120B"


def judge_template_format(config: AIMOTrainingConfig) -> str:

    if config.dummy_test:
        return "chatml"

    return "harmony"


def judge_tool_protocol(config: AIMOTrainingConfig) -> str:

    if config.dummy_test:
        return "olmo_chatml"

    return "harmony"


def judge_moe_backend(config: AIMOTrainingConfig) -> str:

    if config.dummy_test:
        return ""

    return "marlin"


def judge_enable_expert_parallel(config: AIMOTrainingConfig) -> bool:

    return not config.dummy_test
