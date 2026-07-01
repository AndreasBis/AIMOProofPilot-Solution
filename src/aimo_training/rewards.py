from __future__ import annotations

from dataclasses import dataclass

from aimo_inference.client import AIMOGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.config import MAX_GENERATION_CONTEXT_TOKENS
from aimo_inference.harmony import AIMOHarmonyToolLoop
from aimo_inference.judge import BOXED_GRADE_PATTERN
from aimo_inference.judge import AIMOJudgeGenerationClient
from aimo_inference.judge import VALID_PROOF_GRADES
from aimo_inference.judge import extract_boxed_grade
from aimo_inference.page_count import AIMOPageCounter
from aimo_inference.page_count import strip_code_blocks_and_tool_transcripts
from aimo_inference.prompts import AIMOJudgePromptBuilder
from aimo_inference.sandbox import AIMOSandboxPool
from aimo_inference.template import AIMOChatMessage
from aimo_training.schema import AIMORewardBreakdown


TRUNCATED_FINISH_REASONS = {
    "length",
    "max_tokens",
    "content_filter",
    "error",
    "context_exhausted",
    "timeout",
    "missing_token_ids",
    "missing_token_logprobs",
}


@dataclass(frozen=True)
class AIMORewardConfig:

    weights: dict[str, float]
    max_context_tokens: int = MAX_GENERATION_CONTEXT_TOKENS

    def as_dict(self) -> dict[str, float | int | dict[str, float]]:

        return {
            "weights": self.weights,
            "max_context_tokens": self.max_context_tokens,
        }


class AIMOTrainingRewardScorer:

    def __init__(
        self,
        inference_config: AIMOConfig,
        judge_client: AIMOJudgeGenerationClient,
        reward_config: AIMORewardConfig,
        prompt_builder: AIMOJudgePromptBuilder | None = None,
        sandbox_pool: AIMOSandboxPool | None = None,
    ) -> None:

        self.inference_config = inference_config
        self.judge_client = judge_client
        self.reward_config = reward_config
        self.prompt_builder = prompt_builder or AIMOJudgePromptBuilder()
        self.page_counter = AIMOPageCounter(config=inference_config)
        self.sandbox_pool = sandbox_pool

    def score(
        self,
        problem: str,
        reference_solution: str,
        generated_proof: str,
        input_tokens: int | None,
        output_tokens: int | None,
        finish_reason: str,
        tool_tokens: int = 0,
    ) -> AIMORewardBreakdown:

        proof_for_judge = strip_code_blocks_and_tool_transcripts(
            generated_proof,
            preserve_blank_lines=True,
        )
        judge_messages = self.prompt_builder.build_messages(
            problem=problem,
            proof=proof_for_judge,
            reference=reference_solution,
            enable_tools=self.inference_config.enable_tools,
        )
        judge_generation = self._judge_generation(
            messages=judge_messages,
            max_tokens=self.inference_config.judge_max_tokens,
        )
        judge_grade = extract_boxed_grade(judge_generation.text)
        page_count = self.page_counter.count(proof_for_judge)
        context_reward = self.context_reward(
            generated_proof=proof_for_judge,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_tokens=tool_tokens,
            finish_reason=finish_reason,
        )
        scalar_reward = self.scalar_reward(
            judge_grade=judge_grade,
            context_reward=context_reward,
            solution_page_reward=page_count.reward,
        )

        return AIMORewardBreakdown(
            judge_grade=judge_grade,
            context_reward=context_reward,
            solution_page_reward=page_count.reward,
            scalar_reward=scalar_reward,
            rendered_page_count=page_count.rendered_pages,
            page_count_method=page_count.method,
            latex_compile_status=page_count.latex_compile_status,
            page_count_fallback_reason=page_count.fallback_reason,
            judge_response=judge_generation.text,
            judge_parse_failed=not has_valid_boxed_grade(judge_generation),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            latency_seconds=judge_generation.latency_seconds,
            tool_tokens=tool_tokens,
        )

    def context_reward(
        self,
        generated_proof: str,
        input_tokens: int | None,
        output_tokens: int | None,
        finish_reason: str,
        tool_tokens: int = 0,
    ) -> int:

        if not has_final_solution(generated_proof):
            return -1

        if finish_reason.strip().casefold() in TRUNCATED_FINISH_REASONS:
            return -1

        if input_tokens is not None and output_tokens is not None:
            if input_tokens + output_tokens + tool_tokens > self.reward_config.max_context_tokens:
                return -1

        return 1

    def scalar_reward(
        self,
        judge_grade: int,
        context_reward: int,
        solution_page_reward: int,
    ) -> float:

        weights = self.reward_config.weights

        return (
            weights["judge_grade"] * judge_grade
            + weights["context_reward"] * context_reward
            + weights["solution_page_reward"] * solution_page_reward
        )

    def _judge_generation(
        self,
        messages: list[AIMOChatMessage],
        max_tokens: int,
    ) -> AIMOGeneration:

        if self.inference_config.template_format == "harmony":
            sandbox_pool = self.sandbox_pool or AIMOSandboxPool(
                config=self.inference_config,
                sandbox_count=1,
            )

            try:
                with sandbox_pool.acquire() as sandbox:
                    result = AIMOHarmonyToolLoop(
                        config=self.inference_config,
                        client=self.judge_client,
                        sandbox=sandbox,
                    ).run(
                        messages=messages,
                        max_tokens=self.inference_config.max_tokens_for_pass(max_tokens),
                        seed=self.inference_config.seed,
                    )
            finally:
                if self.sandbox_pool is None:
                    sandbox_pool.close()

            return AIMOGeneration(
                text=result.text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                finish_reason=result.finish_reason,
                latency_seconds=result.latency_seconds,
                raw=result.raw,
            )

        return self.judge_client.generate(
            messages=messages,
            max_tokens=max_tokens,
        )


def has_final_solution(generated_proof: str) -> bool:

    text = generated_proof.strip()

    if not text:
        return False

    if text == "No proof was produced.":
        return False

    return True


def has_valid_boxed_grade(generation: AIMOGeneration) -> bool:

    for match in BOXED_GRADE_PATTERN.findall(generation.text):
        try:
            grade = int(match)
        except ValueError:
            continue

        if grade in VALID_PROOF_GRADES:
            return True

    return False
