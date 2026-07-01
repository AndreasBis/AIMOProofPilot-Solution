from __future__ import annotations

from pathlib import Path

from aimo_inference.client import AIMOCompletionGeneration
from aimo_inference.config import AIMOConfig
from aimo_training.rewards import AIMORewardConfig
from aimo_training.rewards import AIMOTrainingRewardScorer


class FakeHarmonyJudgeClient:

    def __init__(self) -> None:

        self.chat_call_count = 0
        self.completion_call_count = 0

    def generate(self, messages: list[object], max_tokens: int) -> object:

        self.chat_call_count += 1

        raise AssertionError("Harmony judge must not use chat completions.")

    def complete_token_ids(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
    ) -> AIMOCompletionGeneration:

        self.completion_call_count += 1

        return AIMOCompletionGeneration(
            text="The proof is complete. \\boxed{7}",
            token_ids=[],
            token_logprobs=[],
            input_tokens=len(prompt_token_ids),
            output_tokens=8,
            finish_reason="stop",
            latency_seconds=0.2,
            entropy=None,
            raw={},
        )


def test_harmony_training_reward_judge_uses_completion_tool_loop(
    tmp_path: Path,
    fake_openai_harmony_module: object,
) -> None:

    client = FakeHarmonyJudgeClient()
    scorer = AIMOTrainingRewardScorer(
        inference_config=AIMOConfig(
            model_path=tmp_path / "judge",
            template_format="harmony",
            use_jupyter_sandbox=False,
            enable_tools=True,
            page_count_method="word_count",
        ),
        judge_client=client,
        reward_config=AIMORewardConfig(
            weights={
                "judge_grade": 1.0,
                "context_reward": 1.0,
                "solution_page_reward": 1.0,
            },
        ),
    )

    reward = scorer.score(
        problem="Problem.",
        reference_solution="Reference.",
        generated_proof="A complete proof.",
        input_tokens=10,
        output_tokens=8,
        finish_reason="stop",
    )

    assert client.chat_call_count == 0
    assert client.completion_call_count == 1
    assert reward.judge_grade == 7
    assert reward.judge_parse_failed is False
