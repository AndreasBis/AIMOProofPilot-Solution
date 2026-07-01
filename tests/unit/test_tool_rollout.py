from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aimo_inference.client import AIMOCompletionGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.sandbox import AIMOSandboxResult
from aimo_training.tool_rollout import AIMOToolRolloutEngine


class FakeTokenizer:

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:

        return [
            ord(character) % 251
            for character in text
        ]


class FakeGenerationClient:

    def __init__(self) -> None:

        self.requests: list[list[int]] = []
        self.generations = [
            AIMOCompletionGeneration(
                text="<function_calls>\npython(code=\"print(2 + 2)\")\n</function_calls>",
                token_ids=[
                    10,
                    11,
                    12,
                ],
                token_logprobs=[
                    -0.3,
                    -0.2,
                    -0.1,
                ],
                input_tokens=4,
                output_tokens=3,
                finish_reason="stop",
                latency_seconds=0.1,
                entropy=None,
                raw={},
            ),
            AIMOCompletionGeneration(
                text="The calculation gives 4, so the proof is complete.",
                token_ids=[
                    13,
                    14,
                ],
                token_logprobs=[
                    -0.5,
                    -0.4,
                ],
                input_tokens=12,
                output_tokens=2,
                finish_reason="stop",
                latency_seconds=0.1,
                entropy=None,
                raw={},
            ),
        ]

    def complete_token_ids(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
    ) -> AIMOCompletionGeneration:

        self.requests.append(prompt_token_ids)

        return self.generations.pop(0)


class FakeSandbox:

    def __init__(self) -> None:

        self.codes: list[str] = []

    def execute(self, code: str) -> AIMOSandboxResult:

        self.codes.append(code)

        return AIMOSandboxResult(
            success=True,
            output="4",
            error="",
            timed_out=False,
        )


def test_olmo_chatml_function_call_executes_and_masks_environment_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer(),
            ),
        ),
    )
    client = FakeGenerationClient()
    sandbox = FakeSandbox()
    config = AIMOConfig(
        model_path=tmp_path / "model",
        enable_tools=True,
        tool_protocol="olmo_chatml",
        max_python_calls=4,
    )

    result = AIMOToolRolloutEngine(
        config=config,
        client=client,
        sandbox=sandbox,
    ).run_problem("Compute 2 + 2.")

    assert sandbox.codes == [
        "print(2 + 2)",
    ]
    assert result.python_calls == 1
    assert result.python_errors == 0
    assert result.env_mask[:3] == [
        1,
        1,
        1,
    ]
    assert result.env_mask[-2:] == [
        1,
        1,
    ]
    assert 0 in result.env_mask[3:-2]
    assert len(result.completion_ids) == len(result.token_logprobs) == len(result.env_mask)
    assert "<function_calls>" not in result.proof_text
