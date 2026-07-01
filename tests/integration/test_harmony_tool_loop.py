from __future__ import annotations

from types import SimpleNamespace

from aimo_inference.client import AIMOCompletionGeneration
from aimo_inference.config import AIMOConfig
from aimo_inference.harmony import AIMOHarmonyToolLoop
from aimo_inference.template import AIMOChatMessage
from conftest import FakeSandbox


class FakeHarmonyMessage:

    def __init__(
        self,
        text: str,
        channel: str,
        recipient: str | None = None,
    ) -> None:

        self.channel = channel
        self.recipient = recipient
        self.content = [
            SimpleNamespace(text=text),
        ]


class FakeHarmonyConversation:

    def __init__(self) -> None:

        self.messages: list[object] = []


class FakeHarmonyEncoding:

    def render_conversation_for_completion(
        self,
        conversation: object,
        role: object,
    ) -> list[int]:

        return [
            1,
            len(getattr(conversation, "messages", [])),
        ]


class FakeHarmonyTemplate:

    def __init__(self) -> None:

        self.encoding = FakeHarmonyEncoding()

    def render_for_completion(
        self,
        messages: list[AIMOChatMessage],
        tool: object,
    ) -> object:

        return SimpleNamespace(
            conversation=FakeHarmonyConversation(),
            prompt_token_ids=[
                1,
            ],
            stop_token_ids=[
                99,
            ],
            input_tokens=1,
        )

    def parse_completion(self, token_ids: list[int]) -> list[object]:

        if token_ids == [
            10,
        ]:
            return [
                FakeHarmonyMessage(
                    text="print(2 + 2)",
                    channel="analysis",
                    recipient="python",
                ),
            ]

        return [
            FakeHarmonyMessage(
                text="Final proof.",
                channel="final",
            ),
        ]

    def is_final_message(self, message: object) -> bool:

        return getattr(message, "channel", None) == "final"

    def is_python_tool_call(self, message: object) -> bool:

        return getattr(message, "recipient", None) == "python"

    def text_from_message(self, message: object) -> str:

        return str(message.content[0].text)

    def text_from_messages(self, messages: list[object]) -> str:

        return "\n".join(
            self.text_from_message(message)
            for message in messages
        )

    def encode_text(self, text: str, allow_special: bool = False) -> list[int]:

        return [
            1,
            2,
        ]


class FakeCompletionClient:

    def __init__(self) -> None:

        self.calls: list[dict[str, object]] = []
        self.generations = [
            AIMOCompletionGeneration(
                text="tool",
                token_ids=[
                    10,
                ],
                input_tokens=1,
                output_tokens=1,
                finish_reason="stop",
                latency_seconds=0.1,
                entropy=0.5,
                raw={
                    "step": 1,
                },
            ),
            AIMOCompletionGeneration(
                text="final",
                token_ids=[
                    20,
                ],
                input_tokens=3,
                output_tokens=1,
                finish_reason="stop",
                latency_seconds=0.2,
                entropy=0.7,
                raw={
                    "step": 2,
                },
            ),
        ]

    def complete_token_ids(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
    ) -> AIMOCompletionGeneration:

        self.calls.append({
            "prompt_token_ids": prompt_token_ids,
            "max_tokens": max_tokens,
            "stop_token_ids": stop_token_ids,
            "seed": seed,
        })

        return self.generations.pop(0)


def test_fake_gpt_oss_tool_loop(
    fake_openai_harmony_module: object,
) -> None:

    sandbox = FakeSandbox()
    client = FakeCompletionClient()
    loop = AIMOHarmonyToolLoop(
        config=AIMOConfig(
            max_python_calls=2,
        ),
        client=client,
        sandbox=sandbox,
        template=FakeHarmonyTemplate(),
    )

    result = loop.run(
        messages=[
            AIMOChatMessage(
                role="system",
                content="System.",
            ),
            AIMOChatMessage(
                role="user",
                content="Problem.",
            ),
        ],
        max_tokens=8,
        seed=123,
    )

    assert result.text == "Final proof."
    assert result.python_calls == 1
    assert result.python_errors == 0
    assert result.tool_tokens == 2
    assert result.output_tokens == 2
    assert abs(result.entropy - 0.6) < 1e-9
    assert sandbox.codes == [
        "print(2 + 2)",
    ]
    assert client.calls[0]["stop_token_ids"] == [
        99,
    ]
