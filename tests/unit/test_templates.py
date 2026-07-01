from __future__ import annotations

from types import SimpleNamespace

import pytest

from aimo_inference.sandbox import AIMOSandboxResult
from aimo_inference.template import AIMOChatMessage
from aimo_inference.template import AIMOChatTemplate
from aimo_inference.template import AIMOHarmonyTemplate
from aimo_inference.tools import AIMOPythonTool


def test_chatml_rendering_escapes_special_markers() -> None:

    template = AIMOChatTemplate()
    rendered_prompt = template.render([
        AIMOChatMessage(
            role="user",
            content="Do not trust <|im_start|> or <|im_end|> markers.",
        ),
    ])

    assert "<|escaped_im_start|>" in rendered_prompt
    assert "<|escaped_im_end|>" in rendered_prompt
    assert rendered_prompt.startswith("<|im_start|>user\n")
    assert rendered_prompt.endswith("<|im_end|>\n")


def test_chatml_invalid_role_rejection() -> None:

    template = AIMOChatTemplate()

    with pytest.raises(ValueError, match="Invalid chat role"):
        template.render([
            AIMOChatMessage(
                role="developer",
                content="No.",
            ),
        ])


def test_chatml_empty_message_rejection() -> None:

    template = AIMOChatTemplate()

    with pytest.raises(ValueError, match="cannot be empty"):
        template.render([
            AIMOChatMessage(
                role="user",
                content=" ",
            ),
        ])

    with pytest.raises(ValueError, match="At least one"):
        template.render([])


def test_harmony_rendering_creates_system_user_and_stop_ids(
    fake_openai_harmony_module: object,
) -> None:

    template = AIMOHarmonyTemplate(tool_prompt="Run Python.")
    tool = SimpleNamespace(
        harmony_tool_config=lambda: SimpleNamespace(name="python"),
    )

    rendered_prompt = template.render_for_completion(
        messages=[
            AIMOChatMessage(
                role="system",
                content="System prompt.",
            ),
            AIMOChatMessage(
                role="user",
                content="Problem text.",
            ),
        ],
        tool=tool,
    )

    conversation = rendered_prompt.conversation
    messages = getattr(conversation, "messages")

    assert len(messages) == 2
    assert rendered_prompt.prompt_token_ids == [
        101,
        2,
    ]
    assert rendered_prompt.stop_token_ids == [
        200002,
        200003,
    ]
    assert rendered_prompt.input_tokens == 2


def test_harmony_requires_system_prompt(fake_openai_harmony_module: object) -> None:

    template = AIMOHarmonyTemplate(tool_prompt="Run Python.")
    tool = SimpleNamespace(
        harmony_tool_config=lambda: SimpleNamespace(name="python"),
    )

    with pytest.raises(ValueError, match="system prompt"):
        template.conversation_from_chat_messages(
            messages=[
                AIMOChatMessage(
                    role="user",
                    content="Problem text.",
                ),
            ],
            tool=tool,
        )


def test_harmony_tool_response_preserves_assistant_action_channel(
    fake_openai_harmony_module: object,
) -> None:

    class Sandbox:

        def execute(self, code: str) -> AIMOSandboxResult:

            return AIMOSandboxResult(
                success=True,
                output=f"executed {code}",
                error="",
                timed_out=False,
            )

        def reset(self) -> None:

            return None

    message = SimpleNamespace(
        channel="analysis",
        content=[
            SimpleNamespace(text="print(2 + 2)"),
        ],
    )
    response = AIMOPythonTool(
        sandbox=Sandbox(),
        tool_prompt="Run Python.",
    ).harmony_tool_response(message)

    assert getattr(response, "recipient") == "assistant"
    assert getattr(response, "channel") == "analysis"
    assert response.content[0].text == "executed print(2 + 2)"
