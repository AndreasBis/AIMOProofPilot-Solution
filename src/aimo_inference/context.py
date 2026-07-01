from __future__ import annotations


def available_output_tokens(max_model_len: int, prompt_tokens: int) -> int:

    return max(0, max_model_len - max(0, prompt_tokens))


def resolve_generation_max_tokens(
    configured_max_tokens: int,
    max_model_len: int,
    prompt_tokens: int,
) -> int:

    available_tokens = available_output_tokens(
        max_model_len=max_model_len,
        prompt_tokens=prompt_tokens,
    )

    if configured_max_tokens <= 0:
        return available_tokens

    return min(configured_max_tokens, available_tokens)
