from __future__ import annotations

from aimo_inference.sandbox import AIMOSandbox


def test_successful_python_execution_and_stdout_capture() -> None:

    sandbox = AIMOSandbox(timeout_seconds=2.0)

    result = sandbox.execute("print(2 + 3)")

    assert result.success is True
    assert result.output == "5"
    assert result.error == ""
    assert result.to_tool_payload() == "5"


def test_exception_cleanup_and_custom_payload() -> None:

    sandbox = AIMOSandbox(timeout_seconds=2.0)

    result = sandbox.execute("raise ValueError(\"bad\")")

    assert result.success is False
    assert "ValueError: bad" in result.error
    assert result.to_tool_payload() == result.error


def test_timeout_handling() -> None:

    sandbox = AIMOSandbox(timeout_seconds=0.05)

    result = sandbox.execute("while True:\n    pass")

    assert result.success is False
    assert result.timed_out is True
    assert result.to_tool_payload() == "Python execution timed out."


def test_output_truncation() -> None:

    sandbox = AIMOSandbox(
        timeout_seconds=2.0,
        max_output_chars=40,
    )

    result = sandbox.execute("print(\"x\" * 200)")

    assert result.success is True
    assert "[Truncated" in result.output
    assert len(result.output) < 100


def test_custom_error_rewrites() -> None:

    sandbox = AIMOSandbox()

    rewritten_error = sandbox._rewrite_error(
        "Traceback (most recent call last):\n"
        "NameError: name 'missing_value' is not defined\n"
    )

    assert "Ensure you define every variable" in rewritten_error
    assert "missing_value" in rewritten_error


def test_reset_and_close_are_safe_noops() -> None:

    sandbox = AIMOSandbox()

    assert sandbox.reset() is None
    assert sandbox.close() is None
