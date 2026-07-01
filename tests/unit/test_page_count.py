from __future__ import annotations

from pathlib import Path

import pytest

import aimo_inference.page_count as page_count_module
from aimo_inference.config import AIMOConfig
from aimo_inference.page_count import AIMOPageCounter
from aimo_inference.page_count import estimate_pages_by_lines
from aimo_inference.page_count import estimate_pages_by_words
from aimo_inference.page_count import sanitize_latex_body
from aimo_inference.page_count import strip_code_blocks_and_tool_transcripts


def test_code_block_and_tool_transcript_stripping() -> None:

    cleaned_solution = strip_code_blocks_and_tool_transcripts(
        "Proof line.\n"
        "```python\nprint(1)\n```\n"
        "stdout: hidden\n"
        "Tool output hidden\n"
        "Final line."
    )

    assert cleaned_solution == "Proof line.\nFinal line."


def test_canonical_latex_template_rendering() -> None:

    counter = AIMOPageCounter(
        config=AIMOConfig(
            page_template="Before <solution> After",
        )
    )

    assert counter._build_latex_document("x & y", sanitize=False) == "Before x & y After"
    assert counter._build_latex_document("x & y", sanitize=True) == "Before x \\& y After"


def test_pdfinfo_page_count_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class CompletedProcess:

        returncode = 0
        stdout = "Title: solution\nPages: 5\n"

    monkeypatch.setattr(page_count_module.shutil, "which", lambda command: "/usr/bin/pdfinfo")
    monkeypatch.setattr(
        page_count_module.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(),
    )
    pdf_path = tmp_path / "solution.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    counter = AIMOPageCounter(config=AIMOConfig())

    assert counter._count_pdf_pages(pdf_path) == 5


def test_page_rewards_for_rendered_page_counts() -> None:

    counter = AIMOPageCounter(config=AIMOConfig())

    assert counter._result("text", 4, "word_count", "not_attempted", "").reward == 1
    assert counter._result("text", 5, "word_count", "not_attempted", "").reward == 1
    assert counter._result("text", 3, "word_count", "not_attempted", "").reward == -1
    assert counter._result("text", 6, "word_count", "not_attempted", "").reward == -1


def test_sanitized_latex_fallback(monkeypatch: pytest.MonkeyPatch) -> None:

    counter = AIMOPageCounter(
        config=AIMOConfig(
            page_count_method="latex",
        )
    )

    def rendered_page_count(solution: str, sanitize: bool) -> int:

        if not sanitize:
            raise RuntimeError("raw render failed")

        return 4

    monkeypatch.setattr(counter, "_rendered_page_count", rendered_page_count)

    result = counter.count("x & y")

    assert result.method == "sanitized_latex"
    assert result.rendered_pages == 4
    assert result.reward == 1
    assert result.latex_compile_status == "success"
    assert result.fallback_reason == "raw render failed"


def test_line_count_and_word_count_fallbacks() -> None:

    line_solution = "\n".join([
        "line"
        for _ in range(90)
    ])
    word_solution = "word " * 2200
    line_result = AIMOPageCounter(
        config=AIMOConfig(
            page_count_method="line_count",
        )
    ).count(line_solution)
    word_result = AIMOPageCounter(
        config=AIMOConfig(
            page_count_method="word_count",
        )
    ).count(word_solution)

    assert estimate_pages_by_lines(line_solution) == 2
    assert estimate_pages_by_words(word_solution) == 4
    assert line_result.method == "line_count"
    assert line_result.rendered_pages == 2
    assert word_result.method == "word_count"
    assert word_result.rendered_pages == 4
    assert word_result.reward == 1


def test_sanitized_latex_and_method_logging() -> None:

    sanitized_body = sanitize_latex_body("Use 50% of $x_1$ and image ![x](a.png).")
    result = AIMOPageCounter(
        config=AIMOConfig(
            page_count_method="word_count",
        )
    ).count("Short proof.")

    assert "50\\%" in sanitized_body
    assert "$x_1$" in sanitized_body
    assert "a.png" not in sanitized_body
    assert result.as_metadata()["page_count_method"] == "word_count"
    assert result.as_metadata()["solution_page_reward"] == -1
