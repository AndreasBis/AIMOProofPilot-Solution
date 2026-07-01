from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from aimo_inference.config import AIMOConfig


PAGE_COUNT_METHODS = {
    "latex",
    "sanitized_latex",
    "line_count",
    "word_count",
}

CANONICAL_PAGE_TEMPLATE = (
    "\\documentclass[12pt]{article}\n"
    "\\usepackage[a4paper,margin=1in]{geometry}\n"
    "\\usepackage{amsmath,amssymb,amsthm}\n"
    "\\setlength{\\parindent}{0pt}\n"
    "\\setlength{\\parskip}{0.6em}\n"
    "\\begin{document}\n"
    "<solution>\n"
    "\\end{document}\n"
)

CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
FUNCTION_CALL_PATTERN = re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL)
ENVIRONMENT_TURN_PATTERN = re.compile(
    r"<\|im_start\|>environment\n.*?<\|im_end\|>",
    re.DOTALL,
)
TOOL_TRANSCRIPT_PATTERN = re.compile(
    r"(?im)^\s*(python execution|tool output|sandbox output|stderr|stdout)\b.*$"
)
MATH_SEGMENT_PATTERN = re.compile(
    r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])",
    re.DOTALL,
)
IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)")
HTML_IMAGE_PATTERN = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
PDF_PAGE_PATTERN = re.compile(rb"/Type\s*/Page\b")
PDFINFO_PAGE_PATTERN = re.compile(r"(?im)^Pages:\s*(\d+)\s*$")


@dataclass(frozen=True)
class AIMOPageCountResult:

    rendered_pages: int
    method: str
    reward: int
    latex_compile_status: str
    fallback_reason: str
    word_count: int
    line_count: int
    solution_character_count: int

    def as_metadata(self) -> dict[str, str | int]:

        return {
            "rendered_pages": self.rendered_pages,
            "page_count_method": self.method,
            "solution_page_reward": self.reward,
            "latex_compile_status": self.latex_compile_status,
            "fallback_reason": self.fallback_reason,
            "word_count": self.word_count,
            "line_count": self.line_count,
            "solution_character_count": self.solution_character_count,
        }


class AIMOPageCounter:

    def __init__(self, config: AIMOConfig) -> None:

        self.config = config

    def count(self, solution: str) -> AIMOPageCountResult:

        cleaned_solution = strip_code_blocks_and_tool_transcripts(solution)
        fallback_reason = ""
        latex_compile_status = "not_attempted"

        for method in page_count_fallback_order(self.config.page_count_method):
            try:
                if method == "latex":
                    rendered_pages = self._rendered_page_count(
                        solution=cleaned_solution,
                        sanitize=False,
                    )
                    latex_compile_status = "success"

                    return self._result(
                        solution=cleaned_solution,
                        rendered_pages=rendered_pages,
                        method=method,
                        latex_compile_status=latex_compile_status,
                        fallback_reason=fallback_reason,
                    )

                if method == "sanitized_latex":
                    rendered_pages = self._rendered_page_count(
                        solution=cleaned_solution,
                        sanitize=True,
                    )
                    latex_compile_status = "success"

                    return self._result(
                        solution=cleaned_solution,
                        rendered_pages=rendered_pages,
                        method=method,
                        latex_compile_status=latex_compile_status,
                        fallback_reason=fallback_reason,
                    )

                if method == "line_count":
                    return self._result(
                        solution=cleaned_solution,
                        rendered_pages=estimate_pages_by_lines(cleaned_solution),
                        method=method,
                        latex_compile_status=latex_compile_status,
                        fallback_reason=fallback_reason,
                    )

                if method == "word_count":
                    return self._result(
                        solution=cleaned_solution,
                        rendered_pages=estimate_pages_by_words(cleaned_solution),
                        method=method,
                        latex_compile_status=latex_compile_status,
                        fallback_reason=fallback_reason,
                    )
            except Exception as error:
                fallback_reason = str(error)
                latex_compile_status = "failed"

        return self._result(
            solution=cleaned_solution,
            rendered_pages=estimate_pages_by_words(cleaned_solution),
            method="word_count",
            latex_compile_status=latex_compile_status,
            fallback_reason=fallback_reason or "All page-count methods failed.",
        )

    def _rendered_page_count(self, solution: str, sanitize: bool) -> int:

        latex_command_path = shutil.which(self.config.latex_command)

        if latex_command_path is None:
            raise FileNotFoundError(f"LaTeX command not found: {self.config.latex_command}")

        document = self._build_latex_document(solution=solution, sanitize=sanitize)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            tex_path = temporary_path / "solution.tex"
            pdf_path = temporary_path / "solution.pdf"
            tex_path.write_text(document, encoding="utf-8")
            completed_process = subprocess.run(
                [
                    latex_command_path,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    tex_path.name,
                ],
                cwd=temporary_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.config.page_count_timeout_seconds,
                check=False,
            )

            if completed_process.returncode != 0 or not pdf_path.exists():
                raise RuntimeError("LaTeX render failed.")

            return self._count_pdf_pages(pdf_path)

    def _build_latex_document(self, solution: str, sanitize: bool) -> str:

        body = sanitize_latex_body(solution) if sanitize else solution
        template = self._resolve_page_template()

        return template.replace("<solution>", body)

    def _resolve_page_template(self) -> str:

        page_template = self.config.page_template
        possible_path = Path(page_template)

        if "\n" not in page_template and possible_path.exists():
            return possible_path.read_text(encoding="utf-8")

        return page_template

    def _count_pdf_pages(self, pdf_path: Path) -> int:

        pdfinfo_command_path = shutil.which(self.config.pdfinfo_command)

        if pdfinfo_command_path is not None:
            completed_process = subprocess.run(
                [
                    pdfinfo_command_path,
                    str(pdf_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.config.page_count_timeout_seconds,
                check=False,
            )
            match = PDFINFO_PAGE_PATTERN.search(completed_process.stdout)

            if completed_process.returncode == 0 and match:
                return int(match.group(1))

        page_count = len(PDF_PAGE_PATTERN.findall(pdf_path.read_bytes()))

        if page_count < 1:
            raise ValueError(f"Unable to count PDF pages: {pdf_path}")

        return page_count

    def _result(
        self,
        solution: str,
        rendered_pages: int,
        method: str,
        latex_compile_status: str,
        fallback_reason: str,
    ) -> AIMOPageCountResult:

        reward = 1 if rendered_pages in {4, 5} else -1

        return AIMOPageCountResult(
            rendered_pages=rendered_pages,
            method=method,
            reward=reward,
            latex_compile_status=latex_compile_status,
            fallback_reason=fallback_reason,
            word_count=count_words(solution),
            line_count=count_non_empty_lines(solution),
            solution_character_count=len(solution),
        )


def strip_code_blocks_and_tool_transcripts(
    solution: str,
    preserve_blank_lines: bool = False,
) -> str:

    without_code_blocks = CODE_BLOCK_PATTERN.sub("", solution)
    without_function_calls = FUNCTION_CALL_PATTERN.sub("", without_code_blocks)
    without_environment_turns = ENVIRONMENT_TURN_PATTERN.sub("", without_function_calls)
    without_tool_transcripts = TOOL_TRANSCRIPT_PATTERN.sub("", without_environment_turns)

    if preserve_blank_lines:
        return "\n".join(
            line.rstrip()
            for line in without_tool_transcripts.splitlines()
        ).strip()

    compact_lines = [
        line.rstrip()
        for line in without_tool_transcripts.splitlines()
        if line.strip()
    ]

    return "\n".join(compact_lines).strip()


def page_count_fallback_order(method: str) -> list[str]:

    if method == "latex":
        return [
            "latex",
            "sanitized_latex",
            "line_count",
            "word_count",
        ]

    if method == "sanitized_latex":
        return [
            "sanitized_latex",
            "line_count",
            "word_count",
        ]

    if method == "line_count":
        return [
            "line_count",
            "word_count",
        ]

    return [
        "word_count",
    ]


def sanitize_latex_body(value: str) -> str:

    text = strip_image_markup(value)
    segments = MATH_SEGMENT_PATTERN.split(text)
    sanitized_segments = [
        segment if is_math_segment(segment) else escape_latex_text(segment)
        for segment in segments
    ]

    return "".join(sanitized_segments)


def strip_image_markup(text: str) -> str:

    return HTML_IMAGE_PATTERN.sub("", IMAGE_MARKDOWN_PATTERN.sub("", text))


def is_math_segment(value: str) -> bool:

    return (
        value.startswith("$")
        or value.startswith("\\(")
        or value.startswith("\\[")
    )


def escape_latex_text(value: str) -> str:

    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }

    return "".join(
        replacements.get(character, character)
        for character in value
    )


def estimate_pages_by_lines(solution: str) -> int:

    return max(1, math.ceil(count_non_empty_lines(solution) / 45))


def estimate_pages_by_words(solution: str) -> int:

    return max(1, math.ceil(count_words(solution) / 550))


def count_non_empty_lines(solution: str) -> int:

    return len([
        line
        for line in solution.splitlines()
        if line.strip()
    ])


def count_words(solution: str) -> int:

    return len(solution.split())


def page_template_from_environment() -> str:

    return os.environ.get("AIMO_PAGE_TEMPLATE", CANONICAL_PAGE_TEMPLATE)
