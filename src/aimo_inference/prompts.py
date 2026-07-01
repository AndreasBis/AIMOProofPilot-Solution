from __future__ import annotations

from aimo_inference.template import AIMOChatMessage


CONTESTANT_SYSTEM_PROMPT = (
    "You are a contestant in the International Mathematical Olympiad. "
    "Solve the given IMO challenge by reducing it to its foundational components "
    "and reasoning from first principles. "
    "Check each lemma, proof step, and calculation until the solution is factual "
    "and coherent. "
    "Write a complete 4 to 5 page text-only solution that explains the reasoning "
    "step by step, with no dead ends or failed attempts. "
    "Use mathematical notation, LaTeX, and markdown when useful. "
    "Do not include code, diagrams, pictures, or images in the final solution. "
    "Internet access is disabled. "
    "Use the Python tool to verify lemmas, arithmetic, examples, and symbolic "
    "or computational claims whenever that verification is useful."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a judge on the International Mathematical Olympiad committee. "
    "Grade the contestant solution using exactly one score from 0, 1, 6, or 7. "
    "Use 0 when the solution is not close and has no meaningful partial progress. "
    "Use 1 when it is not close but contains meaningful partial progress. "
    "Use 6 when it is close to complete but has a material error or omission. "
    "Use 7 only when it is complete without a material error or omission. "
    "Internet access is disabled. "
    "Use the Python tool to verify lemmas, arithmetic, examples, and symbolic "
    "or computational claims whenever that verification is useful."
)

TOOL_PROMPT = (
    "Use this tool to execute Python code in a stateful Jupyter notebook. "
    "The sandbox includes math, statistics, random, collections, itertools, "
    "functools, fractions, decimal, sympy, numpy, mpmath, networkx, and z3. "
    "Use Fraction and Decimal when exact rational or decimal arithmetic is useful. "
    "The tool returns stdout, stderr, a compact error message, or times out "
    "after 10 seconds."
)

CONTESTANT_REPAIR_PROMPT = (
    "Read the provided solution to the IMO challenge and verify whether it solves "
    "the problem. "
    "Identify all mathematical mistakes, gaps, and unsupported claims. "
    "Then rewrite the solution so it is factual, coherent, and solves the problem "
    "correctly.\n\n"
    "Problem:\n{problem_text}\n\n"
    "Solution:\n{previous_pass_solution}"
)

JUDGE_EVALUATION_PROMPT = (
    "Problem:\n{problem}\n\n"
    "Reference solution or rubric:\n{reference}\n\n"
    "Contestant solution:\n{solution}\n\n"
    "Identify the material issues in the solution and evaluate the mathematical "
    "reasoning. "
    "End with exactly one grade from 0, 1, 6, or 7 inside \\boxed{{}}. "
    "Example: \\boxed{{0}}."
)


class AIMOPromptBuilder:

    def build_first_pass_messages(
        self,
        problem_text: str,
        enable_tools: bool,
    ) -> list[AIMOChatMessage]:

        return [
            AIMOChatMessage(
                role="system",
                content=self.build_system_prompt(enable_tools=enable_tools),
            ),
            AIMOChatMessage(
                role="user",
                content=self.build_first_pass_prompt(problem_text=problem_text),
            ),
        ]

    def build_second_pass_messages(
        self,
        problem_text: str,
        first_solution: str,
        first_tool_output: str,
        enable_tools: bool,
    ) -> list[AIMOChatMessage]:

        return [
            AIMOChatMessage(
                role="system",
                content=self.build_system_prompt(enable_tools=enable_tools),
            ),
            AIMOChatMessage(
                role="user",
                content=self.build_second_pass_prompt(
                    problem_text=problem_text,
                    first_solution=first_solution,
                    first_tool_output=first_tool_output,
                ),
            ),
        ]

    def build_third_pass_messages(
        self,
        problem_text: str,
        first_solution: str,
        repaired_solution: str,
        second_tool_output: str,
        enable_tools: bool,
    ) -> list[AIMOChatMessage]:

        return [
            AIMOChatMessage(
                role="system",
                content=self.build_system_prompt(enable_tools=enable_tools),
            ),
            AIMOChatMessage(
                role="user",
                content=self.build_third_pass_prompt(
                    problem_text=problem_text,
                    first_solution=first_solution,
                    repaired_solution=repaired_solution,
                    second_tool_output=second_tool_output,
                ),
            ),
        ]

    def build_system_prompt(self, enable_tools: bool) -> str:

        if enable_tools:
            return f"{CONTESTANT_SYSTEM_PROMPT}\n\n{TOOL_PROMPT}"

        return CONTESTANT_SYSTEM_PROMPT

    def build_first_pass_prompt(self, problem_text: str) -> str:

        return self._clean_text(problem_text)

    def build_second_pass_prompt(
        self,
        problem_text: str,
        first_solution: str,
        first_tool_output: str,
    ) -> str:

        return CONTESTANT_REPAIR_PROMPT.format(
            problem_text=self._clean_text(problem_text),
            previous_pass_solution=self._clean_text(first_solution),
        )

    def build_third_pass_prompt(
        self,
        problem_text: str,
        first_solution: str,
        repaired_solution: str,
        second_tool_output: str,
    ) -> str:

        return CONTESTANT_REPAIR_PROMPT.format(
            problem_text=self._clean_text(problem_text),
            previous_pass_solution=self._clean_text(repaired_solution),
        )

    def _clean_text(self, text: str) -> str:

        return str(text).strip()


class AIMOJudgePromptBuilder:

    def build_messages(
        self,
        problem: str,
        proof: str,
        reference: str = "",
        enable_tools: bool = True,
    ) -> list[AIMOChatMessage]:

        return [
            AIMOChatMessage(
                role="system",
                content=self.build_system_prompt(enable_tools=enable_tools),
            ),
            AIMOChatMessage(
                role="user",
                content=JUDGE_EVALUATION_PROMPT.format(
                    problem=str(problem).strip(),
                    solution=str(proof).strip(),
                    reference=str(reference).strip() or "No reference solution was provided.",
                ),
            ),
        ]

    def build_system_prompt(self, enable_tools: bool) -> str:

        if enable_tools:
            return f"{JUDGE_SYSTEM_PROMPT}\n\n{TOOL_PROMPT}"

        return JUDGE_SYSTEM_PROMPT


class AIMOAnswerPromptBuilder:

    def build_messages(
        self,
        problem: str,
        enable_tools: bool = True,
    ) -> list[AIMOChatMessage]:

        return [
            AIMOChatMessage(
                role="system",
                content=self.build_system_prompt(enable_tools=enable_tools),
            ),
            AIMOChatMessage(
                role="user",
                content=str(problem).strip(),
            ),
        ]

    def build_system_prompt(self, enable_tools: bool) -> str:

        if enable_tools:
            return f"{CONTESTANT_SYSTEM_PROMPT}\n\n{TOOL_PROMPT}"

        return CONTESTANT_SYSTEM_PROMPT
