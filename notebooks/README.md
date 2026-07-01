# Notebooks

This directory keeps the inference notebook iterations that led to the final Kaggle-facing submission notebook.

The final Kaggle notebook is under `kaggle/aimoproofpilot-submission.ipynb`. It is a Kaggle-path and dependency adaptation of `notebooks/submission_v6.ipynb`.

## Version History

`submission_v1.ipynb` was the full tool-assisted prototype. It used a two-pass solve-and-audit flow, exposed a Python tool through vLLM chat tool calling, enabled `--enable-auto-tool-choice` with the `olmo3` tool-call parser, kept fallback parsing for `<function_calls>`, JSON tool calls, `python(...)` calls, and fenced Python blocks, and executed code in a stateful Jupyter kernel. Its metadata tracked `python_calls` and `python_errors`.

`submission_v2.ipynb` removed the Python tool path. Compared with v1, it removed the Jupyter kernel sandbox, tool schemas, tool prompt, tool-call parsing, tool execution loop, vLLM auto-tool flags, and Python-call metadata. It kept the two-pass solve-and-audit structure, switched generation to plain chat text, enabled `VLLM_USE_V2_MODEL_RUNNER`, increased the stream interval default from `128` to `256`, and changed the request timeout default from `1500` to `1560` seconds.

`submission_v3.ipynb` removed the audit pass. Compared with v2, it deleted the second-pass system prompt, second-pass user prompt, and `audit` generation call, leaving one long `solve` pass per problem. It also raised `AIMO_MAX_NUM_BATCHED_TOKENS` from `4096` to `8192`, raised the request timeout from `1560` to `3420` seconds, raised the pass timeout from `1500` to `3300` seconds, and added `--generation-config vllm` to the vLLM launch command.

`submission_v4.ipynb` was a prompt-only revision from v3. It added explicit English-only reasoning and final-answer instructions, required internal translation or restatement for non-English problem text, expanded the proof obligations to include construction, injection, bijection, and recurrence claims, and warned against relying on numerical patterns, asymptotic intuition, or plausibility arguments without proof. The runtime structure stayed as the single-pass v3 pipeline.

`submission_v5.ipynb` changed the default run shape and relaxed the strictest v4 prompt language. Compared with v4, it changed `AIMO_QUICK_RUN` from `true` to `false`, so the notebook no longer defaulted to a two-problem quick run. It kept the English-only instruction but removed the extra construction/injection/bijection/recurrence sentence and the numerical-pattern/asymptotic-intuition warning. The captured v5 output covers 16 problems.

`submission_v6.ipynb` was the final pre-Kaggle notebook. Compared with v5, it kept the same single-pass runtime structure and full-run default, but changed the prompt to explicitly allow standard LaTeX where proper mathematical notation requires it. This is the source notebook for `kaggle/aimoproofpilot-submission.ipynb`.

Captured output text and status JSON from these iterations are documented in `notebook_outputs/`.

Kaggle-specific notebooks live in `kaggle/`. Colab, Google Drive, rclone, and Kaggle Dataset transfer helpers live in `utils/`.
