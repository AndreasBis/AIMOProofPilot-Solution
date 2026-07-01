# Changelog

## 2026-06-25

### Added

- Documented the final repository structure across source modules, Kaggle submission notebooks, development notebook iterations, utility transfer notebooks, and captured notebook outputs.
- Added directory-level documentation for notebook outputs, including run totals and problem-level status summaries derived from the recorded JSON metadata.

### Changed

- Updated package metadata to version `2026.6.25` and release date `2026-06-25`.

## 2026-06-14

### Added

- Added an inference notebook targeting an RTX PRO 6000 Blackwell GPU.

### Changed

- Updated package metadata to version `2026.6.14` and release date `2026-06-14`.

### Known Issues

- Tool calling fails in the inference notebook. Sandbox evidence shows generated outputs can still be legible and coherent enough to have a clear beginning and ending, but no Python tool calls are executed.

## 2026-06-11

### Added

- Added `--dummy_test` and `AIMO_DUMMY_TEST` support so contestant and judge paths resolve to `models/dummy` for SmolLM-3B pipeline checks.
- Added a ChatML judge configuration path for dummy SmolLM-3B runs while keeping the GPT-OSS Harmony judge path for production.

### Changed

- Updated package metadata to version `2026.6.11` and release date `2026-06-11`.
- Renamed the Fields-compatible run and train container definitions to the 2026-06-11 version.
- Reduced the default vLLM `--attention-config` payload to `backend` and `flash_attn_version` only.
- Removed the deprecated `VLLM_MXFP4_USE_MARLIN` environment override and kept Marlin selection on the supported `--moe-backend marlin` argument.

### Verified

- Full verification passed:
  `python -m pytest`

## 2026-06-10

### Changed

- Updated package metadata to version `2026.6.10` and release date `2026-06-10`.
- Renamed the Fields-compatible run and train container definitions to the 2026-06-10 version.
- Removed vLLM help-output compatibility checks from service preflight and container tests because the Fields/NII submission interface does not require them and vLLM help output is not a stable launch preflight surface.
- Separated service preflight diagnostics from vLLM server diagnostics so failures before process launch are reported as preflight failures instead of vLLM runtime failures.

### Removed

- Removed the 2026-06-09 training SIF artifact from `artifacts/`.

## 2026-06-09

### Added

- Added GRPO training-job diagnostics that write `failure_report.json`, `failure_report.txt`, `failure_traceback.txt`, `phase_events.jsonl`, vLLM command files, service preflight reports, and failure artifact inventories when any rank fails.
- Added service preflight checks for model paths, tokenizer and weight files, log writability, port availability, visible GPU count, `/tmp`, `/dev/shm`, and vLLM CLI compatibility.
- Added service start, readiness, and failure markers for judge and contestant rollout ranks so the controller can fail fast when a service dies.
- Added sandbox and dataset preflight coverage before long online rollout phases.

### Changed

- Updated package metadata to version `2026.6.9` and release date `2026-06-09`.
- Updated the production online topology defaults to judge rank 0 on port 8000, contestant rollout rank 1 on port 8001, and trainer rank 2.
- Updated the GPT-OSS-120B judge vLLM command to enable expert parallelism while keeping the Marlin MoE backend.
- Renamed the Fields-compatible run and train container definitions to the 2026-06-09 version.
- Improved controller health polling to report all unhealthy online services instead of stopping at the first failed endpoint.

### Verified

- Full verification passed:
  `python -m pytest`

## 2026-06-04

### Added

- Added 2026-06-04 Fields-compatible run and train container definitions.
- Added FSDP-backed TRL GRPO gradient updates so node 2 shards trainer parameters, gradients, and optimizer state across its 8 H200s instead of running a single-device update or mirrored-only trainer.

### Changed

- Reduced the default MathNet train split from 8192 to 4096 problems, keeping `G=16`, 64-problem updates, and 1024 rollout samples per update for 64 total updates.
- Updated package metadata to version `2026.6.4`, release date `2026-06-04`, `dataset_train_size = 4096`, and `gradient_update_count = 64`.
- Kept the next-batch handoff model-adapter based: each GRPO chunk trains from the current adapter state, publishes a new adapter, and the rollout vLLM service reloads that adapter before generating subsequent groups.

### Rationale

- The sandbox task-duration estimate shows that 8192 problems at `G=16` creates 128 updates and a generation-only lower bound of about 5 days before RSLoRA time, while 4096 problems at `G=16` creates 65,536 trajectories over 64 updates and fits a conservative 100-hour planning budget.
- The June 18, 2026 deadline is 14 days after the 2026-06-04 version date, so the 4096-problem run leaves practical margin for container rebuilds, failed starts, validation, comparison against the base model, and final packaging.

### Verified

- Focused verification passed for the sharded GRPO launch path, training configuration, online training handoff, and container command-shape contracts:
  `PYTHONDONTWRITEBYTECODE=1 pytest tests/unit/test_training.py tests/unit/test_online_training.py tests/contract/test_fields_contracts.py -q`
- Broader unit verification passed:
  `PYTHONDONTWRITEBYTECODE=1 pytest tests/unit -q`

## 2026-06-03

### Added

- Implemented the full AIMO Proof Pilot pipeline across data building, inference, reward judging, online GRPO rollouts, adapter training, container entry points, and result upload packaging.
- Added deterministic MathNet-derived dataset construction with a 16-problem held-out eval split and an 8192-problem topic-stratified training split for 128 updates at 64 problems per update.
- Added OLMo contestant inference paths for proof generation, GPT-OSS judge inference paths for boxed reward grades, and GPT-OSS Harmony answer-mode tooling for Kaggle-style answer extraction.
- Added vLLM-backed rollout orchestration with durable complete-group queues, sandboxed Python tool execution, selected-token logprob capture, and top-logprob alternatives disabled for lower CPU overhead.
- Added GRPO training support through queued rollout payloads, Rank-Stabilized LoRA adapter output, selected-token logprob propagation, and trainable-token masking for tool transcripts.
- Added online adapter versioning with adapter state files, rollout readiness checks, adapter hash validation, and per-sample policy adapter metadata.
- Added detailed training and online logs covering run metadata, reward configuration, source dataset manifests, per-step rewards, gradient-update reward summaries, per-sample reward components, judge parse failures, rollout failures, skipped groups, checkpoint hashes, and final evaluation summaries.
- Added Fields-compatible `/app/run.py` and `/app/train.py` wrappers plus Singularity definitions for the 2026-06-03 run and train containers.
- Added upload packaging through `execution.upload`, including upload manifests, archive creation, S3 upload, and receipt writing.
- Added unit, integration, and contract tests for data building, inference clients, server command shape, page counting, judging, training queues, online rollout scheduling, adapter readiness, container wrappers, and fake-service smoke paths.

### Changed

- Dataset generation now defaults to `--eval_size 16` and `--train_size 8192` instead of a small eval split with the full remaining training set.
- Training sequence construction keeps tool-output tokens in the model input while excluding `env_mask=0` tokens from loss and old-policy logprob comparison.
- Online GRPO handoff now records the rollout adapter state and trained adapter state in manifests and emits phase-level events for failure diagnosis.

### Verified

- Focused verification passed for data building, training artifacts, online training handoff, and fake inference integration:
  `PYTHONPATH=src pytest tests/unit/test_data_builder.py tests/unit/test_training.py tests/unit/test_online_training.py tests/integration/test_fake_inference.py`
