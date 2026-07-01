# Container Definitions

This directory contains Singularity/Apptainer definition files for the AIMOProofPilot runtime images.

The files were hand-authored from the container requirements in `sandbox/EXECUTION_PLAN.md`. They are not generated from `venv-container`, and they are not built images. They are recipes consumed by `apptainer build`.

## Files

- `aimo-proof_run_20260611.def` builds the inference image and runs `/app/run.py`.
- `aimo-proof_train_20260611.def` builds the training image and runs `/app/train.py`, using `torchrun` when multi-GPU or multi-node settings are present.

## Template

Both definitions use the same structure:

- `Bootstrap: docker`
- `From: pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel`
- copy `requirements_container.txt` into `/app`
- copy `venv-container/tiktoken_encodings` into `/app/tiktoken_encodings`
- copy `src` into `/app/src`
- copy the relevant entry point into `/app`
- install Debian/Ubuntu system packages inside the image with `apt-get`
- install Python dependencies from `/app/requirements_container.txt`
- set runtime cache, tiktoken, and telemetry environment variables
- run a lightweight import test in `%test`
- define the entry point in `%runscript`

The local Fedora `venv-container` setup is separate. It is used to resolve and freeze `requirements_container.txt` and to store the tiktoken encoding files copied into the image. It is not activated during `apptainer build`.

## Build

Build the training image from the repository root:

```bash
mkdir -p artifacts/apptainer-tmp artifacts/apptainer-cache

env \
    APPTAINER_TMPDIR="$PWD/artifacts/apptainer-tmp" \
    APPTAINER_CACHEDIR="$PWD/artifacts/apptainer-cache" \
    TMPDIR="$PWD/artifacts/apptainer-tmp" \
    apptainer build \
        artifacts/aimo-proof_train_20260611.sif \
        containers/aimo-proof_train_20260611.def
```

Build the inference image similarly:

```bash
mkdir -p artifacts/apptainer-tmp artifacts/apptainer-cache

env \
    APPTAINER_TMPDIR="$PWD/artifacts/apptainer-tmp" \
    APPTAINER_CACHEDIR="$PWD/artifacts/apptainer-cache" \
    TMPDIR="$PWD/artifacts/apptainer-tmp" \
    apptainer build \
        artifacts/aimo-proof_run_20260611.sif \
        containers/aimo-proof_run_20260611.def
```
