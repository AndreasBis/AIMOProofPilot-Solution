# Apptainer Training Image

## Goal

Build and verify the training `.sif` image from the repository root:

```bash
artifacts/aimo-proof_train_20260611.sif
```

## Preconditions

The image recipe is:

```text
containers/aimo-proof_train_20260611.def
```

The recipe installs Python packages from `requirements_container.txt`, so that file must already contain the pinned container dependencies.

The recipe also copies local tiktoken encoding files from `venv-container/tiktoken_encodings` into `/app/tiktoken_encodings` and sets `TIKTOKEN_ENCODINGS_BASE` inside the image. Make sure `venv-container/tiktoken_encodings` exists before building.

The host `venv-container` does not need to be activated for the Apptainer build. Activate it only when changing or refreezing `requirements_container.txt`, or when preparing the local tiktoken encoding directory.

## Build

Create repository-local Apptainer temp and cache directories:

```bash
mkdir -p artifacts/apptainer-tmp artifacts/apptainer-cache
```

Build with all temporary Apptainer paths under `artifacts/`:

```bash
env \
    APPTAINER_TMPDIR="$PWD/artifacts/apptainer-tmp" \
    APPTAINER_CACHEDIR="$PWD/artifacts/apptainer-cache" \
    TMPDIR="$PWD/artifacts/apptainer-tmp" \
    apptainer build \
        artifacts/aimo-proof_train_20260611.sif \
        containers/aimo-proof_train_20260611.def
```

## Verify

Run the image test after the build completes:

```bash
apptainer test artifacts/aimo-proof_train_20260611.sif
```

Confirm that the image exists and has a plausible size:

```bash
ls -lh artifacts/aimo-proof_train_20260611.sif
```

## Cleanup After Success

After a successful build and verification, keep the `.sif` and remove only the build temp/cache directories:

```bash
rm -rf artifacts/apptainer-tmp artifacts/apptainer-cache
```

Confirm the remaining artifact state:

```bash
du -sh artifacts ~/.apptainer
find artifacts -maxdepth 2 -print
```

## Cleanup After Failure

If the build fails, remove the partial target first, then remove the build temp/cache directories:

```bash
rm -f artifacts/aimo-proof_train_20260611.sif
rm -rf artifacts/apptainer-tmp artifacts/apptainer-cache
```

Confirm the cleanup state:

```bash
du -sh artifacts ~/.apptainer
find artifacts -maxdepth 2 -print
```
