# Kaggle Notebooks

This directory contains the Kaggle-facing notebooks.

| File | Purpose |
| --- | --- |
| `aimoproofpilot-submission.ipynb` | Final inference notebook prepared for the AIMO Proof Pilot Kaggle environment. It is a minor Kaggle-path and dependency adaptation of `notebooks/submission_v6.ipynb`. |
| `aimoproofpilot-utils.ipynb` | Kaggle utility notebook that downloads the wheel cache used by the submission environment and writes the matching package snapshot. |

The submission notebook expects the Kaggle competition input path, the mirrored OLMo model dataset, and the wheel cache produced by the utility notebook.
