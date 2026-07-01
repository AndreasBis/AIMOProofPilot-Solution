# Tests

This directory is scoped to the Python modules under `src/`.

It intentionally excludes tests for Jupyter notebooks, Kaggle notebooks, utility notebooks, containers, and publication documentation.

Run the suite from the repository root with `venv-local` activated or by calling its Python directly.

```bash
source venv-local/bin/activate
PYTHONPATH=src python -m pytest -q -p no:cacheprovider
```

Equivalent direct command:

```bash
PYTHONPATH=src venv-local/bin/python -m pytest -q -p no:cacheprovider
```

`venv-local` must include `pyarrow` and `pytest`.
