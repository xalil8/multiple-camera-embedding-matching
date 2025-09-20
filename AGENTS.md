# Repository Guidelines

## Project Structure & Module Organization
- Core package in `dinov3/` with submodules: `train/`, `eval/`, `models/` (architectures), `data/` (datasets, loaders), `configs/` (YAML), `hub/` (Torch Hub), `utils/`, `logging/`.
- Checkpoints and example weights live in `models/` (do not commit new large files).
- Notebooks in `notebooks/`. Example datasets may be referenced in `imagenet/`.
- Entry points: `dinov3/train/train.py` (training), `dinov3/eval/*.py` (evaluation), `hubconf.py` (Torch Hub).

## Build, Test, and Development Commands
- Environment (recommended):
  ```bash
  micromamba env create -f conda.yaml && micromamba activate dinov3
  ```
- Install editable package + dev tools:
  ```bash
  pip install -e . -r requirements-dev.txt
  ```
- Lint and type-check:
  ```bash
  ruff check .
  mypy dinov3
  pylint dinov3
  ```
- Run training/eval locally (ensure module path):
  ```bash
  PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py --config-file dinov3/configs/train/vitl_im1k_lin834.yaml
  PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/knn.py model.pretrained_weights=<PATH>
  ```

## Coding Style & Naming Conventions
- Python 3.11, PEP 8, 4-space indentation, max line length 120 (`ruff`).
- Use type hints for public APIs; keep functions small and pure when possible.
- Naming: modules `lower_snake_case.py`, classes `CapWords`, functions/vars `snake_case`, constants `UPPER_SNAKE_CASE`.
- Prefer explicit imports; avoid circular dependencies. Place configs in `dinov3/configs/`.

## Testing Guidelines
- No formal test suite is included. New tests should go under `dinov3/tests/` and be named `test_*.py`.
- Prefer `pytest` with deterministic, fast unit tests. Target ≥80% coverage for new/changed code.
- Example:
  ```bash
  pytest -q
  ```

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject; use prefixes when helpful (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
- PRs: include scope/goal, key changes, usage notes, configs touched, and sample commands; link issues. Add before/after metrics for training/eval changes.
- Keep PRs focused and small; update docs/README snippets and configs as needed.

## Security & Configuration Tips
- Do not commit model weights, datasets, or credentials. Use paths/URLs via config.
- Always prefix CLI runs with `PYTHONPATH=${PWD}` to ensure local package resolution.
- Pin dependencies via `conda.yaml`; prefer CUDA-enabled PyTorch when training/evaluating.
