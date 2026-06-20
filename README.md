# silentspeechoe

Project skeleton for a master thesis on earable sensor-based cross-domain
silent speech recognition.

## Scope

- Research topic: cross-domain closed-set sentence recognition
- Domains: normal speech, whisper speech, silent speech
- Primary direction: whisper-to-silent transfer
- Sensors: bone acceleration, IMU, barometer, and low-rate merged streams
- Future extension: multi-sensor fusion with PyTorch-based experiments

## Current Status

This repository is intentionally scaffold-only.

- No model, training, or preprocessing logic is implemented yet
- Python modules contain docstrings and TODO markers only
- Hydra-style YAML configs are placeholders for later experiments
- Docker, Dev Container, CI, and test scaffolding are ready for extension

## Quick Start

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r .devcontainer/requirements-dev.txt -e .
ruff check .
ruff format --check .
pytest
```

## Suggested Workflow

1. Start in the Dev Container for a reproducible environment.
2. Refine `configs/` before implementing data pipelines or models.
3. Add functionality incrementally under `src/silentspeechoe/`.
4. Keep tests focused on structure and import safety until behavior exists.

## Dev Container Modes

- Default: `.devcontainer/devcontainer.json` does not require NVIDIA runtime.
- GPU: `.devcontainer/gpu/devcontainer.json` adds `gpus: all` for CUDA work.

## Repository Layout

```text
src/silentspeechoe/   Python package placeholders
configs/              Hydra-style configuration placeholders
scripts/              Entry-point placeholders
tests/                Basic scaffold validation
.devcontainer/        Dev Container and Docker setup
.agent/               Agent guidance for Codex and Claude Code
```
