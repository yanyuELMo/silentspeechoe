"""Structural tests for the project scaffold."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "Makefile",
    ".devcontainer/devcontainer.json",
    ".devcontainer/Dockerfile",
    ".devcontainer/docker-compose.yml",
    ".devcontainer/requirements.txt",
    ".devcontainer/requirements-dev.txt",
    ".github/workflows/ci.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".vscode/settings.json",
    ".vscode/extensions.json",
    "configs/config.yaml",
    "configs/data/openearable.yaml",
    "configs/experiment/silent_to_silent.yaml",
    "configs/experiment/normal_to_silent.yaml",
    "configs/experiment/whisper_to_silent.yaml",
    "configs/experiment/normal_whisper_to_silent.yaml",
    "configs/model/bone_acc_temporal_cnn.yaml",
    "configs/model/imu_temporal_cnn.yaml",
    "configs/train/default.yaml",
    "configs/train/debug.yaml",
    "scripts/preprocess.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/inspect_data.py",
    "src/silentspeechoe/__init__.py",
    "src/silentspeechoe/config.py",
    "src/silentspeechoe/data/dataset.py",
    "src/silentspeechoe/features/filters.py",
    "src/silentspeechoe/models/__init__.py",
    "src/silentspeechoe/models/build.py",
    "src/silentspeechoe/models/tcn.py",
    "src/silentspeechoe/training/trainer.py",
    "src/silentspeechoe/evaluation/metrics.py",
    "src/silentspeechoe/utils/checkpoint.py",
    "tests/test_imports.py",
    "tests/test_project_structure.py",
    "tests/test_config_files.py",
    "outputs/runs/.gitkeep",
    "outputs/checkpoints/.gitkeep",
    "outputs/logs/.gitkeep",
    "outputs/figures/.gitkeep",
]


def test_required_paths_exist() -> None:
    """Ensure the scaffold keeps its expected layout."""

    missing_paths = [
        relative_path
        for relative_path in REQUIRED_PATHS
        if not (REPO_ROOT / relative_path).exists()
    ]
    assert not missing_paths, f"Missing scaffold paths: {missing_paths}"
