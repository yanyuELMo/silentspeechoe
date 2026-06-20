"""Import smoke tests for the scaffolded package."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "silentspeechoe",
    "silentspeechoe.config",
    "silentspeechoe.data",
    "silentspeechoe.data.preprocessing",
    "silentspeechoe.models",
    "silentspeechoe.models.imu_cnn",
    "silentspeechoe.models.imu_mlp",
    "silentspeechoe.models.fusion_cnn",
    "silentspeechoe.evaluation",
    "silentspeechoe.evaluation.metrics",
    "silentspeechoe.evaluation.plots",
    "silentspeechoe.utils",
    "silentspeechoe.utils.io",
    "silentspeechoe.utils.logger",
    "silentspeechoe.utils.checkpoint",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Ensure placeholder modules remain importable."""

    module = importlib.import_module(module_name)
    assert module is not None
