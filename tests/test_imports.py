"""Import smoke tests for the scaffolded package."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

MODULES = [
    "silentspeechoe",
    "silentspeechoe.config",
    "silentspeechoe.data",
    "silentspeechoe.data.dataset",
    "silentspeechoe.data.preprocessing",
    "silentspeechoe.data.collate",
    "silentspeechoe.features",
    "silentspeechoe.features.filters",
    "silentspeechoe.features.envelope",
    "silentspeechoe.models",
    "silentspeechoe.models.bone_cnn",
    "silentspeechoe.models.imu_cnn",
    "silentspeechoe.models.fusion_cnn",
    "silentspeechoe.models.build",
    "silentspeechoe.training",
    "silentspeechoe.training.trainer",
    "silentspeechoe.training.losses",
    "silentspeechoe.evaluation",
    "silentspeechoe.evaluation.metrics",
    "silentspeechoe.evaluation.plots",
    "silentspeechoe.utils",
    "silentspeechoe.utils.seed",
    "silentspeechoe.utils.io",
    "silentspeechoe.utils.logger",
    "silentspeechoe.utils.checkpoint",
]

TORCH_REQUIRED_MODULES = {
    "silentspeechoe.data.dataset",
    "silentspeechoe.data.collate",
    "silentspeechoe.models.bone_cnn",
    "silentspeechoe.models.build",
    "silentspeechoe.training.trainer",
    "silentspeechoe.training.losses",
}


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Ensure placeholder modules remain importable."""

    if (
        module_name in TORCH_REQUIRED_MODULES
        and importlib.util.find_spec("torch") is None
    ):
        pytest.skip(f"{module_name} requires optional PyTorch dependency")

    module = importlib.import_module(module_name)
    assert module is not None
