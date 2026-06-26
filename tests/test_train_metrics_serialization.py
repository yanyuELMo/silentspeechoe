"""Tests for training metrics serialization."""

from __future__ import annotations

import json

from scripts.train import _classification_metrics_to_json


def test_classification_metrics_to_json_handles_nested_metrics() -> None:
    """Nested validation metrics should remain JSON serializable."""

    metrics = {
        "overall": {
            "accuracy": 0.875,
            "top3_accuracy": 0.9375,
            "open_set_identification": {
                "dir": 0.8125,
                "fpir": 0.001,
                "folds": [{"dir": 0.8}, {"dir": 0.825}],
            },
        },
        "by_group": {
            "normal": {
                "accuracy": 0.9,
                "open_set_identification": {"dir": 0.85},
            },
            "silent": {
                "accuracy": 0.85,
                "open_set_identification": {"dir": 0.8},
            },
        },
        "val_loss": 0.7821,
    }

    result = _classification_metrics_to_json(
        metrics,
        epoch=34,
        selection_metric="overall_accuracy",
        selection_value=0.875,
    )

    json.dumps(result)
    assert result["overall"]["accuracy"] == 0.875
    assert result["overall"]["open_set_identification"]["dir"] == 0.8125
    assert result["overall"]["open_set_identification"]["folds"][0]["dir"] == 0.8
    assert result["by_domain"]["normal"]["open_set_identification"]["dir"] == 0.85
    assert result["epoch"] == 34
