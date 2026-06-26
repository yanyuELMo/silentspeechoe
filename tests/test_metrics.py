"""Tests for evaluation metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from silentspeechoe.evaluation.metrics import (
    compute_attack_success_rate as compat_compute_attack_success_rate,
)
from silentspeechoe.evaluation.metrics import (
    compute_authentication_metrics as compat_compute_authentication_metrics,
)
from silentspeechoe.evaluation.metrics_attack import compute_attack_success_rate
from silentspeechoe.evaluation.metrics_authentication import (
    compute_authentication_metrics,
)
from silentspeechoe.evaluation.metrics_identification import (
    compute_classification_metrics,
    compute_dir_at_fpir_leave_one_user_out,
    compute_grouped_classification_metrics,
)

NUM_CLASSES = 36
NUM_SAMPLES = 108  # 3 samples per class to guarantee macro-F1 coverage


def test_compat_metrics_module_reexports_authentication_metrics():
    """The old metrics module should keep working while code migrates."""
    assert compat_compute_authentication_metrics is compute_authentication_metrics


def test_compat_metrics_module_reexports_attack_metrics():
    """The compatibility module should expose attack metrics."""
    assert compat_compute_attack_success_rate is compute_attack_success_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perfect_data(
    num_samples: int,
    num_classes: int = NUM_CLASSES,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build perfect-prediction arrays where every class appears at least once.

    Args:
        num_samples: Total number of samples (must be >= num_classes).
        num_classes: Number of classes.
        seed: Random seed.

    Returns:
        (y_true, y_pred, y_score) with perfect predictions.
    """
    rng = np.random.default_rng(seed)
    # Ensure every class appears at least once
    assert num_samples >= num_classes
    y_true = np.empty(num_samples, dtype=np.int64)
    y_true[:num_classes] = np.arange(num_classes)
    y_true[num_classes:] = rng.integers(0, num_classes, size=num_samples - num_classes)
    rng.shuffle(y_true)

    y_pred = y_true.copy()
    y_score = np.full((num_samples, num_classes), -100.0, dtype=np.float32)
    y_score[np.arange(num_samples), y_true] = 100.0
    return y_true, y_pred, y_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_preds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions that exactly match ground truth, covering all classes."""
    return _make_perfect_data(NUM_SAMPLES, NUM_CLASSES, seed=42)


@pytest.fixture
def wrong_preds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions that are always off by one class (modulo num_classes)."""
    y_true, _, _ = _make_perfect_data(NUM_SAMPLES, NUM_CLASSES, seed=1)
    y_pred = (y_true + 1) % NUM_CLASSES
    y_score = np.full((NUM_SAMPLES, NUM_CLASSES), -100.0, dtype=np.float32)
    y_score[np.arange(NUM_SAMPLES), y_pred] = 100.0
    return y_true, y_pred, y_score


@pytest.fixture
def top3_preds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions where the correct class is ranked 2nd (in top-3 but not top-1)."""
    y_true, _, _ = _make_perfect_data(NUM_SAMPLES, NUM_CLASSES, seed=2)
    y_pred = (y_true + 1) % NUM_CLASSES  # wrong top-1
    y_score = np.full((NUM_SAMPLES, NUM_CLASSES), -100.0, dtype=np.float32)
    # correct class gets second-highest logit
    y_score[np.arange(NUM_SAMPLES), y_pred] = 100.0
    y_score[np.arange(NUM_SAMPLES), y_true] = 50.0
    return y_true, y_pred, y_score


@pytest.fixture
def grouped_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Data with 3 speech-mode groups, each covering all 36 classes."""
    # 36 samples per group — one sample per class, guaranteeing coverage
    samples_per_group = NUM_CLASSES
    total = samples_per_group * 3
    y_true = np.tile(np.arange(NUM_CLASSES), 3)
    rng = np.random.default_rng(123)
    rng.shuffle(y_true[:samples_per_group])  # shuffle within normal
    rng.shuffle(y_true[samples_per_group : 2 * samples_per_group])  # whisper
    rng.shuffle(y_true[2 * samples_per_group :])  # silent

    modes = (
        ["normal"] * samples_per_group
        + ["whisper"] * samples_per_group
        + ["silent"] * samples_per_group
    )
    y_pred = y_true.copy()
    y_score = np.full((total, NUM_CLASSES), -100.0, dtype=np.float32)
    y_score[np.arange(total), y_true] = 100.0
    return y_true, y_pred, y_score, modes


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_perfect_predictions_all_one(perfect_preds):
    """All metrics should be 1.0 when predictions are perfect."""
    y_true, y_pred, y_score = perfect_preds
    metrics = compute_classification_metrics(y_true, y_pred, y_score)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["dir_at_fpir_0p1pct"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["top3_accuracy"] == 1.0


def test_wrong_predictions_low_metrics(wrong_preds):
    """Completely wrong predictions should yield low accuracy and f1."""
    y_true, y_pred, y_score = wrong_preds
    metrics = compute_classification_metrics(y_true, y_pred, y_score)
    assert metrics["top1_accuracy"] < 0.2
    assert metrics["accuracy"] < 0.2
    assert metrics["macro_f1"] < 0.2
    # balanced accuracy is also low for systematic wrong predictions
    assert metrics["balanced_accuracy"] < 0.2
    # top-3: the correct class IS rank 1 in the logits (we set it to -100
    # and wrong to 100), so top-3 accuracy is also low — the correct class
    # is not in the top 3 because the model is confidently wrong.
    assert metrics["top3_accuracy"] < 0.2


def test_top3_accuracy_catches_correct_in_top3(top3_preds):
    """Top-3 accuracy should be high when correct class is in the top 3,
    even though top-1 accuracy is zero."""
    y_true, y_pred, y_score = top3_preds
    metrics = compute_classification_metrics(y_true, y_pred, y_score)
    assert metrics["accuracy"] < 0.2  # top-1 is wrong
    assert metrics["top3_accuracy"] > 0.9  # correct is rank 2


def test_returns_expected_keys():
    """Returned dict should include compact and compatibility metrics."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, NUM_CLASSES, size=10)
    y_pred = y_true.copy()
    y_score = np.zeros((10, NUM_CLASSES))
    y_score[np.arange(10), y_true] = 1.0
    result = compute_classification_metrics(y_true, y_pred, y_score)
    assert {
        "top1_accuracy",
        "accuracy",
        "macro_f1",
        "dir_at_fpir_0p1pct",
        "dir_at_fpir_target",
        "fpir_at_dir_threshold",
        "fpir_target",
        "open_set_identification",
        "balanced_accuracy",
        "top3_accuracy",
    }.issubset(result)
    for key in (
        "top1_accuracy",
        "accuracy",
        "macro_f1",
        "dir_at_fpir_0p1pct",
        "balanced_accuracy",
        "top3_accuracy",
    ):
        assert isinstance(result[key], float)


# ---------------------------------------------------------------------------
# NumPy array support
# ---------------------------------------------------------------------------


def test_numpy_arrays(perfect_preds):
    """Metrics should work with plain NumPy arrays."""
    y_true, y_pred, y_score = perfect_preds
    _ = compute_classification_metrics(y_true, y_pred, y_score)


def test_numpy_1d_2d_shapes():
    """1-D labels and 2-D scores should work."""
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, NUM_CLASSES, size=20)
    y_pred = y_true.copy()
    y_score = np.random.randn(20, NUM_CLASSES).astype(np.float32)
    result = compute_classification_metrics(y_true, y_pred, y_score)
    assert result["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Grouped metrics
# ---------------------------------------------------------------------------


def test_grouped_metrics_structure(grouped_data):
    """Grouped output should contain overall and by_group with three modes."""
    y_true, y_pred, y_score, groups = grouped_data
    result = compute_grouped_classification_metrics(y_true, y_pred, y_score, groups)
    assert "overall" in result
    assert "by_group" in result
    # Each speech mode should appear
    assert set(result["by_group"].keys()) == {"normal", "whisper", "silent"}
    # Overall should have all metric keys
    for key in (
        "top1_accuracy",
        "accuracy",
        "macro_f1",
        "dir_at_fpir_0p1pct",
        "balanced_accuracy",
        "top3_accuracy",
    ):
        assert key in result["overall"]
    # Each group should have all metric keys
    for mode in ("normal", "whisper", "silent"):
        for key in (
            "top1_accuracy",
            "accuracy",
            "macro_f1",
            "dir_at_fpir_0p1pct",
            "balanced_accuracy",
            "top3_accuracy",
        ):
            assert key in result["by_group"][mode]


def test_grouped_metrics_perfect(grouped_data):
    """Grouped metrics with perfect predictions should be all 1.0 everywhere."""
    y_true, y_pred, y_score, groups = grouped_data
    result = compute_grouped_classification_metrics(y_true, y_pred, y_score, groups)
    for key in (
        "top1_accuracy",
        "accuracy",
        "macro_f1",
        "dir_at_fpir_0p1pct",
        "balanced_accuracy",
        "top3_accuracy",
    ):
        assert result["overall"][key] == 1.0
    for mode_metrics in result["by_group"].values():
        for key in (
            "top1_accuracy",
            "accuracy",
            "macro_f1",
            "dir_at_fpir_0p1pct",
            "balanced_accuracy",
            "top3_accuracy",
        ):
            assert mode_metrics[key] == 1.0


def test_grouped_metrics_with_extra_group():
    """If a group appears only once, it should still get its own entry."""
    rng = np.random.default_rng(99)
    modes = ["normal"] * 5 + ["whisper"] * 5 + ["silent"] * 5 + ["shout"]
    y_true = rng.integers(0, NUM_CLASSES, size=len(modes))
    y_pred = y_true.copy()
    y_score = np.random.randn(len(modes), NUM_CLASSES).astype(np.float32)
    y_score[np.arange(len(modes)), y_true] = 100.0
    result = compute_grouped_classification_metrics(y_true, y_pred, y_score, modes)
    assert set(result["by_group"].keys()) == {"normal", "whisper", "silent", "shout"}
    assert result["by_group"]["shout"]["accuracy"] == 1.0


def test_single_sample_input_is_valid():
    """A one-sample batch should remain 1-D and compute metrics."""
    y_true = np.array([3])
    y_pred = np.array([3])
    y_score = np.full((1, NUM_CLASSES), -100.0, dtype=np.float32)
    y_score[0, 3] = 100.0

    result = compute_classification_metrics(y_true, y_pred, y_score)

    assert result["accuracy"] == 1.0
    assert result["top3_accuracy"] == 1.0
    assert np.isnan(result["dir_at_fpir_0p1pct"])


def test_dir_at_fpir_leave_one_user_out_perfect():
    """DIR@FPIR should use each subject once as the unknown identity."""
    y_true = np.repeat(np.arange(4), 3)
    y_score = np.full((y_true.shape[0], 4), -10.0, dtype=np.float32)
    y_score[np.arange(y_true.shape[0]), y_true] = 10.0

    result = compute_dir_at_fpir_leave_one_user_out(
        y_true,
        y_score,
        fpir_target=0.001,
    )

    assert result["dir"] == 1.0
    assert result["fpir"] == 0.0
    assert result["num_folds"] == 4.0
    assert {fold["unknown_label"] for fold in result["folds"]} == {0, 1, 2, 3}


def test_dir_at_fpir_leave_one_user_out_rejects_high_unknown_scores():
    """A high unknown score should force a stricter threshold in its fold."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_score = np.array(
        [
            [0.8, 0.2, 0.1],
            [0.7, 0.3, 0.2],
            [0.2, 0.8, 0.1],
            [0.3, 0.7, 0.2],
            [0.95, 0.9, 0.1],  # class 2 looks like enrolled classes when unknown
            [0.96, 0.85, 0.1],
        ],
        dtype=np.float32,
    )

    result = compute_dir_at_fpir_leave_one_user_out(
        y_true,
        y_score,
        fpir_target=0.001,
    )

    assert result["fpir"] == 0.0
    assert 0.0 <= result["dir"] < 1.0


def test_balanced_accuracy_uses_fixed_class_set():
    """Missing classes should contribute zero recall to balanced accuracy."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_score = np.full((2, 4), -100.0, dtype=np.float32)
    y_score[np.arange(2), y_true] = 100.0

    result = compute_classification_metrics(y_true, y_pred, y_score)

    assert result["balanced_accuracy"] == 0.5


def test_grouped_metrics_accept_single_sample_group():
    """Grouped metrics should handle groups with exactly one sample."""
    y_true = np.array([0, 1, 2])
    y_pred = y_true.copy()
    y_score = np.full((3, NUM_CLASSES), -100.0, dtype=np.float32)
    y_score[np.arange(3), y_true] = 100.0
    groups = ["normal", "whisper", "silent"]

    result = compute_grouped_classification_metrics(y_true, y_pred, y_score, groups)

    assert set(result["by_group"].keys()) == {"normal", "whisper", "silent"}
    assert result["by_group"]["silent"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Authentication metrics
# ---------------------------------------------------------------------------


def test_authentication_metrics_perfect_one_vs_all():
    """Perfect subject scores should give zero EER/FAR/FRR."""
    y_true = np.repeat(np.arange(4), 5)
    y_score = np.full((y_true.shape[0], 4), -10.0, dtype=np.float32)
    y_score[np.arange(y_true.shape[0]), y_true] = 10.0

    result = compute_authentication_metrics(y_true, y_score, top_k=2)

    assert result["top1_accuracy"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["dir_at_fpir_0p1pct"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["top3_accuracy"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["eer"] == 0.0
    assert result["authentication_summary"] == {"eer": 0.0, "roc_auc": 1.0}
    assert result["far"] == 0.0
    assert result["frr"] == 0.0
    assert result["frr_at_far_1pct"] == 0.0
    assert set(result["by_subject"].keys()) == {"0", "1", "2", "3"}


def test_authentication_metrics_one_vs_all_counts():
    """Each subject should be treated as positive against all others."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_score = np.full((6, 3), -5.0, dtype=np.float32)
    y_score[np.arange(6), y_true] = 5.0

    result = compute_authentication_metrics(y_true, y_score, top_k=2)

    for subject in ("0", "1", "2"):
        metrics = result["by_subject"][subject]
        assert metrics["num_positive"] == 2.0
        assert metrics["num_negative"] == 4.0
        assert metrics["roc_auc"] == 1.0


def test_authentication_metrics_nonperfect_scores_are_bounded():
    """Authentication metrics should remain valid for imperfect scores."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_score = np.array(
        [
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.5, 2.0, 0.0],  # class 1 sample confused with class 0
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    result = compute_authentication_metrics(y_true, y_score, top_k=2)

    for key in (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "top3_accuracy",
        "roc_auc",
        "eer",
        "far",
        "frr",
        "frr_at_far_1pct",
    ):
        assert 0.0 <= result[key] <= 1.0
    assert result["authentication"]["far_target"] == 0.01
    assert result["authentication"]["num_subjects"] == 3.0


def test_authentication_metrics_custom_far_target():
    """The FRR at a requested FAR operating point should be exposed."""
    y_true = np.repeat(np.arange(3), 4)
    y_score = np.full((y_true.shape[0], 3), -2.0, dtype=np.float32)
    y_score[np.arange(y_true.shape[0]), y_true] = 2.0

    result = compute_authentication_metrics(y_true, y_score, top_k=2, far_target=0.05)

    assert result["authentication"]["far_target"] == 0.05
    assert result["frr_at_far_target"] == 0.0
    assert result["far_at_far_target"] <= 0.05


# ---------------------------------------------------------------------------
# Attack metrics
# ---------------------------------------------------------------------------


def test_attack_success_rate_uses_best_of_attempt_budget():
    """ASR should succeed when any attempt within the budget crosses threshold."""
    scores = np.array(
        [
            0.1,
            0.2,
            0.9,  # target u0, trial t0 succeeds only when A >= 3
            0.8,
            0.1,
            0.1,  # target u0, trial t1 succeeds already at A = 1
            0.2,
            0.3,
            0.4,  # target u1, trial t0 never succeeds
            0.1,
            0.7,
            0.2,  # target u1, trial t1 succeeds when A >= 3
        ],
        dtype=np.float32,
    )
    target_ids = np.array(["u0"] * 6 + ["u1"] * 6)
    trial_ids = np.array(["t0"] * 3 + ["t1"] * 3 + ["t0"] * 3 + ["t1"] * 3)
    thresholds = {"u0": 0.75, "u1": 0.65}

    result = compute_attack_success_rate(
        scores,
        target_ids,
        trial_ids,
        thresholds,
        attempt_budgets=(1, 3),
    )

    assert result["by_target"]["u0"]["1"]["asr"] == 0.5
    assert result["by_target"]["u0"]["3"]["asr"] == 1.0
    assert result["by_target"]["u1"]["1"]["asr"] == 0.0
    assert result["by_target"]["u1"]["3"]["asr"] == 0.5
    assert result["overall"]["1"]["asr"] == 0.25
    assert result["overall"]["3"]["asr"] == 0.75


def test_attack_success_rate_reports_attacker_and_group_breakdowns():
    """Optional attacker and condition labels should produce breakdowns."""
    scores = np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float32)
    target_ids = np.array(["target", "target", "target", "target"])
    trial_ids = np.array(["a_trial", "a_trial", "b_trial", "b_trial"])
    attackers = np.array(["attacker_a", "attacker_a", "attacker_b", "attacker_b"])
    groups = np.array(["normal", "normal", "silent", "silent"])

    result = compute_attack_success_rate(
        scores,
        target_ids,
        trial_ids,
        {"target": 0.85},
        attempt_budgets=(1, 2),
        attack_subject_ids=attackers,
        groups=groups,
    )

    assert result["by_attacker"]["attacker_a"]["overall"]["1"]["asr"] == 1.0
    assert result["by_attacker"]["attacker_b"]["overall"]["1"]["asr"] == 0.0
    assert result["by_group"]["normal"]["overall"]["1"]["asr"] == 1.0
    assert result["by_group"]["silent"]["overall"]["2"]["asr"] == 0.0


# ---------------------------------------------------------------------------
# Input validation — ValueErrors
# ---------------------------------------------------------------------------


def test_mismatched_true_pred_raises():
    """Mismatched lengths for y_true and y_pred should raise ValueError."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1])
    y_score = np.random.randn(3, NUM_CLASSES)
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_score)


def test_mismatched_true_score_raises():
    """Mismatched lengths for y_true and y_score rows should raise ValueError."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_score = np.random.randn(5, NUM_CLASSES)
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_score)


def test_top_k_exceeds_classes_raises():
    """If top_k > num_classes, a ValueError should be raised."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 5, size=10)
    y_pred = y_true.copy()
    y_score = np.random.randn(10, 5)
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_score, top_k=10)


def test_top_k_zero_raises():
    """If top_k is not positive, a ValueError should be raised."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_score = np.random.randn(3, NUM_CLASSES)
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_score, top_k=0)


def test_y_score_not_2d_raises():
    """If y_score is 1-D, a ValueError should be raised."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_score = np.array([1.0, 2.0, 3.0])  # 1-D
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_score)


def test_y_true_not_1d_raises():
    """If y_true is 2-D (more than one row), a ValueError should be raised."""
    y_true = np.array([[0, 1], [2, 3]])  # shape (2, 2) — stays 2-D after squeeze
    y_pred = np.array([0, 1])
    y_score = np.random.randn(2, NUM_CLASSES)
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_score)


def test_mismatched_groups_raises():
    """Mismatched group length should raise ValueError."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_score = np.random.randn(3, NUM_CLASSES)
    groups = ["normal", "whisper"]
    with pytest.raises(ValueError):
        compute_grouped_classification_metrics(y_true, y_pred, y_score, groups)


def test_label_outside_score_classes_raises():
    """Labels must fit the class dimension from y_score."""
    y_true = np.array([0, 1, 36])
    y_pred = np.array([0, 1, 2])
    y_score = np.random.randn(3, NUM_CLASSES)
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred, y_score)


def test_authentication_metrics_invalid_far_target_raises():
    """FAR target must be a probability."""
    y_true = np.array([0, 1, 2])
    y_score = np.eye(3)
    with pytest.raises(ValueError):
        compute_authentication_metrics(y_true, y_score, far_target=1.5)


def test_authentication_metrics_requires_2d_scores():
    """Authentication scores must be a [N, C] matrix."""
    y_true = np.array([0, 1, 2])
    y_score = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        compute_authentication_metrics(y_true, y_score)
