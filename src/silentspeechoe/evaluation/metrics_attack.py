"""Attack metrics for authentication robustness experiments.

The main metric here follows the ArtiPass-style attack success rate (ASR):
for a target user and an attempt budget ``A``, one attack trial succeeds if
any of the first ``A`` attack attempts scores above the target user's
authentication threshold.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._utils import as_1d_array

__all__ = ["compute_attack_success_rate", "compute_all_attempt_asr"]


def _scalar(value: Any) -> Any:
    """Return a plain Python scalar when NumPy exposes ``item``."""
    return value.item() if hasattr(value, "item") else value


def _threshold_for_target(
    thresholds: Mapping[Any, float] | Sequence[float] | np.ndarray,
    target_id: Any,
) -> float:
    """Resolve the decision threshold for one target user."""
    if isinstance(thresholds, Mapping):
        if target_id in thresholds:
            return float(thresholds[target_id])
        target_key = str(target_id)
        if target_key in thresholds:
            return float(thresholds[target_key])
        raise ValueError(f"Missing threshold for target {target_id!r}")

    try:
        return float(thresholds[int(target_id)])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing threshold for target {target_id!r}") from exc


def _validate_attempt_budgets(attempt_budgets: Sequence[int]) -> tuple[int, ...]:
    """Return sorted unique positive attempt budgets."""
    budgets = tuple(sorted({int(budget) for budget in attempt_budgets}))
    if not budgets:
        raise ValueError("attempt_budgets must contain at least one value")
    if budgets[0] <= 0:
        raise ValueError("attempt_budgets must be positive")
    return budgets


def _asr_for_trials(
    trials: list[np.ndarray],
    threshold: float,
    attempt_budgets: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    """Compute best-of-A ASR for one collection of attack trials."""
    if not trials:
        raise ValueError("ASR requires at least one attack trial")

    result: dict[str, dict[str, float]] = {}
    for budget in attempt_budgets:
        successes = []
        short_trials = 0
        for scores in trials:
            usable = scores[:budget]
            if usable.shape[0] < budget:
                short_trials += 1
            successes.append(float(np.max(usable) >= threshold))
        result[str(budget)] = {
            "asr": float(np.mean(successes)),
            "num_trials": float(len(trials)),
            "num_short_trials": float(short_trials),
            "threshold": float(threshold),
        }
    return result


def compute_attack_success_rate(
    attack_scores: Any,
    target_ids: Any,
    trial_ids: Any,
    thresholds: Mapping[Any, float] | Sequence[float] | np.ndarray,
    *,
    attempt_budgets: Sequence[int] = (1, 3, 5, 7, 10),
    attack_subject_ids: Any | None = None,
    groups: Any | None = None,
) -> dict[str, Any]:
    """Compute per-user best-of-A attack success rate.

    Args:
        attack_scores: One scalar verification score per attack attempt. Each
            score must be the score against the claimed target user's template.
        target_ids: Target user identifier for each attack attempt.
        trial_ids: Attack trial identifier. Attempts with the same
            ``(target_id, trial_id)`` are treated as repeated tries for one
            attack trial and are evaluated in input order.
        thresholds: Per-target decision thresholds. A mapping can be keyed by
            either the original target id or its string representation. A
            sequence is indexed by integer target ids.
        attempt_budgets: Attempt budgets ``A`` for best-of-A ASR.
        attack_subject_ids: Optional attacker user identifier per attempt.
            When provided, the result includes ASR grouped by attacker.
        groups: Optional condition label per attempt, such as speech mode or
            attacker mixture condition. When provided, the result includes ASR
            grouped by condition.

    Returns:
        A dictionary with macro-ASR by attempt budget, per-target ASR, and
        optional per-attacker/per-group breakdowns. The top-level ``asr`` is
        the macro average over target users, matching the usual way of
        aggregating ``ASR_{A,u}`` across users. ``micro_asr`` is also reported
        over all attack trials pooled together.
    """
    scores = as_1d_array(attack_scores, "attack_scores").astype(np.float64)
    targets = as_1d_array(target_ids, "target_ids")
    trials = as_1d_array(trial_ids, "trial_ids")

    if not (scores.shape[0] == targets.shape[0] == trials.shape[0]):
        raise ValueError("attack_scores, target_ids, and trial_ids must align")

    budgets = _validate_attempt_budgets(attempt_budgets)
    attackers = None
    if attack_subject_ids is not None:
        attackers = as_1d_array(attack_subject_ids, "attack_subject_ids")
        if attackers.shape[0] != scores.shape[0]:
            raise ValueError("attack_subject_ids must align with attack_scores")

    group_values = None
    if groups is not None:
        group_values = as_1d_array(groups, "groups")
        if group_values.shape[0] != scores.shape[0]:
            raise ValueError("groups must align with attack_scores")

    by_target_trial: dict[Any, dict[Any, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_attacker_target_trial: dict[Any, dict[Any, dict[Any, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    by_group_target_trial: dict[Any, dict[Any, dict[Any, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for index, score in enumerate(scores):
        target = _scalar(targets[index])
        trial = _scalar(trials[index])
        by_target_trial[target][trial].append(float(score))

        if attackers is not None:
            attacker = _scalar(attackers[index])
            by_attacker_target_trial[attacker][target][trial].append(float(score))

        if group_values is not None:
            group = _scalar(group_values[index])
            by_group_target_trial[group][target][trial].append(float(score))

    by_target: dict[str, dict[str, dict[str, float]]] = {}
    pooled_trials_by_budget: dict[str, list[float]] = {
        str(budget): [] for budget in budgets
    }

    for target, target_trials in sorted(
        by_target_trial.items(),
        key=lambda item: str(item[0]),
    ):
        threshold = _threshold_for_target(thresholds, target)
        trial_arrays = [
            np.asarray(values, dtype=np.float64) for values in target_trials.values()
        ]
        target_metrics = _asr_for_trials(trial_arrays, threshold, budgets)
        by_target[str(target)] = target_metrics
        for budget in budgets:
            budget_key = str(budget)
            for scores_for_trial in trial_arrays:
                pooled_trials_by_budget[budget_key].append(
                    float(np.max(scores_for_trial[:budget]) >= threshold)
                )

    overall: dict[str, dict[str, float]] = {}
    for budget in budgets:
        budget_key = str(budget)
        target_asrs = [
            target_metrics[budget_key]["asr"] for target_metrics in by_target.values()
        ]
        overall[budget_key] = {
            "asr": float(np.mean(target_asrs)),
            "macro_asr": float(np.mean(target_asrs)),
            "micro_asr": float(np.mean(pooled_trials_by_budget[budget_key])),
            "num_targets": float(len(by_target)),
            "num_trials": float(len(pooled_trials_by_budget[budget_key])),
        }

    result: dict[str, Any] = {
        "attempt_budgets": list(budgets),
        "overall": overall,
        "by_target": by_target,
    }

    if attackers is not None:
        result["by_attacker"] = _compute_nested_asr(
            by_attacker_target_trial,
            thresholds,
            budgets,
        )
    if group_values is not None:
        result["by_group"] = _compute_nested_asr(
            by_group_target_trial,
            thresholds,
            budgets,
        )

    return result


def _compute_nested_asr(
    nested_trials: dict[Any, dict[Any, dict[Any, list[float]]]],
    thresholds: Mapping[Any, float] | Sequence[float] | np.ndarray,
    attempt_budgets: tuple[int, ...],
) -> dict[str, Any]:
    """Compute ASR for attacker/group -> target -> trials mappings."""
    output: dict[str, Any] = {}
    for outer_key, target_trials in sorted(
        nested_trials.items(),
        key=lambda item: str(item[0]),
    ):
        by_target = {}
        overall_by_budget: dict[str, list[float]] = {
            str(budget): [] for budget in attempt_budgets
        }
        for target, trials in sorted(
            target_trials.items(),
            key=lambda item: str(item[0]),
        ):
            threshold = _threshold_for_target(thresholds, target)
            trial_arrays = [
                np.asarray(values, dtype=np.float64) for values in trials.values()
            ]
            metrics = _asr_for_trials(trial_arrays, threshold, attempt_budgets)
            by_target[str(target)] = metrics
            for budget in attempt_budgets:
                overall_by_budget[str(budget)].append(metrics[str(budget)]["asr"])
        output[str(outer_key)] = {
            "overall": {
                str(budget): {
                    "asr": float(np.mean(overall_by_budget[str(budget)])),
                    "num_targets": float(len(by_target)),
                }
                for budget in attempt_budgets
            },
            "by_target": by_target,
        }
    return output


def compute_all_attempt_asr(
    attack_scores: Any,
    target_ids: Any,
    trial_ids: Any,
    thresholds: Mapping[Any, float] | Sequence[float] | np.ndarray,
    *,
    attack_subject_ids: Any | None = None,
) -> dict[str, Any]:
    """Compute attack success rate using **all** attempts per trial.

    One trial = one ``(attacker → target)`` pair.  The attacker presents
    **every** available attempt sample against the target template.  The
    trial succeeds if **any** attempt score crosses the target's decision
    threshold.  ASR = successful trials / total trials.

    Additionally, ``attempt_asr`` is reported: the fraction of individual
    attempt scores across all trials that cross their respective thresholds.

    This is the "unlimited‑budget" variant kept for compatibility with the
    evaluation protocol used in ``scripts/evaluate_imu_tcn_templates.py``.

    Args:
        attack_scores: 1‑D array of per‑attempt verification scores.
        target_ids: 1‑D array of target user identifiers (same length).
        trial_ids: 1‑D array of trial identifiers — typically the attacker
            subject id.
        thresholds: Per‑target decision thresholds (mapping or sequence).
        attack_subject_ids: Optional 1‑D attacker user id per attempt.
            When provided the result includes per‑attacker breakdowns.

    Returns:
        Dict with ``trial_asr``, ``attempt_asr``, ``num_trials``,
        ``num_attempts``, ``by_target``, and optional ``by_attacker``.
    """
    scores = as_1d_array(attack_scores, "attack_scores").astype(np.float64)
    targets = as_1d_array(target_ids, "target_ids")
    trials = as_1d_array(trial_ids, "trial_ids")

    if not (scores.shape[0] == targets.shape[0] == trials.shape[0]):
        raise ValueError("attack_scores, target_ids, and trial_ids must align")

    attackers = None
    if attack_subject_ids is not None:
        attackers = as_1d_array(attack_subject_ids, "attack_subject_ids")
        if attackers.shape[0] != scores.shape[0]:
            raise ValueError("attack_subject_ids must align with attack_scores")

    # Group attempts by (target, trial).
    by_target_trial: dict[Any, dict[Any, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_attacker_target_trial: dict[Any, dict[Any, dict[Any, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for idx in range(scores.shape[0]):
        target = _scalar(targets[idx])
        trial = _scalar(trials[idx])
        score = float(scores[idx])
        by_target_trial[target][trial].append(score)

        if attackers is not None:
            attacker = _scalar(attackers[idx])
            by_attacker_target_trial[attacker][target][trial].append(score)

    # Evaluate
    trial_successes: list[int] = []
    total_attempts = 0
    successful_attempts = 0

    by_target: dict[str, dict[str, float]] = {}
    for target, target_trials in sorted(
        by_target_trial.items(),
        key=lambda item: str(item[0]),
    ):
        threshold = _threshold_for_target(thresholds, target)
        target_successes: list[int] = []
        for trial_scores in target_trials.values():
            arr = np.asarray(trial_scores, dtype=np.float64)
            success = int(np.max(arr) >= threshold)
            target_successes.append(success)
            trial_successes.append(success)
            total_attempts += int(arr.shape[0])
            successful_attempts += int(np.sum(arr >= threshold))

        by_target[str(target)] = {
            "asr": float(np.mean(target_successes)) if target_successes else 0.0,
            "num_trials": float(len(target_successes)),
            "threshold": float(threshold),
        }

    result: dict[str, Any] = {
        "trial_asr": float(np.mean(trial_successes)) if trial_successes else 0.0,
        "attempt_asr": (
            float(successful_attempts / total_attempts) if total_attempts > 0 else 0.0
        ),
        "num_trials": len(trial_successes),
        "num_attempts": total_attempts,
        "successful_trials": int(sum(trial_successes)),
        "successful_attempts": int(successful_attempts),
        "by_target": by_target,
    }

    if attackers is not None:
        by_attacker: dict[str, dict[str, float]] = {}
        for attacker, a_target_trials in sorted(
            by_attacker_target_trial.items(),
            key=lambda item: str(item[0]),
        ):
            a_successes: list[int] = []
            a_attempts = 0
            for target, target_trials in a_target_trials.items():
                threshold = _threshold_for_target(thresholds, target)
                for trial_scores in target_trials.values():
                    arr = np.asarray(trial_scores, dtype=np.float64)
                    a_successes.append(int(np.max(arr) >= threshold))
                    a_attempts += int(arr.shape[0])
            by_attacker[str(attacker)] = {
                "asr": float(np.mean(a_successes)) if a_successes else 0.0,
                "num_trials": float(len(a_successes)),
                "num_attempts": float(a_attempts),
            }
        result["by_attacker"] = by_attacker

    return result
