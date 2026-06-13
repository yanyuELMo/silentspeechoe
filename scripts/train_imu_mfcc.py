"""Train a StandardScaler + LogisticRegression baseline on MFCC features.

Reads pre‑computed MFCC features from disk, splits by subject and domain
(no data leakage), fits a StandardScaler on the training set only, and
evaluates overall and per‑domain metrics.

Usage::

    python scripts/train_imu_mfcc.py \\
        --features-dir data/processed/features/imu_mfcc_left_200hz_raw9 \\
        --experiment configs/experiment/imu_left_normal_to_all.yaml

    python scripts/train_imu_mfcc.py \\
        --features-dir data/processed/features/imu_mfcc_left_200hz_raw9 \\
        --experiment configs/experiment/imu_left_whisper_to_all.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ensure the package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from silentspeechoe.evaluation.metrics import (  # noqa: E402
    compute_grouped_classification_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_OUTPUTS = _PROJECT_ROOT / "outputs" / "runs"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MFCC baseline for IMU sentence classification."
    )
    p.add_argument(
        "--features-dir",
        required=True,
        help="Directory containing MFCC .pt files and manifest.json.",
    )
    p.add_argument(
        "--experiment",
        required=True,
        help="Path to experiment YAML config (validation_subjects, domains).",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory under outputs/runs/.  Auto‑generated if omitted.",
    )
    p.add_argument(
        "--C", type=float, default=1.0, help="Inverse regularization strength."
    )
    p.add_argument("--max-iter", type=int, default=2000, help="Solver max iterations.")
    return p.parse_args(argv)


def _load_experiment_cfg(yaml_path: str | Path) -> dict:
    """Load a YAML experiment config (Hydra‑compatible subset)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(yaml_path)
    return {
        "validation_subjects": list(cfg.get("validation_subjects", [])),
        "source_domain": cfg.get("source_domain", "normal"),
        "source_domains": (
            list(cfg["source_domains"]) if "source_domains" in cfg else None
        ),
        "target_domains": list(
            cfg.get("target_domains", ["normal", "whisper", "silent"])
        ),
        "name": cfg.get("name", Path(yaml_path).stem),
    }


def _raw_subject(subject_id: str) -> str:
    return str(subject_id).removeprefix("sub_")


def _resolve_source_domains(exp: dict) -> set[str]:
    if exp.get("source_domains"):
        return set(exp["source_domains"])
    return {exp.get("source_domain", "normal")}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    features_dir = Path(args.features_dir)
    exp = _load_experiment_cfg(args.experiment)

    logger.info("Experiment   : %s", exp["name"])
    logger.info("Features dir : %s", features_dir)
    logger.info("Val subjects : %s", exp["validation_subjects"])

    # ---- load manifest -------------------------------------------------------
    manifest_path = features_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r") as f:
        manifest = json.load(f)

    records = manifest["records"]
    total = len(records)
    logger.info("Total samples: %d", total)

    # ---- load all features ---------------------------------------------------
    X_all: list[np.ndarray] = []
    y_all: list[int] = []
    domains_all: list[str] = []
    subjects_all: list[str] = []

    for rec in records:
        pt_path = features_dir / rec["file"]
        if not pt_path.exists():
            logger.debug("Missing feature file: %s — skipping", pt_path)
            continue
        data = torch.load(pt_path, weights_only=True)
        X_all.append(data["x"].numpy().astype(np.float64))
        y_all.append(int(data["y"]))
        domains_all.append(str(data["domain"]))
        subjects_all.append(str(data["subject_id"]))

    X_all_np = np.stack(X_all, axis=0)
    y_all_np = np.array(y_all, dtype=int)
    domains_all_np = np.array(domains_all)
    subjects_all_np = np.array(subjects_all)

    logger.info("Feature dim  : %d", X_all_np.shape[1])

    # ---- split ---------------------------------------------------------------
    val_subjects_raw = set(exp["validation_subjects"])
    source_domains = _resolve_source_domains(exp)
    target_domains = set(exp["target_domains"])

    train_mask = np.array(
        [
            _raw_subject(s) not in val_subjects_raw and d in source_domains
            for s, d in zip(subjects_all_np, domains_all_np, strict=True)
        ]
    )
    val_mask = np.array(
        [
            _raw_subject(s) in val_subjects_raw and d in target_domains
            for s, d in zip(subjects_all_np, domains_all_np, strict=True)
        ]
    )

    X_train = X_all_np[train_mask]
    y_train = y_all_np[train_mask]
    X_val = X_all_np[val_mask]
    y_val = y_all_np[val_mask]
    val_domains = domains_all_np[val_mask].tolist()

    logger.info(
        "Train: %d (domains=%s)  Val: %d (domains=%s)",
        len(X_train),
        source_domains,
        len(X_val),
        target_domains,
    )

    # ---- normalize (fit ONLY on train) ---------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ---- train classifier ----------------------------------------------------
    clf = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train_scaled, y_train)
    logger.info("Classifier trained — classes: %d", len(clf.classes_))

    # ---- evaluate ------------------------------------------------------------
    y_pred = clf.predict(X_val_scaled)
    y_score = clf.predict_proba(X_val_scaled)

    import torch as _torch

    metrics = compute_grouped_classification_metrics(
        _torch.tensor(y_val),
        _torch.tensor(y_pred),
        _torch.from_numpy(y_score.astype(np.float32)),
        val_domains,
        top_k=3,
    )

    # ---- save artifacts ------------------------------------------------------
    run_name = args.run_name or f"imu_mfcc_{exp['name']}"
    run_dir = _OUTPUTS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # final_metrics.json
    metrics_json: dict = {
        "overall": {k: float(v) for k, v in metrics["overall"].items()},
        "by_domain": {},
    }
    for domain, dmetrics in metrics["by_group"].items():
        metrics_json["by_domain"][domain] = {k: float(v) for k, v in dmetrics.items()}
    with (run_dir / "final_metrics.json").open("w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info("Metrics saved to %s", run_dir / "final_metrics.json")

    # model.joblib
    import joblib

    joblib.dump(
        {"scaler": scaler, "classifier": clf, "config": exp},
        run_dir / "model.joblib",
    )
    logger.info("Model saved to %s", run_dir / "model.joblib")

    # config snapshot
    with (run_dir / "config.yaml").open("w") as f:
        from omegaconf import OmegaConf

        f.write(OmegaConf.to_yaml(OmegaConf.create(exp)))
    logger.info("Config saved to %s", run_dir / "config.yaml")

    # ---- final report --------------------------------------------------------
    logger.info("=== Final Validation ===")
    logger.info("  Overall:")
    for k, v in metrics["overall"].items():
        logger.info("    %s: %.4f", k, v)
    logger.info("  By domain:")
    for domain, dmetrics in metrics["by_group"].items():
        logger.info("    %s:", domain)
        for k, v in dmetrics.items():
            logger.info("      %s: %.4f", k, v)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
