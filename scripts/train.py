"""Training entry point — Hydra‑backed experiment launcher.

Usage::

    python scripts/train.py \\
        experiment=bone_binaural_all_modes \\
        model=bone_cnn \\
        train=debug \\
        train.device=cpu \\
        train.max_epochs=1

    python scripts/train.py \\
        experiment=bone_binaural_all_modes \\
        model=bone_cnn \\
        train=default

    python scripts/train.py \\
        experiment=bone_binaural_all_modes \\
        model=bone_raw_tcn \\
        train=default \\
        train.input_type=raw_bone \\
        train.processed_dir=raw_bone_binaural
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Make the src package importable without installing.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from silentspeechoe.data.collate import pad_collate  # noqa: E402
from silentspeechoe.data.dataset import (  # noqa: E402
    BoneBinauralFeatureDataset,
    BoneBinauralPrecomputedDataset,
    BoneRawPrecomputedDataset,
    build_binaural_event_records,
)
from silentspeechoe.models.build import build_model  # noqa: E402
from silentspeechoe.training.losses import build_loss  # noqa: E402
from silentspeechoe.training.trainer import run_training  # noqa: E402
from silentspeechoe.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

_CONFIG_PATH = str(_PROJECT_ROOT / "configs")


def _raw_subject(subject_id: str) -> str:
    """Strip ``sub_`` so config ``"07"`` matches events.csv ``"sub_07"``."""
    return str(subject_id).removeprefix("sub_")


def _resolve_processed_dir(path_value: str, *, default_subdir: str) -> Path:
    """Resolve a processed-data directory from a config value.

    Accepts a short subdirectory name such as ``"raw_bone_binaural"``
    or a project-relative path such as ``"data/processed/raw_bone_binaural"``.
    """
    value = path_value or default_subdir
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:2] == ("data", "processed"):
        return _PROJECT_ROOT / path
    return _PROJECT_ROOT / "data" / "processed" / path


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    # Disable struct mode so config group overrides can add arbitrary keys.
    OmegaConf.set_struct(cfg, False)

    exp_cfg = cfg.experiment
    model_cfg = cfg.model
    train_cfg = cfg.train

    logger.info("Experiment : %s", exp_cfg.name)
    logger.info("Model      : %s", model_cfg.name)
    logger.info("Train cfg  : %s", train_cfg.name)

    # ---- seed --------------------------------------------------------------
    set_seed(int(train_cfg.seed))

    # ---- data --------------------------------------------------------------
    all_records = build_binaural_event_records(
        events_path=str(_PROJECT_ROOT / "data" / "metadata" / "events.csv"),
        raw_dir=str(_PROJECT_ROOT / "data" / "raw"),
    )
    logger.info("Total paired records: %d", len(all_records))

    # Split by subject — strip "sub_" prefix from events.csv IDs.
    val_subjects_raw = set(exp_cfg.validation_subjects)  # e.g. {"07", "10", ...}
    train_recs = [
        r for r in all_records if _raw_subject(r["subject_id"]) not in val_subjects_raw
    ]
    val_recs = [
        r for r in all_records if _raw_subject(r["subject_id"]) in val_subjects_raw
    ]
    logger.info("Train records: %d  Val records: %d", len(train_recs), len(val_recs))

    # Log per-domain counts for validation.
    val_domains: dict[str, int] = {}
    for r in val_recs:
        val_domains[r["domain"]] = val_domains.get(r["domain"], 0) + 1
    logger.info("Val per domain: %s", val_domains)

    precomputed = bool(train_cfg.get("precomputed", False))
    input_type = str(train_cfg.get("input_type", "feature_bone"))

    if input_type == "raw_bone":
        # ── Raw bone‑acc multi‑axis path ────────────────────────────────
        raw_dir = _resolve_processed_dir(
            str(train_cfg.get("processed_dir", "raw_bone_binaural")),
            default_subdir="raw_bone_binaural",
        )
        manifest_path = raw_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"input_type=raw_bone requires pre‑computed raw data, "
                f"but {manifest_path} not found.  "
                f"Run: python scripts/precompute_raw_bone.py"
            )
        logger.info("Using pre‑computed raw bone windows from %s", raw_dir)
        full_ds = BoneRawPrecomputedDataset(manifest_path, raw_dir)
        val_subjects_raw = set(exp_cfg.validation_subjects)
        train_indices: list[int] = []
        val_indices: list[int] = []
        for i, r in enumerate(full_ds.records):
            if _raw_subject(r["subject_id"]) in val_subjects_raw:
                val_indices.append(i)
            else:
                train_indices.append(i)
        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        logger.info("Raw split: %d train / %d val", len(train_ds), len(val_ds))

    elif input_type == "feature_bone":
        # ── Engineered feature path ─────────────────────────────────────
        features_dir = _resolve_processed_dir(
            str(train_cfg.get("processed_dir", "features/bone_binaural")),
            default_subdir="features/bone_binaural",
        )
        if precomputed and (features_dir / "manifest.json").exists():
            logger.info("Using pre‑computed features from %s", features_dir)
            full_ds = BoneBinauralPrecomputedDataset(
                features_dir / "manifest.json", features_dir
            )
            val_subjects_raw = set(exp_cfg.validation_subjects)
            train_indices: list[int] = []
            val_indices: list[int] = []
            for i, r in enumerate(full_ds.records):
                if _raw_subject(r["subject_id"]) in val_subjects_raw:
                    val_indices.append(i)
                else:
                    train_indices.append(i)
            train_ds = torch.utils.data.Subset(full_ds, train_indices)
            val_ds = torch.utils.data.Subset(full_ds, val_indices)
            logger.info(
                "Precomputed split: %d train / %d val",
                len(train_ds),
                len(val_ds),
            )
        else:
            if precomputed:
                logger.warning(
                    "precomputed=true but %s not found — falling back "
                    "to on‑the‑fly feature extraction.  "
                    "Run: python scripts/precompute_features.py",
                    features_dir / "manifest.json",
                )
            frame_ms = float(train_cfg.get("frame_ms", 50.0))
            hop_ms = float(train_cfg.get("hop_ms", 10.0))
            train_ds = BoneBinauralFeatureDataset(
                train_recs, frame_ms=frame_ms, hop_ms=hop_ms
            )
            val_ds = BoneBinauralFeatureDataset(
                val_recs, frame_ms=frame_ms, hop_ms=hop_ms
            )
    else:
        raise ValueError(
            f"Unknown train.input_type: {input_type!r}. "
            f"Expected 'feature_bone' or 'raw_bone'."
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(train_cfg.num_workers),
        collate_fn=pad_collate,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=int(train_cfg.num_workers),
        collate_fn=pad_collate,
        drop_last=False,
    )

    logger.info(
        "Train batches: %d  Val batches: %d", len(train_loader), len(val_loader)
    )

    # ---- model -------------------------------------------------------------
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = build_model(cfg).to(device)

    # ---- optimizer & loss --------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.learning_rate),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    criterion = build_loss()

    # ---- train -------------------------------------------------------------
    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_epochs=int(train_cfg.max_epochs),
        log_interval=1,
    )

    # ---- save artifacts ----------------------------------------------------
    # Hydra has already changed cwd to the run directory (outputs/logs/...).
    output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    val_metrics = history.get("val_metrics", [])

    # Best checkpoint.
    if val_metrics:
        best_idx = min(
            range(len(val_metrics)),
            key=lambda i: val_metrics[i]["val_loss"],
        )
        best_ckpt = {
            "epoch": best_idx + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics[best_idx],
        }
        torch.save(best_ckpt, output_dir / "best_checkpoint.pt")
        logger.info("Best checkpoint saved (epoch %d)", best_idx + 1)

    # Final metrics JSON.
    if val_metrics:
        final = val_metrics[-1]
        # Convert tensor values to float.
        metrics_json = {
            "overall": {k: float(v) for k, v in final["overall"].items()},
            "by_domain": {},
        }
        for domain, dmetrics in final["by_group"].items():
            metrics_json["by_domain"][domain] = {
                k: float(v) for k, v in dmetrics.items()
            }
        with (output_dir / "final_metrics.json").open("w") as f:
            json.dump(metrics_json, f, indent=2)
        logger.info("Final metrics saved to %s", output_dir / "final_metrics.json")

    # Config snapshot.
    with (output_dir / "config.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info("Config snapshot saved to %s", output_dir / "config.yaml")

    # ---- final report ------------------------------------------------------
    if val_metrics:
        final = val_metrics[-1]
        logger.info("=== Final Validation ===")
        logger.info("  Overall:")
        for k, v in final["overall"].items():
            logger.info("    %s: %.4f", k, v)
        logger.info("  By domain:")
        for domain, dmetrics in final["by_group"].items():
            logger.info("    %s:", domain)
            for k, v in dmetrics.items():
                logger.info("      %s: %.4f", k, v)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
