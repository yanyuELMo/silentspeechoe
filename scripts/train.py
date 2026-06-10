"""Training entry point — Hydra‑backed experiment launcher.

Usage::

    python scripts/train.py \\
        experiment=bone_binaural_all_modes \\
        model=bone_cnn \\
        train=default \\
        train.batch_size=16
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

# Make the src package importable without installing.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from silentspeechoe.data.collate import pad_collate  # noqa: E402
from silentspeechoe.data.dataset import (  # noqa: E402
    BoneBinauralDataset,
    build_binaural_records,
)
from silentspeechoe.models.build import build_model  # noqa: E402
from silentspeechoe.training.losses import build_loss  # noqa: E402
from silentspeechoe.training.trainer import run_training  # noqa: E402
from silentspeechoe.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)


_CONFIG_PATH = str(_PROJECT_ROOT / "configs")


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    # ---- resolve experiment config -----------------------------------------
    exp_cfg = cfg.experiment
    model_cfg = cfg.model
    train_cfg = cfg.train

    logger.info("Experiment : %s", exp_cfg.name)
    logger.info("Model      : %s", model_cfg.name)
    logger.info("Train cfg  : %s", train_cfg.name)

    # ---- seed --------------------------------------------------------------
    set_seed(train_cfg.seed)

    # ---- data --------------------------------------------------------------
    val_subjects = frozenset(exp_cfg.validation_subjects)
    train_recs, val_recs = build_binaural_records(
        base_dir=str(_PROJECT_ROOT),
        val_subjects=val_subjects,
    )

    padding_sec = float(train_cfg.get("padding_sec", 0.0))

    train_ds = BoneBinauralDataset(
        train_recs, padding_sec=padding_sec, base_dir=str(_PROJECT_ROOT)
    )
    val_ds = BoneBinauralDataset(
        val_recs, padding_sec=padding_sec, base_dir=str(_PROJECT_ROOT)
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

    logger.info("Train samples: %d  batches: %d", len(train_ds), len(train_loader))
    logger.info("Val samples  : %d  batches: %d", len(val_ds), len(val_loader))

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

    # ---- final report ------------------------------------------------------
    if history["val_metrics"]:
        final = history["val_metrics"][-1]
        logger.info("=== Final Validation ===")
        logger.info("  Overall:")
        for k, v in final["overall"].items():
            logger.info("    %s: %.4f", k, v)
        logger.info("  By speech mode:")
        for mode, mode_metrics in final["by_group"].items():
            logger.info("    %s:", mode)
            for k, v in mode_metrics.items():
                logger.info("      %s: %.4f", k, v)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
