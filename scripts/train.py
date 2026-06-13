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
from silentspeechoe.data.imu_preprocessing import (  # noqa: E402
    IMUDataset,
    MFCCFeatureDataset,
    PrecomputedIMUDataset,
    build_imu_records,
    imu_pad_collate,
    mfcc_collate,
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


def _resolve_source_domains(exp_cfg: DictConfig) -> set[str]:
    """Read source domain(s) from experiment config.

    Supports a single ``source_domain`` string (legacy) or a list
    ``source_domains``.
    """
    if "source_domains" in exp_cfg:
        return set(exp_cfg.source_domains)
    if "source_domain" in exp_cfg:
        return {exp_cfg.source_domain}
    return {"normal"}


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


def _resolve_channel_indices(model_cfg: DictConfig) -> list[int] | None:
    """Read optional ``channel_indices`` from the model config.

    Returns ``None`` when the config does not specify a subset, so the
    dataset returns all available channels.
    """
    if "channel_indices" in model_cfg:
        return [int(i) for i in model_cfg.channel_indices]
    return None


class _FeatureVectorSubset(torch.utils.data.Dataset):
    """Subset wrapper for fixed vectors with optional targets and scaling."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        indices: list[int],
        targets: dict[int, int] | None = None,
        *,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.targets = targets
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        source_idx = self.indices[idx]
        item = dict(self.dataset[source_idx])
        if self.feature_mean is not None and self.feature_std is not None:
            item["x"] = (item["x"].float() - self.feature_mean) / self.feature_std
        if self.targets is not None:
            item["y"] = int(self.targets[source_idx])
        return item


def _compute_feature_stats(
    dataset: torch.utils.data.Dataset,
    indices: list[int],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute train-only mean/std for fixed-length feature vectors."""
    if not indices:
        raise ValueError("Cannot compute feature stats for an empty index set")
    xs = [dataset[i]["x"].float() for i in indices]
    x = torch.stack(xs, dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(eps)
    return mean, std


def _clone_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Clone a state dict to CPU for portable checkpoint export."""
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _save_encoder(
    model: torch.nn.Module,
    cfg: DictConfig,
    input_type: str,
    output_dir: Path,
    *,
    model_state_dict: dict[str, torch.Tensor] | None = None,
    selection_metric: str | None = None,
    selection_epoch: int | None = None,
    selection_value: float | None = None,
) -> None:
    """Export an encoder checkpoint for downstream feature extraction.

    Only active for IMU models.  Saves the full model state plus metadata
    so a feature‑extraction pipeline can reload it without the training
    config.
    """
    if not (input_type.startswith("imu") or input_type.startswith("mfcc")):
        return

    encoder_dir = _PROJECT_ROOT / "outputs" / "encoders"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg.model
    imu_cfg = cfg.data.imu

    # Resolve channels (model config overrides data/imu defaults).
    model_channels = list(model_cfg.get("channels", []))
    all_channels = model_channels or list(
        imu_cfg.get(
            "channels",
            [
                "acc.x",
                "acc.y",
                "acc.z",
                "gyro.x",
                "gyro.y",
                "gyro.z",
                "mag.x",
                "mag.y",
                "mag.z",
            ],
        )
    )
    in_channels = int(
        imu_cfg.get("in_channels", model_cfg.get("in_channels", len(all_channels)))
    )
    channel_indices = _resolve_channel_indices(model_cfg)
    if channel_indices is not None:
        selected_channels = [all_channels[i] for i in channel_indices]
        channel_suffix = "_accgyro" if in_channels == 6 else "_ch" + str(in_channels)
    else:
        selected_channels = all_channels
        channel_suffix = ""

    # Build a human‑readable filename.
    side = "_".join(sorted(imu_cfg.get("sides", ["left"])))
    source_domains = _resolve_source_domains(cfg.experiment)
    if len(source_domains) > 1:
        source = "_".join(sorted(source_domains))
    else:
        source = next(iter(source_domains))
    fname = f"{model_cfg.name}_{side}_{source}{channel_suffix}_encoder.pt"

    # Resolve processed_dir for metadata.
    train_cfg = cfg.train
    processed_dir = str(train_cfg.get("processed_dir", "imu_windows/left_200hz_raw9"))

    # Extract encoder part (everything except the classifier).
    full_state = _clone_state_dict(model_state_dict or model.state_dict())
    encoder_state = {
        k: v for k, v in full_state.items() if not k.startswith("classifier.")
    }

    metadata: dict = {
        "model_name": model_cfg.name,
        "encoder_state_dict": encoder_state,
        "full_model_state_dict": full_state,
        "in_channels": in_channels,
        "embedding_dim": int(
            model_cfg.get(
                "hidden2",
                model_cfg.get("conv3_channels", model_cfg.get("conv2_channels", 128)),
            )
        ),
        "in_features": int(model_cfg.get("in_features", 0)),
        "target_sample_rate": float(imu_cfg.get("target_sample_rate", 200.0)),
        "side": side,
        "train_domains": sorted(source_domains),
        "eval_domains": cfg.experiment.get(
            "target_domains", ["normal", "whisper", "silent"]
        ),
        "validation_subjects": cfg.experiment.get("validation_subjects", []),
        "channels": selected_channels,
        "source_processed_dir": f"data/processed/{processed_dir}",
    }
    if selection_metric is not None:
        metadata["selection_metric"] = selection_metric
    if selection_epoch is not None:
        metadata["selection_epoch"] = int(selection_epoch)
    if selection_value is not None:
        metadata["selection_value"] = float(selection_value)

    torch.save(metadata, encoder_dir / fname)
    logger.info("Encoder exported to %s", encoder_dir / fname)


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
    val_subjects_raw = set(
        exp_cfg.get("validation_subjects", [])
    )  # e.g. {"07", "10", ...}
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
    collate_fn = pad_collate  # default; overridden by imu branch

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
    elif input_type == "imu":
        # ── IMU single‑side path ─────────────────────────────────────────
        imu_cfg = cfg.data.imu
        sides = list(imu_cfg.get("sides", ["left"]))
        target_sr = float(imu_cfg.get("target_sample_rate", 200.0))
        padding = float(imu_cfg.get("padding_sec", 0.0))
        normalize = bool(imu_cfg.get("normalize", False))

        all_records = build_imu_records(
            events_path=str(_PROJECT_ROOT / cfg.data.metadata_file),
            raw_dir=str(_PROJECT_ROOT / cfg.data.root_dir),
            sides=sides,
        )
        logger.info("Total IMU records: %d (sides=%s)", len(all_records), sides)

        # Domain filtering per experiment config.
        source_domains = _resolve_source_domains(exp_cfg)
        target_domains = set(
            exp_cfg.get("target_domains", ["normal", "whisper", "silent"])
        )

        train_recs = [
            r
            for r in all_records
            if _raw_subject(r["subject_id"]) not in val_subjects_raw
            and r["domain"] in source_domains
        ]
        val_recs = [
            r
            for r in all_records
            if _raw_subject(r["subject_id"]) in val_subjects_raw
            and r["domain"] in target_domains
        ]
        logger.info(
            "IMU split — train: %d (domains=%s), val: %d (domains=%s)",
            len(train_recs),
            source_domains,
            len(val_recs),
            target_domains,
        )

        train_ds = IMUDataset(
            train_recs,
            target_sample_rate=target_sr,
            padding_sec=padding,
            normalize=normalize,
        )
        val_ds = IMUDataset(
            val_recs,
            target_sample_rate=target_sr,
            padding_sec=padding,
            normalize=normalize,
        )
        collate_fn = imu_pad_collate

    elif input_type in ("imu_processed", "imu_precomputed"):
        # ── Pre‑computed / processed IMU windows path ────────────────────
        imu_dir = _resolve_processed_dir(
            str(train_cfg.get("processed_dir", "imu_windows/left_200hz_raw9")),
            default_subdir="imu_windows/left_200hz_raw9",
        )
        manifest_path = imu_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"input_type={input_type} requires pre‑computed IMU data, "
                f"but {manifest_path} not found."
            )
        channel_indices = _resolve_channel_indices(model_cfg)

        logger.info("Using pre‑computed IMU windows from %s", imu_dir)
        full_ds = PrecomputedIMUDataset(
            manifest_path,
            imu_dir,
            channel_indices=channel_indices,
        )
        logger.info(
            "Loaded %d precomputed IMU samples (sides=%s, sr=%.0f Hz, channels=%d)",
            len(full_ds),
            full_ds.sides,
            full_ds.target_sample_rate,
            full_ds.num_channels,
        )

        # Domain filtering + subject split.
        source_domains = _resolve_source_domains(exp_cfg)
        target_domains = set(
            exp_cfg.get("target_domains", ["normal", "whisper", "silent"])
        )
        val_subjects_raw = set(exp_cfg.validation_subjects)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for i, r in enumerate(full_ds.records):
            sid = _raw_subject(r["subject_id"])
            dom = r["domain"]
            if sid not in val_subjects_raw and dom in source_domains:
                train_indices.append(i)
            elif sid in val_subjects_raw and dom in target_domains:
                val_indices.append(i)

        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        collate_fn = imu_pad_collate

        # Log per-domain counts.
        train_doms: dict[str, int] = {}
        for i in train_indices:
            d = full_ds.records[i]["domain"]
            train_doms[d] = train_doms.get(d, 0) + 1
        val_doms: dict[str, int] = {}
        for i in val_indices:
            d = full_ds.records[i]["domain"]
            val_doms[d] = val_doms.get(d, 0) + 1
        logger.info(
            "IMU prec split — train: %d %s, val: %d %s",
            len(train_ds),
            train_doms,
            len(val_ds),
            val_doms,
        )

    elif input_type == "mfcc_processed":
        # ── Pre‑computed MFCC feature vectors ────────────────────────────
        mfcc_dir = _resolve_processed_dir(
            str(train_cfg.get("processed_dir", "features/imu_mfcc_left_200hz_raw9")),
            default_subdir="features/imu_mfcc_left_200hz_raw9",
        )
        manifest_path = mfcc_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"input_type=mfcc_processed requires MFCC features, "
                f"but {manifest_path} not found."
            )
        logger.info("Using MFCC features from %s", mfcc_dir)
        full_ds = MFCCFeatureDataset(manifest_path, mfcc_dir)
        logger.info(
            "Loaded %d MFCC samples (dim=%d)",
            len(full_ds),
            full_ds.feature_dim,
        )

        # Domain filtering + subject split.
        source_domains = _resolve_source_domains(exp_cfg)
        target_domains = set(
            exp_cfg.get("target_domains", ["normal", "whisper", "silent"])
        )
        val_subjects_raw = set(exp_cfg.validation_subjects)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for i, r in enumerate(full_ds.records):
            sid = _raw_subject(r["subject_id"])
            dom = r["domain"]
            if sid not in val_subjects_raw and dom in source_domains:
                train_indices.append(i)
            elif sid in val_subjects_raw and dom in target_domains:
                val_indices.append(i)

        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        collate_fn = mfcc_collate

        train_doms = {}
        for i in train_indices:
            d = full_ds.records[i]["domain"]
            train_doms[d] = train_doms.get(d, 0) + 1
        val_doms = {}
        for i in val_indices:
            d = full_ds.records[i]["domain"]
            val_doms[d] = val_doms.get(d, 0) + 1
        logger.info(
            "MFCC split — train: %d %s, val: %d %s",
            len(train_ds),
            train_doms,
            len(val_ds),
            val_doms,
        )

    elif input_type == "imu_feature_processed":
        # ── Pre-computed fixed-length IMU feature vectors ────────────────
        feature_dir = _resolve_processed_dir(
            str(
                train_cfg.get(
                    "processed_dir",
                    "features/imu_te_binaural_lrdiff_200hz_raw9",
                )
            ),
            default_subdir="features/imu_te_binaural_lrdiff_200hz_raw9",
        )
        manifest_path = feature_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"input_type=imu_feature_processed requires feature vectors, "
                f"but {manifest_path} not found."
            )
        target_feature_dir = feature_dir
        target_processed_dir = train_cfg.get("target_processed_dir", None)
        if target_processed_dir:
            target_feature_dir = _resolve_processed_dir(
                str(target_processed_dir),
                default_subdir="features/imu_te_binaural_lrdiff_200hz_raw9",
            )
        target_manifest_path = target_feature_dir / "manifest.json"
        if not target_manifest_path.exists():
            raise FileNotFoundError(
                f"input_type=imu_feature_processed target features require "
                f"{target_manifest_path}, but it was not found."
            )

        logger.info("Using IMU train feature vectors from %s", feature_dir)
        full_ds = MFCCFeatureDataset(manifest_path, feature_dir)
        logger.info(
            "Loaded %d IMU train feature samples (dim=%d)",
            len(full_ds),
            full_ds.feature_dim,
        )
        target_full_ds = full_ds
        if target_feature_dir.resolve() != feature_dir.resolve():
            logger.info("Using IMU target feature vectors from %s", target_feature_dir)
            target_full_ds = MFCCFeatureDataset(
                target_manifest_path,
                target_feature_dir,
            )
            logger.info(
                "Loaded %d IMU target feature samples (dim=%d)",
                len(target_full_ds),
                target_full_ds.feature_dim,
            )
            if int(target_full_ds.feature_dim) != int(full_ds.feature_dim):
                raise ValueError(
                    "IMU train and target feature dimensions must match, got "
                    f"{full_ds.feature_dim} and {target_full_ds.feature_dim}"
                )

        source_domains = _resolve_source_domains(exp_cfg)
        target_domains = set(
            exp_cfg.get("target_domains", ["normal", "whisper", "silent"])
        )
        source_sentence_types = set(exp_cfg.get("source_sentence_types", []))
        target_sentence_types = set(exp_cfg.get("target_sentence_types", []))

        train_candidates: list[int] = []
        val_candidates: list[int] = []
        for i, r in enumerate(full_ds.records):
            dom = r["domain"]
            sentence_type = r.get("sentence_type", "")
            is_source_type = (
                not source_sentence_types or sentence_type in source_sentence_types
            )
            if dom in source_domains and is_source_type:
                train_candidates.append(i)
        for i, r in enumerate(target_full_ds.records):
            dom = r["domain"]
            sentence_type = r.get("sentence_type", "")
            is_target_type = (
                not target_sentence_types or sentence_type in target_sentence_types
            )
            if dom in target_domains and is_target_type:
                val_candidates.append(i)

        task = str(exp_cfg.get("task", "closed_set_sentence_classification"))
        if task == "subject_identification":
            train_subjects = {
                full_ds.records[i]["subject_id"] for i in train_candidates
            }
            val_subjects = {
                target_full_ds.records[i]["subject_id"] for i in val_candidates
            }
            subjects = sorted(train_subjects & val_subjects)
            subject_to_label = {subject: idx for idx, subject in enumerate(subjects)}
            train_indices = [
                i
                for i in train_candidates
                if full_ds.records[i]["subject_id"] in subject_to_label
            ]
            val_indices = [
                i
                for i in val_candidates
                if target_full_ds.records[i]["subject_id"] in subject_to_label
            ]
            train_targets = {
                i: subject_to_label[full_ds.records[i]["subject_id"]]
                for i in train_indices
            }
            val_targets = {
                i: subject_to_label[target_full_ds.records[i]["subject_id"]]
                for i in val_indices
            }
            expected_classes = len(subjects)
            configured_classes = int(model_cfg.get("num_classes", expected_classes))
            if configured_classes != expected_classes:
                raise ValueError(
                    "model.num_classes must match the number of identification "
                    f"subjects ({expected_classes}), got {configured_classes}"
                )
            logger.info("Subject classes: %d (%s)", len(subjects), subjects)
        else:
            train_indices = train_candidates
            val_indices = val_candidates
            train_targets = None
            val_targets = None

        feature_mean = None
        feature_std = None
        if bool(train_cfg.get("standardize_features", True)):
            feature_mean, feature_std = _compute_feature_stats(full_ds, train_indices)
            logger.info("Feature standardization: train-only z-score enabled")

        train_ds = _FeatureVectorSubset(
            full_ds,
            train_indices,
            train_targets,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        val_ds = _FeatureVectorSubset(
            target_full_ds,
            val_indices,
            val_targets,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

        collate_fn = mfcc_collate

        train_doms = {}
        train_types = {}
        for i in train_indices:
            rec = full_ds.records[i]
            train_doms[rec["domain"]] = train_doms.get(rec["domain"], 0) + 1
            stype = rec.get("sentence_type", "")
            train_types[stype] = train_types.get(stype, 0) + 1
        val_doms = {}
        val_types = {}
        for i in val_indices:
            rec = target_full_ds.records[i]
            val_doms[rec["domain"]] = val_doms.get(rec["domain"], 0) + 1
            stype = rec.get("sentence_type", "")
            val_types[stype] = val_types.get(stype, 0) + 1
        logger.info(
            "IMU feature split — train: %d domains=%s types=%s, "
            "val: %d domains=%s types=%s",
            len(train_ds),
            train_doms,
            train_types,
            len(val_ds),
            val_doms,
            val_types,
        )

    elif input_type == "mfcc_2d":
        # ── MFCC features reshaped to [C, T] for 1‑D CNN ───────────────
        mfcc_dir = _resolve_processed_dir(
            str(train_cfg.get("processed_dir", "features/imu_mfcc_left_200hz_raw9")),
            default_subdir="features/imu_mfcc_left_200hz_raw9",
        )
        manifest_path = mfcc_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"input_type=mfcc_2d requires MFCC features, "
                f"but {manifest_path} not found."
            )
        logger.info("Using MFCC features (2D reshape) from %s", mfcc_dir)
        full_ds = MFCCFeatureDataset(manifest_path, mfcc_dir, reshape_2d=True)
        logger.info(
            "Loaded %d MFCC samples (shape=[%d, %d])",
            len(full_ds),
            full_ds.num_channels,
            full_ds.feature_dim // full_ds.num_channels,
        )

        source_domains = _resolve_source_domains(exp_cfg)
        target_domains = set(
            exp_cfg.get("target_domains", ["normal", "whisper", "silent"])
        )
        val_subjects_raw = set(exp_cfg.validation_subjects)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for i, r in enumerate(full_ds.records):
            sid = _raw_subject(r["subject_id"])
            dom = r["domain"]
            if sid not in val_subjects_raw and dom in source_domains:
                train_indices.append(i)
            elif sid in val_subjects_raw and dom in target_domains:
                val_indices.append(i)

        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        collate_fn = imu_pad_collate

        train_doms = {}
        for i in train_indices:
            d = full_ds.records[i]["domain"]
            train_doms[d] = train_doms.get(d, 0) + 1
        val_doms = {}
        for i in val_indices:
            d = full_ds.records[i]["domain"]
            val_doms[d] = val_doms.get(d, 0) + 1
        logger.info(
            "MFCC 2D split — train: %d %s, val: %d %s",
            len(train_ds),
            train_doms,
            len(val_ds),
            val_doms,
        )

    else:
        raise ValueError(
            f"Unknown train.input_type: {input_type!r}. "
            f"Expected 'feature_bone', 'raw_bone', 'imu', "
            f"'imu_processed', 'mfcc_processed', 'imu_feature_processed', "
            f"or 'mfcc_2d'."
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(train_cfg.num_workers),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=False,
        num_workers=int(train_cfg.num_workers),
        collate_fn=collate_fn,
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
    best_loss_snapshot = history.get("best_val_loss")
    best_accuracy_snapshot = history.get("best_accuracy")

    # Best checkpoint by validation loss.
    if best_loss_snapshot is not None:
        best_ckpt = {
            "epoch": int(best_loss_snapshot["epoch"]),
            "selection_metric": "val_loss",
            "selection_value": float(best_loss_snapshot["value"]),
            "model_state_dict": best_loss_snapshot["model_state_dict"],
            "optimizer_state_dict": best_loss_snapshot["optimizer_state_dict"],
            "val_metrics": best_loss_snapshot["val_metrics"],
        }
        torch.save(best_ckpt, output_dir / "best_checkpoint.pt")
        logger.info(
            "Best validation-loss checkpoint saved (epoch %d, val_loss %.4f)",
            best_ckpt["epoch"],
            best_ckpt["selection_value"],
        )

    # Best checkpoint by overall validation accuracy.
    if best_accuracy_snapshot is not None:
        best_acc_ckpt = {
            "epoch": int(best_accuracy_snapshot["epoch"]),
            "selection_metric": "overall_accuracy",
            "selection_value": float(best_accuracy_snapshot["value"]),
            "model_state_dict": best_accuracy_snapshot["model_state_dict"],
            "optimizer_state_dict": best_accuracy_snapshot["optimizer_state_dict"],
            "val_metrics": best_accuracy_snapshot["val_metrics"],
        }
        torch.save(best_acc_ckpt, output_dir / "best_accuracy_checkpoint.pt")
        logger.info(
            "Best validation-accuracy checkpoint saved (epoch %d, accuracy %.4f)",
            best_acc_ckpt["epoch"],
            best_acc_ckpt["selection_value"],
        )

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

    # ---- encoder export ------------------------------------------------------
    _save_encoder(
        model=model,
        cfg=cfg,
        input_type=input_type,
        output_dir=output_dir,
        model_state_dict=(
            best_accuracy_snapshot["model_state_dict"]
            if best_accuracy_snapshot is not None
            else None
        ),
        selection_metric=(
            "overall_accuracy" if best_accuracy_snapshot is not None else None
        ),
        selection_epoch=(
            int(best_accuracy_snapshot["epoch"])
            if best_accuracy_snapshot is not None
            else None
        ),
        selection_value=(
            float(best_accuracy_snapshot["value"])
            if best_accuracy_snapshot is not None
            else None
        ),
    )

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
