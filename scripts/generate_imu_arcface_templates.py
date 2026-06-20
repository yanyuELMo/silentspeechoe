"""Generate subject authentication templates from an ArcFace IMU encoder.

The script loads a trained ``imu_mlp_arcface`` checkpoint, extracts
L2-normalized embeddings from precomputed binaural IMU feature vectors, and
stores one normalized centroid template per subject.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from silentspeechoe.models.imu_mlp import IMUMLPArcFace  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ArcFace IMU subject templates."
    )
    parser.add_argument(
        "--encoder",
        default="outputs/encoders/imu_mlp_arcface_left_normal_silent_whisper_encoder.pt",
        help="ArcFace encoder checkpoint exported by scripts/train.py.",
    )
    parser.add_argument(
        "--feature-dir",
        default=(
            "data/processed/features/"
            "imu_te_binaural_lrdiff_200hz_raw9_gaussian_noise_005"
        ),
        help="Feature directory used for enrollment/template samples.",
    )
    parser.add_argument(
        "--stats-feature-dir",
        default=None,
        help=(
            "Feature directory used to recompute train-time z-score stats. "
            "Defaults to the encoder checkpoint source_processed_dir."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs/close_set_evaluation/"
            "imu_mlp_arcface_normal_silent_whisper_noisy_nonsemantic"
        ),
        help=(
            "Authentication experiment directory. Template files are written "
            "inside this folder."
        ),
    )
    parser.add_argument(
        "--out-name",
        default="templates",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--sentence-type",
        default="non_semantic",
        help="Sentence type used as enrollment samples.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["normal", "whisper", "silent"],
        help="Domains used as enrollment samples.",
    )
    parser.add_argument(
        "--subject-ids",
        nargs="+",
        default=None,
        help="Optional subject IDs to enroll, e.g. sub_13 sub_14.",
    )
    parser.add_argument(
        "--stats-domains",
        nargs="+",
        default=None,
        help=(
            "Domains used to recompute train-time z-score stats. Defaults to "
            "the encoder checkpoint train_domains when available."
        ),
    )
    parser.add_argument(
        "--stats-sentence-types",
        nargs="+",
        default=None,
        help=(
            "Sentence types used to recompute train-time z-score stats. "
            "Defaults to encoder checkpoint source_sentence_types. An empty "
            "checkpoint list means all sentence types."
        ),
    )
    parser.add_argument(
        "--stats-subject-ids",
        nargs="+",
        default=None,
        help=(
            "Optional subject IDs used to recompute train-time z-score stats. "
            "Defaults to encoder checkpoint train_subjects when available."
        ),
    )
    parser.add_argument(
        "--max-sentences-per-subject",
        type=int,
        default=None,
        help=(
            "If set, keep only the first N sentence_id values per subject "
            "after sentence_id sorting. All repeats of selected sentences "
            "are retained."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding extraction batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for embedding extraction.",
    )
    return parser.parse_args(argv)


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _load_manifest(feature_dir: Path) -> dict:
    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r") as f:
        return json.load(f)


def _canonical_subject_ids(subject_ids: list[str] | None) -> set[str] | None:
    if subject_ids is None:
        return None
    canonical = set()
    for subject_id in subject_ids:
        value = str(subject_id)
        canonical.add(value if value.startswith("sub_") else f"sub_{value}")
    return canonical


def _filter_records(
    records: list[dict],
    *,
    sentence_types: set[str] | None,
    domains: set[str],
    subject_ids: set[str] | None = None,
) -> list[dict]:
    return [
        record
        for record in records
        if (sentence_types is None or record.get("sentence_type") in sentence_types)
        and record.get("domain") in domains
        and (subject_ids is None or str(record.get("subject_id")) in subject_ids)
    ]


def _sentence_sort_key(record: dict) -> tuple:
    sentence_id = str(record.get("sentence_id", ""))
    prefix, _, suffix = sentence_id.rpartition("_")
    sentence_number = int(suffix) if suffix.isdigit() else -1
    return (
        prefix,
        sentence_number,
        sentence_id,
        int(record.get("repeat_id", -1)),
        str(record.get("file", "")),
    )


def _limit_records_by_subject_sentence(
    records: list[dict],
    max_sentences_per_subject: int | None,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Keep only the first N sentence IDs per subject, retaining repeats."""
    if max_sentences_per_subject is None:
        return records, {}
    if max_sentences_per_subject <= 0:
        raise ValueError(
            "max-sentences-per-subject must be positive when provided, got "
            f"{max_sentences_per_subject}"
        )

    records_by_subject: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_subject[str(record["subject_id"])].append(record)

    limited: list[dict] = []
    selected_sentence_ids: dict[str, list[str]] = {}
    for subject_id in sorted(records_by_subject):
        subject_records = sorted(
            records_by_subject[subject_id],
            key=_sentence_sort_key,
        )
        ordered_sentence_ids = []
        seen = set()
        for record in subject_records:
            sentence_id = str(record.get("sentence_id", ""))
            if sentence_id not in seen:
                ordered_sentence_ids.append(sentence_id)
                seen.add(sentence_id)

        selected = ordered_sentence_ids[:max_sentences_per_subject]
        selected_set = set(selected)
        selected_sentence_ids[subject_id] = selected
        limited.extend(
            record
            for record in subject_records
            if str(record.get("sentence_id", "")) in selected_set
        )

    return limited, selected_sentence_ids


def _load_feature_matrix(feature_dir: Path, records: list[dict]) -> torch.Tensor:
    features = []
    for record in records:
        sample = torch.load(
            feature_dir / record["file"],
            map_location="cpu",
            weights_only=True,
        )
        features.append(sample["x"].float())
    if not features:
        raise ValueError("No feature records selected for template generation")
    return torch.stack(features, dim=0)


def _compute_feature_stats(
    feature_dir: Path,
    records: list[dict],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _load_feature_matrix(feature_dir, records)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(eps)
    return mean, std


def _build_model_from_checkpoint(checkpoint: dict) -> IMUMLPArcFace:
    state = checkpoint["full_model_state_dict"]
    fc1_weight = state["fc1.weight"]
    fc2_weight = state["fc2.weight"]
    classifier_weight = state["classifier.weight"]
    model = IMUMLPArcFace(
        in_features=int(fc1_weight.shape[1]),
        hidden1=int(fc1_weight.shape[0]),
        hidden2=int(fc2_weight.shape[0]),
        num_classes=int(classifier_weight.shape[0]),
    )
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def _extract_embeddings(
    model: IMUMLPArcFace,
    x: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    model.to(device)
    mean = mean.to(device)
    std = std.to(device)

    embeddings = []
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size].to(device)
        batch = (batch - mean) / std
        emb = model.extract_features(batch)
        emb = F.normalize(emb, dim=1)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def _make_templates(
    records: list[dict],
    embeddings: torch.Tensor,
) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    by_subject: dict[str, list[torch.Tensor]] = defaultdict(list)
    for record, embedding in zip(records, embeddings, strict=True):
        by_subject[str(record["subject_id"])].append(embedding)

    subject_ids = sorted(by_subject)
    templates = []
    counts = {}
    for subject_id in subject_ids:
        subject_embeddings = torch.stack(by_subject[subject_id], dim=0)
        template = F.normalize(subject_embeddings.mean(dim=0, keepdim=True), dim=1)
        templates.append(template.squeeze(0))
        counts[subject_id] = int(subject_embeddings.shape[0])
    return torch.stack(templates, dim=0), subject_ids, counts


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    domains = set(args.domains)
    subject_ids = _canonical_subject_ids(args.subject_ids)
    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}")

    encoder_path = _resolve_project_path(args.encoder)
    feature_dir = _resolve_project_path(args.feature_dir)
    out_dir = _resolve_project_path(args.out_dir)

    checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=False)
    stats_domains = set(
        args.stats_domains
        if args.stats_domains is not None
        else checkpoint.get("train_domains", sorted(domains))
    )
    checkpoint_source_types = checkpoint.get("source_sentence_types", None)
    if args.stats_sentence_types is not None:
        stats_sentence_types = set(args.stats_sentence_types)
    elif checkpoint_source_types is None:
        stats_sentence_types = {args.sentence_type}
    else:
        stats_sentence_types = set(checkpoint_source_types) or None
    stats_subject_ids = _canonical_subject_ids(
        args.stats_subject_ids
        if args.stats_subject_ids is not None
        else checkpoint.get("train_subjects") or None
    )
    stats_feature_dir = (
        _resolve_project_path(args.stats_feature_dir)
        if args.stats_feature_dir
        else _resolve_project_path(
            checkpoint.get(
                "source_processed_dir",
                "data/processed/features/imu_te_binaural_lrdiff_200hz_raw9",
            )
        )
    )

    feature_manifest = _load_manifest(feature_dir)
    stats_manifest = _load_manifest(stats_feature_dir)

    enrollment_records = _filter_records(
        feature_manifest["records"],
        sentence_types={args.sentence_type},
        domains=domains,
        subject_ids=subject_ids,
    )
    enrollment_records, selected_sentence_ids = _limit_records_by_subject_sentence(
        enrollment_records,
        args.max_sentences_per_subject,
    )
    stats_records = _filter_records(
        stats_manifest["records"],
        sentence_types=stats_sentence_types,
        domains=stats_domains,
        subject_ids=stats_subject_ids,
    )
    if not stats_records:
        raise ValueError("No records selected for feature standardization stats")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    model = _build_model_from_checkpoint(checkpoint)
    mean, std = _compute_feature_stats(stats_feature_dir, stats_records)
    x = _load_feature_matrix(feature_dir, enrollment_records)
    embeddings = _extract_embeddings(
        model,
        x,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        device=device,
    )
    templates, subject_ids, counts = _make_templates(enrollment_records, embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{args.out_name}.pt"
    json_path = out_dir / f"{args.out_name}.json"

    payload = {
        "templates": templates,
        "subject_ids": subject_ids,
        "subject_to_index": {subject: i for i, subject in enumerate(subject_ids)},
        "counts": counts,
        "feature_mean": mean,
        "feature_std": std,
        "encoder_checkpoint": str(encoder_path),
        "feature_dir": str(feature_dir),
        "stats_feature_dir": str(stats_feature_dir),
        "sentence_type": args.sentence_type,
        "domains": sorted(domains),
        "stats_domains": sorted(stats_domains),
        "stats_sentence_types": (
            sorted(stats_sentence_types) if stats_sentence_types is not None else None
        ),
        "subject_ids_filter": sorted(subject_ids) if subject_ids is not None else None,
        "stats_subject_ids": (
            sorted(stats_subject_ids) if stats_subject_ids is not None else None
        ),
        "max_sentences_per_subject": args.max_sentences_per_subject,
        "selected_sentence_ids": selected_sentence_ids,
        "embedding_dim": int(templates.shape[1]),
        "template_method": ("mean_of_l2_normalized_embeddings_then_l2_normalize"),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_epoch": checkpoint.get("selection_epoch"),
        "selection_value": checkpoint.get("selection_value"),
    }
    torch.save(payload, pt_path)

    manifest = {
        "name": args.out_name,
        "template_file": str(pt_path),
        "num_subjects": len(subject_ids),
        "num_enrollment_samples": len(enrollment_records),
        "embedding_dim": int(templates.shape[1]),
        "subject_ids": subject_ids,
        "counts": counts,
        "encoder_checkpoint": str(encoder_path),
        "feature_dir": str(feature_dir),
        "stats_feature_dir": str(stats_feature_dir),
        "sentence_type": args.sentence_type,
        "domains": sorted(domains),
        "stats_domains": sorted(stats_domains),
        "stats_sentence_types": (
            sorted(stats_sentence_types) if stats_sentence_types is not None else None
        ),
        "subject_ids_filter": sorted(subject_ids) if subject_ids is not None else None,
        "stats_subject_ids": (
            sorted(stats_subject_ids) if stats_subject_ids is not None else None
        ),
        "max_sentences_per_subject": args.max_sentences_per_subject,
        "selected_sentence_ids": selected_sentence_ids,
        "feature_standardization": "stats_feature_dir train-style z-score",
        "embedding_normalization": "l2",
        "template_method": payload["template_method"],
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_epoch": checkpoint.get("selection_epoch"),
        "selection_value": checkpoint.get("selection_value"),
    }
    with json_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved templates: {pt_path}")
    print(f"Saved manifest : {json_path}")
    print(f"Subjects       : {len(subject_ids)}")
    print(f"Samples        : {len(enrollment_records)}")
    print(f"Template shape : {tuple(templates.shape)}")


if __name__ == "__main__":
    main()
