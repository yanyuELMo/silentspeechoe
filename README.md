# ArticuEar

ArticuEar is a PyTorch research codebase for earable sensor-based silent speech
experiments. The project is built around OpenEarable 2.0 recordings and focuses
on learning compact sensor encoders from ear-worn motion and vibration signals.

The repository is intended for master thesis research on cross-domain silent
speech and speaker-related sensing. The main experimental setting studies how
models trained from non-semantic utterances and different speaking modes can
generalize to semantic utterances, especially across normal speech, whisper
speech, and silent speech.

The Python package namespace is currently `silentspeechoe` for historical
compatibility, while the project name is ArticuEar.

## Research Goal

Earable devices can capture subtle articulatory and head-motion cues during
speech-like behavior. ArticuEar uses these signals to study whether lightweight
neural encoders can learn stable representations from in-ear sensors under
different speaking conditions.

The current codebase supports:

- IMU and bone-conduction accelerometer data handling.
- Window-level preprocessing and sensor dataset utilities.
- IMU augmentation for robustness to wearing angle, rhythm, amplitude, and
  sensor noise.
- Temporal neural models such as TCN, CNN, MLP, LSTM, ResNet, and feature-token
  Transformer variants.
- Classification, ArcFace-style metric learning, and template-oriented
  evaluation metrics.
- Hydra-configured experiment, model, and training recipes.

## Tracked Repository Structure

Only Git-tracked project files are described here. Large local artifacts such as
raw sensor recordings, processed tensors, checkpoints, outputs, and ad-hoc local
scripts are intentionally excluded from version control.

```text
.
├── .devcontainer/
│   ├── Dockerfile
│   ├── devcontainer.json
│   ├── docker-compose.yml
│   ├── requirements-dev.txt
│   └── requirements.txt
├── .github/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/ci.yml
├── configs/
│   ├── config.yaml
│   ├── data/
│   ├── experiment/
│   ├── model/
│   └── train/
├── data/
│   ├── README.md
│   ├── instruction.txt
│   ├── instruction2.txt
│   └── metadata/
├── src/
│   └── silentspeechoe/
│       ├── config.py
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       ├── training/
│       └── utils/
├── tests/
├── LICENSE
├── pyproject.toml
└── README.md
```

## Configuration

The project uses Hydra for experiment management.

```text
configs/data/        Dataset and sensor-data configuration.
configs/experiment/  Experimental splits, domains, subjects, and modes.
configs/model/       Model architecture choices and hyperparameters.
configs/train/       Training recipes, optimizer settings, augmentation, and loss.
```

Experiment names follow the project convention:

```text
<sensor>_<ear>_<model>_<sentence_type>_<modes>
```

Examples include:

```text
imu_binaural_tcn_subjects36_nonsemantic_all
imu_right_tcn_subjects36_nonsemantic_all
imu_binaural_tcn_subjects36_nonsemantic_normal_whisper
```

## Source Package

The tracked source package is under `src/silentspeechoe/`.

```text
data/        Dataset classes, preprocessing utilities, labels, collation, and
             IMU augmentation.
evaluation/  Classification, authentication, attack, and plotting metrics.
features/    Hand-crafted feature extraction helpers for IMU and bone-acc data.
models/      Neural network definitions and model builders.
training/    Training loop, losses, and optimization utilities.
utils/       Checkpointing, logging, I/O, and seed helpers.
```

## Data Policy

Tracked data is limited to metadata and lightweight documentation under
`data/`.

Raw recordings, processed tensors, model checkpoints, and experiment outputs are
not tracked. This keeps the repository suitable for GitHub and makes the code
reviewable without requiring the full local dataset.

The tracked metadata includes utterance-level event information such as subject,
ear, session, sentence type, label, speaking mode, repeat index, and time window
boundaries.

## Models and Losses

The model package contains compact research baselines for earable time-series
experiments:

- Temporal CNN / TCN encoders for raw or preprocessed sensor windows.
- Feature MLP and CNN models for precomputed feature vectors.
- LSTM and ResNet variants for temporal and feature baselines.
- Feature-token Transformer variants for structured feature inputs.

The training package supports standard cross-entropy classification and
ArcFace-style metric learning losses. ArcFace is used when the experiment needs
an embedding space with stronger angular class separation.

## Development

Install development dependencies:

```bash
make install-dev
```

Run the standard checks:

```bash
make check
```

Equivalent manual checks:

```bash
ruff check .
ruff format --check .
pytest
```

CI is intentionally lightweight and should not require GPU access, full dataset
preprocessing, or complete training runs.

## Project Status

ArticuEar is an active research codebase. The tracked files provide the reusable
configuration, package code, tests, and metadata needed to reproduce and inspect
the research workflow. Local experiment artifacts are deliberately kept outside
Git tracking.
