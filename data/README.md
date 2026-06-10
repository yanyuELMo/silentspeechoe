# Data README

This directory contains OpenEarable 2.0 sensor streams, manual annotation files, and future preprocessed training samples.

The `data/` directory is intentionally ignored by Git. This README is a local dataset note and is not meant to be committed unless the ignore policy changes.

## Data Source

The dataset was collected with OpenEarable 2.0 while participants stayed still. The setting is intended to simulate a stable registration-like scenario where the head does not move significantly.

OpenEarable 2.0 records both ears, so each valid participant may have two sides:

```text
left/
right/
```

The experiment contains three speaking modes:

```text
normal   normal speech
whisper  whispered speech
silent   silent speech
```

The current project focuses on closed-set silent speech sentence recognition, especially cross-domain recognition such as:

```text
whisper -> silent
```

## Current Scope

The full experiment may include single-character and two-character recordings, but the current project only uses sentence-level data:

- `non-semantic`: 20 meaningless sentences
- `semantic`: 16 meaningful sentences

The current label convention is:

```text
1-20   non-semantic sentences
21-36  semantic sentences
```

Single-character and two-character data are not used at this stage.

## Directory Layout

Recommended local structure:

```text
data/
  raw/
    left/
      <subject_id>/
        non-semantic/
        semantic/
    right/
      <subject_id>/
        non-semantic/
        semantic/

  metadata/
    left/
      labels_left_nw.xlsx
      labels_left_silent.xlsx
    right/
      labels_right_nw.xlsx
      labels_right_silent.xlsx

  processed/
```

Meaning:

- `raw/` stores original sensor-stream CSV files only.
- `metadata/` stores manual annotations and aligned label sheets.
- `processed/` stores future segmented, resampled, feature-ready training samples.

If a participant is missing either `non-semantic/` or `semantic/`, that missing subset can be skipped for now.

## Raw Sensor Streams

The `raw/` directory currently keeps only sensor-stream data. It does not store annotation sheets or preprocessed train/validation/test splits.

Each participant's `non-semantic/` and `semantic/` folders usually contain sensor files like:

```text
sensor_<recording_id>_<random_id>__barometer.csv
sensor_<recording_id>_<random_id>__bone_acc.csv
sensor_<recording_id>_<random_id>__imu.csv
sensor_<recording_id>_<random_id>__lowrate_merged.csv
```

For left-right alignment, the second random ID in the file name may differ. Use `recording_id` from `sensor_<recording_id>_*` as the main matching key.

The first modeling baseline should focus on:

```text
bone_acc
acc
gyro
```

`acc` and `gyro` usually come from the IMU stream. Do not include `barometer` in the first baseline. Magnetometer channels can be considered later as an ablation.

## Label Files

Current annotation files:

```text
data/metadata/left/labels_left_nw.xlsx
data/metadata/left/labels_left_silent.xlsx
data/metadata/right/labels_right_nw.xlsx
data/metadata/right/labels_right_silent.xlsx
data/metadata/events.csv
```

Naming convention:

- `nw` means normal + whisper.
- `silent` means silent speech.
- `left` / `right` identifies the ear side.

The label sheets store start and end timestamps for each utterance window.

Current notes mention that timestamps are frame-based, with approximately:

```text
1 second = 30 frames
```

Preprocessing scripts should convert this consistently into seconds or sensor sample indices.

### Event CSV Schema

`data/metadata/events.csv` is the normalized event mapping generated from the
left/right Excel annotation sheets. It uses the following fields:

```text
subject_id     Participant ID, for example sub_00.
session_id     Raw recording session ID parsed from the sensor filename.
ear            left or right.
event_id       Slot index within one subject and one ear.
sentence_type  semantic or non_semantic.
sentence_id    sem_001..sem_016 or nonsem_001..nonsem_020.
label_id       Zero-based class label for 36-way classification.
domain         normal, whisper, or silent.
repeat_id      Repetition index, 1 or 2.
start_time     Label-window start time in seconds.
end_time       Label-window end time in seconds.
```

The `session_id` comes from raw filenames such as:

```text
sensor_002_2083961914__bone_acc.csv -> 002_2083961914
```

The `label_id` is the zero-based class index:

```text
nonsem_001 -> 0
...
nonsem_020 -> 19
sem_001    -> 20
...
sem_016    -> 35
```

### Row and Column Semantics

In `labels_*_nw.xlsx`, each `sentence_id` uses two rows:

```text
first row   = normal speech
second row  = whispered speech
```

The difference between one repetition and two repetitions is represented by columns, not by additional mode rows.

For sentences `1-10`, each mode is read twice. These rows may contain two start/end windows:

```text
start_1 / end_1
start_2 / end_2
```

For the remaining sentence IDs, each mode is read once. In those cases, only the first start/end window is expected to be filled.

In `labels_*_silent.xlsx`, the first row for each `sentence_id` represents silent speech. It follows the same timestamp-window idea, where columns indicate whether one or two repetitions are available.

### Mapping Labels to Raw Sensor Streams

Each Excel sheet name is the participant ID, for example `00`, `07`, or `21`.

Each row maps to one speaking mode for one sentence ID:

```text
labels_*_nw.xlsx:
  sentence row 1 = normal speech
  sentence row 2 = whispered speech

labels_*_silent.xlsx:
  sentence row 1 = silent speech
```

Sentence IDs determine which raw subset folder should be used:

```text
1-20   -> raw/<side>/<subject_id>/non-semantic/
21-36  -> raw/<side>/<subject_id>/semantic/
```

For example, if sheet `00` in `labels_left_nw.xlsx` has sentence ID `1` with two normal-speech windows:

```text
start_1 = a, end_1 = b
start_2 = c, end_2 = d
```

then both windows should be cut from:

```text
data/raw/left/00/non-semantic/*__bone_acc.csv
```

The same rule applies to right-ear labels and right-ear raw files:

```text
data/metadata/right/... -> data/raw/right/<subject_id>/<subset>/
```

Empty rows or empty timestamp windows indicate missing readings or unused repetitions. They should be skipped rather than imputed for this baseline.

The label timestamps use a 30-frame-per-second convention. Convert a label timestamp to seconds with:

```text
seconds = minutes * 60 + seconds + frames / 30
```

Raw sensor files may use a different timestamp representation or sampling rate. Preprocessing code should map the converted label time window onto the raw sensor stream's time axis before slicing.

`event_id` is assigned by spreadsheet slot order, not by compacting only
non-empty rows. For sentence `1`, the expected slot order is:

```text
event_id 0 = normal repeat 1
event_id 1 = normal repeat 2
event_id 2 = whisper repeat 1
event_id 3 = whisper repeat 2
event_id 4 = silent repeat 1
event_id 5 = silent repeat 2
```

This preserves left/right alignment even if one annotation window is missing.

## Recording Order

Each sentence is read in normal, whisper, and silent modes. Repetition count depends on the sentence ID.

Sentences `1-10` are usually read twice per mode. The standard order is:

```text
normal -> gap -> normal -> gap ->
whisper -> gap -> whisper -> gap ->
silent -> gap -> silent -> gap
```

Sentences `11-36` are usually read once per mode. The standard order is:

```text
normal -> gap -> whisper -> gap -> silent -> gap
```

Participant `08` is a special case. For repeated sentences, the order is closer to two one-pass blocks:

```text
normal -> gap -> whisper -> gap -> silent -> gap ->
normal -> gap -> whisper -> gap -> silent -> gap
```

Gap durations are not fixed.

## Silent Label Completion Rules

Silent speech has no clear acoustic signal, so it cannot be annotated using only the audio waveform like normal speech.

Current heuristic rules:

- The silent segment usually appears after the current sentence's whisper segment and before the next sentence's normal segment.
- Normal, whisper, and silent durations should be roughly similar for the same sentence.
- If an envelope signal shows a weak motion pattern similar to a smaller normal-speech pattern, use it as evidence.
- If the envelope is unclear, estimate the silent interval from the previous normal/whisper duration.
- If the candidate gap is much shorter than the previous mode duration, for example less than about 50%, the silent utterance may be missing and can be skipped.
- If the next normal utterance is misread or repeated, the candidate gap may be unusually long. In that case, use the envelope to locate the likely silent segment.

Left-ear labels for subjects `00-21` were the main initial annotation target for normal/whisper and silent. Right-ear labels can be generated by aligning corresponding left-right recordings and shifting the left-ear labels by the estimated offset.

## Suggested Preprocessing Pipeline

First baseline pipeline:

1. Read annotation sheets from `metadata/`.
2. Segment each utterance using label windows.
3. For the initial baseline, do not resample yet.
4. Add optional padding, for example `0.2s`, before and after each segment.
5. Keep variable-length windows and use batch padding in the collate function, or crop/pad to a fixed length as a simple first baseline.
6. Build basic channels for each modality:

```text
bone_acc.x/y/z
bone_acc_norm
bone_acc_diff_norm

acc.x/y/z
acc_norm
acc_diff_norm

gyro.x/y/z
gyro_norm
gyro_diff_norm
```

7. Save preprocessed samples under `data/processed/`.

Do not feed raw variable-length, mixed-sampling-rate CSV files directly into the model.

Later baselines can add resampling to a shared time grid, such as `100 Hz` or `200 Hz`, once the label parser and raw window extraction are stable.

## Suggested Task Definitions

Recommended task stages:

1. `normal` vs `whisper`
2. `normal` vs `whisper` vs `silent`
3. closed-set `sentence_id` recognition
4. multi-task `sentence_id + speech_mode`

If the goal is sentence content recognition, use `sentence_id` as the label.

If the goal is speaking-mode recognition, use:

```text
normal
whisper
silent
```

## Suggested Model Input

The first model should use multi-sensor time-series fusion:

```text
bone_acc sequence -> encoder -> embedding
acc sequence      -> encoder -> embedding
gyro sequence     -> encoder -> embedding

[bone_acc_embedding, acc_embedding, gyro_embedding] -> classifier
```

A simple 1D CNN encoder is recommended for the first baseline. Build a clear baseline before adding more complex models.

Suggested ablations:

- `bone_acc only`
- `bone_acc + acc`
- `bone_acc + gyro`
- `bone_acc + acc + gyro`
- later: add `mag` or `barometer`

## Data Split

The dataset has not been formally split into train/validation/test sets yet.

When splits are created, do not randomly split windows into train/validation/test sets. That can leak the same participant into multiple splits.

Use subject-wise splits:

```text
train subjects / validation subjects / test subjects
```

Example:

```text
train: 00-15
val:   17-19
test:  20-21
```

A stricter protocol can use leave-one-subject-out evaluation.

## Notes

- Do not commit `data/` to Git.
- Annotation sheets belong in `metadata/`, not `raw/`.
- Keep `raw/` for original sensor-stream data only.
- No official train/validation/test split currently exists in `data/`.
- Generated `.npz`, `.parquet`, `.pt`, logs, and intermediate outputs should go under `processed/` or `outputs/`.
- Scripts should not hard-code local absolute paths.
- The repository is still scaffold-first. Preprocessing, training, and evaluation logic should be implemented incrementally.
