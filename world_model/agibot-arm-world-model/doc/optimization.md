# Optimization Notes

## 1. Scenario

This sample targets robotic arm world model training and inference on AgiBotWorld-style data. The design focuses on making the sample understandable, reusable, and suitable for Ascend NPU execution.

## 2. Training Design

### 2.1 49-frame training organization

The training dataset uses 49-frame samples. When a single clip is shorter than the target frame count, the loader attempts to merge consecutive clips from the same episode and then sample a contiguous training window.

Why:

- short clips alone provide limited temporal context
- robotic arm motion prediction benefits from longer temporal continuity
- clip concatenation improves motion consistency across local boundaries

Benefit:

- longer temporal context than a single short clip
- better alignment between control trajectory and generated motion
- cleaner sample organization for this scenario

### 2.2 Conditioning design

The training pipeline uses:

- `control_video`
- `reference_image`
- `input_image`
- text prompt

Why:

- `control_video` carries rendered trajectory supervision
- `reference_image` preserves scene appearance information
- `input_image` helps maintain temporal continuation
- text prompt keeps the sample compatible with language-conditioned generation

Benefit:

- better controllability for robotic arm motion generation
- explicit separation of scene appearance and motion control

## 3. Inference Design

### 3.1 Chunk-based autoregressive inference

Inference is performed chunk by chunk instead of generating the full sequence in a single pass.

Why:

- long-horizon generation is more memory-intensive
- chunking is easier to manage on practical NPU hardware
- robotic arm rollouts naturally fit sequential continuation

Method:

- chunk 0 uses the ground-truth first frame as `input_image`
- later chunks use the previous generated last frame as the next `input_image`
- the first frame remains the fixed `reference_image`
- trajectory maps are sliced per chunk

Benefit:

- lower practical inference pressure than one-shot long video generation
- clearer continuation logic for rollout-style prediction

## 4. Ascend Adaptation

The sample includes Ascend-oriented launcher scripts and default runtime environment variables for NPU execution.

Why:

- users need a runnable reference instead of only Python entrypoints
- multi-card training and single-card inference require different defaults

Included adaptation points:

- NPU-oriented shell launchers
- multi-card training defaults
- single-card inference defaults
- explicit model path, dataset path, and output path variables

## 5. Current Verification Scope

Current validation focuses on lightweight engineering checks:

- training script import smoke test
- inference script import smoke test
- shell script syntax check

Full performance and accuracy results should be added after formal end-to-end runs in the target release environment.
