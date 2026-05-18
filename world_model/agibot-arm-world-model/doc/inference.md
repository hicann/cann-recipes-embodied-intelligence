# Inference

This sample performs chunked autoregressive inference for AgiBotWorld test episodes using text conditioning, reference-image conditioning, and action trajectory maps.

## Summary

- Device: Ascend NPU
- Default sample topology: single-card inference
- Main inputs:
  - checkpoint
  - pretrained model directory
  - test episode directory
  - text prompt
  - first-frame reference image
  - action-derived trajectory maps

## Environment Preparation

Before running inference, complete the environment setup in the top-level [README.md](../README.md).

Recommended workspace layout:

```text
workspace/
  DiffSynth-Studio/
  agibot-arm-world-model/
```

Run this inference sample from:

```text
workspace/agibot-arm-world-model/infer_with_torch/src
```

The sample does not need to be copied into the `DiffSynth-Studio` source tree as long as `pip install -e .` has already been executed in `DiffSynth-Studio`.

## Input Preparation

Prepare:

- `CHECKPOINT_PATH`: fine-tuned checkpoint path
- `MODEL_DIR`: `Wan2.1-Fun-V1.1-1.3B-Control` model directory
- `TEST_ROOT`: test dataset root

Expected episode-level test inputs include files such as:

```text
task_id/episode_id/
  frame.png
  text.txt
  proprio_stats.h5
  head_intrinsic_params.json
  head_extrinsic_params_aligned.json
```

## Run Steps

1. Enter the source directory.
2. Set the checkpoint, model, test root, and output directory variables.
3. Run the launcher script.

Example:

```bash
cd infer_with_torch/src
CHECKPOINT_PATH=/path/to/step-18000.safetensors \
MODEL_DIR=/path/to/Wan2.1-Fun-V1.1-1.3B-Control \
TEST_ROOT=/path/to/test/info_dataset \
OUTPUT_DIR=./outputs/inference \
CHUNK_SIZE=49 \
CFG_SCALE=3.0 \
bash run_infer.sh
```

## Key Parameters

- `CHECKPOINT_PATH`: fine-tuned checkpoint
- `MODEL_DIR`: pretrained model directory
- `TEST_ROOT`: test dataset root
- `OUTPUT_DIR`: inference output directory
- `CHUNK_SIZE`: chunk length for autoregressive generation
- `CFG_SCALE`: guidance scale

## Output

- Frame results under `OUTPUT_DIR/<task>/<episode>/<rollout>/video/`
- Optional mp4 results under `OUTPUT_DIR/mp4/`
- Optional legacy metadata file in `OUTPUT_DIR/meta_info.txt` when explicitly enabled


## Dependency

- Install the official `modelscope/DiffSynth-Studio` repository separately and make sure `python` can import `diffsynth`.
- See [../doc/optimization.md](../doc/optimization.md) for design rationale and optimization notes.
