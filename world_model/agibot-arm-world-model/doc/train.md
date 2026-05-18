# Training

This training sample fine-tunes `Wan2.1-Fun-V1.1-1.3B-Control` on the AgiBotWorld clip dataset with text conditioning and 49-frame samples.

## Summary

- Device: Ascend NPU
- Default sample topology: 8 NPUs
- Main inputs:
  - clip frames
  - text prompt
  - action-derived trajectory maps
  - reference image
- Entry files:
  - `src/train_fun_control_1_3b_text_49frames.py`
  - `src/run_train.sh`

## Environment Preparation

Before running training, complete the environment setup in the top-level [README.md](../README.md).

Recommended workspace layout:

```text
workspace/
  DiffSynth-Studio/
  agibot-arm-world-model/
```

Run this training sample from:

```text
workspace/agibot-arm-world-model/train/src
```

The sample does not need to be copied into the `DiffSynth-Studio` source tree as long as `pip install -e .` has already been executed in `DiffSynth-Studio`.

## Model Preparation

Prepare the `Wan2.1-Fun-V1.1-1.3B-Control` model directory and make sure the following files are available under `MODEL_DIR`:

```text
diffusion_pytorch_model.safetensors
models_t5_umt5-xxl-enc-bf16.pth
Wan2.1_VAE.pth
models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
```

## Data Preparation

Expected data layout under `DATA_ROOT`:

```text
task_id/episode_id/clip_idx/
  text.txt
  annotation.json
  proprio_stats.h5
  head_intrinsic_params.json
  head_extrinsic_params_aligned.json
  videos/
```

Optional split file:

- `split.json`
- or a custom split file passed through `SPLIT_FILE`

## Run Steps

1. Enter the source directory.
2. Set `MODEL_DIR`, `DATA_ROOT`, `SPLIT_FILE`, and `OUTPUT_DIR`.
3. Confirm that `NUM_PROCESSES` matches the accelerate config.
4. Launch the training script.

Example:

```bash
cd train/src
MODEL_DIR=/path/to/Wan2.1-Fun-V1.1-1.3B-Control \
DATA_ROOT=/path/to/agibot_world_dataset \
SPLIT_FILE=/path/to/split.json \
OUTPUT_DIR=./outputs/agibot_arm_world_model \
NUM_PROCESSES=8 \
bash run_train.sh
```

## Key Parameters

- `MODEL_DIR`: pretrained model directory
- `DATA_ROOT`: AgiBotWorld clip dataset root
- `SPLIT_FILE`: split file path
- `OUTPUT_DIR`: checkpoint output directory
- `NUM_PROCESSES`: number of NPUs used by `accelerate launch`

## Output

Training outputs are written to `OUTPUT_DIR`, typically including saved checkpoints at configured step intervals.

## Notes

- `run_train.sh` defaults to 8 Ascend NPUs.
- Install the official `modelscope/DiffSynth-Studio` repository separately and make sure `python` can import `diffsynth`.
- The launch script and accelerate config must use the same process count.
- See [../doc/optimization.md](../doc/optimization.md) for design rationale and optimization notes.
