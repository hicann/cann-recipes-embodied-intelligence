# Copyright (c) 2026 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 Shanghai Jiao Tong University
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
import logging
import os
import sys
from pathlib import Path
import time

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torch.onnx

from lerobot.policies.pi05.modeling_pi05_part1 import PI05Policy

logging.basicConfig(level=logging.INFO)

# Configuration
PRETRAINED_POLICY_PATH = "models/pi05-libero"
OUTPUT_DIR = Path("output/onnx_models/pi05")
ONNX_FILENAME = "pi05-part1.onnx"
PREFIX_EMBS_PATH = Path("prefix_embs.pt")
RUNTIME_SAVE_DIR = Path("runtime_save")
PAST_KEY_VALUE_SAVE_PATH = RUNTIME_SAVE_DIR / "past_kv_tensor.pth"
PREFIX_PAD_MASK_SAVE_PATH = RUNTIME_SAVE_DIR / "prefix_pad_masks.pth"

# DEVICE = torch.device("cuda:0")  if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
OPSET_VERSION = 14
BATCH_SIZE = 1
LANG_TOKENS_LEN = 200


def prepare_base_tensors_mock(device: torch.device):
    """Prepare example state/image/lang tensors used as model inputs."""
    state = torch.zeros((1, 8), dtype=torch.float32, device=device)               # (batch, state_dim)
    image = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device) / 255  # normalized
    image2 = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device) / 255  # normalized
    lang_tokens = torch.zeros((BATCH_SIZE, LANG_TOKENS_LEN), dtype=torch.long, device=device)
    lang_masks = torch.zeros((BATCH_SIZE, LANG_TOKENS_LEN), dtype=torch.bool, device=device)
    prefix_att_masks = torch.zeros((BATCH_SIZE, 872), dtype=torch.bool, device=device)
    prefix_att_2d_masks_4d = torch.zeros((BATCH_SIZE, 1, 872, 872), dtype=torch.float32, device=device)
    

    return {
        "observation.state": state,
        "observation.images.image": image,
        "observation.images.image2": image2,
        "observation.language.tokens": lang_tokens,
        "observation.language.attention_mask": lang_masks,
        "prefix_att_masks": prefix_att_masks,
        "prefix_att_2d_masks_4d": prefix_att_2d_masks_4d,
    }

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


def prepare_base_tensors(device: torch.device):
    """Prepare example state/image/lang tensors used as model inputs."""
    data = torch.load("input_data/start_obs_0.pt", map_location=device) 

    image = data['observation.images.image']          # [1, 3, 256, 256]
    image2 = data['observation.images.image2']         # [1, 3, 256, 256]
    state = data['observation.state']                # [1, 8]
    lang_tokens = data['observation.language.tokens']  # 原始文本指令
    lang_masks = data['observation.language.attention_mask']  # [1, 77]
    prefix_att_masks = torch.zeros((BATCH_SIZE, 968), dtype=torch.bool, device=device)
    
    num_images = 3 # need to match the number of images used in training
    image_tokens_per_img = 256
    
    def _prepare_attention_masks_4d(prefix_att_masks):
        image_masks = torch.full(
            (1, num_images * image_tokens_per_img), 
            1, 
            dtype=lang_masks.dtype, 
            device=lang_masks.device
        )
        start_idx = 2 * image_tokens_per_img
        end_idx = 3 * image_tokens_per_img
        image_masks[:, start_idx:end_idx] = 0
        prefix_pad_masks = torch.cat([image_masks, lang_masks], dim=1)

        def make_att_2d_masks(pad_masks, att_masks): 
            if att_masks.ndim != 2:
                raise ValueError(att_masks.ndim)
            if pad_masks.ndim != 2:
                raise ValueError(pad_masks.ndim)

            att_masks = att_masks.to(torch.int32)
            cumsum = torch.cumsum(att_masks, dim=1)
            cumsum = cumsum.to(torch.bool)
            att_masks = att_masks.to(torch.bool)
            att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
            pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
            return att_2d_masks & pad_2d_masks
        
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks = prefix_att_2d_masks[:, None, :, :]
        prefix_att_2d_masks = prefix_att_2d_masks.to(dtype=torch.bool)
        return torch.where(prefix_att_2d_masks, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        
    prefix_att_2d_masks_4d = _prepare_attention_masks_4d(prefix_att_masks)
    
    
    return {
        "observation.state": state,
        "observation.images.image": image,
        "observation.images.image2": image2,
        "observation.language.tokens": lang_tokens,
        "observation.language.attention_mask": lang_masks,
        "prefix_att_masks": prefix_att_masks,
        "prefix_att_2d_masks_4d": prefix_att_2d_masks_4d,
    }


def parse_args():
    """Parse optional CLI overrides for output directory and filename."""
    parser = argparse.ArgumentParser(description="Export PI05 part1 to ONNX")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to write ONNX model (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--onnx-filename",
        type=str,
        default=ONNX_FILENAME,
        help=f"ONNX filename (default: {ONNX_FILENAME})",
    ) 
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=PRETRAINED_POLICY_PATH,
        help=f"Directory to write ONNX model (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


class ONNXWrapper(torch.nn.Module):
    """Wrapper to expose the policy.select_action(...) as a single forward op for ONNX export."""
    def __init__(self, policy: PI05Policy, example_observation: dict, device: torch.device):
        super().__init__()
        self.policy = policy.to(device)
        self.device = device
        self._keys = list(example_observation.keys())

    def forward(self, *args):
        # Map positional args back to the expected input dict and coerce dtypes
        if len(args) != len(self._keys):
            raise ValueError(f"Expected {len(self._keys)} inputs, got {len(args)}")
        input_dict = {}
        for key, tensor in zip(self._keys, args):
            if key == "observation.language.tokens":
                input_dict[key] = tensor.to(torch.long)
            elif key == "observation.language.attention_mask":
                input_dict[key] = tensor.to(torch.bool)
            elif key == "prefix_att_masks":
                input_dict[key] = tensor.to(torch.bool)
            else:
                input_dict[key] = tensor.to(torch.float32)

        # Use eval and no_grad for deterministic outputs
        self.policy.eval()
        with torch.no_grad():
            past_kv_tensor, prefix_pad_masks = self.policy.select_action(input_dict)
            # prefix_embs = self.policy.select_action(input_dict)
        
        return past_kv_tensor, prefix_pad_masks


def export_onnx(wrapper: ONNXWrapper, observation: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_keys = list(observation.keys())
    observation_values = [observation[k] for k in dummy_keys]

    torch.onnx.export(
        wrapper,
        tuple(observation_values),
        str(output_path),
        opset_version=OPSET_VERSION,
        input_names=dummy_keys,
        output_names=["past_kv_tensor", "prefix_pad_masks"],
        do_constant_folding=True,
        verbose=False,
        dynamo=True,
    )
    logging.info(f"Exported ONNX model to {output_path}")


def verify_onnx(
        onnx_path: Path,
        wrapper: ONNXWrapper,
        observation: dict,
        pytorch_past_kv: torch.Tensor,
        pytorch_prefix_mask: torch.Tensor
    ):
    # PyTorch outputs
    dummy_keys = list(observation.keys())
    observation_values = [observation[k] for k in dummy_keys]

    # Save runtime tensors (for debugging)
    RUNTIME_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(pytorch_past_kv, PAST_KEY_VALUE_SAVE_PATH)
    torch.save(pytorch_prefix_mask, PREFIX_PAD_MASK_SAVE_PATH)

    # Run ONNXRuntime
    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    valid_input_names = {inp.name for inp in sess.get_inputs()}

    onnx_inputs = {}
    for name, val in zip(dummy_keys, observation_values):
        if name not in valid_input_names:
            continue
        if isinstance(val, torch.Tensor):
            arr = val.cpu().numpy()
            if val.dtype == torch.bool:
                arr = arr.astype(bool)
            onnx_inputs[name] = arr
        else:
            onnx_inputs[name] = val

    start = time.time()
    onnx_outputs = sess.run(None, onnx_inputs)
    logging.info(f"pytorch_past_kv: {pytorch_past_kv}")
    logging.info(f"onnx_outputs: {onnx_outputs}")
    elapsed = time.time() - start

    onnx_past_kv = np.array(onnx_outputs[0])
    onnx_prefix_mask = np.array(onnx_outputs[1])

    pyt_np = pytorch_past_kv.cpu().numpy()
    pref_np = pytorch_prefix_mask.cpu().numpy()

    logging.info(f"ONNX inference time: {elapsed:.4f}s")
    logging.info("=== Compare past_kv_tensor ===")
    logging.info(f"Shapes: {pyt_np.shape}, {onnx_past_kv.shape}")
    logging.info(f"Max diff: {float(np.abs(pyt_np - onnx_past_kv).max())}")
    logging.info(f"Mean diff: {float(np.abs(pyt_np - onnx_past_kv).mean())}")
    

    try:
        torch_tensor = torch.from_numpy(pyt_np).float()
        onnx_tensor = torch.from_numpy(onnx_past_kv).float()

        if onnx_tensor.ndim < 2 or torch_tensor.ndim < 2:
            logging.warning("警告: past_kv_tensor 维度 < 2，无法计算逐行余弦相似度。")
        elif onnx_tensor.shape != torch_tensor.shape:
            raise ValueError(
                f"past_kv_tensor shape mismatch for cosine similarity: "
                f"onnx={tuple(onnx_tensor.shape)}, pytorch={tuple(torch_tensor.shape)}"
            )
        else:
            cos_sim = F.cosine_similarity(onnx_tensor, torch_tensor, dim=-1, eps=1e-8)
            mean_cos = cos_sim.mean().item()
            min_cos = cos_sim.min().item()
            max_cos = cos_sim.max().item()
            cos_std = cos_sim.std().item()

            logging.info("Cosine Similarity (ONNX vs PyTorch, past_kv_tensor):")
            logging.info(f"  Mean: {mean_cos:.6f}, Min: {min_cos:.6f}, Max: {max_cos:.6f}, Std: {cos_std:.6f}")
    except (TypeError, ValueError) as err:
        logging.warning(
            "跳过 past_kv_tensor 余弦相似度计算: "
            "err_type=%s, err=%s, pyt_shape=%s, onnx_shape=%s, pyt_dtype=%s, onnx_dtype=%s",
            type(err).__name__,
            err,
            tuple(pyt_np.shape),
            tuple(onnx_past_kv.shape),
            pyt_np.dtype,
            onnx_past_kv.dtype,
        )
    except RuntimeError as err:
        logging.exception(
            "计算 past_kv_tensor 余弦相似度时发生运行时错误: "
            "err_type=%s, err=%s, pyt_shape=%s, onnx_shape=%s, pyt_dtype=%s, onnx_dtype=%s. "
            "请检查张量设备/类型兼容性与输入有效性。",
            type(err).__name__,
            err,
            tuple(pyt_np.shape),
            tuple(onnx_past_kv.shape),
            pyt_np.dtype,
            onnx_past_kv.dtype,
        )
    
    # Avoid boolean subtract error: convert boolean masks to integer before numeric diff,
    # or use logical comparison to report mismatches.
    if np.issubdtype(pref_np.dtype, np.bool_) or np.issubdtype(onnx_prefix_mask.dtype, np.bool_):
        pref_num = pref_np.astype(np.int8)
        onnx_pref_num = onnx_prefix_mask.astype(np.int8)
        logging.info(f"Max diff: {float(np.abs(pref_num - onnx_pref_num).max())}")
        logging.info(f"Num mismatches: {int((pref_num != onnx_pref_num).sum())}")
    else:
        logging.info(f"Max diff: {float(np.abs(pref_np - onnx_prefix_mask).max())}")
        logging.info(f"Mean diff: {float(np.abs(pref_np - onnx_prefix_mask).mean())}")


def main():
    args = parse_args()
    output_dir = args.output_dir
    onnx_filename = args.onnx_filename
    pretrain_path = args.pretrain_path
    # Load policy
    policy = PI05Policy.from_pretrained(pretrain_path, local_files_only=True, strict=False)
    policy.to(DEVICE)
    policy.model = policy.model.half()
    policy.eval()

    # policy.select_action(prepare_base_tensors(DEVICE))  # check
    
    # Load prefix embeddings if needed (optional)
    if PREFIX_EMBS_PATH.exists():
        try:
            _ = torch.load(PREFIX_EMBS_PATH, map_location=DEVICE)
        except Exception:
            logging.warning("Warning: failed to load prefix_embs.pt; continuing without it.")
            
    observation = prepare_base_tensors(DEVICE)
    dummy_keys = list(observation.keys())
    observation_values = [observation[k] for k in dummy_keys]

    wrapper = ONNXWrapper(policy, observation, DEVICE)
    with torch.no_grad():
        pytorch_past_kv, pytorch_prefix_mask = wrapper(*observation_values)

    onnx_path = output_dir / onnx_filename
    export_onnx(wrapper, observation, onnx_path)
    
    logging.info("Verifying ONNX output vs PyTorch output...")
    verify_onnx(onnx_path, wrapper, observation, pytorch_past_kv, pytorch_prefix_mask)


if __name__ == "__main__":
    main()
