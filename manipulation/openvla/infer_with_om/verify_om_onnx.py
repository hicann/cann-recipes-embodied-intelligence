# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (R) @huawei.com; all rights reserved
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import argparse
import logging
from dataclasses import dataclass
from time import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

import acl
import acllite_utils as utils
import constants as const
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_resource import resource_list

# Import common utilities
from vla_validation_utils import (
    compare_outputs,
    show_compare,
    get_image_size_from_processor,
    make_dummy_image,
    post_process_action,
)


# ==================== Data Classes ====================

@dataclass
class ActionConfig:
    """Container for action-related configuration."""
    action_dim: int
    vocab_size: int
    bin_centers: np.ndarray
    action_norm_stats: Dict


@dataclass
class ModelPaths:
    """Container for model file paths."""
    prefill_model_path: str
    decode_model_path: str


class AclLiteResource:
    """
    ACL runtime resource manager.
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.run_mode = None

    def init(self) -> None:
        """
        Init ACL resources.
        """
        logging.info("init resource stage:")
        _ = acl.init()

        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)

        self.run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)

        logging.info("Init resource success")

    def __del__(self):
        # Keep original behavior
        logging.info("acl resource release all resource")
        resource_list.destroy()
        if self.stream:
            logging.info("acl resource release stream")
            acl.rt.destroy_stream(self.stream)

        if self.context:
            logging.info("acl resource release context")
            acl.rt.destroy_context(self.context)

        logging.info("Reset acl device ", self.device_id)
        acl.rt.reset_device(self.device_id)
        logging.info("Release acl resource success")


class VisionModel:
    """
    Wrapper for vision-related OM models (vision_backbone/projector/embedding).
    """

    def __init__(self, model_path: str, model_width: int, model_height: int):
        self.model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self.device_id = 0
        self._dvpp = None
        self._model = None

    def init(self) -> int:
        """
        Initialize DVPP and load OM model.
        """
        self._dvpp = AclLiteImageProc()
        self._model = AclLiteModel(self.model_path)
        return const.SUCCESS

    @utils.display_time
    def inference(self, input_list):
        """
        Run model inference.

        Args:
            input_list: List of input arrays

        Returns:
            First output array from the model
        """
        outputs = self._model.execute(input_list)
        return outputs[0]


class OpenVLA:
    """
    Wrapper for OpenVLA OM inference (prefill + decode) and post-process actions.
    """

    def __init__(
        self,
        model_paths: ModelPaths,
        vision_backbone: VisionModel,
        projector: VisionModel,
        embedding: VisionModel,
        action_config: ActionConfig,
    ):
        self.model_paths = model_paths
        self.prefill_model_path = model_paths.prefill_model_path
        self.decode_model_path = model_paths.decode_model_path
        self.device_id = 0
        self._dvpp = None

        self.vision_backbone = vision_backbone
        self.projector = projector
        self.embedding = embedding

        self.prefill_model = None
        self.decode_model = None

        self.action_config = action_config
        self.action_dim = action_config.action_dim
        self.vocab_size = action_config.vocab_size
        self.bin_centers = action_config.bin_centers
        self.action_norm_stats = action_config.action_norm_stats

        self.max_new_tokens = action_config.action_dim  # keep original behavior

    def init(self) -> int:
        """
        Initialize DVPP and load OM models.
        """
        self._dvpp = AclLiteImageProc()
        self.prefill_model = AclLiteModel(self.prefill_model_path)
        self.decode_model = AclLiteModel(self.decode_model_path)
        return const.SUCCESS

    @utils.display_time
    def pre_process(self, input_list):
        """
        Pre-process inputs: ensure input_ids ends with pad token (29871).

        Args:
            input_list: Dictionary containing input_ids, pixel_values, attention_mask

        Returns:
            Tuple of (input_ids, pixel_values, attention_mask)
        """
        input_ids = input_list["input_ids"]
        pixel_values = input_list["pixel_values"]
        attention_mask = input_list["attention_mask"]

        if not torch.all(input_ids[:, -1] == 29871):
            pad_token_tensor = torch.tensor(
                [[29871]], dtype=torch.long, device=input_ids.device
            )
            input_ids = torch.cat([input_ids, pad_token_tensor], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
        return input_ids, pixel_values, attention_mask

    @utils.display_time
    def inference(self, input_ids, pixel_values, attention_mask, target_seq_len: int = 288):
        """
        Perform model inference with all models (vision_backbone + projector + embedding + prefill + decode).

        Args:
            input_ids: Input token IDs
            pixel_values: Pixel values
            attention_mask: Attention mask
            target_seq_len: Target sequence length for padding
        """

        # Process vision through vision backbone and projector
        pixel_values = pixel_values.numpy()
        patch_features = self.vision_backbone.inference([pixel_values])
        projected_patch_embeddings = self.projector.inference(patch_features)
        projected_patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        # Get input embeddings from language model
        input_ids = input_ids.numpy()
        input_embeddings = self.embedding.inference([input_ids])
        input_embeddings = input_embeddings[:, : input_ids.shape[1], :]

        # Create multimodal embeddings and attention mask
        multimodal_embeddings = np.concatenate(
            [
                input_embeddings[:, :1, :],
                projected_patch_embeddings,
                input_embeddings[:, 1:, :],
            ],
            axis=1,
        )
        multimodal_attention_mask = torch.cat(
            [
                attention_mask[:, :1],
                projected_patch_attention_mask,
                attention_mask[:, 1:],
            ],
            dim=1,
        )

        # Pad to target sequence length
        current_seq_len = multimodal_embeddings.shape[1]
        padding_len = target_seq_len - current_seq_len
        original_effective_seq_len = current_seq_len

        if padding_len > 0:
            # Get pad embedding
            pad_token_id = 32000
            pad_tensor = torch.tensor(
                [[pad_token_id]], device=attention_mask.device
            )
            pad_emb = self.embedding.inference([pad_tensor.numpy()])
            pad_emb = pad_emb[:, :1, :]

            # Expand pad_emb to match batch size and repeat for padding_len
            pad_embeddings = np.tile(
                pad_emb, (multimodal_embeddings.shape[0], padding_len, 1)
            )
            multimodal_embeddings = np.concatenate(
                [multimodal_embeddings, pad_embeddings], axis=1
            )

            # Create padding for attention mask
            pad_attention_mask = torch.zeros(
                multimodal_attention_mask.shape[0],
                padding_len,
                dtype=multimodal_attention_mask.dtype,
                device=multimodal_attention_mask.device,
            )
            multimodal_attention_mask = torch.cat(
                [multimodal_attention_mask, pad_attention_mask], dim=1
            )

        # Create position IDs
        effective_position_ids = torch.arange(
            original_effective_seq_len,
            dtype=torch.long,
            device=attention_mask.device,
        )
        pad_position_ids = torch.full(
            (padding_len,),
            fill_value=original_effective_seq_len - 1,
            dtype=torch.long,
            device=attention_mask.device,
        )
        sequence_position_ids = torch.cat(
            [effective_position_ids, pad_position_ids], dim=0
        )
        multimodal_position_ids = sequence_position_ids.unsqueeze(0).expand(
            multimodal_embeddings.shape[0], -1
        )

        # Prefill stage
        multimodal_attention_mask = multimodal_attention_mask.numpy()
        multimodal_position_ids = multimodal_position_ids.numpy()
        prefill_input_list = [
            multimodal_attention_mask,
            multimodal_position_ids,
            multimodal_embeddings,
        ]
        prefill_outputs = self.prefill_model.execute(prefill_input_list)

        logits = prefill_outputs[0]
        past_key_values = prefill_outputs[1:]

        # Get the last valid token position and extract logits
        last_valid_token_indices = np.sum(multimodal_attention_mask, axis=1) - 1
        logits_last_valid_token = logits[
            np.arange(logits.shape[0]), last_valid_token_indices, :
        ]
        next_token_id = np.argmax(logits_last_valid_token, axis=-1)[:, None]
        generated_ids = next_token_id

        # Decode loop
        for i in range(self.max_new_tokens - 1):
            # Prepare decoder inputs: use only one valid token (matching export shape)
            decoder_input_ids = next_token_id.astype(np.int64)

            # Update attention mask: set the position for the new token
            multimodal_attention_mask[:, original_effective_seq_len + i - 1] = 1
            # Add one new mask segment for the new token
            new_mask = np.array([[1]], dtype=np.int64)
            dummy_attention_mask = np.concatenate(
                [multimodal_attention_mask, new_mask], axis=1
            )

            # Position ID for the single new token
            new_pos_id = np.array(
                [[original_effective_seq_len + i]], dtype=np.int64
            )

            decoder_inputs = (
                [decoder_input_ids, dummy_attention_mask, new_pos_id]
                + list(past_key_values)
            )
            decoder_outputs = self.decode_model.execute(decoder_inputs)

            logits = decoder_outputs[0]
            past_key_values = decoder_outputs[1:]

            # Post-process past_key_values: update kv cache and truncate
            def replace_kv_index(past_key_values_local, index, seq_len):
                """Replace kv cache at index with value from position seq_len."""
                out = []
                for arr in past_key_values_local:
                    if arr.shape[2] > seq_len:
                        arr = arr.copy()  # Avoid in-place side effects
                        arr[:, :, index, :] = arr[:, :, seq_len, :]
                    out.append(arr)
                return tuple(out)

            # Replace kv cache at original_effective_seq_len + i with value from position target_seq_len
            past_key_values = replace_kv_index(
                past_key_values,
                index=original_effective_seq_len + i,
                seq_len=target_seq_len,
            )

            # Truncate past_key_values to length target_seq_len
            past_key_values = [
                x[:, :, :target_seq_len, :] if x.shape[2] >= target_seq_len else x
                for x in past_key_values
            ]

            # Extract next token: use logits[:, 0, :] (single token output)
            next_token_id = np.argmax(logits[:, 0, :], axis=-1)[:, np.newaxis]
            generated_ids = np.concatenate([generated_ids, next_token_id], axis=1)

        return generated_ids

    @utils.display_time
    def post_process(self, generated_ids):
        """
        Post-process generated token IDs to obtain actions.

        Args:
            generated_ids: Generated token IDs (batch_size, seq_len)

        Returns:
            Post-processed actions
        """
        return post_process_action(
            generated_ids=generated_ids,
            action_dim=self.action_dim,
            vocab_size=self.vocab_size,
            bin_centers=self.bin_centers,
            action_norm_stats=self.action_norm_stats,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenVLA OM inference demo (vision_backbone/projector/embedding + prefill/decode) with Ascend ACL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local OpenVLA HF model directory (e.g., models/openvla-7b-finetuned-libero-object).",
    )
    parser.add_argument(
        "--unnorm-key",
        default="libero_object",
        help="Unnormalize key used by OpenVLA for action stats (e.g., libero_object).",
    )
    parser.add_argument(
        "--prompt",
        default="In: What action should the robot take to pick up the alphabet soup and place it in the basket?\nOut:",
        help="Prompt text for inference.",
    )

    # OM paths
    parser.add_argument("--vision-backbone-om", required=True, help="Path to vision_backbone.om")
    parser.add_argument("--projector-om", required=True, help="Path to projector.om")
    parser.add_argument("--embedding-om", required=True, help="Path to embedding.om")
    parser.add_argument("--prefill-om", required=True, help="Path to prefill OM model (e.g., openvla_prefill.om)")
    parser.add_argument("--decode-om", required=True, help="Path to decode OM model (e.g., openvla_decode.om)")

    # ACL device
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device id")

    # Dummy image controls (keep original behavior: use random dummy image)
    parser.add_argument(
        "--image-width",
        type=int,
        default=0,
        help="Override dummy image width (0 = use processor config)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=0,
        help="Override dummy image height (0 = use processor config)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dummy image (0 = do not set)")

    # Model size hints (not used by OM inference logic, kept for compatibility)
    parser.add_argument("--model-width", type=int, default=224, help="Model input width hint")
    parser.add_argument("--model-height", type=int, default=224, help="Model input height hint")

    # Sequence length
    parser.add_argument(
        "--target-seq-len",
        type=int,
        default=288,
        help="Target sequence length for padding (default: 288)",
    )

    return parser.parse_args()


# _get_image_size_from_processor and _make_dummy_image are now imported from vla_validation_utils


def main() -> None:
    args = _parse_args()

    local_model_path = args.model_path
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Model path not found: {local_model_path}.")

    logging.info(f"Loading model from: {local_model_path}")
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    vla_model = AutoModelForVision2Seq.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    unnorm_key = args.unnorm_key

    # Determine dummy image size
    h0, w0 = get_image_size_from_processor(processor)
    height = args.image_height if args.image_height > 0 else h0
    width = args.image_width if args.image_width > 0 else w0

    dummy_image = make_dummy_image(width=width, height=height, seed=args.seed)

    prompt = args.prompt
    logging.info(f"Using a dummy prompt: '{prompt}'")
    logging.info(f"Using a dummy random image of size: {dummy_image.size}")

    inputs = processor(text=prompt, images=dummy_image, return_tensors="pt").to(dtype=torch.float16)

    # Init ACL resources
    acl_resource = AclLiteResource(device_id=args.device_id)
    acl_resource.init()

    # Load OM models
    vision_backbone = VisionModel(args.vision_backbone_om, args.model_width, args.model_height)
    ret = vision_backbone.init()
    utils.check_ret("vision_backbone.init ", ret)

    projector = VisionModel(args.projector_om, args.model_width, args.model_height)
    ret = projector.init()
    utils.check_ret("projector.init ", ret)

    embedding = VisionModel(args.embedding_om, args.model_width, args.model_height)
    ret = embedding.init()
    utils.check_ret("embedding.init ", ret)

    # Extract action-related metadata from PyTorch model (unchanged)
    action_dim = vla_model.get_action_dim(unnorm_key)
    vocab_size = vla_model.vocab_size
    bin_centers = vla_model.bin_centers
    action_norm_stats = vla_model.get_action_stats(unnorm_key)

    # Create action config
    action_config = ActionConfig(
        action_dim=action_dim,
        vocab_size=vocab_size,
        bin_centers=bin_centers,
        action_norm_stats=action_norm_stats,
    )

    # Create model paths
    model_paths = ModelPaths(
        prefill_model_path=args.prefill_om,
        decode_model_path=args.decode_om,
    )

    # Init OpenVLA OM pipeline
    openvla = OpenVLA(
        model_paths=model_paths,
        vision_backbone=vision_backbone,
        projector=projector,
        embedding=embedding,
        action_config=action_config,
    )
    ret = openvla.init()
    utils.check_ret("openvla.init ", ret)

    input_ids, pixel_values, attention_mask = openvla.pre_process(inputs)
    om_generated_ids = openvla.inference(
        input_ids, pixel_values, attention_mask, target_seq_len=args.target_seq_len
    )
    om_action = openvla.post_process(om_generated_ids)

    # PyTorch reference (unchanged)
    with torch.no_grad():
        original_action = vla_model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    # Convert to numpy for comparison
    original_action_np = original_action.cpu().numpy() if torch.is_tensor(original_action) else original_action
    om_action_np = om_action if isinstance(om_action, np.ndarray) else np.array(om_action)

    show_compare("actions", original_action_np, om_action_np)
    from vla_validation_utils import ComparisonConfig
    comparison_config = ComparisonConfig(rtol=1e-3, atol=1e-3, mean_diff_threshold=None)
    compare_outputs(original_action_np, om_action_np, name="action", config=comparison_config)


if __name__ == "__main__":
    main()
