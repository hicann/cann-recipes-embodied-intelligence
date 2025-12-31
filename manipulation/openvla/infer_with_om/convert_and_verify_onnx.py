# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
OpenVLA ONNX Model Converter and Validator

This script converts PyTorch OpenVLA models to ONNX format and validates
the conversion by comparing outputs between PyTorch and ONNX models.
"""

import argparse
import gc
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Suppress ONNX Runtime warnings
ort.set_default_logger_severity(3)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# Import common utilities
from vla_validation_utils import (
    compare_outputs,
    print_diff_only,
    get_image_size_from_processor,
    make_dummy_image,
    post_process_action,
    ComparisonConfig,
)


# ==================== Data Classes ====================

@dataclass
class ActionPostProcessConfig:
    """Container for action post-processing configuration."""
    action_dim: int
    vocab_size: int
    bin_centers: np.ndarray
    action_norm_stats: Dict


@dataclass
class ModelInputs:
    """Container for model input tensors."""
    input_ids: torch.LongTensor
    pixel_values: torch.FloatTensor
    attention_mask: torch.LongTensor


@dataclass
class ExportConfig:
    """Container for export and validation configuration."""
    vision_export_dir: str
    llama_prefill_export_dir: str
    llama_decoder_export_dir: str
    unnorm_key: str
    target_seq_len: int = 288
    provider: str = "CPUExecutionProvider"


@dataclass
class MultimodalInputs:
    """Container for multimodal model inputs."""
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor


@dataclass
class MultimodalInputsNumpy:
    """Container for multimodal model inputs (numpy arrays)."""
    input_ids: np.ndarray
    pixel_values: np.ndarray
    attention_mask: np.ndarray


@dataclass
class MultimodalConfig:
    """Container for multimodal input preparation configuration."""
    pad_token_id: int = 32000
    target_seq_len: int = 288


@dataclass
class OnnxSessions:
    """Container for ONNX Runtime sessions."""
    vision: ort.InferenceSession
    projector: ort.InferenceSession
    embedding: ort.InferenceSession


@dataclass
class OnnxModelPaths:
    """Container for ONNX model file paths."""
    vision_backbone_path: str
    projector_path: str
    embedding_path: str
    prefill_path: str
    decoder_path: str


@dataclass
class OnnxGeneratorConfig:
    """Container for ONNX generator configuration."""
    model_paths: OnnxModelPaths
    num_hidden_layers: int
    provider: str = "CPUExecutionProvider"


@dataclass
class GenerationConfig:
    """Container for token generation configuration."""
    max_new_tokens: int
    pad_token_id: int = 32000
    target_seq_len: int = 288


@dataclass
class ValidationConfig:
    """Container for validation configuration."""
    unnorm_key: str
    target_seq_len: int = 288
    provider: str = "CPUExecutionProvider"


# ==================== Wrapper Classes ====================

class LlamaPrefillWrapper(nn.Module):
    """Wrapper for LLaMA prefill model to enable ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        multimodal_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        multimodal_embeddings: torch.Tensor,
    ):
        """Forward pass for prefill stage."""
        outputs = self.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.logits, *outputs.past_key_values


class LlamaDecoderWrapper(nn.Module):
    """Wrapper for LLaMA decoder model to enable ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        multimodal_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *past_key_values,
    ):
        """Forward pass for decoder stage."""
        num_layers = self.model.config.num_hidden_layers
        past = tuple(
            (past_key_values[i * 2], past_key_values[i * 2 + 1])
            for i in range(num_layers)
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            inputs_embeds=None,
            labels=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        # Flatten past_key_values for ONNX export
        present = []
        for k, v in outputs.past_key_values:
            present.extend([k, v])
        return (outputs.logits, *present)


# ==================== Helper Functions ====================

def prepare_multimodal_inputs_torch(
    model,
    inputs: MultimodalInputs,
    config: Optional[MultimodalConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Prepare multimodal embeddings, attention mask, and position IDs for PyTorch models.

    Args:
        model: The VLA model
        inputs: Multimodal model inputs
        config: Multimodal configuration (optional, uses defaults if None)

    Returns:
        Tuple of (multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids, original_effective_seq_len)
    """
    if config is None:
        config = MultimodalConfig()
    
    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values
    attention_mask = inputs.attention_mask
    # Process through vision backbone and projector
    patch_features = model.vision_backbone(pixel_values)
    projected_patch_embeddings = model.projector(patch_features)
    projected_patch_attention_mask = torch.full(
        (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
        fill_value=1,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    # Get input embeddings
    input_embeddings = model.get_input_embeddings()(input_ids)

    # Create multimodal embeddings and attention mask
    multimodal_embeddings = torch.cat(
        [
            input_embeddings[:, :1, :],
            projected_patch_embeddings,
            input_embeddings[:, 1:, :],
        ],
        dim=1,
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
    padding_len = config.target_seq_len - current_seq_len
    original_effective_seq_len = current_seq_len

    if padding_len > 0:
        pad_emb = model.get_input_embeddings()(
            torch.tensor(
                [[config.pad_token_id]], device=multimodal_embeddings.device
            )
        ).squeeze(0)
        embedding_dim = multimodal_embeddings.shape[2]
        pad_embeddings = pad_emb.unsqueeze(0).expand(
            multimodal_embeddings.shape[0], padding_len, embedding_dim
        )
        multimodal_embeddings = torch.cat([multimodal_embeddings, pad_embeddings], dim=1)

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
        device=multimodal_embeddings.device,
    )
    # Pad position IDs use original_effective_seq_len - 1 (not 0)
    pad_position_ids = torch.full(
        (padding_len,),
        fill_value=original_effective_seq_len - 1,
        dtype=torch.long,
        device=multimodal_embeddings.device,
    )
    sequence_position_ids = torch.cat([effective_position_ids, pad_position_ids], dim=0)
    multimodal_position_ids = sequence_position_ids.unsqueeze(0).expand(
        multimodal_embeddings.shape[0], -1
    )

    return (
        multimodal_embeddings,
        multimodal_attention_mask,
        multimodal_position_ids,
        original_effective_seq_len,
    )


def prepare_multimodal_inputs_numpy(
    sessions: OnnxSessions,
    inputs: MultimodalInputsNumpy,
    config: Optional[MultimodalConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Prepare multimodal embeddings, attention mask, and position IDs using ONNX models.

    Args:
        sessions: ONNX Runtime sessions
        inputs: Multimodal model inputs (numpy arrays)
        config: Multimodal configuration (optional, uses defaults if None)

    Returns:
        Tuple of (multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids, original_effective_seq_len)
    """
    if config is None:
        config = MultimodalConfig()
    
    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values
    attention_mask = inputs.attention_mask
    # Step 1: Process vision through ONNX models
    vision_inputs = {"pixel_values": pixel_values}
    patch_features = sessions.vision.run(None, vision_inputs)[0]

    projector_inputs = {"patch_features": patch_features}
    projected_patch_embeddings = sessions.projector.run(None, projector_inputs)[0]

    # Step 2: Get input embeddings
    embedding_inputs = {"input_ids": input_ids}
    input_embeddings = sessions.embedding.run(None, embedding_inputs)[0]

    # Step 3: Create multimodal embeddings
    projected_patch_attention_mask = np.ones(
        (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
        dtype=attention_mask.dtype,
    )
    multimodal_embeddings = np.concatenate(
        [
            input_embeddings[:, :1, :],
            projected_patch_embeddings,
            input_embeddings[:, 1:, :],
        ],
        axis=1,
    )
    multimodal_attention_mask = np.concatenate(
        [
            attention_mask[:, :1],
            projected_patch_attention_mask,
            attention_mask[:, 1:],
        ],
        axis=1,
    )

    # Step 4: Pad to target sequence length
    current_seq_len = multimodal_embeddings.shape[1]
    padding_len = config.target_seq_len - current_seq_len
    original_effective_seq_len = current_seq_len

    if padding_len > 0:
        # Get pad embedding
        pad_emb_inputs = {
            "input_ids": np.array([[config.pad_token_id]], dtype=np.int64)
        }
        pad_emb_output = sessions.embedding.run(None, pad_emb_inputs)[0]
        # pad_emb_output shape: (1, 1, embedding_dim)
        # Squeeze batch dimension to get (1, embedding_dim)
        pad_emb = pad_emb_output[0]  # Shape: (1, embedding_dim)
        embedding_dim = multimodal_embeddings.shape[2]
        # Expand to (batch_size, padding_len, embedding_dim)
        pad_embeddings = np.tile(
            pad_emb[np.newaxis, :],  # (1, embedding_dim) -> (1, 1, embedding_dim)
            (multimodal_embeddings.shape[0], padding_len, 1),
        )
        multimodal_embeddings = np.concatenate(
            [multimodal_embeddings, pad_embeddings], axis=1
        )

        pad_attention_mask = np.zeros(
            (multimodal_attention_mask.shape[0], padding_len),
            dtype=multimodal_attention_mask.dtype,
        )
        multimodal_attention_mask = np.concatenate(
            [multimodal_attention_mask, pad_attention_mask], axis=1
        )

    # Step 5: Create position IDs
    effective_position_ids = np.arange(
        original_effective_seq_len, dtype=np.int64
    )
    # Pad position IDs use original_effective_seq_len - 1 (not 0)
    pad_position_ids = np.full(
        padding_len, fill_value=original_effective_seq_len - 1, dtype=np.int64
    )
    sequence_position_ids = np.concatenate([effective_position_ids, pad_position_ids], axis=0)
    multimodal_position_ids = np.tile(
        sequence_position_ids[np.newaxis, :],
        (multimodal_embeddings.shape[0], 1),
    )

    return (
        multimodal_embeddings,
        multimodal_attention_mask,
        multimodal_position_ids,
        original_effective_seq_len,
    )


# ==================== Export Functions ====================

def export_vision_models(
    model,
    export_dir: str,
    dummy_inputs: Dict,
) -> Dict[str, str]:
    """
    Export vision-related models (vision_backbone, projector, embedding).

    Args:
        model: The VLA model
        export_dir: Directory to save ONNX models
        dummy_inputs: Dummy inputs dictionary

    Returns:
        Dictionary mapping model names to their ONNX file paths
    """
    logging.info("\n" + "=" * 60)
    logging.info("Exporting Vision Models")
    logging.info("=" * 60)

    os.makedirs(export_dir, exist_ok=True)
    model_paths = {}

    # 1. Export Vision Backbone
    logging.info("\n[1/3] Exporting Vision Backbone...")
    vision_backbone_model = model.vision_backbone.eval()
    dummy_pixel_values = dummy_inputs["pixel_values"]

    vision_onnx_path = os.path.join(export_dir, "vision_backbone.onnx")
    with torch.no_grad():
        torch.onnx.export(
            vision_backbone_model,
            (dummy_pixel_values,),
            vision_onnx_path,
            input_names=["pixel_values"],
            output_names=["patch_features"],
            opset_version=14,
            export_params=True,
        )
    logging.info(f"✓ Vision backbone saved to: {vision_onnx_path}")
    model_paths["vision_backbone"] = vision_onnx_path

    # 2. Export Projector
    logging.info("\n[2/3] Exporting Projector...")
    patch_features = model.vision_backbone(dummy_pixel_values)
    projector_model = model.projector.eval()

    projector_onnx_path = os.path.join(export_dir, "projector.onnx")
    with torch.no_grad():
        torch.onnx.export(
            projector_model,
            (patch_features,),
            projector_onnx_path,
            input_names=["patch_features"],
            output_names=["projected_patch_embeddings"],
            opset_version=14,
            export_params=True,
        )
    logging.info(f"✓ Projector saved to: {projector_onnx_path}")
    model_paths["projector"] = projector_onnx_path

    # 3. Export Embedding
    logging.info("\n[3/3] Exporting Embedding...")
    dummy_input_ids = dummy_inputs["input_ids"]
    embedding_model = model.get_input_embeddings().eval()

    embedding_onnx_path = os.path.join(export_dir, "embedding.onnx")
    dynamic_axes = {"input_ids": {1: "seqlen"}}
    with torch.no_grad():
        torch.onnx.export(
            embedding_model,
            (dummy_input_ids,),
            embedding_onnx_path,
            input_names=["input_ids"],
            output_names=["input_embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            export_params=True,
        )
    logging.info(f"✓ Embedding saved to: {embedding_onnx_path}")
    model_paths["embedding"] = embedding_onnx_path

    return model_paths


def export_llama_models(
    model,
    prefill_export_dir: str,
    decoder_export_dir: str,
    dummy_inputs: Dict,
    target_seq_len: int = 288,
) -> Dict[str, str]:
    """
    Export LLaMA models (prefill and decoder).

    Args:
        model: The VLA model
        prefill_export_dir: Directory to save prefill ONNX model
        decoder_export_dir: Directory to save decoder ONNX model
        dummy_inputs: Dummy inputs dictionary
        target_seq_len: Target sequence length for padding

    Returns:
        Dictionary mapping model names to their ONNX file paths
    """
    logging.info("\n" + "=" * 60)
    logging.info("Exporting LLaMA Models")
    logging.info("=" * 60)

    os.makedirs(prefill_export_dir, exist_ok=True)
    os.makedirs(decoder_export_dir, exist_ok=True)
    model_paths = {}

    device = torch.device("cpu")
    model.to(device).eval()
    llama_model = model.language_model.eval()

    # Get model configuration
    language_config = model.language_model.config
    main_config = model.config

    if not hasattr(main_config, "image_sizes") or not main_config.image_sizes:
        raise AttributeError(
            "The main model config does not have a valid 'image_sizes' attribute."
        )

    num_hidden_layers = language_config.num_hidden_layers
    hidden_size = language_config.hidden_size
    num_attention_heads = language_config.num_attention_heads
    num_key_value_heads = getattr(
        language_config, "num_key_value_heads", num_attention_heads
    )
    head_dim = hidden_size // num_attention_heads
    vocab_size = language_config.vocab_size

    logging.info("\n--- Model Configuration ---")
    logging.info(f"Num Hidden Layers: {num_hidden_layers}")
    logging.info(f"Num Key/Value Heads: {num_key_value_heads}")
    logging.info(f"Head Dimension: {head_dim}")
    logging.info(f"Vocab Size: {vocab_size}")
    logging.info("---------------------------\n")

    # Prepare inputs
    dummy_input_ids = dummy_inputs["input_ids"]
    dummy_pixel_values = dummy_inputs["pixel_values"]
    dummy_attention_mask = dummy_inputs["attention_mask"]

    # Prepare multimodal inputs using helper function
    multimodal_inputs = MultimodalInputs(
        input_ids=dummy_input_ids,
        pixel_values=dummy_pixel_values,
        attention_mask=dummy_attention_mask,
    )
    multimodal_config = MultimodalConfig(target_seq_len=target_seq_len)
    (
        multimodal_embeddings,
        multimodal_attention_mask,
        multimodal_position_ids,
        original_effective_seq_len,
    ) = prepare_multimodal_inputs_torch(
        model=model,
        inputs=multimodal_inputs,
        config=multimodal_config,
    )

    # 1. Export Prefill Model
    logging.info("\n[1/2] Exporting Prefill Model...")
    prefill_wrapper = LlamaPrefillWrapper(llama_model)
    prefill_onnx_path = os.path.join(prefill_export_dir, "vla_prefill.onnx")

    prefill_input_names = [
        "multimodal_attention_mask",
        "position_ids",
        "multimodal_embeddings",
    ]
    prefill_output_names = ["logits"] + [
        f"present_{i}_{kv}"
        for i in range(num_hidden_layers)
        for kv in ["key", "value"]
    ]

    with torch.no_grad():
        torch.onnx.export(
            prefill_wrapper,
            (multimodal_attention_mask, multimodal_position_ids, multimodal_embeddings),
            prefill_onnx_path,
            input_names=prefill_input_names,
            output_names=prefill_output_names,
            opset_version=14,
            export_params=True,
        )
    logging.info(f"✓ Prefill model saved to: {prefill_onnx_path}")
    model_paths["prefill"] = prefill_onnx_path

    # 2. Export Decoder Model
    logging.info("\n[2/2] Exporting Decoder Model...")
    decoder_wrapper = LlamaDecoderWrapper(llama_model)
    decoder_onnx_path = os.path.join(decoder_export_dir, "vla_decoder.onnx")

    dummy_batch_size = 1
    # dummy_cache_len should equal target_seq_len because prefill outputs
    # past_key_values with sequence length equal to target_seq_len
    dummy_cache_len = target_seq_len

    # Use only one valid token for decoder input 
    dummy_decoder_input_ids = torch.randint(
        0, vocab_size, (dummy_batch_size, 1), dtype=torch.long, device=device
    )
    # Add one new mask segment for the new token
    new_mask_segment = torch.tensor(
        [[1]], dtype=torch.long, device=device
    )
    dummy_attention_mask_decoder = torch.cat(
        (multimodal_attention_mask, new_mask_segment), dim=1
    )
    # Position ID for the single new token
    new_pos_id = torch.tensor(
        [[original_effective_seq_len]],
        dtype=multimodal_position_ids.dtype,
        device=device,
    )

    dummy_past_key_values_flat = [
        torch.randn(
            dummy_batch_size,
            num_key_value_heads,
            dummy_cache_len,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
        for _ in range(num_hidden_layers * 2)
    ]

    decoder_input_names = (
        ["input_ids", "multimodal_attention_mask", "position_ids"]
        + [
            f"past_{i}_{kv}"
            for i in range(num_hidden_layers)
            for kv in ["key", "value"]
        ]
    )
    decoder_output_names = ["logits"] + [
        f"present_{i}_{kv}"
        for i in range(num_hidden_layers)
        for kv in ["key", "value"]
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        with torch.no_grad():
            torch.onnx.export(
                decoder_wrapper,
                (
                    dummy_decoder_input_ids,
                    dummy_attention_mask_decoder,
                    new_pos_id,
                    *dummy_past_key_values_flat,
                ),
                decoder_onnx_path,
                input_names=decoder_input_names,
                output_names=decoder_output_names,
                opset_version=14,
                export_params=True,
            )
    logging.info(f"✓ Decoder model saved to: {decoder_onnx_path}")
    model_paths["decoder"] = decoder_onnx_path

    return model_paths


# ==================== Validation Functions ====================

# compare_outputs and print_diff_only are now imported from vla_validation_utils

# ==================== ONNX Full Pipeline Inference ====================

class OnnxVLAGenerator:
    """Complete ONNX inference generator for VLA models."""

    def __init__(self, config: OnnxGeneratorConfig):
        """
        Initialize ONNX inference sessions.

        Args:
            config: ONNX generator configuration
        """
        logging.info(f"Loading ONNX models with provider: {config.provider}...")
        session_options = ort.SessionOptions()
        self.ort_session_vision = ort.InferenceSession(
            config.model_paths.vision_backbone_path,
            sess_options=session_options,
            providers=[config.provider],
        )
        self.ort_session_projector = ort.InferenceSession(
            config.model_paths.projector_path,
            sess_options=session_options,
            providers=[config.provider],
        )
        self.ort_session_embedding = ort.InferenceSession(
            config.model_paths.embedding_path,
            sess_options=session_options,
            providers=[config.provider],
        )
        self.ort_session_prefill = ort.InferenceSession(
            config.model_paths.prefill_path,
            sess_options=session_options,
            providers=[config.provider],
        )
        self.ort_session_decoder = ort.InferenceSession(
            config.model_paths.decoder_path,
            sess_options=session_options,
            providers=[config.provider],
        )
        self.num_hidden_layers = config.num_hidden_layers
        self.decoder_input_names = [
            inp.name for inp in self.ort_session_decoder.get_inputs()
        ]
        logging.info("ONNX models loaded successfully.")

    def generate(
        self,
        inputs: MultimodalInputsNumpy,
        generation_config: GenerationConfig,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate tokens using ONNX models.

        Args:
            inputs: Multimodal model inputs (numpy arrays)
            generation_config: Token generation configuration

        Returns:
            Tuple of (generated_ids, original_effective_seq_len)
        """
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values
        attention_mask = inputs.attention_mask
        # Step 1-5: Prepare multimodal inputs using helper function
        sessions = OnnxSessions(
            vision=self.ort_session_vision,
            projector=self.ort_session_projector,
            embedding=self.ort_session_embedding,
        )
        multimodal_inputs = MultimodalInputsNumpy(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        multimodal_config = MultimodalConfig(
            pad_token_id=generation_config.pad_token_id,
            target_seq_len=generation_config.target_seq_len,
        )
        (
            multimodal_embeddings,
            multimodal_attention_mask,
            multimodal_position_ids,
            original_effective_seq_len,
        ) = prepare_multimodal_inputs_numpy(
            sessions=sessions,
            inputs=multimodal_inputs,
            config=multimodal_config,
        )

        # Step 6: Prefill stage
        prefill_inputs = {
            "multimodal_attention_mask": multimodal_attention_mask.astype(np.int64),
            "position_ids": multimodal_position_ids.astype(np.int64),
            "multimodal_embeddings": multimodal_embeddings.astype(np.float16),
        }
        prefill_outputs = self.ort_session_prefill.run(None, prefill_inputs)
        logits = prefill_outputs[0]
        past_key_values = prefill_outputs[1:]

        # Get the last valid token position
        last_valid_token_indices = np.sum(multimodal_attention_mask, axis=1) - 1
        logits_last_valid_token = logits[
            np.arange(logits.shape[0]), last_valid_token_indices, :
        ]
        next_token_id = np.argmax(logits_last_valid_token, axis=-1)[:, np.newaxis]
        generated_ids = next_token_id

        # Step 7: Decode loop
        for i in range(generation_config.max_new_tokens - 1):
            # Prepare decoder inputs
            # Use only one valid token (matching export shape)
            decoder_input_ids = next_token_id.astype(np.int64)
            
            # Update attention mask: set the position for the new token
            multimodal_attention_mask[:, original_effective_seq_len + i - 1] = 1
            # Add one new mask segment for the new token
            new_mask = np.array([[1]], dtype=np.int64)
            dummy_attention_mask = np.concatenate([multimodal_attention_mask, new_mask], axis=1)
            
            decoder_inputs = {
                "input_ids": decoder_input_ids,
                "multimodal_attention_mask": dummy_attention_mask.astype(np.int64),
            }

            # Position ID for the single new token
            new_pos_id = np.array(
                [[original_effective_seq_len + i]], dtype=np.int64
            )
            decoder_inputs["position_ids"] = new_pos_id

            # Add past key values
            for idx, name in enumerate(self.decoder_input_names[3:]):  # Skip input_ids, attention_mask, position_ids
                decoder_inputs[name] = past_key_values[idx]

            decoder_outputs = self.ort_session_decoder.run(None, decoder_inputs)
            logits = decoder_outputs[0]
            past_key_values = decoder_outputs[1:]

            # Post-process past_key_values: manually update kv cache and fill padding slots
            def replace_kv_index(past_key_values, index, seq_len):
                """Replace kv cache at index with value from position seq_len."""
                out = []
                for arr in past_key_values:
                    if arr.shape[2] > seq_len:
                        arr = arr.copy()  # Avoid in-place side effects
                        arr[:, :, index, :] = arr[:, :, seq_len, :]
                    out.append(arr)
                return out

            # Replace kv cache at original_effective_seq_len + i with value from position target_seq_len
            past_key_values = replace_kv_index(
                past_key_values,
                index=original_effective_seq_len + i,
                seq_len=generation_config.target_seq_len,
            )

            # Truncate past_key_values to length target_seq_len
            past_key_values = [
                x[:, :, :generation_config.target_seq_len, :]
                if x.shape[2] >= generation_config.target_seq_len
                else x
                for x in past_key_values
            ]

            # Extract next token: use logits[:, 0, :] (single token output)
            next_token_id = np.argmax(logits[:, 0, :], axis=-1)[:, np.newaxis]
            generated_ids = np.concatenate([generated_ids, next_token_id], axis=1)

        return generated_ids, original_effective_seq_len


def predict_action_with_onnx(
    onnx_generator: OnnxVLAGenerator,
    action_config: ActionPostProcessConfig,
    model_inputs: ModelInputs,
    target_seq_len: int = 288,
) -> np.ndarray:
    """
    Predict action using ONNX models.

    Args:
        onnx_generator: ONNX generator instance
        action_config: Action post-processing configuration
        model_inputs: Model input tensors
        target_seq_len: Target sequence length for padding

    Returns:
        Predicted actions
    """
    # Preprocess: ensure input_ids ends with pad token (29871)
    input_ids = model_inputs.input_ids
    pixel_values = model_inputs.pixel_values
    attention_mask = model_inputs.attention_mask

    if not torch.all(input_ids[:, -1] == 29871):
        pad_token_tensor = torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)
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

    # Convert to numpy
    input_ids_np = input_ids.cpu().numpy().astype(np.int64)
    pixel_values_np = pixel_values.cpu().numpy().astype(np.float16)
    attention_mask_np = attention_mask.cpu().numpy().astype(np.int64)

    # Generate tokens
    generated_ids, _ = onnx_generator.generate(
        input_ids=input_ids_np,
        pixel_values=pixel_values_np,
        attention_mask=attention_mask_np,
        max_new_tokens=action_config.action_dim,
        target_seq_len=target_seq_len,
    )

    # Post-process: convert tokens to actions
    actions = post_process_action(
        generated_ids=generated_ids,
        action_dim=action_config.action_dim,
        vocab_size=action_config.vocab_size,
        bin_centers=action_config.bin_centers,
        action_norm_stats=action_config.action_norm_stats,
    )

    return actions


def validate_vision_models(
    model,
    model_paths: Dict[str, str],
    dummy_inputs: Dict,
    provider: str = "CPUExecutionProvider",
) -> bool:
    """
    Validate vision models by comparing PyTorch and ONNX outputs.

    Args:
        model: The VLA model
        model_paths: Dictionary of model paths
        dummy_inputs: Dummy inputs
        provider: ONNX Runtime provider

    Returns:
        True if all validations pass
    """
    logging.info("\n" + "=" * 60)
    logging.info("Validating Vision Models")
    logging.info("=" * 60)

    session_options = ort.SessionOptions()
    all_match = True

    # 1. Validate Vision Backbone
    logging.info("\n[1/3] Validating Vision Backbone...")
    ort_session_vision = ort.InferenceSession(
        model_paths["vision_backbone"],
        sess_options=session_options,
        providers=[provider],
    )
    dummy_pixel_values = dummy_inputs["pixel_values"]

    torch_patch_features = (
        model.vision_backbone(dummy_pixel_values).detach().cpu().numpy()
    )
    vision_inputs = {"pixel_values": dummy_pixel_values.numpy()}
    onnx_patch_features = ort_session_vision.run(None, vision_inputs)[0]

    # Only print diff, don't check match
    print_diff_only(
        torch_patch_features, onnx_patch_features, name="patch_features"
    )

    # 2. Validate Projector
    logging.info("\n[2/3] Validating Projector...")
    ort_session_projector = ort.InferenceSession(
        model_paths["projector"],
        sess_options=session_options,
        providers=[provider],
    )
    patch_features = model.vision_backbone(dummy_pixel_values)

    torch_projected = model.projector(patch_features).detach().cpu().numpy()
    projector_inputs = {"patch_features": patch_features.detach().numpy()}
    onnx_projected = ort_session_projector.run(None, projector_inputs)[0]

    # Only print diff, don't check match
    print_diff_only(
        torch_projected, onnx_projected, name="projected_patch_embeddings"
    )

    # 3. Validate Embedding
    logging.info("\n[3/3] Validating Embedding...")
    ort_session_embedding = ort.InferenceSession(
        model_paths["embedding"],
        sess_options=session_options,
        providers=[provider],
    )
    dummy_input_ids = dummy_inputs["input_ids"]

    torch_embeddings = (
        model.get_input_embeddings()(dummy_input_ids).detach().numpy()
    )
    embedding_inputs = {"input_ids": dummy_input_ids.numpy()}
    onnx_embeddings = ort_session_embedding.run(None, embedding_inputs)[0]

    comparison_config = ComparisonConfig()
    match = compare_outputs(
        torch_embeddings, onnx_embeddings, name="input_embeddings", config=comparison_config
    )
    all_match = all_match and match

    return all_match


def validate_full_pipeline(
    model,
    model_paths: Dict[str, str],
    dummy_inputs: Dict,
    validation_config: ValidationConfig,
) -> bool:
    """
    Validate the full inference pipeline by comparing PyTorch and ONNX outputs.

    Args:
        model: The VLA model
        model_paths: Dictionary of model paths
        dummy_inputs: Dummy inputs
        unnorm_key: Key for action normalization
        target_seq_len: Target sequence length for padding
        provider: ONNX Runtime provider

    Returns:
        True if validation passes
    """
    logging.info("\n" + "=" * 60)
    logging.info("Validating Full Inference Pipeline")
    logging.info("=" * 60)

    # Run PyTorch inference
    logging.info("\n[1/2] Running PyTorch inference...")
    model.eval()
    inputs_for_pytorch = {k: v.clone() if torch.is_tensor(v) else v for k, v in dummy_inputs.items()}
    
    for k, v in inputs_for_pytorch.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs_for_pytorch[k] = v.to(model.dtype)
    
    with torch.no_grad():
        torch_action = model.predict_action(
            **inputs_for_pytorch,
            unnorm_key=validation_config.unnorm_key,
            do_sample=False,
        )
    
    # Extract post-processing configs
    post_process_configs = {
        "action_dim": model.get_action_dim(validation_config.unnorm_key),
        "vocab_size": model.vocab_size,
        "bin_centers": model.bin_centers,
        "action_norm_stats": model.get_action_stats(validation_config.unnorm_key),
        "num_hidden_layers": model.language_model.config.num_hidden_layers,
    }

    # Convert to numpy for shape inspection
    torch_action_np = (
        torch_action.cpu().numpy()
        if torch.is_tensor(torch_action)
        else torch_action
    )

    # Run ONNX inference
    logging.info("\n[2/2] Running ONNX inference...")
    onnx_model_paths = OnnxModelPaths(
        vision_backbone_path=model_paths["vision_backbone"],
        projector_path=model_paths["projector"],
        embedding_path=model_paths["embedding"],
        prefill_path=model_paths["prefill"],
        decoder_path=model_paths["decoder"],
    )
    onnx_generator_config = OnnxGeneratorConfig(
        model_paths=onnx_model_paths,
        num_hidden_layers=post_process_configs["num_hidden_layers"],
        provider=validation_config.provider,
    )
    onnx_generator = OnnxVLAGenerator(config=onnx_generator_config)

    action_config = ActionPostProcessConfig(
        action_dim=post_process_configs["action_dim"],
        vocab_size=post_process_configs["vocab_size"],
        bin_centers=post_process_configs["bin_centers"],
        action_norm_stats=post_process_configs["action_norm_stats"],
    )
    model_inputs = ModelInputs(
        input_ids=dummy_inputs["input_ids"],
        pixel_values=dummy_inputs["pixel_values"],
        attention_mask=dummy_inputs["attention_mask"],
    )
    onnx_action = predict_action_with_onnx(
        onnx_generator=onnx_generator,
        action_config=action_config,
        model_inputs=model_inputs,
        target_seq_len=validation_config.target_seq_len,
    )

    # Compare results
    logging.info("\n[3/3] Comparing results...")
    comparison_config = ComparisonConfig(
        rtol=1e-3,
        atol=1e-3,
        mean_diff_threshold=1e-2,
    )
    match = compare_outputs(
        torch_action_np,
        onnx_action,
        name="full_pipeline_action",
        config=comparison_config,
    )

    # Print full actions after comparison
    logging.info("\nPyTorch action:")
    logging.info(torch_action_np)
    logging.info("\nONNX action:")
    logging.info(onnx_action)

    # Cleanup
    del onnx_generator
    gc.collect()

    if match:
        logging.info("\n✅ Full pipeline validation passed!")
    else:
        logging.info("\n❌ Full pipeline validation failed!")

    return match


# ==================== Main Export Function ====================

def export_and_validate(
    model_path: str,
    export_config: ExportConfig,
    prompt: str = None,
    export: bool = True,
    validate: bool = True,
):
    """
    Main function to export and validate ONNX models.

    Args:
        model_path: Path to PyTorch model
        export_config: Export and validation configuration
        prompt: Text prompt (default: standard prompt)
        export: Whether to export models (if False, load existing models)
        validate: Whether to run validation
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logging.info(f"Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vla_model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Prepare dummy inputs
    if prompt is None:
        prompt = (
            "In: What action should the robot take to pick up the alphabet soup "
            "and place it in the basket?\nOut:"
        )

    height, width = get_image_size_from_processor(processor)
    dummy_image = make_dummy_image(width=width, height=height, seed=0)

    logging.info(f"Using prompt: '{prompt}'")
    logging.info(f"Using dummy image of size: {dummy_image.size}")

    inputs = processor(
        text=prompt, images=dummy_image, return_tensors="pt"
    ).to(dtype=torch.float16)

    # Export or load models
    if export:
        vision_paths = export_vision_models(
            vla_model, export_config.vision_export_dir, inputs
        )

        llama_paths = export_llama_models(
            vla_model,
            export_config.llama_prefill_export_dir,
            export_config.llama_decoder_export_dir,
            inputs,
            target_seq_len=export_config.target_seq_len,
        )

        all_model_paths = {**vision_paths, **llama_paths}
    else:
        # Load existing model paths
        logging.info("\n" + "=" * 60)
        logging.info("Loading Existing ONNX Models")
        logging.info("=" * 60)
        
        vision_paths = {
            "vision_backbone": os.path.join(
                export_config.vision_export_dir, "vision_backbone.onnx"
            ),
            "projector": os.path.join(
                export_config.vision_export_dir, "projector.onnx"
            ),
            "embedding": os.path.join(
                export_config.vision_export_dir, "embedding.onnx"
            ),
        }
        
        llama_paths = {
            "prefill": os.path.join(
                export_config.llama_prefill_export_dir, "vla_prefill.onnx"
            ),
            "decoder": os.path.join(
                export_config.llama_decoder_export_dir, "vla_decoder.onnx"
            ),
        }
        
        all_model_paths = {**vision_paths, **llama_paths}
        
        # Check if all models exist
        missing_models = [
            name for name, path in all_model_paths.items() if not os.path.exists(path)
        ]
        if missing_models:
            raise FileNotFoundError(
                f"ONNX models not found: {missing_models}. "
                f"Please run export first or check the export directories."
            )
        
        logging.info("✓ All ONNX models found")

    # Validate models
    if validate:
        logging.info("\n" + "=" * 60)
        logging.info("Starting Validation")
        logging.info("=" * 60)

        vision_valid = validate_vision_models(
            vla_model,
            vision_paths,
            inputs,
            provider=export_config.provider,
        )

        if vision_valid:
            logging.info("\n✅ All vision model validations passed!")
        else:
            logging.info("\n❌ Some vision model validations failed!")

        # Full pipeline validation
        full_valid = validate_full_pipeline(
            vla_model,
            all_model_paths,
            inputs,
            export_config.unnorm_key,
            target_seq_len=export_config.target_seq_len,
            provider=export_config.provider,
        )

    # Cleanup
    del vla_model
    gc.collect()
    torch.cuda.empty_cache()

    if export:
        logging.info("\n" + "=" * 60)
        logging.info("Export Complete")
        logging.info("=" * 60)
        logging.info(f"Vision models saved to: {export_config.vision_export_dir}")
        logging.info(
            f"LLaMA prefill model saved to: {export_config.llama_prefill_export_dir}"
        )
        logging.info(
            f"LLaMA decoder model saved to: {export_config.llama_decoder_export_dir}"
        )
    else:
        logging.info("\n" + "=" * 60)
        logging.info("Validation Complete")
        logging.info("=" * 60)
        logging.info(f"Vision models validated from: {export_config.vision_export_dir}")
        logging.info(
            f"LLaMA prefill model validated from: {export_config.llama_prefill_export_dir}"
        )
        logging.info(
            f"LLaMA decoder model validated from: {export_config.llama_decoder_export_dir}"
        )


# ==================== CLI ====================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert OpenVLA PyTorch models to ONNX format and validate"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the PyTorch model directory",
    )
    parser.add_argument(
        "--vision-export-dir",
        type=str,
        required=True,
        help="Directory to save vision ONNX models",
    )
    parser.add_argument(
        "--llama-prefill-export-dir",
        type=str,
        required=True,
        help="Directory to save LLaMA prefill ONNX model",
    )
    parser.add_argument(
        "--llama-decoder-export-dir",
        type=str,
        required=True,
        help="Directory to save LLaMA decoder ONNX model",
    )
    parser.add_argument(
        "--unnorm-key",
        type=str,
        required=True,
        help="Key for action normalization (e.g., 'libero_object')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for testing (default: standard prompt)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after export",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing ONNX models without exporting",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX Runtime provider (default: CPUExecutionProvider)",
    )
    parser.add_argument(
        "--target-seq-len",
        type=int,
        default=288,
        help="Target sequence length for padding (default: 288)",
    )

    args = parser.parse_args()

    # Validate-only mode requires validation
    if args.validate_only and args.no_validate:
        parser.error("--validate-only and --no-validate cannot be used together")

    try:
        export_config = ExportConfig(
            vision_export_dir=args.vision_export_dir,
            llama_prefill_export_dir=args.llama_prefill_export_dir,
            llama_decoder_export_dir=args.llama_decoder_export_dir,
            unnorm_key=args.unnorm_key,
            target_seq_len=args.target_seq_len,
            provider=args.provider,
        )
        export_and_validate(
            model_path=args.model_path,
            export_config=export_config,
            prompt=args.prompt,
            export=not args.validate_only,
            validate=not args.no_validate,
        )
        logging.info("\n✅ All operations completed successfully!")
    except Exception as e:
        logging.info(f"\n❌ Error occurred: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())