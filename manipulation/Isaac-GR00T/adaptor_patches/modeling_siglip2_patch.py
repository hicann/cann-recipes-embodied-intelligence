# Adapted from
# https://github.com/NVIDIA/Isaac-GR00T
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2025 The HuggingFace Inc. team
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import sys
import logging
import traceback
from typing import Optional, Tuple
from dataclasses import dataclass
import importlib

import torch
import torch.nn as nn
import torch_npu

# NPU max heads limit
MAX_HEADS = 256

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NPU-Patch")


@dataclass
class FlashAttentionArgs:
    """FlashAttention NPU execution parameter wrapper."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    num_heads: int
    scale: float
    input_layout: str = "BSND"
    pre_tokens: int = 65535
    next_tokens: int = 65535
    
    def __str__(self):
        return (
            f"FlashAttentionArgs(\n"
            f"  num_heads={self.num_heads}, scale={self.scale:.6f},\n"
            f"  input_layout={self.input_layout}, pre_tokens={self.pre_tokens},\n"
            f"  next_tokens={self.next_tokens}, device={self.q.device}\n"
            f")"
        )


@dataclass
class FlashAttentionForwardParams:
    module: torch.nn.Module
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    dropout: float = 0.0
    scaling: Optional[float] = None
    kwargs: dict = None


class NpuRMSNorm(nn.Module):
    """NPU-optimized RMSNorm implementation using npu_rms_norm fused operator.
    
    Note: bias parameter is kept for compatibility with pretrained LayerNorm weights but is not used.
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Compatibility with LayerNorm: add bias parameter to load pretrained weights (not used)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # NPU optimization: use fused operator
        if hidden_states.device.type == "npu" and hasattr(torch_npu, "npu_rms_norm"):
            # npu_rms_norm returns (output, rstd), we only need output
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)
            return result[0] if isinstance(result, tuple) else result
        
        # Fallback to standard implementation
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
        # Note: RMSNorm does not use bias, ignore self.bias


def apply_rope_npu(xq, xk, freqs_cis):
    """NPU-optimized RoPE implementation using npu_rotary_mul fused operator."""
    cos = freqs_cis.real.unsqueeze(-2)
    sin = freqs_cis.imag.unsqueeze(-2)
    xq_out = torch_npu.npu_rotary_mul(xq, cos, sin)
    xk_out = torch_npu.npu_rotary_mul(xk, cos, sin)
    return xq_out, xk_out


def patched_apply_rope(xq, xk, freqs_cis):
    """Replace original apply_rope function, adding NPU optimized path."""
    if xq.device.type == "npu" and hasattr(torch_npu, "npu_rotary_mul"):
        return apply_rope_npu(xq, xk, freqs_cis)
    
    # Fallback to original implementation
    for name, mod in sys.modules.items():
        if 'modeling_siglip2' in name and hasattr(mod, '_original_apply_rope'):
            return mod._original_apply_rope(xq, xk, freqs_cis)
    
    # Built-in implementation
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def patched_flash_attention_forward_for_packing(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None, ** kwargs,
) -> Tuple[torch.Tensor, None]:
    """NPU-adapted FlashAttention forward function (supports head sharding + FIA operator)."""
    # Wrap parameters
    forward_params = FlashAttentionForwardParams(
        module=module, query=query, key=key, value=value,
        attention_mask=attention_mask, dropout=dropout, scaling=scaling, kwargs=kwargs
    )
    
    # Data type conversion
    q, k, v = forward_params.query.bfloat16(), forward_params.key.bfloat16(), forward_params.value.bfloat16()
    bsz, q_len, num_heads, head_dim = q.shape
    scale = forward_params.scaling or (1.0 / math.sqrt(head_dim))
    
    # FIA requirement: head_dim must be multiple of 16, auto padding
    head_dim_padded = ((head_dim + 15) // 16) * 16
    if head_dim_padded != head_dim:
        pad_size = head_dim_padded - head_dim
        q = torch.nn.functional.pad(q, (0, pad_size), mode='constant', value=0)
        k = torch.nn.functional.pad(k, (0, pad_size), mode='constant', value=0)
        v = torch.nn.functional.pad(v, (0, pad_size), mode='constant', value=0)
    
    # Wrap arguments
    fa_args = FlashAttentionArgs(q=q, k=k, v=v, num_heads=num_heads, scale=scale)
    outputs = []

    try:
        # Shard by MAX_HEADS
        for i in range(0, fa_args.num_heads, MAX_HEADS):
            end = min(i + MAX_HEADS, fa_args.num_heads)
            curr_heads = end - i
            
            out, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query=fa_args.q[:, :, i:end, :].contiguous(),
                key=fa_args.k[:, :, i:end, :].contiguous(),
                value=fa_args.v[:, :, i:end, :].contiguous(),
                num_heads=curr_heads,
                num_key_value_heads=curr_heads,
                input_layout=fa_args.input_layout,
                scale=fa_args.scale,
                atten_mask=None,
                sparse_mode=0,
            )
            outputs.append(out)

        # Concatenate results
        final_output = torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
        
        # Remove padding, restore original dimension
        if head_dim_padded != head_dim:
            final_output = final_output[..., :head_dim]
        
        return final_output, None

    # Enhanced exception handling
    except torch.cuda.OutOfMemoryError:
        logger.error("[NPU OOM] Memory limit reached. Device: %s, Total Heads: %d",
                     fa_args.q.device, fa_args.num_heads)
        raise
    except RuntimeError as e:
        error_trace = traceback.format_exc()
        logger.error("\n============================================\n"
                     "[NPU FIA Error] Args: %s\nQuery Shape: %s\nTrace: %s\n"
                     "============================================",
                     fa_args, list(fa_args.q.shape), error_trace)
        raise e
    except Exception as e:
        logger.error("[NPU Unexpected Error] %s: %s", type(e).__name__, e)
        raise e


_patched_modules = set()


class NpuPatchFinder(importlib.abc.MetaPathFinder):
    """Custom import finder that automatically applies patches after modeling_siglip2 is imported."""
    
    def find_spec(self, fullname, path, target=None):
        if 'modeling_siglip2' in fullname and 'adaptor_patches' not in fullname:
            for finder in sys.meta_path:
                if finder is self:
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None:
                        spec.loader = NpuPatchLoader(spec.loader)
                        return spec
                except (AttributeError, TypeError):
                    continue
        return None


class NpuPatchLoader(importlib.abc.Loader):
    """Custom loader that applies patches after the module is loaded."""
    
    def __init__(self, original_loader=None):
        self.original_loader = original_loader
    
    def create_module(self, spec):
        if self.original_loader:
            return self.original_loader.create_module(spec)
        return None
    
    def exec_module(self, module):
        if self.original_loader:
            self.original_loader.exec_module(module)
        else:
            importlib._bootstrap._exec_module(module)
        
        if 'modeling_siglip2' in module.__name__ and 'adaptor_patches' not in module.__name__:
            apply_patch_to_module(module)


def apply_patch_to_module(mod):
    """Apply patches to the given module."""
    global _patched_modules
    
    if mod.__name__ in _patched_modules:
        return
    
    patched_count = 0
    
    # 1. Patch FlashAttention functions
    for attr in ['flash_attention_forward_for_packing', '_flash_attention_forward']:
        if hasattr(mod, attr):
            setattr(mod, attr, patched_flash_attention_forward_for_packing)
            patched_count += 1
    
    # 2. Patch apply_rope
    if hasattr(mod, 'apply_rope') and not hasattr(mod, '_original_apply_rope'):
        mod._original_apply_rope = mod.apply_rope
        mod.apply_rope = patched_apply_rope
        patched_count += 1
    
    # 3. Patch Siglip2Attention.forward
    if hasattr(mod, 'Siglip2Attention'):
        original_attn_class = mod.Siglip2Attention
        
        def make_patched_forward(original_forward):
            def patched_forward(self, *args, **kwargs):
                if 'rope_freqs_cis' in kwargs and kwargs['rope_freqs_cis'] is not None:
                    original_apply_rope = mod.apply_rope if hasattr(mod, 'apply_rope') else None
                    mod.apply_rope = patched_apply_rope
                    try:
                        return original_forward(self, *args, **kwargs)
                    finally:
                        if original_apply_rope:
                            mod.apply_rope = original_apply_rope
                return original_forward(self, *args, **kwargs)
            return patched_forward
        
        if hasattr(original_attn_class, 'forward'):
            original_attn_class._original_forward = original_attn_class.forward
            original_attn_class.forward = make_patched_forward(original_attn_class.forward)
            patched_count += 1
    
    # 4. Patch Siglip2EncoderLayer (replace RMSNorm)
    if hasattr(mod, 'Siglip2EncoderLayer'):
        original_layer_class = mod.Siglip2EncoderLayer
        original_init = original_layer_class.__init__
        
        def patched_layer_init(self, config):
            original_init(self, config)
            embed_dim = config.hidden_size
            eps = config.layer_norm_eps
            self.layer_norm1 = NpuRMSNorm(embed_dim, eps=eps)
            self.layer_norm2 = NpuRMSNorm(embed_dim, eps=eps)
        
        original_layer_class.__init__ = patched_layer_init
        patched_count += 1
    
    # 5. Patch Siglip2VisionTransformer (replace post_layernorm)
    if hasattr(mod, 'Siglip2VisionTransformer'):
        original_vit_class = mod.Siglip2VisionTransformer
        original_vit_init = original_vit_class.__init__
        
        def patched_vit_init(self, config):
            original_vit_init(self, config)
            embed_dim = config.hidden_size
            eps = config.layer_norm_eps
            self.post_layernorm = NpuRMSNorm(embed_dim, eps=eps)
        
        original_vit_class.__init__ = patched_vit_init
        patched_count += 1
    
    # 6. Patch Siglip2MultiheadAttentionPoolingHead
    if hasattr(mod, 'Siglip2MultiheadAttentionPoolingHead'):
        original_head_class = mod.Siglip2MultiheadAttentionPoolingHead
        original_head_init = original_head_class.__init__
        
        def patched_head_init(self, config):
            original_head_init(self, config)
            eps = config.layer_norm_eps
            self.layernorm = NpuRMSNorm(config.hidden_size, eps=eps)
        
        original_head_class.__init__ = patched_head_init
        patched_count += 1
    
    # 7. Patch Siglip2VisionEmbeddings.split_patch_embeddings_to_windows_with_meta
    if hasattr(mod, 'Siglip2VisionEmbeddings'):
        original_emb_class = mod.Siglip2VisionEmbeddings
        if hasattr(original_emb_class, 'split_patch_embeddings_to_windows_with_meta'):
            original_method = original_emb_class.split_patch_embeddings_to_windows_with_meta

            def patched_split_method(self, patch_embeds, batch_hw, window_size):
                """NPU-optimized window splitting method."""
                from collections import defaultdict
                from itertools import accumulate
                import torch.nn.functional as F

                batch_hw = batch_hw.tolist()
                counts = [H * W for (H, W) in batch_hw]
                starts = [0] + list(accumulate(counts))[:-1]

                size2info = defaultdict(list)
                for img_idx, ((H, W), start) in enumerate(zip(batch_hw, starts)):
                    size2info[(H, W)].append((img_idx, start))

                all_windows = []
                all_meta = []

                for (H, W), info in size2info.items():
                    H, W = int(H), int(W)
                    B = len(info)
                    C = patch_embeds.shape[-1]
                    img_idxs, img_starts = zip(*info)

                    imgs = []
                    for st in img_starts:
                        # NPU optimization: view+permute instead of transpose+reshape
                        flat = patch_embeds[0, st: st + H * W].view(H, W, C)
                        imgs.append(flat.permute(2, 0, 1))
                    batch_tensor = torch.stack(imgs, dim=0)

                    pad_h = (window_size - H % window_size) % window_size
                    pad_w = (window_size - W % window_size) % window_size
                    batch_padded = F.pad(batch_tensor, (0, pad_w, 0, pad_h))

                    H_pad, W_pad = H + pad_h, W + pad_w
                    n_h = H_pad // window_size
                    n_w = W_pad // window_size
                    n_windows = n_h * n_w

                    patches_unf = F.unfold(
                        batch_padded,
                        kernel_size=(window_size, window_size),
                        stride=(window_size, window_size)
                    )

                    patches = (
                        patches_unf
                        .view(B, C, window_size * window_size, n_windows)
                        .permute(0, 3, 2, 1)
                        .reshape(-1, window_size * window_size, C)
                    )
                    all_windows.append(patches)

                    for b, img_idx in enumerate(img_idxs):
                        for win_id in range(n_windows):
                            i, j = divmod(win_id, n_w)
                            h0, w0 = i * window_size, j * window_size
                            h1 = min(h0 + window_size, H)
                            w1 = min(w0 + window_size, W)
                            all_meta.append({
                                'img_idx': img_idx,
                                'patch_hw': (H, W),
                                'win_xy': (h0, w0),
                                'win_hw': (h1 - h0, w1 - w0),
                            })

                sorted_idx = sorted(
                    range(len(all_meta)),
                    key=lambda k: (
                        all_meta[k]['img_idx'],
                        all_meta[k]['win_xy'][0],
                        all_meta[k]['win_xy'][1]
                    )
                )
                all_windows = torch.cat(all_windows, dim=0)
                all_windows = all_windows[sorted_idx]
                win_meta_list = [all_meta[i] for i in sorted_idx]

                windows_list = []
                for meta, win in zip(win_meta_list, all_windows):
                    h_eff, w_eff = meta['win_hw']
                    valid_num = h_eff * w_eff
                    if valid_num == window_size * window_size:
                        windows_list.append(win)
                    else:
                        win = win.view(window_size, window_size, -1)[:h_eff, :w_eff, :].reshape(h_eff * w_eff, -1)
                        windows_list.append(win)

                all_tokens = torch.cat(windows_list, dim=0).unsqueeze(0)

                counts = [H * W for H, W in batch_hw]
                starts = [0] + list(accumulate(counts))[:-1]
                total_patches = sum(counts)

                mapping = [None] * total_patches
                offset = 0

                for meta in win_meta_list:
                    img_idx = meta['img_idx']
                    H, W = meta['patch_hw']
                    h0, w0 = meta['win_xy']
                    h_eff, w_eff = meta['win_hw']
                    base = starts[img_idx]

                    for u in range(h_eff):
                        for v in range(w_eff):
                            orig_idx = base + (h0 + u) * W + (w0) + v
                            p = u * w_eff + v
                            mapping[orig_idx] = offset + p
                    offset += h_eff * w_eff

                reverse_mapping = torch.tensor(mapping, dtype=torch.long)

                return all_tokens, win_meta_list, reverse_mapping

            original_emb_class.split_patch_embeddings_to_windows_with_meta = patched_split_method
            patched_count += 1

    # 8. Patch _init_weights
    if hasattr(mod, '_init_weights'):
        original_init_weights = mod._init_weights

        def patched_init_weights(self, module):
            if isinstance(module, NpuRMSNorm):
                module.weight.data.fill_(1.0)
            else:
                original_init_weights(self, module)

        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and hasattr(attr, '_init_weights'):
                attr._init_weights = patched_init_weights

    if patched_count > 0:
        _patched_modules.add(mod.__name__)
        logger.info("[NPU-Patch] Applied %d patches to %s", patched_count, mod.__name__)


# Register import hook
sys.meta_path[:] = [NpuPatchFinder()] + sys.meta_path
logger.info("[NPU-Patch] Hook registered (device=%s)",
            torch.npu.get_device_name(0) if torch.npu.is_available() else 'N/A')


def patch_transformers_all_attention_functions():
    """Patch the ALL_ATTENTION_FUNCTIONS registry in transformers library."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        
        # Register NPU version of flash_attention_2
        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = patched_flash_attention_forward_for_packing
        ALL_ATTENTION_FUNCTIONS['flash_attention_2_packing'] = patched_flash_attention_forward_for_packing
        
        logger.info("[NPU-Patch] Patched transformers ALL_ATTENTION_FUNCTIONS")
        return True
    except ImportError as e:
        logger.warning("[NPU-Patch] transformers.modeling_utils not found: %s", e)
        return False
    except Exception as e:
        logger.error("[NPU-Patch] Failed to patch ALL_ATTENTION_FUNCTIONS: %s", e)
        return False


def patch_transformers_flash_attention_utils():
    """Patch the _flash_attention_forward function in transformers.modeling_flash_attention_utils."""
    try:
        import transformers.modeling_flash_attention_utils as fa_utils
        
        if hasattr(fa_utils, '_flash_attention_forward'):
            fa_utils._original_flash_attention_forward = fa_utils._flash_attention_forward
            fa_utils._flash_attention_forward = patched_flash_attention_forward_for_packing
            logger.info("[NPU-Patch] Patched transformers.modeling_flash_attention_utils._flash_attention_forward")
            return True
        else:
            logger.warning("[NPU-Patch] _flash_attention_forward not found in modeling_flash_attention_utils")
            return False
    except ImportError:
        logger.info("[NPU-Patch] transformers.modeling_flash_attention_utils not available (may be older version)")
        return False
    except Exception as e:
        logger.error("[NPU-Patch] Failed to patch modeling_flash_attention_utils: %s", e)
        return False


def patch_transformers_integrations_flash_attention():
    """Patch the transformers.integrations.flash_attention module."""
    try:
        import transformers.integrations.flash_attention as fa_integration
        
        patched_count = 0
        
        # Patch flash_attention_forward function
        if hasattr(fa_integration, 'flash_attention_forward'):
            fa_integration._original_flash_attention_forward = fa_integration.flash_attention_forward
            fa_integration.flash_attention_forward = patched_flash_attention_forward_for_packing
            patched_count += 1
        
        logger.info("[NPU-Patch] Patched transformers.integrations.flash_attention (%d functions)", patched_count)
        return patched_count > 0
    except ImportError:
        logger.info("[NPU-Patch] transformers.integrations.flash_attention not available")
        return False
    except Exception as e:
        logger.error("[NPU-Patch] Failed to patch integrations.flash_attention: %s", e)
        return False


def apply_patch():
    """
    Apply all NPU patches - must be called before model loading.
    """
    patched_count = 0
    
    if patch_transformers_all_attention_functions():
        patched_count += 1
    if patch_transformers_flash_attention_utils():
        patched_count += 1
    if patch_transformers_integrations_flash_attention():
        patched_count += 1
    
    if patched_count > 0:
        logger.info("[NPU-Patch] apply_patch() completed: %d transformers patches applied", patched_count)
    else:
        logger.warning("[NPU-Patch] apply_patch() completed: no transformers patches applied")
    
    return patched_count
