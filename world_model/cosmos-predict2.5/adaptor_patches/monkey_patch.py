import sys
import math
from typing import List, Optional, Tuple, Union

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from einops import rearrange  # 提前导入，避免后续报错
from torch.distributed.device_mesh import DeviceMesh
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache


def setup_npu_device():
    import os
    if torch.npu.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.set_default_device(f'npu:{local_rank}')
        torch.npu.set_device(local_rank)


# 内置RoPE实现
def apply_rotary_pos_emb(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    radians = freqs.transpose(0, 1)  # [1,S2,1,D2]
    # 核心旋转计算
    cos = torch.cos(radians)  # [1,S2,1,128]
    sin = torch.sin(radians)  # [1,S2,1,128]
    # 用torch_npu融合算子进行快速使能
    res_rot = torch_npu.npu_rotary_mul(x, cos, sin)
    return res_rot


# 重载Attention类的compute_qkv方法（去掉多余参数）
def patched_compute_qkv(self, x, context=None, rope_emb=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = self.q_proj(x)
    context = x if context is None else context
    k = self.k_proj(context)
    v = self.v_proj(context)
    q, k, v = map(
        lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
        (q, k, v),
    )

    def apply_norm_and_rotary_pos_emb(q, k, v, rope_emb):
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        original_dtype = q.dtype
        if self.is_selfattn and rope_emb is not None:
            if self.use_wan_fp32_strategy:
                q = q.to(torch.float32)
                k = k.to(torch.float32)
            # 仅传q/k和rope_emb，去掉tensor_format和fused参数
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)
            if self.use_wan_fp32_strategy:
                q = q.to(original_dtype)
                k = k.to(original_dtype)
        return q, k, v

    q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)
    return q, k, v


# ------------------------- 修复minimal_v4_dit -------------------------
def patch_minimal_v4_dit():
    import cosmos_predict2._src.predict2.networks.minimal_v4_dit as m


    m.apply_rotary_pos_emb = apply_rotary_pos_emb

    # 1. 重载Attention类的compute_qkv方法
    m.Attention.compute_qkv = patched_compute_qkv
    
    # npu_rms_norm融合算子版本RMSNorm
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))

        def reset_parameters(self):
            torch.nn.init.ones_(self.weight)

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = torch_npu.npu_rms_norm(
                x,
                self.weight.float(),
                self.eps
            )[0]
            return result

    # 仅对核心重载逻辑做异常捕获（try/except紧邻）
    try:
        # 2. 设置Attention后端为minimal_a2a
        original_attn_init = m.Attention.__init__

        def new_attn_init(
            self,
            query_dim,
            context_dim=None,
            n_heads=8,
            head_dim=64,
            dropout=0.0,
            qkv_format="bshd",
            backend="minimal_a2a",
            use_wan_fp32_strategy=False,
        ):
            # 使用minimal_a2a后端，
            original_attn_init(
                self,
                query_dim,
                context_dim,
                n_heads,
                head_dim,
                dropout,
                qkv_format,
                backend,
                use_wan_fp32_strategy,
            )
            # 替换norm为本地实现
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.v_norm = torch.nn.Identity()
        
        m.Attention.__init__ = new_attn_init
        
        # 3. 修复MiniTrainDIT中的t_embedding_norm
        original_mini_init = m.MiniTrainDIT.__init__

        def new_mini_init(self, *args, **kwargs):
            original_mini_init(self, *args, **kwargs)
            # 替换TE的RMSNorm为本地实现
            self.t_embedding_norm = RMSNorm(self.model_channels, eps=1e-6)
        
        m.MiniTrainDIT.__init__ = new_mini_init
        
        # 4. 修复I2VCrossAttention的k_img_norm
        original_i2v_init = m.I2VCrossAttention.__init__
        
        def new_i2v_init(self, *args, **kwargs):
            original_i2v_init(self, *args, **kwargs)
            self.k_img_norm = RMSNorm(self.head_dim, eps=1e-6)
        
        m.I2VCrossAttention.__init__ = new_i2v_init

        if not hasattr(m, 'RMSNorm'):
            m.RMSNorm = RMSNorm
    except Exception as e:
        import traceback
        traceback.print_exc()


# ---------------------- 修复qwen2_5_vl的FlashAttention2 ----------------------
def patch_qwen2_5_vl():
    """适配qwen2_5_vl到NPU"""
    try:
        import cosmos_predict2._src.reason1.networks.qwen2_5_vl as qwen2_5_vl


        original_flash_attention2_init = qwen2_5_vl.Qwen2_5_VLFlashAttention2.__init__

        def new_flash_attention2_init(self, *args, **kwargs):
            original_flash_attention2_init(self, *args, **kwargs)
            self.target_device = torch.device("npu")
            # 直接在NPU上创建张量，无设备间拷贝，效率更高
            self.atten_mask_npu = torch.triu(torch.ones(2048, 2048, device=self.target_device), diagonal=1).bool()

        qwen2_5_vl.Qwen2_5_VLFlashAttention2.__init__ = new_flash_attention2_init

        # 强制关闭 flash_attn 检查（关键：避免走 flash 路径）
        def disable_flash_attn_check():
            return False
                
        qwen2_5_vl.is_flash_attn_2_available = disable_flash_attn_check
        
        def new_npu_flash_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            cp_mesh: DeviceMesh | None = None,
        ):
            """
            Args:
                cp_mesh (DeviceMesh, optional): Device mesh over which context parallelism is done.
            """
            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

            # Because the input can be padded, the absolute sequence length depends on the max position id.
            cos, sin = position_embeddings
            query_states, key_states = qwen2_5_vl.apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
            )

            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }  # Specific to RoPE models
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = qwen2_5_vl.repeat_kv(key_states, self.num_key_value_groups)
            value_states = qwen2_5_vl.repeat_kv(value_states, self.num_key_value_groups)

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in float16 just to be sure everything works as expected.
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = getattr(self.config, "_pre_quantization_dtype")
                else:
                    target_dtype = self.q_proj.weight.dtype

                qwen2_5_vl.logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. "
                    f"We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)


            # 确认输入布局：为BNSD
            query_bnsd = query_states.contiguous()
            key_bnsd = key_states.contiguous()
            value_bnsd = value_states.contiguous()

            scale = 1.0 / math.sqrt(self.head_dim)
            head_num = query_bnsd.shape[1]
            attn_output_bnsd = torch_npu.npu_fusion_attention(
                                query_bnsd, key_bnsd, value_bnsd, head_num, input_layout="BNSD", 
                                pse=None,
                                atten_mask=self.atten_mask_npu,
                                scale=scale,
                                pre_tockens=2147483647,
                                next_tockens=2147483647,
                                keep_prob=1,
                                sparse_mode=2)[0]

            # 转换输出布局：BNSD 转为 BSND（恢复原格式）
            attn_output = attn_output_bnsd.permute(0, 2, 1, 3)

            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value
        
        # 替换forward方法
        if hasattr(qwen2_5_vl, 'Qwen2_5_VLFlashAttention2'):
            qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = new_npu_flash_forward
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return


# 应用补丁
def apply_all_patches():
    setup_npu_device()  # 启用NPU设备初始化
    patch_minimal_v4_dit()
    patch_qwen2_5_vl()


if __name__ == "__main__":
    apply_all_patches()