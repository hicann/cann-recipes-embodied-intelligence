# pi0机器人VLA大模型推理昇腾迁移-性能优化说明
<br>


## pi0昇腾迁移适配
基于torch_npu，可以实现原始代码的昇腾npu自动适配。同时，将代码中npu不支持的float64数据类型替换为float32数据类型，可以实现推理快速适配打通，并保证推理精度达标。部分npu迁移导入语句如下所示：
```bash
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
<br>

## pi0昇腾迁移优化
### 优化目标
在昇腾单机单卡计算平台上实现pi0模型的推理性能优化，缩短推理时间。测试的输入输出信息如下所示：
- 输入：双路RGB图像 + 机器人本体状态 + 语言文本指令
- 输出：50x6关节角度序列

### 优化流程
- pi0原始开箱推理profiling分析：
  
  结合torch_npu.profiler及MindStudio Insight计算流水可视化工具，得到pi0推理过程中的流水图及算子计算统计图。需要注意pi0在整个推理过程中，计算时间和空闲时间所占的比例。pi0原始开箱推理的流水图中，空闲时间占据的比例过大，由此可以得出是host-bound问题，即device侧一直在等待运算指令的下发。
  
  同时，也需要关注推理过程中AI_VECTOR_CORE和AI_CORE中算子所占的比例，pi0原始开箱推理过程中，AI_CORE算子占据的比例较小，由此可以得出计算过程中昇腾npu亲和的融合大算子所占比例较少，没能充分发挥npu的Cube（矩阵单元）的计算性能，原始的pi0代码推理计算逻辑不适合Cube。

- pi0推理代码结构分析：
  
  根据pi0推理代码，慢系统调用PaliGemma的视觉编码器对单帧图像提取高维表征；快系统则是一个基于Transformer的Flow-Matching策略模型，负责并行去噪生成动作序列。代码中的旋转编码部分、FA部分、layer-norm(add)部分、QKV计算部分均是通过小算子拼接或多步骤串接而成，导致在昇腾npu上计算不亲和。因此，从代码逻辑、昇腾亲和融合算子替换、图模式等手段出发，对pi0在昇腾npu上的推理性能优化具有重要意义。以下是对几点优化策略的简介：
  
- 优化策略——旋转编码sin/cos计算逻辑优化：
  
  通过修改`paligemma_with_expert.py`中`class PaliGemmaWithExpertModel(PreTrainedModel)`中forward函数中的旋转编码计算逻辑，修改apply_rope函数中的输入输出，利用torch_npu.npu_rotary_mul融合算子，并将Q/K_states进行融合计算拆分，同时将sin/cos计算提到for循环外部进行统一计算，减少其重复计算次数等手段，可以实现旋转编码部分的优化加速。优化的部分代码片段如下所示：
  ```
  def apply_rope(query_states, key_states, cos, sin):
    N_q = query_states.shape[2]
    N_k = key_states.shape[2]
    
    merged_states = torch.cat([query_states, key_states], dim=2)  # 维度为[B, S, N_q + N_k, D]
    
    merged_rot = torch_npu.npu_rotary_mul(merged_states, cos, sin)
    
    q_rot, k_rot = merged_rot.split([N_q, N_k], dim=2)

    return q_rot, k_rot

  ```
  

- 优化策略——flash attention的npu原生算子优化：
  
  通过替换`paligemma_with_expert.py`中FA部分为npu_prompt_flash_attention融合算子，复用融合算子在npu上的快速计算能力，提升此模块的计算性能。同时，需要注意pi0中原始FA计算方式与npu_prompt_flash_attention融合算子在数学公式上的实现差别，尤其是在attention_mask上的数学公式实现差异（bool类型True位置转换为负无穷大），并进行相应的适配修改，将一些计算过程提到循环外面进行提前计算，可以实现FA部分的优化加速。优化的部分代码片段如下所示：
  ```bash
  attention_mask = torch.logical_not(attention_mask).to(dtype=torch.int8, memory_format=torch.contiguous_format)
  attention_mask = attention_mask[:, None, :, :]

  ......

  def eager_attention_forward(
      self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
  ):
      att_output = torch_npu.npu_prompt_flash_attention(
          query_states,
          key_states.contiguous(),
          value_states.contiguous(),
          num_heads=self.num_att_heads_forward,
          input_layout="BSND",
          scale_value=self.scale_value,
          pre_tokens=65535,
          next_tokens=65535,
          atten_mask=attention_mask,
          num_key_value_heads=self.num_key_value_heads_forward
      )

      att_output = att_output.reshape(batch_size, -1, self.num_att_heads_forward * head_dim)

      return att_output
  ```


- 优化策略——layernorm部分npu原生算子优化：
  
  将first residule计算中的layernorm和add运算替换为npu算子npu_add_rms_norm融合算子，复用融合算子在npu上的快速计算能力，提升此模块的计算性能。同时，需要注意pi0中layernorm计算方式和npu_add_rms_norm融合算子在数学公式乘积系数上的实现差别，并进行相应的适配修改。此外，可以将`paligemma_with_expert.py`中`class PaliGemmaWithExpertModel(PreTrainedModel)`中forward函数中原生的layer.input_layernorm计算部分用昇腾原生的npu_rms_norm进行替换，可以实现layernorm部分的优化加速。优化的部分代码片段如下所示：
  ```bash
  hidden_states = torch_npu.npu_rms_norm(
      hidden_states,
      layer.input_layernorm.weight.add(self.ones_add),
      1e-6
  )[0]

  ......

  out_emb, _, after_first_residual = torch_npu.npu_add_rms_norm(
      layer.self_attn.o_proj(att_output[:, start:end]),
      hidden_states.to(torch.bfloat16),
      layer.post_attention_layernorm.weight.add(self.ones_add),
      1e-6
  )
  ```


- 优化策略——qkv计算部分npu原生算子优化：
  
  将pi0模型前向计算过程中q/k/v的分离计算过程，转换为qkv融合权重进行计算，然后再对融合计算结果进行split切分，节省缓存换入换出，用大矩阵乘法提升Cube的利用率。通过对`paligemma_with_expert.py`中的q/k/v计算部分进行对应优化，可以实现qkv计算部分的优化加速。优化的部分代码片段如下所示：
  ```bash
  # 将q/k/v权重融合为单个线性层qkv
  @torch.no_grad()
  def fuse_qkv_weights(self):
      """
      将每层 self_attn 的 q/k/v 权重 concat 成单个 qkv 线性层
      """
      for model in self.models:  # paligemma + gemma_expert
          for layer in model.layers:      
              attn = layer.self_attn
              w_q = attn.q_proj.weight
              w_k = attn.k_proj.weight
              w_v = attn.v_proj.weight

              w_fused = torch.cat([w_q, w_k, w_v], dim=0).bfloat16()

              # 创建 qkv 线性层
              attn.qkv = nn.Linear(w_fused.shape[1], w_fused.shape[0], bias=False, device=w_q.device).to(torch.bfloat16)

              # 将拼接后的权重赋值给 qkv 权重              
              attn.qkv.weight.data.copy_(w_fused)

  ......

  qkv = layer.self_attn.qkv(hidden_states)

  q_proj, k_proj, v_proj = qkv.split([q_out, kv_out, kv_out], dim=-1)
  ```


- 优化策略——图模式优化：
  
  原生开箱的pi0推理性能，在npu上存在严重的host-bound问题。因此，为了减少CPU到NPU的逐算子下发与同步开销，结合torch_npu的torchair图模式编译模块，可以在`modeling_pi0.py`中将pi0的快系统与慢系统部分尽可能进行整图编译，进而实现整体的计算优化加速。优化的部分代码片段如下所示：
  ```bash
  import torch_npu
  import torchair as tng
  from torchair.configs.compiler_config import CompilerConfig

  config = CompilerConfig()
  config.experimental_config.frozen_parameter = True
  config.experimental_config.tiling_schedule_optimize = True

  npu_backend = tng.get_npu_backend(compiler_config=config)

  ......

  self.compiled_embed_and_model_forward = torch.compile(
      self.embed_and_model_forward,
      dynamic=False,
      fullgraph=True,
      backend=npu_backend
  )

  self.denoise_step_all_compile = torch.compile(
      self.denoise_step_all.forward,
      dynamic=False,
      fullgraph=True,
      backend=npu_backend
  )

  ......

  (
      prefix_pad_masks,
      dt,
      x_t,
      time,
      past_key_values
  ) = self.compiled_embed_and_model_forward(
      noise, images, img_masks, lang_tokens, lang_masks
  )

  for step in range(self.config.num_steps):
      self.denoise_step_all_compile(
          state,
          prefix_pad_masks,
          past_key_values,
          x_t,
          bsize,
          time,
          dt
      )
  ```


- 优化策略结果总结：
  
  基于上述昇腾迁移-优化策略，对单次推理过程进行基于MindStudio Insight的profiling分析。其中，昇腾npu计算时间总占比提升至94%，AI_CORE类型昇腾融合算子计算时间占比提升至50%，计算总耗时为71.6毫秒，空闲时间为4.58毫秒。分析结果充分显示出迁移优化策略提升了pi0对昇腾的亲和程度，充分发挥了昇腾npu的融合算子优势和算力优势，并充分显示出具身智能pi0模型在昇腾上迁移的可行性及易用性。
<br>