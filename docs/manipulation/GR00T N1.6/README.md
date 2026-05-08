# GR00T N1.6机器人VLA大模型推理昇腾迁移-性能优化说明

## GR00T N1.6昇腾迁移适配
基于torch_npu，可以实现原始代码的昇腾npu自动适配。同时，将代码中flash attention相关算子使用昇腾融合算子进行替换，并npu不支持的float64数据类型替换为float32数据类型，可以实现推理快速适配打通，并保证推理精度达标。部分npu迁移导入语句如下所示：
```bash
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
<br>

## GR00T N1.6昇腾迁移优化
### 优化目标
在昇腾A3平台上单卡实现GR00T N1.6模型的推理性能优化，降低推理时间。测试的输入输出信息如下所示：
- 输入：单视角RGB图像 + 机器人本体状态 + 语言文本指令，默认采用demo_data所提供的gr1.PickNPlace数据集，4去噪步。
- 输出：每步推理输出16x29关节角度序列，包含双手、双臂、腰等关节部位信息。

### 优化流程
- 开箱推理profiling分析：
  
  结合torch_npu.profiler及MindStudio Insight计算流水可视化工具，得到GR00T N1.6在推理过程中的流水图及算子计算统计表，尤其注意GR00T N1.6在整个推理过程中，计算时间与空闲时间所占的比例、各算子类型占比等信息。GR00T N1.6在开箱推理的流水图中，空闲时间比例过大，计算占比不到20%，device侧一直在等待运算指令的下发，存在host-bound问题。
  
  同时，也需要关注推理过程中AI_VECTOR_CORE和AI_CORE中算子所占的比例，在GR00T N1.6的开箱推理过程中，AI_CORE算子占据的比例较小，由此可以得出计算过程中昇腾NPU亲和的融合大算子所占比例较少，没能充分发挥NPU的Cube单元的计算性能。此外，在算子计算统计表中，transpose、cast等算子占比较大，分别达到了16%和10%，同样未能发挥出NPU的计算优势。

- GR00T N1.6推理结构分析：
  
  GR00T N1.6的网络分为快系统和慢系统两部分组成，慢系统采用的是Cosmos-Reason-2B，包括视觉编码器、多模态投影适配器、LLM 基座解码器、跨模态双向融合层四部分串行组成；快系统则是一个扩散Transformer(DiT)模型，负责并行去噪生成动作序列。代码中的旋转编码部分、FA部分、layer-norm(add)部分均是通过小算子拼接或多步骤串接而成，导致在昇腾NPU上计算不亲和。因此，从代码逻辑优化、昇腾融合算子替换、图模式使能等手段出发，可以加速GR00T N1.6在昇腾A3上的推理优化。以下是对几点优化策略的说明：
  

- 优化策略——图模式使能：

  由于开箱存在host-bound问题，可以对GR00T N1.6整网或者部分网络进行编译后一次性下发指令，减少device的等待。GR00T N1.6的网络主要由backbone和action head两部分组成，由于backbone存在大量动态操作，无法使用torchair进行编译，在`standalone_inference_script.py`中对`action head`部分使用torchair实现图模式使能的代码如下所示，使能后会在第一步推理时增加两分钟左右的图编译时间，但是后续推理步的速度会大幅提升：
  ```
    if local_model_path is not None:
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=local_model_path,
        device="npu" if torch.npu.is_available() else "cpu",
    )

    if hasattr(policy.model, 'action_head') and hasattr(policy.model.action_head, 'model'):
        policy.model.action_head.model = torch.compile(
            policy.model.action_head.model,
            backend="npugraphs",
            fullgraph=True,
            dynamic=False,
        )
        logging.info(" npugraphs mode enabled for action_head (dynamic=True)")
  ```

- 优化策略——flash attention融合算子替换：

  通过使用猴子补丁的方式在`modeling_siglip2.py`中使用`npu_fused_infer_attention_score`融合算子，复用融合算子在npu上的快速计算能力，提升此模块的计算性能。

  ```
  ...
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
  ```

- 优化策略——旋转编码sin/cos计算逻辑优化：
  
  通过修改`modeling_siglip2.py`中forward函数中的旋转编码计算逻辑，修改apply_rope函数中的输入输出，利用torch_npu.npu_rotary_mul融合算子，并将Q/K_states进行融合计算拆分，同时将sin/cos计算提到for循环外部进行统一计算，减少其重复计算次数等手段，可以实现旋转编码部分的优化加速。优化的部分代码片段如下所示：
  ```
    def apply_rope_npu(xq, xk, freqs_cis):
        """NPU-optimized RoPE implementation using npu_rotary_mul fused operator."""
        cos = freqs_cis.real.unsqueeze(-2)
        sin = freqs_cis.imag.unsqueeze(-2)
        xq_out = torch_npu.npu_rotary_mul(xq, cos, sin)
        xk_out = torch_npu.npu_rotary_mul(xk, cos, sin)
        return xq_out, xk_out
  ```


- 优化策略——layernorm部分npu原生算子优化：
  
  将计算中的layernorm和add运算替换为npu算子npu_add_rms_norm融合算子，复用融合算子在npu上的快速计算能力，提升此模块的计算性能。此外，可以将`modeling_siglip2.py`中中forward函数中原生的layer.input_layernorm计算部分用昇腾原生的`npu_rms_norm`进行替换，可以实现layernorm部分的优化加速。优化的部分代码片段如下所示：
  ```bash
    class NpuRMSNorm(nn.Module):        
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            # NPU optimization: use fused operator
            if hidden_states.device.type == "npu" and hasattr(torch_npu, "npu_rms_norm"):
                # npu_rms_norm returns (output, rstd), we only need output
                result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)
                return result[0] if isinstance(result, tuple) else result
            
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
  ```


- 优化策略结果总结：
  
  基于上述昇腾迁移-优化策略，对单次推理过程进行基于MindStudio Insight的profiling分析。其中，在A3上计算时间总占比提升至70%，计算总耗时为82ms，相较于开箱性能的280ms有了大幅提升。
<br>