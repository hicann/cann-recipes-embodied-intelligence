# Cosmos 昇腾 NPU 多卡并行优化说明

## 1. 优化概述

本次优化针对 Cosmos 系列世界基础模型在昇腾 NPU 平台上的多卡并行推理能力进行了系统性增强，主要涵盖两个模型：
- **Cosmos-Transfer2.5-2B**: 视频风格转换多控制网络模型
- **Cosmos-Predict2.5-2B**: 视频生成世界基础模型

优化重点聚焦于**使能多卡并行功能**,包括 CFG（Classifier-Free Guidance）并行、上下文并行（Context Parallelism）以及 NPU 设备管理，实现在昇腾多卡环境下的分布式高效推理。

此外，针对 NPU 特性，还进行了相关优化，包括 Flash Attention 替换、RMSNorm 融合算子适配以及 Rotary 位置编码优化。

---

## 2. 多卡并行使能

### 2.1 Cosmos在NPU上的多卡并行说明
目前的 Cosmos-Predict2.5 与 Cosmos-Transfer2.5 通过运行 `npu_adapt.sh` 脚本即可在 NPU 上正常进行多卡并行推理。


### 2.2 CFG并行修复

Cosmos-Transfer2.5 原生支持多种控制模态（深度图、语义分割、边缘检测等）的视频到视频风格迁移。为提升大规模推理效率，需实现以下并行策略：

1. **CFG 并行（Classifier-Free Guidance Parallelism）**：将 NPU 分为两组，分别处理条件（conditional）和无条件（unconditional）去噪任务，提升大规模集群扩展性
2. **上下文并行（Context Parallelism）**：跨设备分配长序列视频帧，支持超长视频生成

### 2.3 核心修改内容

#### 2.3.1 配置层：修改 `cosmos_transfer2/config.py`

在 `SetupArguments` 数据类中添加新的并行控制参数：

```python
# 在 SetupArguments 数据类中添加新参数
enable_cfg_parallel: bool = False
"""Enable Classifier-Free Guidance parallelism for better scaling across more NPUs. 
Splits NPUs into two groups for conditional/unconditional denoising."""
```

#### 2.3.2 推理层：重构 `Control2WorldInference.__init__` 方法

**修改文件**: `cosmos_transfer2/inference.py`

**Patch 文件**: `adaptor_patches/inference_patch.py`

**关键代码变更**:

```python
# 原始代码 (官方版本)
self.device_rank = 0
process_group = None
if args.context_parallel_size > 1:
    from megatron.core import parallel_state
    distributed.init()
    parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
    process_group = parallel_state.get_context_parallel_group()

# 优化后代码 (昇腾适配版)
self.device_rank = 0
cfg_parallel = args.enable_cfg_parallel  # 新增：读取 CFG 并行标志
process_group = None
if args.context_parallel_size > 1:
    from megatron.core import parallel_state
    
    distributed.init()
    
    # 根据 cfg_parallel 决定上下文并行规模
    if cfg_parallel:
        # CFG 并行模式：将总卡数对半分，一半用于 condition，一半用于 unconditional
        parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size // 2)
    else:
        # 标准模式：使用全部卡进行上下文并行
        parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
    
    process_group = parallel_state.get_context_parallel_group()
```

**逻辑说明**:
1. **CFG 并行模式** (`enable_cfg_parallel=True`):
   - 假设总卡数为 8，则 `context_parallel_size=4`
   - 4 卡处理条件去噪分支，4 卡处理无条件去噪分支
   

2. **标准并行模式** (`enable_cfg_parallel=False`):
   - 8 卡全部用于上下文并行


3. **传递 cfg_parallel 标志**:
   ```python
   self.inference_pipeline = ControlVideo2WorldInference(
       ...
       cfg_parallel=cfg_parallel,  # 传递给下游流水线
   )
   ```
---

## 3. NPU 算子性能优化

### 3.1 Flash Attention（FA）替换

#### 3.1.1 优化说明

使用 torch_npu 中的 `npu_fusion_attention` 融合算子替换源代码中的 FlashAttention 算子实现。关于 `npu_fusion_attention` 的详细说明，可见 [昇腾社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_fusion_attention.md)。

#### 3.1.2 实现方式

（1）在 Cosmos-Predict2.5-2B 中使用了 `torch_npu` 接口调用方式：
```python
attn_output_bnsd = torch_npu.npu_fusion_attention(
    query_bnsd, key_bnsd, value_bnsd, head_num, input_layout="BNSD", 
    pse=None,
    atten_mask=self.atten_mask_npu,
    scale=scale,
    pre_tockens=2147483647,
    next_tockens=2147483647,
    keep_prob=1,
    sparse_mode=2
)[0]
```
（2）在 Cosmos-Transfer2.5-2B 中使用了原生 SDPA 接口调用：
```python
attn_output_bnsd = F.scaled_dot_product_attention(
    query_bnsd, 
    key_bnsd, 
    value_bnsd, 
    attn_mask=None, 
    dropout_p=0.0,
    is_causal=True
)
```

#### 3.1.3 优化位置

- **文件**：
  - `cosmos-predict2.5/cosmos_predict2/_src/reason1/networks/qwen2_5_vl.py`
  - `cosmos-transfer2.5/cosmos_transfer2/_src/reason1/networks/qwen2_5_vl.py`


### 3.2 RMSNorm 算子优化

#### 3.2.1 优化说明

使用 torch_npu 内置的 `npu_rms_norm` 融合算子替换源代码中的自定义实现。关于 `npu_rms_norm` 的详细说明，可见 [昇腾设计文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md)。

#### 3.2.2 实现方式

（1）原始实现：
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```
（2）优化后实现：
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch_npu.npu_rms_norm(x, self.weight.float(), epsilon=self.eps)[0]
        return output
```

#### 3.2.3 优化位置

- **文件**：
  - `cosmos-predict2.5/cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py`
  - `cosmos-transfer2.5/cosmos_transfer2/_src/predict2/networks/minimal_v4_dit.py`

### 3.3 Rotary 融合算子适配

#### 3.3.1 优化说明

使用 torch_npu 内置的 `npu_rotary_mul` 融合算子替换源代码中由 `transformer_engine` 导入的 `apply_rotary_pos_emb`。关于 `npu_rotary_mul` 的详细说明，可见 [昇腾设计文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0030.html)。

#### 3.3.2 实现方式

```python
def apply_rotary_pos_emb(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    radians = freqs.transpose(0, 1) 
    cos = torch.cos(radians) 
    sin = torch.sin(radians) 
    res_rot = torch_npu.npu_rotary_mul(x, cos, sin)
    return res_rot
```

#### 3.3.3 优化位置

- **文件**：
  - `cosmos-predict2.5/cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py`
  - `cosmos-transfer2.5/cosmos_transfer2/_src/predict2/networks/minimal_v4_dit.py`

## 4. 总结

本次优化成功实现了 Cosmos 系列模型在昇腾 NPU 平台上的多卡并行推理能力与优化：

### 4.1 多卡并行优化

1. **Cosmos-Transfer2.5**：
   - 新增 `enable_cfg_parallel` 参数，支持 CFG 并行和上下文并行的灵活组合
   - 通过 `inference_patch.py` 动态修改初始化逻辑，无需侵入式修改源码

2. **Cosmos-Predict2.5**：
   - 通过 Monkey Patch 机制动态应用 NPU 适配补丁

### 4.2 通用特性

- 支持 `torchrun` 启动的多卡分布式推理
- 灵活的并行策略配置

### 4.3 融合算子优化

- **Flash Attention**：使用 `npu_fusion_attention` 替代标准 Flash Attention
- **RMSNorm**：使用 `npu_rms_norm` 融合算子提升归一化性能
- **Rotary 位置编码**：使用 `npu_rotary_mul` 加速旋转位置编码计算


