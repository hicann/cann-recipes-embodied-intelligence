# Alpamayo-R1 智驾VLA模型推理昇腾迁移-性能优化说明

本文档总结了Alpamayo-R1模型在昇腾NPU上的优化策略：

## 1. Flash Attention (FA) 替换

### 优化说明
本样例使用torch_npu内置的npu_fusion_attention融合算子替换源代码中的小算子实现，npu_fusion_attention详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_fusion_attention.md)。

### 实现方式
在模型配置中设置以下参数，会调用torch_npu内置的npu_fusion_attention融合算子：

```python
config.attn_implementation = 'flash_attention_2'
```

### 优化位置
- 文件：`src/alpamayo_r1/models/base_model.py`
- 修改：在模型初始化时设置attention实现方式

---

## 2. Transformers库中布尔索引优化

### 优化说明
原始实现中使用了布尔索引操作，效率较低。通过优化索引计算逻辑，避免不必要的布尔运算和类型转换。

### 实现方式

**原始实现：**
```python
if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
```

**优化后实现：**
创建了独立的`patched_get_placeholder_mask`函数，优化了image和video token的mask生成逻辑：
```python
if image_features is not None:
    n_masked_elements = special_image_mask.sum().item()
    if n_masked_elements != image_features.numel():
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
        )
```

### 优化位置
- 新增文件：`src/alpamayo_r1/qwen_patches/qwen3vl_optimization.py`
- 新增：`patched_get_placeholder_mask`函数
- 新增：`apply_qwen3vl_patches()`函数用于应用patch

---

## 3. RmsNorm算子优化

### 优化说明
本样例使用torch_npu内置的npu_rms_norm融合算子替换源代码中的小算子实现，npu_rms_norm详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md)。在`src/alpamayo_r1/models/action_in_proj.py`的`RmsNorm`类中使能`npu_rms_norm`融合算子：

### 实现方式

**原始实现：**
```python
def _norm(self, x):
    """Normalize the input tensor."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    """Normalize the input tensor."""
    output = self._norm(x.float()).type_as(x)
    return output * self.weight
```

**优化后实现：**
```python
def forward(self, x):
    input_dtype = x.dtype
    weight_dtype = self.weight.dtype

    if input_dtype != weight_dtype:
        weight_dtype = self.weight.to(input_dtype)
    else:
        weight = self.weight

    # npu_rms_norm returns
    output = torch_npu.npu_rms_norm(x, weight, self.eps)[0]
    return output
```

### 优化位置
- 文件：`src/alpamayo_r1/models/action_in_proj.py`
- 类：`RmsNorm`
- 修改：使用`torch_npu.npu_rms_norm`替换原始实现

---

## 4. ADD算子优化：使用融合算子torch.addcmul

### 优化说明
在扩散模型的积分过程中遇到了`x = x + dt * v`这样的操作。使用昇腾NPU提供的融合算子`torch.addcmul`，可以将加法和乘法融合为一个算子，减少计算步骤和内存访问。

### 实现方式

**原始实现：**
```python
x = x + dt * v
```

**优化后实现：**
```python
x = torch.addcmul(x, dt, v)
```

### 优化位置
- 文件：`src/alpamayo_r1/diffusion/flow_matching.py`
- 函数：`flow_matching_euler`积分步进
- 修改：使用`torch.addcmul`替换原始的加法和乘法组合

---

## 5. Concat优化：预分配+索引赋值

### 优化说明
在处理多特征拼接时，原始实现使用`torch.cat`进行多次拼接，这会导致多次内存分配和数据拷贝。优化方案是预先分配完整大小的tensor，然后通过索引直接赋值，避免拼接操作。

### 实现方式

**原始实现：**
```python
action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
timestep_feats = timestep_feats.repeat(1, T, 1)
x = torch.cat((action_feats, timestep_feats), dim=-1)
```

**优化后实现：**
```python
batch_feats = torch.empty(B, T, self.num_input_feats, device=x.device, dtype=x.dtype)
offset = 0
for i, s in enumerate(self.sinus):
    feat = s(x[:, :, i])
    batch_feats[:, :, offset:offset+feat.size(-1)] = feat
    offset += feat.size(-1)
timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
batch_feats[:, :, offset:] = timestep_feats.repeat(1, T, 1)
x = batch_feats
```

### 优化位置
- 文件：`src/alpamayo_r1/models/action_in_proj.py`
- 类：`PerWaypointActionInProjV2`
- 方法：`forward`
- 修改：预分配`batch_feats`，使用索引赋值替代`torch.cat`

---

## 优化总结

| 优化项 | 优化类型 | 主要技术 | 性能收益 |
|--------|----------|----------|----------|
| FA替换 | 注意力优化 | torch_npu.npu_fusion_attention | 降低显存、提升速度、支持更长序列 |
| Index算子替换 | 算子优化 | 优化布尔索引逻辑 | 减少冗余计算、提升索引效率 |
| RmsNorm优化 | 算子融合 | torch_npu.npu_rms_norm | 减少kernel launch、优化内存访问 |
| ADD算子优化 | 算子融合 | torch.addcmul | 减少内存读写、减少kernel launch |
| Slice+Concat优化 | 内存优化 | 预分配+索引赋值 | 减少内存分配和拷贝、提升缓存效率 |