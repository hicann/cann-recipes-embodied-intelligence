# Pi0 OM 推理 Debug 经验

本文档记录 Pi0 在昇腾 310P 上进行 ONNX 导出、ATC 转 OM、OM 精度验证与端到端推理时的常见问题和排查路径。建议先按 [Pi0 OM 推理使用指南](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/manipulation/pi0/infer_with_om/README.md) 完成基础环境准备，再按本文档逐段定位问题。

## 1. 先固定排查边界

Pi0 OM 链路建议固定以下前提，否则很多报错会表现为配置类、shape 类或权重加载类问题：

- LeRobot 代码版本使用 `577cd10974b84bea1f06b6472eb9e5e74e07f77a`。
- 模型目录中应包含 `config.json`、权重文件和 tokenizer 相关文件。
- ONNX 导出和 OM 校验使用同一套 Pi0 适配文件。
- VLM 与 action expert 分别导出，action expert 依赖 VLM 导出阶段保存的中间张量。
- 自定义模型需要确认 `config.json.input_features`、`config.json.output_features`、`chunk_size` 与实际数据一致。

最小目录检查：

```bash
ls /path/to/pi0_model/config.json
ls manipulation/pi0/infer_with_om/lerobot_modify/
ls manipulation/pi0/infer_with_om/lib/modeling_gemma.patch
```

## 2. 为什么拆成两段 OM

Pi0 的推理链路由两部分构成：

- VLM 模块：处理图像、语言和本体状态，生成前缀特征与 KV cache。
- Action expert 模块：基于 VLM 生成的 KV cache 和动作噪声，执行多步 denoise，输出动作序列。

本样例没有把完整 Pi0 一次性导出成单个 OM，而是拆成 VLM 和 action expert 两段：

- 降低 ATC 工具在转换时的内存开销。Pi0 完整图包含视觉编码器、语言/图像前缀处理、KV cache 和 action expert 多步去噪，整体图较大，一次性转换更容易触发内存压力或编译失败。
- 固定 VLM 输出的中间张量边界。VLM 段输出 `past_kv_tensor` 和 `prefix_pad_masks`，action expert 段把它们作为显式输入，有利于单独验证两段模型的 ONNX/OM 精度。
- 方便定位问题。如果 VLM 段校验正常而 action expert 段异常，可以直接聚焦 KV cache、attention mask、position ids 或动作后处理，而不需要在完整端到端图里混合排查。
- 复用 Pi0 的推理结构。Pi0 原始推理本身也是先计算 image/language prefix KV cache，再在 action expert 中多步 denoise；两段 OM 保留了这个边界。

对应文件：

- `modeling_pi0_vlm.py`：导出 VLM 前缀编码图，输出 `past_kv_tensor` 和 `prefix_pad_masks`。
- `modeling_pi0_action_expert.py`：导出 action expert 单步 denoise 图，输入上一段生成的 KV cache。
- `run_om_e2e.py`：端到端串联两段 OM，循环调用 action expert。

## 3. 主要算子与逻辑改造

Pi0 OM 版本与昇腾 torch 推理优化版本的目标不同：

- torch 推理优化版本优先使用 `torch_npu` 融合算子提升运行性能。
- OM 版本优先保证 ONNX 导出、ATC 编译和 OM Runtime 校验稳定，因此部分融合算子会改回更容易导出和编译的基础算子组合。

### 3.1 KV cache 展平

原始推理中，`past_key_values` 通常是 Python list/dict 结构。ONNX/OM 不适合直接表达这种嵌套对象，因此 OM 版本将其展平为单个 Tensor：

```text
list[{"key_states": Tensor, "value_states": Tensor}]
-> Tensor(L, 2, B, S, H, D)
```

相关函数：

- `flatten_kv`
- `unflatten_kv`

这样可以让 VLM OM 的输出直接作为 action expert OM 的输入。

### 3.2 RoPE 旋转编码

昇腾 torch 推理优化版本中可使用：

```python
torch_npu.npu_rotary_mul(...)
```

OM 版本中改为 ONNX 友好的手写 RoPE：

```python
sin = torch.sin(radians)
cos = torch.sin(radians + math.pi / 2)
res[..., :d_half] = x1 * cos - x2 * sin
res[..., d_half:] = x2 * cos + x1 * sin
```

原因：

- `npu_rotary_mul` 适合 NPU eager/graph 推理优化，但不一定适合当前 ONNX 导出和 ATC 编译路径。
- 手写版本由基础算子组成，更容易做 ONNX/OM 精度对齐。
- 使用 `sin(x + pi/2)` 表达 cos，可减少部分环境中 cos 精度或导出差异带来的不一致。

### 3.3 Attention

昇腾 torch 推理优化版本中可使用：

```python
torch_npu.npu_prompt_flash_attention(...)
```

OM 版本中改为基础 attention 计算：

```python
att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
att_weights *= head_dim**-0.5
masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
probs = nn.functional.softmax(masked_att_weights, dim=-1)
att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
```

原因：

- `npu_prompt_flash_attention` 是运行时融合算子，适合 torch_npu 路径的性能优化。
- OM 导出更关注 ATC 可编译性和输入输出稳定性，基础 attention 更容易排查 mask、shape 和 dtype 问题。
- VLM 段和 action expert 段的 attention mask 形状不同，基础实现更方便分别验证。

### 3.4 RMSNorm 与 residual add

昇腾 torch 推理优化版本中可使用：

```python
torch_npu.npu_rms_norm(...)
torch_npu.npu_add_rms_norm(...)
```

OM 版本中使用普通模块调用和显式残差：

```python
hidden_states = layer.input_layernorm(hidden_states)
out_emb = layer.self_attn.o_proj(att_output[:, start:end])
out_emb += hidden_states
after_first_residual = out_emb.clone()
out_emb = layer.post_attention_layernorm(out_emb)
out_emb = layer.mlp(out_emb)
out_emb += after_first_residual
```

原因：

- 融合 RMSNorm/add 算子对 torch_npu 性能更友好。
- OM 导出时保留普通算子组合更利于 ONNXRuntime 和 OM 输出逐段对齐。
- 显式残差路径便于定位 layernorm、MLP 或 dtype 导致的误差。

### 3.5 dtype、mask 和 position ids

OM 路径中需要特别注意 dtype：

- 图像和 state 在 VLM 输入中通常保持 `float32`。
- action expert 中的 state、KV、time、noise 多使用 `float16`。
- `lang_tokens` 使用 `int64`。
- `lang_masks`、`prefix_pad_masks` 使用 bool。
- `cumsum` 前需要明确将 mask 转成整数类型，再转回 bool 参与 mask 计算。

原因：

- NPU/ATC 对部分 dtype 组合更敏感，隐式类型转换容易导致导出图不稳定。
- attention mask 和 position ids 是两段 OM 最容易出现 shape/dtype 不一致的位置。
- 显式 dtype 能降低 ONNXRuntime 与 OM Runtime 之间的输入差异。

## 4. LeRobot 版本或补丁不一致

现象：

- `PI0Policy.from_pretrained` 加载时报 config 字段缺失。
- `modeling_pi0_vlm` 或 `modeling_pi0_action_expert` 无法 import。
- ONNX 导出阶段出现输入字段、模型结构或 tokenizer 相关报错。

处理：

1. 确认 LeRobot commit：

```bash
cd /path/to/lerobot
git rev-parse HEAD
```

2. 确认已复制 Pi0 OM 适配文件：

```bash
export LEROBOT=/path/to/cann-recipes-embodied-ai/manipulation/lerobot
export INFER_WITH_OM=/path/to/cann-recipes-embodied-ai/manipulation/pi0/infer_with_om

cp "$INFER_WITH_OM/lerobot_modify/modeling_pi0_vlm.py" "$LEROBOT/src/lerobot/policies/pi0/"
cp "$INFER_WITH_OM/lerobot_modify/modeling_pi0_action_expert.py" "$LEROBOT/src/lerobot/policies/pi0/"
cp "$INFER_WITH_OM/lerobot_modify/paligemma_with_expert_fp16.py" "$LEROBOT/src/lerobot/policies/pi0/"
cp "$INFER_WITH_OM/lerobot_modify/normalize.py" "$LEROBOT/src/lerobot/policies/"
```

3. 确认 `transformers` 的 Gemma 补丁可以应用：

```bash
cd /path/to/conda/envs/lerobot/lib/python3.10/site-packages/transformers/models/gemma
git apply --check -p1 "$INFER_WITH_OM/lib/modeling_gemma.patch"
git apply -p1 "$INFER_WITH_OM/lib/modeling_gemma.patch"
```

如果 `git apply --check` 失败，不建议继续导出。应先确认当前 `transformers` 版本和 patch 适配关系。

## 5. Hugging Face 模型下载或 gated model 问题

现象：

- 下载 `google/paligemma-3b-pt-224` 超时、403 或卡住。
- 离线环境中启动时仍尝试访问 Hugging Face。

处理：

- 先在网络可用机器上下载模型，再把模型目录拷贝到运行环境。
- 如果必须走镜像，可临时设置 `HF_ENDPOINT=https://hf-mirror.com`。
- `google/paligemma-3b-pt-224` 是 gated model，需要提前在 Hugging Face 页面完成访问授权。
- 离线环境中，将 LeRobot 代码里硬编码的 `google/paligemma-3b-pt-224` 替换成本地路径，README 中已列出涉及位置。

建议优先使用本地路径，减少导出和验证过程中的网络变量。

## 6. action expert 导出缺少 `runtime_save`

现象：

- `convert_verify_onnx_action_expert.py` 报 `past_kv_tensor.pth` 或 `prefix_pad_masks.pth` 不存在。
- 单独运行 action expert 导出失败。

原因：

Pi0 OM 链路拆成两段：

- VLM 导出阶段会生成 `runtime_save/past_kv_tensor.pth` 和 `runtime_save/prefix_pad_masks.pth`。
- action expert 导出阶段需要读取这两个文件，才能复用与 VLM 一致的中间特征和 mask。

处理：

```bash
cd manipulation/pi0/infer_with_om

./run_pi0_export.sh \
  --pretrained-policy-path /path/to/pi0_model \
  --runtime-save-dir runtime_save

ls runtime_save/past_kv_tensor.pth
ls runtime_save/prefix_pad_masks.pth
```

如果需要分步运行，应先执行 `convert_verify_onnx_vlm.py`，再执行 `convert_verify_onnx_action_expert.py`，并确保两步使用相同的 `--runtime-save-dir`。

## 7. ONNX 导出成功但 ATC 转换失败

现象：

- ATC 编译报算子、shape、dtype 或 `soc_version` 相关错误。
- OM 生成后执行时报 `acl.mdl.execute error 507011`。

排查顺序：

1. 确认设备型号：

```bash
npu-smi info
```

2. 确认 ATC 使用的 `--soc_version` 与设备一致。例如设备是 310P1，则使用：

```bash
--soc_version=Ascend310P1
```

3. 保持 README 中的精度配置：

```bash
--precision_mode_v2=origin
```

4. 确认 ONNX 文件路径和输出名。ATC 的 `--output=outputs/om/pi0_vlm` 会生成 `outputs/om/pi0_vlm.om`，后续脚本应使用实际生成的文件名。

如果更换了模型、输入分辨率或动作维度，需要重新导出 ONNX，再重新 ATC 转换。不要复用旧 OM。

## 8. ACL / AclLite 依赖找不到

现象：

- `import acl` 失败。
- `acllite_utils` 或 `acllite_model` 找不到。
- OM 校验脚本启动即失败。

处理：

```bash
source /path/to/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/path/to/Ascend/ascend-toolkit/latest/thirdpart/python/:$PYTHONPATH
```

如果路径不确定，可在机器上查找：

```bash
find /path/to/Ascend -name acllite_utils.py
```

`PYTHONPATH` 应指向 `acllite_utils.py` 所在目录的上一级或实际可 import 的目录。

## 9. ONNX 与 OM 精度不一致

现象：

- `verify_om_onnx_vlm.py` 或 `verify_om_onnx_action_expert.py` 输出较大的 `max_abs_error`、`mean_abs_error`。
- `prefix_pad_masks mismatches` 不为 0。
- OM 端到端输出动作明显异常。

优先检查 dtype 和输入顺序：

- VLM 输入中，图像和 state 使用 `float32`，`lang_tokens` 使用 `int64`，`lang_masks` 使用 bool。
- action expert 输入中，state、`past_kv_tensor`、`time`、`noise` 使用 `float16`，tokens 使用 `int64`，mask 使用 bool。
- OM 输入顺序必须与 ONNX input order 一致。验证脚本会先读取 ONNX input order，再按相同顺序喂给 OM。

推荐先分别验证两段：

```bash
python3 verify_om_onnx_vlm.py \
  --onnx-model-path outputs/onnx/pi0-vlm.onnx \
  --om-model-path outputs/om/pi0_vlm.om

python3 verify_om_onnx_action_expert.py \
  --onnx-model-path outputs/onnx/pi0-action_expert.onnx \
  --om-model-path outputs/om/pi0_action_expert.om \
  --past-kv-path runtime_save/past_kv_tensor.pth \
  --prefix-mask-path runtime_save/prefix_pad_masks.pth
```

如果 VLM 精度正常而 action expert 异常，重点检查 `runtime_save` 是否来自同一次 VLM 导出，以及 action expert 的 `--lang-len`、`--noise-shape` 是否与导出时一致。

## 10. 端到端动作输出异常

现象：

- `run_om_e2e.py` 可以运行，但动作范围明显异常。
- 仿真中机器人动作抖动、幅度不合理或无法完成任务。

常见原因：

- 使用自定义模型时没有提供对应的动作归一化统计量。
- `output_features.action.shape` 与模型训练时不一致。
- 动作维度从 32 截断到 14 时，与目标机器人自由度映射不一致。

处理：

如果使用 README 中示例模型，可直接使用脚本内默认 mean/std。若使用自定义模型，应提供训练时保存的动作统计量：

```bash
python ./run_om_e2e.py \
  --mean-path /path/to/mean.pt \
  --std-path /path/to/std.pt \
  --vlm-model-path ./outputs/om/pi0_vlm.om \
  --action-expert-model-path ./outputs/om/pi0_action_expert.om
```

若更换机器人或动作维度，不应只依赖默认的 `_to_action_14` 截断逻辑，需要按实际自由度和 action key 做映射检查。

## 11. 仿真评测环境问题

现象：

- MuJoCo / Aloha 环境无法启动。
- 无显示环境下渲染失败。
- 评测脚本尝试联网下载模型或环境依赖。

处理：

```bash
export MUJOCO_GL=osmesa
```

如系统缺少 OSMesa：

```bash
sudo apt-get install libosmesa6 libosmesa6-dev
```

评测时尽量使用本地模型路径，并提前准备依赖。无 GPU 渲染需求时，OSMesa 是更稳定的选择。

## 12. 推荐排查顺序

遇到问题时，不建议直接跑完整仿真，应按以下顺序缩小范围：

1. 检查 LeRobot commit、`transformers` 补丁和 Pi0 适配文件。
2. 使用本地模型路径加载 `PI0Policy.from_pretrained`。
3. 运行 VLM ONNX 导出，确认 `runtime_save` 生成。
4. 运行 action expert ONNX 导出，确认 ONNXRuntime 对比通过。
5. 使用 ATC 分别生成 VLM 和 action expert OM。
6. 分别运行 ONNX vs OM 验证脚本。
7. 运行 `run_om_e2e.py` 做端到端动作输出检查。
8. 最后再接入 `eval_pi0_ascend.py` 或实际仿真环境。`eval_pi0_ascend.py` 基于 LeRobot eval 流程接入 VLM OM 与 action expert OM，可用于 Aloha 仿真评测；其他仿真或实机闭环需要按环境接口进一步适配。

这样可以把问题定位到“环境/权重加载”、“ONNX 导出”、“ATC 编译”、“OM Runtime”或“动作后处理”中的具体一段，避免在端到端仿真里混合排查。
