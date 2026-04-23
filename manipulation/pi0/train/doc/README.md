# PI0 LIBERO 昇腾训练说明

## 1. 训练与评测结论
### 1.1 Atlas A2 训练
- 模型：PI0
- 数据集：`HuggingFaceVLA/libero`
- 训练方式：离线模仿学习
- 推荐硬件：昇腾 Atlas A2 8 卡
- 长训配置：`pi0_libero.yaml`
- 训练步数：`30000`
- 结论：可在昇腾 Atlas A2 集群上 8 卡并行进行 PI0 模型训练
- 训练吞吐对比：

| 场景 | 平台 | 配置 | 全局 batch | 统计口径 / 优化项 | 吞吐 |
| --- | --- | --- | --- | --- | --- |
| smoke 稳定段 | Atlas A2 `8` 卡 | 每卡 `batch_size=16`，`steps=100` | `128` | `samples/s = global_batch / (train/update_s + train/dataloading_s)` | `66.86 samples/s` |
| smoke 稳定段（纯计算） | Atlas A2 `8` 卡 | 每卡 `batch_size=16`，`steps=100` | `128` | `global_batch / train/update_s` | `67.59 samples/s` |
| 优化对比基线 | Atlas A2 `8` 卡 | 每卡 `batch_size=24`，`steps=30` | `192` | baseline | `66 samples/s` |
| 优化对比 | Atlas A2 `8` 卡 | 每卡 `batch_size=24`，`steps=30` | `192` | `NPU fusion attention` | `71.11 samples/s` |
| 优化对比 | Atlas A2 `8` 卡 | 每卡 `batch_size=24`，`steps=30` | `192` | `disable outer suffix checkpoint` | `77.64 samples/s` |
| 优化对比 | Atlas A2 `8` 卡 | 每卡 `batch_size=24`，`steps=30` | `192` | 两者同时开启 | `81.77 samples/s` |

- 吞吐参考（smoke 验证，`batch_size=16` 每卡，全局 batch `128`）：
  - `steps=100`
  - 稳定阶段（`step>=10`，100 step debug）平均吞吐：`66.86 samples/s`
  - 对应纯计算吞吐：`67.59 samples/s`
  - 统计口径：`samples/s = global_batch / (train/update_s + train/dataloading_s)`
- 训练优化验证（Atlas A2，8 卡，`batch_size=24` 每卡，全局 batch `192`，`steps=30`）：
  - 基线：约 `66 samples/s`
  - `NPU fusion attention`：约 `71.11 samples/s`
  - `disable outer suffix checkpoint`：约 `77.64 samples/s`
  - 二者同时开启：约 `81.77 samples/s`
  - 相比基线端到端提升约 `24%`

### 1.2 当前归档评测结果
- 评测口径：四个 suite 共 `40` 个 case（每个 task 跑 `1` 次）
- 聚合成功率：`92.5%`（`37/40`）

### 1.3 当前样例包含的配置
- `pi0_libero_smoke.yaml`：用于验证链路
- `pi0_libero.yaml`：用于正式长训

## 2. 当前样例依赖的 lerobot commit
固定版本：

```text
58f70b6bd370864139a3795ac3497a9eae8c42d5  # 2025-11-27
```

说明：
- 本样例以该 commit 作为基线；
- 通用 Ascend patch 与 PI0 专属 patch 都基于该 commit 提取；
- 若使用其他 commit，可参考补丁中的修改点进行手工迁移。

## 3. 关键迁移点
### 3.1 本地化预训练模型与 tokenizer
PI0 训练会访问两类外部模型：
- `pi0_base`
- `google/paligemma-3b-pt-224`

在无网或镜像不稳定环境下，建议提前准备本地目录，并按推荐工作区布局放置。

配置位置：
- `policy.pretrained_path` 在 YAML 中指定本地 `pi0_base`；
- `LEROBOT_PI0_TOKENIZER_PATH` 在运行脚本中指定本地 `paligemma-3b-pt-224`。

### 3.2 NPU 不支持 INT64 `cumsum`
PI0 原始实现里，attention mask 和 `position_ids` 的构造路径会触发 `torch.cumsum`。在 Ascend 上，如果输入或输出走 `int64`，可能报错。

当前补丁的处理方式：
- `att_masks` 统一转成 `torch.int32`；
- `torch.cumsum(..., dtype=torch.int32)`；
- `pad_masks` 相关逻辑统一转成 `bool` 后再做 `logical_and`。

这样可以规避以下典型报错：
- `aclnnCumsum failed ... DT_INT64`
- `aclnnBitwiseAndTensor failed ... self not implemented for DT_FLOAT`

### 3.3 PI0 fused attention 迁移方式
PI0 的 fused attention 不能直接用全局 monkey-patch `Gemma eager_attention_forward` 的方式替换。当前补丁采用局部接法：
- 在 PI0 内部增加 `_repeat_kv_heads`，先将 `key/value` 扩展到与 `query` 相同的 head 数；
- 增加 `_npu_fusion_attention_forward`，只在 PI0 的 `compute_layer_complete` 调用点切换到 `torch_npu.npu_fusion_attention`；
- 通过 `use_npu_fusion_attention` 实例级开关控制，不影响其他 Gemma 调用路径。

这一做法已在 8 卡 Atlas A2 上验证可稳定跑通，并取得额外训练吞吐收益。

### 3.4 评测推荐 CPU 仿真 + NPU 推理
- `MUJOCO_GL=osmesa`
- `--policy.device=npu`

也就是：
- 仿真与渲染在 CPU 侧执行；
- policy 前向推理继续放在 NPU。

## 4. 配置建议
在当前样例中，`run_train.sh` 会直接从 YAML 里解析 `output_dir` 和 `job_name`。为避免解析歧义，建议这两个字段遵循以下写法：

```yaml
output_dir: ../ckpt/pi0_libero
job_name: pi0_libero
```

约束：
- 保持单行简单键值格式；
- 避免多行值；
- 避免嵌套到其他 YAML 结构中；
- 如需注释，优先使用单行行内注释。

说明：
- 当前脚本已支持过滤 `output_dir` / `job_name` 上的单行行内注释；
- 后续若需要进一步提升复杂 YAML 场景下的健壮性，可考虑改为使用 Python YAML 解析库替代当前 `awk` 方案。

### 4.1 `pi0_libero_smoke.yaml`
建议用途：
- 首次拉起环境；
- 验证本地数据集路径和模型路径；
- 验证 8 卡分布式训练脚本是否可正常启动。

关键项：
- `batch_size: 16`（每卡，8 卡时全局 batch `128`）
- `steps: 20`
- `wandb.enable: false`
- `save_freq: 20`

### 4.2 `pi0_libero.yaml`
建议用途：
- 长训或 resume 训练

关键项：
- `policy.dtype: bfloat16`
- `policy.gradient_checkpointing: true`
- `policy.compile_model: false`
- `batch_size: 8`（每卡；全局 batch = `batch_size × 卡数`）
- `steps: 30000`

`run_train.sh` 默认会导出以下已验证优化开关：
- `LEROBOT_PI0_USE_FAST_BETA_SAMPLE=1`
- `LEROBOT_PI0_USE_NPU_FUSION_ATTENTION=1`
- `LEROBOT_PI0_DISABLE_OUTER_SUFFIX_CHECKPOINT=1`
- `save_freq: 10000`

## 5. 常见问题
### 5.1 `pi0_base` 或 `paligemma` 在线下载失败
现象：
- 启动阶段尝试访问 Hugging Face 或镜像站
- 无网环境中出现超时、403 或长时间重试

处理：
- 提前将 `pi0_base` 和 `paligemma-3b-pt-224` 下载到本地；
- 按推荐工作区布局放到 `../models/...`；
- 训练前确认 `policy.pretrained_path` 和 `LEROBOT_PI0_TOKENIZER_PATH` 指向本地目录。

### 5.2 `torch_npu` / 平台栈没有准备好
现象：
- `setup.sh` 末尾提示无法 `import torch, torch_npu`；
- 或训练启动时直接报 NPU 依赖缺失。

处理：
- 方案 1：先激活一个已验证可用的 Ascend 训练环境，再执行 `setup.sh`；
- 方案 2：执行 `setup.sh` 时传入本地 wheel：

```bash
./manipulation/pi0/train/src/scripts/setup.sh \
  --create-conda \
  --env-name lerobot-pi0 \
  --python-version 3.10 \
  --torch-wheel /path/to/torch.whl \
  --torchvision-wheel /path/to/torchvision.whl \
  --torch-npu-wheel /path/to/torch_npu.whl
```

### 5.4 `libOSMesa.so.0` 缺失
现象：
- 评测阶段提示找不到 `libOSMesa.so.0`

处理：
- 需要安装 OSMesa 相关运行库；
- 常见包名为 `libosmesa6`、`libosmesa6-dev` 或发行版同类包；
- 安装完成后重新设置 `MUJOCO_GL=osmesa` 再执行评测。

## 6. 推荐启动方式
### 6.1 smoke

```bash
source /path/to/conda.sh
conda activate <your-ascend-train-env>
source /path/to/Ascend/set_env.sh
./manipulation/pi0/train/src/scripts/run_train.sh pi0_libero_smoke --port 29510
```

### 6.2 正式训练

```bash
source /path/to/conda.sh
conda activate <your-ascend-train-env>
source /path/to/Ascend/set_env.sh
./manipulation/pi0/train/src/scripts/run_train.sh pi0_libero --port 29510
```

### 6.3 resume

```bash
./manipulation/pi0/train/src/scripts/run_train.sh pi0_libero --resume --port 29510
```
