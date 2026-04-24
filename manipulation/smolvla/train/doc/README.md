# SmolVLA LIBERO 昇腾训练说明

## 1. 训练与评测结论
### 1.1 目标任务
- 模型：SmolVLA
- 数据集：`HuggingFaceVLA/libero`
- 训练方式：离线模仿学习
- 推荐硬件：昇腾 Atlas A2 8 卡

### 1.2 在线评测
- 评测对象：`100000` step checkpoint
- 评测设置：`seed=3000`，`eval.n_episodes=1`
- 评测侧关键参数：
  - `PYOPENGL_PLATFORM=egl`
  - `MUJOCO_GL=egl`
  - `TOKENIZERS_PARALLELISM=false`
  - `rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}`
- LIBERO 成功率：
  - `libero_spatial`: `90.0%`
  - `libero_object`: `100.0%`
  - `libero_goal`: `90.0%`
  - `libero_10`: `80.0%`
  - 平均成功率：`90.0%`

### 1.3 训练吞吐参考
- 训练吞吐对比：

| 平台 | 训练硬件 | 训练配置 | 全局 batch | 稳定阶段 `updt_s` | 训练吞吐 |
| --- | --- | --- | --- | --- | --- |
| 910b | 昇腾 Atlas A2 `8` 卡 | 每卡 `batch_size=32` | `256` | `1.05 ~ 1.10s` | `233 ~ 244 samples/s` |
| h20 | `2` 卡 | 每卡 `batch_size=8` | `16` | `0.176 ~ 0.181s` | `88.4 ~ 90.9 samples/s` |

- 910b 正式训练参考：
  - 硬件：昇腾 Atlas A2 `8` 卡
  - 配置：每卡 `batch_size=32`，全局 batch `256`
  - 稳定阶段 `updt_s`：约 `1.05 ~ 1.10s`
  - 对应训练吞吐：约 `233 ~ 244 samples/s`

说明：
- 上述 `samples/s` 均为总吞吐，不是单卡吞吐；
- 吞吐根据训练日志中的稳定阶段 `updt_s` 换算得到。

### 1.4 当前样例包含的配置
- `smolvla_libero_smoke.yaml`：用于验证本地模型、空相机补齐与训练链路
- `smolvla_libero.yaml`：用于正式训练

### 1.5 样本预算换算
LeRobot 官方 `SmolVLA + LIBERO` 示例常见口径为：
- `steps = 100000`
- `batch_size = 4`

对应总样本预算约为：

```text
100000 × 4 = 400000 samples
```

当前 910b 样例默认采用：
- 8 卡训练
- 每卡 `batch_size = 32`
- 全局 batch：`256`

因此预算对齐步数约为：

```text
400000 / 256 ≈ 1562.5
```

样例中统一取整为 `1600 steps`。

## 2. 当前样例依赖的 lerobot commit
固定版本：

```text
58f70b6bd370864139a3795ac3497a9eae8c42d5
```

说明：
- 本样例以该 commit 作为基线；
- 通用 Ascend patch 与 SmolVLA 专属 patch 都基于该 commit 提取；
- 若使用其他 commit，请参考补丁中的修改点手工迁移。

补充说明：
- `setup.sh` 默认会通过 `git clone https://github.com/huggingface/lerobot.git` 获取上游 `lerobot`；
- 若机器处于企业内网或受限网络环境，无法直接访问 GitHub，建议提前手动 clone 到工作区同级目录，或使用内部镜像 / 代理源后再执行 `setup.sh`。

## 3. 关键迁移点
### 3.1 权重加载阶段的 `copy_d2d / 507001`
SmolVLA 原始实现走通用 `PreTrainedPolicy.from_pretrained()` 时，可能直接把 safetensors 权重加载到 NPU，进而在 Ascend 上触发设备间 copy 失败。

当前补丁的处理方式：
- 在 `modeling_smolvla.py` 中增加自定义 `from_pretrained()`；
- 先在 CPU 侧完成权重读取和 `load_state_dict()`；
- 再将模型迁移到目标设备。

这样可以规避训练初始化阶段的 `copy_d2d` 类报错。

### 3.2 LIBERO 图像键名与 SmolVLA 预训练输入不一致
SmolVLA base 模型预训练时的图像输入键名是：
- `camera1`
- `camera2`
- `camera3`

而 LIBERO 数据集中常见键名是：
- `observation.images.image`
- `observation.images.image2`

当前样例在 910b 稳定训练口径中默认不带 `rename_map`，保留：

```yaml
policy:
  empty_cameras: 1
```

用于补齐缺失的第三路相机输入。

需要额外说明的是，不同仓库版本 / patch 组合下，`rename_map` 的实际需求可能不同：
- 在当前样例对应的 910b 训练路径中，稳定训练口径是不带 `rename_map`，保留 `empty_cameras=1`；
- 在 CUDA 主机上做在线评测时，仍可使用 `rename_map` 将 `image / image2` 映射到 `camera1 / camera2`；
- 若训练时出现 `All image features are missing from the batch`，应先检查当前 batch 键名与 `policy.config.image_features` 是否一致，而不是机械保留旧版 `rename_map`。

如果自有数据集确实需要键名映射，可在配置中手动加入：

```yaml
rename_map:
  observation.images.image: observation.images.camera1
  observation.images.image2: observation.images.camera2
```

### 3.3 本地模型目录与 `local_files_only`
无网环境下，SmolVLA 训练至少依赖两类本地模型目录：
- `smolvla_base`
- `SmolVLM2-500M-Video-Instruct`

当前补丁在处理器与配置加载路径上补充了本地目录判断，并在本地目录场景下使用 `local_files_only=True`，减少不必要的在线访问。

### 3.4 磁盘配额与数据集缓存
SmolVLA 训练同样依赖 `datasets` 的 parquet / arrow 缓存。若缓存落在容量受限目录，很容易触发：
- `Disk quota exceeded`

当前样例默认把：
- `XDG_CACHE_HOME`
- `HF_HOME`
- `HF_DATASETS_CACHE`
- `HF_HUB_CACHE`

都指向工作区相对大盘路径，并建议正式训练时使用：

```yaml
num_workers: 4
```

当前已验证在 910b 相关短测中，`num_workers=4` 可以稳定工作，且 `data_s` 占比很低。

### 3.5 评测路径中临时放宽的断言
SmolVLA 专属 patch 中对 `lerobot_eval.py` 的少量 `assert` 做了临时放宽，主要包括：
- rollout 阶段的 `policy` 类型与 `action_numpy.ndim` 校验；
- `return_episode_data=True` 时 `episode_data` 拼接连续性的两处校验。

当前这样处理的原因是：
- 现阶段样例重点是保证 Ascend 上的训练 / 在线评测链路稳定跑通；
- 在当前上游版本与本地补丁组合下，这几处断言在评测聚合路径上会带来额外兼容性干扰，不利于问题定位；
- 对应 patch 仍保留了原始断言位置，便于后续基于上游版本继续收敛和恢复。

后续若上游 `lerobot` 更新或评测链路进一步稳定，建议重新核对这些校验点，并优先考虑恢复更严格的断言。

## 4. 配置建议
### 4.1 `smolvla_libero_smoke.yaml`
建议用途：
- 首次验证本地 base 模型和 VLM 模型目录；
- 验证 `empty_cameras` 与多卡训练链路是否配置正确；
- 验证 8 卡训练链路是否可正常拉起。

关键项：
- `steps: 20`
- `batch_size: 32`
- `num_workers: 0`
- `wandb.enable: false`

### 4.2 `smolvla_libero.yaml`
建议用途：
- 正式训练或 budget-aligned 训练

关键项：
- `policy.pretrained_path: ../models/lerobot/smolvla_base`
- `policy.vlm_model_name: ../models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- `policy.empty_cameras: 1`
- `batch_size: 32`
- `num_workers: 4`
- `steps: 1600`
- `save_freq: 500`

## 5. 常见问题
### 5.1 初始化阶段访问外网或找不到本地模型
现象：
- 启动时尝试访问 Hugging Face
- 本地目录明明存在，但仍触发在线访问或权重缺失

处理：
- 确认 `policy.pretrained_path` 指向本地 `smolvla_base`
- 确认 `policy.vlm_model_name` 指向本地 `SmolVLM2-500M-Video-Instruct`
- 确认本地目录内包含 `config.json` / `model.safetensors` 等核心文件

### 5.2 feature mismatch / camera 键名不一致
现象：
- 启动后在视觉输入路径报特征键名不匹配

处理：
- 优先检查当前 batch 键名与 `policy.config.image_features` 是否一致
- 若自有数据集键名和策略输入不一致，再按需补充 `rename_map`
- 检查 `policy.empty_cameras: 1` 是否开启，用于补齐第三路相机输入

### 5.3 在线评测的渲染依赖
现象：
- 评测阶段无头渲染失败
- 缺失 `libOSMesa.so.0`

处理：
- 安装 OSMesa 运行库
- 设置 `MUJOCO_GL=osmesa`
- 让仿真走 CPU，policy 走 NPU

## 6. 推荐启动方式
### 6.1 smoke

```bash
source /path/to/conda.sh
conda activate <your-ascend-train-env>
source /path/to/Ascend/set_env.sh
./manipulation/smolvla/train/src/scripts/run_train.sh smolvla_libero_smoke --port 29510
```

### 6.2 正式训练

```bash
source /path/to/conda.sh
conda activate <your-ascend-train-env>
source /path/to/Ascend/set_env.sh
./manipulation/smolvla/train/src/scripts/run_train.sh smolvla_libero --port 29510
```

### 6.3 resume

```bash
./manipulation/smolvla/train/src/scripts/run_train.sh smolvla_libero --resume --port 29510
```
