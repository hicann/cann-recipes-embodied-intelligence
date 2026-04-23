# PI0 在昇腾 Atlas A2 上的训练样例

本目录提供 PI0 训练样例，基于 LeRobot 在 LIBERO 数据集上完成分布式训练，并提供配套的环境准备、训练启动和评测脚本。

当前样例遵循以下原则：
- `cann-recipes` 仓库中仅保存训练样例目录、配置、脚本、文档和补丁；
- `lerobot` 作为外部依赖仓单独 clone；
- `setup.sh` 固定 `lerobot` commit id，并依次应用通用 Ascend 训练补丁与 PI0 专属补丁；
- 数据集、预训练权重和缓存目录均通过工作区相对路径组织。

## 1. 适用场景
- 硬件：昇腾 Atlas A2
- CANN：8.3.0 及以上
- 任务：LIBERO 离线训练
- 数据集：`HuggingFaceVLA/libero`

## 2. 外部依赖与固定版本
本样例不内嵌 `lerobot` 源码，默认使用如下 commit：

```text
58f70b6bd370864139a3795ac3497a9eae8c42d5
```

## 3. 目录说明

```text
manipulation/pi0/train/
├── README.md
├── doc/
│   └── README.md
└── src/
    ├── configs/
    │   ├── pi0_libero.yaml
    │   └── pi0_libero_smoke.yaml
    ├── patches/
    │   ├── lerobot_ascend_train_common.patch
    │   └── lerobot_pi0_ascend.patch
    └── scripts/
        ├── run_eval.sh
        ├── run_train.sh
        └── setup.sh
```

## 4. 环境准备
### 4.1 clone 代码

```bash
git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git
cd cann-recipes-embodied-intelligence
```

### 4.2 准备 `lerobot`
执行：

```bash
chmod +x manipulation/pi0/train/src/scripts/setup.sh
./manipulation/pi0/train/src/scripts/setup.sh
```

该脚本会：
- 在 `cann-recipes` 同级目录下准备 `lerobot` 代码仓；
- checkout 到固定 commit `58f70b6bd370864139a3795ac3497a9eae8c42d5`；
- 应用通用 Ascend 训练补丁与 PI0 补丁；
- 安装 LeRobot 通用依赖、LIBERO 训练依赖和 PI0 所需的 `transformers` 分支；
- 已包含 `tqdm>=4.66.0,<5.0.0`，可覆盖通用训练补丁中新增的 tqdm 进度条依赖；
- 默认复用当前已激活环境中的 `torch` / `torch_npu`；
- 如需在新环境中执行，可通过参数创建 conda 环境，并通过本地 wheel 注入平台相关的 `torch` / `torchvision` / `torch_npu`。

常见用法：

```bash
# 查看脚本帮助
./manipulation/pi0/train/src/scripts/setup.sh --help

# 复用当前已经准备好的 Ascend 训练环境
./manipulation/pi0/train/src/scripts/setup.sh

# 创建新 conda 环境，并从本地 wheel 安装平台栈
./manipulation/pi0/train/src/scripts/setup.sh \
  --create-conda \
  --env-name lerobot-pi0 \
  --python-version 3.10 \
  --torch-wheel /path/to/torch.whl \
  --torchvision-wheel /path/to/torchvision.whl \
  --torch-npu-wheel /path/to/torch_npu.whl
```

说明：
- 平台栈建议由已有训练环境复用，或由使用者自行提供本地 wheel；
- 如已提前确认平台栈可用，也可以追加 `--skip-torch-check` 跳过末尾导入校验。
- whl 链接可参考 https://github.com/Ascend/pytorch

### 4.3 推荐工作区布局
当前配置默认使用工作区相对路径，推荐布局如下：

```text
<workspace>/
├── cann-recipes-embodied-intelligence/
├── lerobot/
├── dataset/
│   └── HuggingFaceVLA/
│       └── libero/
├── models/
│   ├── google/
│   │   └── paligemma-3b-pt-224/
│   └── lerobot/
│       └── pi0_base/
└── ckpt/
```

其中：
- `dataset/HuggingFaceVLA/libero` 需要直接指向包含 `data/`、`meta/` 的数据集根目录；
- `models/lerobot/pi0_base` 需要是本地可直接加载的 PI0 预训练模型目录；
- `models/google/paligemma-3b-pt-224` 需要是本地可直接加载的 PaliGemma tokenizer / processor 目录。模型来自 `https://huggingface.co/google/paligemma-3b-pt-224`

注意：`dataset`、`models`、`ckpt` 这三个目录需要自行创建和填充，本仓库只负责训练脚本和配置。

### 4.4 无网环境下的模型准备
PI0 训练至少依赖以下本地模型目录：
- `../models/lerobot/pi0_base`
- `../models/google/paligemma-3b-pt-224`

默认脚本会优先使用：

```text
../models/google/paligemma-3b-pt-224
```

对应的环境变量名为：

```bash
export LEROBOT_PI0_TOKENIZER_PATH=/path/to/paligemma-3b-pt-224
```

如果未设置该变量，且默认目录不存在，则 processor 会回退到 `google/paligemma-3b-pt-224` 在线仓库名称。

## 5. 训练配置
### 5.0 YAML 格式约束
`run_train.sh` 会用简单的 `awk` 逻辑读取配置里的 `output_dir` 与 `job_name`，因此这两个字段建议保持单行简单键值格式，例如：

```yaml
output_dir: ../ckpt/pi0_libero
job_name: pi0_libero
```

约束建议：
- 不要将 `output_dir` / `job_name` 写成多行值（如 `|` 或 `>` 块标量）；
- 不要把它们写到嵌套结构里；
- 不要使用 YAML 锚点引用（`&anchor` / `*alias`）来定义这两个字段；
- 可以保留单行行内注释，但建议避免复杂 YAML 展开写法。

说明：
- 当前脚本已支持过滤 `output_dir` / `job_name` 上的单行行内注释；
- 后续若需要进一步提升复杂 YAML 场景下的健壮性，可考虑改为使用 Python YAML 解析库替代当前 `awk` 方案。

### 5.1 smoke 配置
- 配置文件：`src/configs/pi0_libero_smoke.yaml`
- 作用：快速验证环境、依赖、数据集路径和多卡训练链路
- 关键参数：
  - `batch_size: 16`（每卡），8 卡时全局 batch `128`
  - `steps: 20`
  - `compile_model: false`
  - `wandb.enable: false`

启动：

```bash
./manipulation/pi0/train/src/scripts/run_train.sh pi0_libero_smoke --port 29510
```

### 5.2 长训配置
- 配置文件：`src/configs/pi0_libero.yaml`
- 关键参数：
  - `steps: 30000`
  - `batch_size: 8`（每卡；全局 batch = `batch_size × 卡数`）
  - `num_workers: 16`
  - `policy.dtype: bfloat16`
  - `policy.compile_model: false`
  - `wandb.enable: true`

启动：

```bash
./manipulation/pi0/train/src/scripts/run_train.sh pi0_libero --port 29510
```

说明：
- 样例默认面向 8 卡 Atlas A2；
- 如果使用更少卡数或更小显存的平台，请先下调 `batch_size` 和 `num_workers`。
- `run_train.sh` 默认开启以下已验证优化：
  - `LEROBOT_PI0_USE_FAST_BETA_SAMPLE=1`
  - `LEROBOT_PI0_USE_NPU_FUSION_ATTENTION=1`
  - `LEROBOT_PI0_DISABLE_OUTER_SUFFIX_CHECKPOINT=1`

## 6. 评测说明
`run_eval.sh` 是对 `lerobot-eval` 的轻量封装，参数直接透传。

在线 LIBERO 评测推荐：
- CPU 执行 MuJoCo 仿真与渲染；
- NPU 执行 policy 推理；
- 设置 `MUJOCO_GL=osmesa`。

示例：

```bash
export MUJOCO_GL=osmesa
./manipulation/pi0/train/src/scripts/run_eval.sh \
  --policy.path=/path/to/checkpoint/pretrained_model \
  --policy.device=npu \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=100 \
  --eval.batch_size=1 \
  --output_dir=/path/to/eval_out
```

## 7. 已验证结果摘要
当前已归档的一组参考结果：
- 训练吞吐对比：

| 场景 | 平台 | 优化项 | 吞吐 |
| --- | --- | --- | --- |
| 优化对比基线 | Atlas A2 `8` 卡  | baseline | `66 samples/s` |
| 优化对比 | Atlas A2 `8` 卡 | `NPU fusion attention` | `71.11 samples/s` |
| 优化对比 | Atlas A2 `8` 卡 | `disable outer suffix checkpoint` | `77.64 samples/s` |
| 优化对比 | Atlas A2 `8` 卡 | 两者同时开启 | `81.77 samples/s` |

- 训练任务：PI0 on `HuggingFaceVLA/libero`
- 评测方式：LIBERO 四个 suite 粗评
- 评测配置：`eval.batch_size=1`，`policy.n_action_steps=10`
- 评测口径：`4` 个 suite × 每个 suite `10` 个 task × 每个 task `1` 次，共 `40` 个 case
- 评测结果：
  - `libero_spatial`: `80.0%` (`8/10`)
  - `libero_object`: `100.0%` (`10/10`)
  - `libero_goal`: `100.0%` (`10/10`)
  - `libero_10`: `90.0%` (`9/10`)
  - 聚合结果：`37/40 = 92.5%`
  
libero_spatial任务示例

![spatial.gif](https://raw.gitcode.com/user-images/assets/8784759/f46d5b89-3c59-46d6-9852-e5697ed5631c/spatial.gif 'spatial.gif')

libero_object任务示例

![object.gif](https://raw.gitcode.com/user-images/assets/8784759/44e1e40a-aafc-46d0-bffd-139bdbf84d1c/object.gif 'object.gif')

libero_goal任务示例

![goal.gif](https://raw.gitcode.com/user-images/assets/8784759/e7a20895-5fe4-4b3f-8b0e-a4bb40b6d106/goal.gif 'goal.gif')

libero_10任务示例

![10.gif](https://raw.gitcode.com/user-images/assets/8784759/cc2f70fc-fa1f-43bd-b751-d3498080f086/10.gif '10.gif')

更详细的历史路径、wandb 链接和评测说明见：
- [doc/README.md](doc/README.md)

## 8. W&B 记录占位
![image.png](https://raw.gitcode.com/user-images/assets/8784759/7ee72d9f-05a8-4604-8c34-9bb698050f99/image.png 'wandb日志')

## 9. 常用命令
### 查看训练日志

```bash
# 以下路径以推荐工作区中的 `cann-recipes-embodied-intelligence/` 目录为起点
cd ../lerobot
tail -f ../ckpt/logs/train_pi0_libero_*.log
```

### resume 训练

```bash
./manipulation/pi0/train/src/scripts/run_train.sh pi0_libero --resume --port 29510
```

## 10. 常见问题索引
更详细的迁移说明、配置原因和问题处理建议见：[doc/README.md](doc/README.md)
