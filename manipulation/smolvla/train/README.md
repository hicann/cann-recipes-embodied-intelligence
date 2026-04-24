# SmolVLA 在昇腾 Atlas A2 上的训练样例

本目录提供 SmolVLA 在 LIBERO 数据集上的昇腾训练样例，包含环境初始化、分布式训练、评测启动脚本以及 Ascend 侧专用 patch。

当前样例基于以下原则整理：
- `cann-recipes` 仓库仅保存 recipe、配置、文档和补丁；
- `lerobot` 作为外部依赖仓单独 clone，并固定到已验证 commit；
- 通用 Ascend 训练补丁与 SmolVLA 专属补丁分离管理；
- 数据集、基础模型和 VLM 权重均通过工作区相对路径引用。

## 1. 适用场景
- 硬件：昇腾 Atlas A2
- CANN：8.3.0 及以上
- 任务：LIBERO 离线训练
- 数据集：`HuggingFaceVLA/libero`
- 外部训练框架：`huggingface/lerobot`

## 2. 外部依赖与固定版本
本样例不内嵌 `lerobot` 源码，默认使用如下 commit：

```text
58f70b6bd370864139a3795ac3497a9eae8c42d5
```

## 3. 目录说明

```text
manipulation/smolvla/train/
├── README.md
├── doc/
│   └── README.md
└── src/
    ├── configs/
    │   ├── smolvla_libero.yaml
    │   └── smolvla_libero_smoke.yaml
    ├── patches/
    │   ├── lerobot_ascend_train_common.patch
    │   └── lerobot_smolvla_ascend.patch
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

```bash
chmod +x manipulation/smolvla/train/src/scripts/setup.sh
./manipulation/smolvla/train/src/scripts/setup.sh
```

该脚本会：
- 在 `cann-recipes` 同级目录下准备 `lerobot` 代码仓；
- checkout 到固定 commit `58f70b6bd370864139a3795ac3497a9eae8c42d5`；
- 应用通用 Ascend 训练补丁与 SmolVLA 专属补丁；
- 安装 LeRobot 通用依赖、LIBERO 训练依赖和 SmolVLA 依赖；
- 默认复用当前已激活环境中的 `torch` / `torch_npu`；
- 支持 `--create-conda` 与本地 wheel 注入的初始化方式。

说明：
- `setup.sh` 默认通过 `git clone https://github.com/huggingface/lerobot.git` 拉取上游 `lerobot`；
- 若处于企业内网或受限网络环境，无法直接访问 GitHub，可提前在工作区同级目录手动准备 `lerobot/`，或改用内部镜像 / 代理源，再重新执行 `setup.sh`；
- 若 `LEROBOT_ROOT` 已存在且是有效 git 仓库，脚本会直接复用该目录。
- 若检测到本地 `lerobot` 工作树已被其他改动覆盖、补丁无法干净应用，`setup.sh` 会直接报错退出，避免在未完成 Ascend 适配的状态下继续训练。

常见用法：

```bash
./manipulation/smolvla/train/src/scripts/setup.sh --help
./manipulation/smolvla/train/src/scripts/setup.sh
./manipulation/smolvla/train/src/scripts/setup.sh \
  --create-conda \
  --env-name lerobot-smolvla \
  --python-version 3.10 \
  --torch-wheel /path/to/torch.whl \
  --torchvision-wheel /path/to/torchvision.whl \
  --torch-npu-wheel /path/to/torch_npu.whl
```

### 4.3 推荐工作区布局

```text
<workspace>/
├── cann-recipes-embodied-intelligence/
├── lerobot/
├── dataset/
│   └── HuggingFaceVLA/
│       └── libero/
├── models/
│   ├── HuggingFaceTB/
│   │   └── SmolVLM2-500M-Video-Instruct/
│   └── lerobot/
│       └── smolvla_base/
└── ckpt/
```

其中：
- `dataset/HuggingFaceVLA/libero` 需要直接指向包含 `data/`、`meta/` 的数据集根目录；
- `models/lerobot/smolvla_base` 需要是本地可直接加载的 SmolVLA base 模型目录；
- `models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 需要是本地可直接加载的 VLM 权重目录。

注意：`dataset`、`models`、`ckpt` 这三个目录需要自行创建和填充，本仓库只负责训练脚本和配置。

## 5. 训练配置
### 5.0 YAML 格式约束
`run_train.sh` 会读取配置里的 `output_dir` 与 `job_name` 并自动追加时间戳。当前脚本仅支持解析这两个字段的“单行简单键值”写法，因此这里不是建议，而是必须遵守，例如：

```yaml
output_dir: ../ckpt/smolvla_libero
job_name: smolvla_libero
```

约束要求：
- 必须将 `output_dir` / `job_name` 写成顶层单行简单键值；
- 不要将 `output_dir` / `job_name` 写成多行值（如 `|` 或 `>` 块标量）；
- 不要把它们写到嵌套结构里；
- 不要使用 YAML 锚点引用（`&anchor` / `*alias`）来定义这两个字段；
- 可以保留单行行内注释，但建议避免复杂 YAML 展开写法。

若配置不满足上述格式，`run_train.sh` 会直接报错并退出。

### 5.1 smoke 配置
- 配置文件：`src/configs/smolvla_libero_smoke.yaml`
- 作用：验证本地模型目录、空相机补齐以及多卡训练链路
- 关键参数：
  - `steps: 20`
  - `batch_size: 32`
  - `num_workers: 0`
  - `wandb.enable: false`

启动：

```bash
./manipulation/smolvla/train/src/scripts/run_train.sh smolvla_libero_smoke --port 29510
```

### 5.2 长训配置
- 配置文件：`src/configs/smolvla_libero.yaml`
- 关键参数：
  - `batch_size: 32`
  - `num_workers: 4`
  - `steps: 1600`
  - `policy.empty_cameras: 1`
  - `policy.pretrained_path` / `policy.vlm_model_name` 均使用本地目录

启动：

```bash
./manipulation/smolvla/train/src/scripts/run_train.sh smolvla_libero --port 29510
```

说明：
- 当前长训配置按官方 `100k steps @ batch_size=4` 的样本预算换算；
- 在 8 卡、每卡 `batch_size=32` 时，全局 batch 为 `256`，对应约 `1563` steps；
- 样例中取整为 `1600` steps，便于统一管理；
- 当前已验证在 910b 上可进一步放开到 `num_workers=4`，且不会成为主要瓶颈；
- 当前训练配置默认不带 `rename_map`，保留 `policy.empty_cameras: 1`；若自有数据集键名不一致，可按 `doc/README.md` 的说明手动补充映射。

## 6. 评测说明
`run_eval.sh` 对 `lerobot-eval` 做了轻量封装，并默认设置：
- `MUJOCO_GL=osmesa`
- 仿真在 CPU 侧执行，policy 在 NPU 侧执行
- 当传入 `--policy.device=npu` / `--policy.device=npu:x` 时，脚本会在启动前同步设置 `LEROBOT_EVAL_NPU_DEVICE`，避免评测脚本内部硬编码设备号。

若需要在 CUDA 主机上对训练出的 checkpoint 做 LIBERO 在线评测，可参考以下命令：

```bash
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

REPO=<workspace>/lerobot
EVAL_BIN=$(command -v lerobot-eval)
CKPT=<workspace>/ckpt/smolvla_libero/checkpoints/100000/pretrained_model
VLM=<workspace>/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
OUT_ROOT=<workspace>/evals/smolvla_libero_$(date +%Y%m%d_%H%M%S)

cd "$REPO"

for suite in libero_spatial libero_object libero_goal libero_10; do
  "$EVAL_BIN" \
    --policy.path="$CKPT" \
    --policy.device=cuda \
    --policy.vlm_model_name="$VLM" \
    --env.type=libero \
    --env.task="$suite" \
    --seed=3000 \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --output_dir="$OUT_ROOT/$suite"
done
```

## 7. 已验证结果摘要
- 910b 上已完成 smoke 和按预算对齐的正式训练启动；
- 训练链路已验证的关键配置为：
  - `empty_cameras=1`
  - 本地 `smolvla_base`
  - 本地 `SmolVLM2-500M-Video-Instruct`
  - `num_workers=4`
  - 当前 910b 训练口径不带 `rename_map`
- 训练吞吐对比：

| 平台 | 训练硬件 | 训练配置 | 全局 batch | 稳定阶段 `updt_s` | 训练吞吐 |
| --- | --- | --- | --- | --- | --- |
| 910b | 昇腾 Atlas A2 `8` 卡 | 每卡 `batch_size: 32` | `256` | `1.05 ~ 1.10s` | `233 ~ 244 samples/s` |
| h20 | `2` 卡 | 每卡 `batch_size: 8` | `16` | `0.176 ~ 0.181s` | `88.4 ~ 90.9 samples/s` |

- 910b 训练吞吐参考：
  - 训练硬件：昇腾 Atlas A2 `8` 卡
  - 训练配置：每卡 `batch_size: 32`，全局 batch `256`
  - 稳定阶段 `updt_s`：约 `1.05 ~ 1.10s`
  - 对应训练吞吐：约 `233 ~ 244 samples/s`
- 基于 `100000` step checkpoint、`seed=3000`、`eval.n_episodes=1` 的一轮 LIBERO 在线评测结果：
  - `libero_spatial`: `90.0%`
  - `libero_object`: `100.0%`
  - `libero_goal`: `90.0%`
  - `libero_10`: `80.0%`
  - 平均成功率：`90.0%`

说明：
- 上述 `samples/s` 为总吞吐，不是单卡吞吐；
- `910b` 吞吐来自 `8卡 x 每卡32` 的稳定阶段日志换算；
- 当前不同仓库版本对 `rename_map` 的需要不完全一致，建议以样例目录内配置与日志结论为准。

## 8. W&B 记录占位
![image.png](https://raw.gitcode.com/user-images/assets/8784759/f580c994-11ee-4594-aa00-c9d436ec99ed/image.png 'wandb.png')

## 9. 常用命令
### 查看训练日志

```bash
# 以下路径以推荐工作区中的 `cann-recipes-embodied-intelligence/` 目录为起点
cd ../lerobot
tail -f ../ckpt/logs/train_smolvla_*.log
```

### resume 训练

```bash
./manipulation/smolvla/train/src/scripts/run_train.sh smolvla_libero --resume --port 29510
```

## 10. 相关说明
- 当前样例目录不包含 `lerobot` 源码；
- 详细的迁移原因、关键问题和配置建议见：[doc/README.md](doc/README.md)；
- 若后续 `lerobot` 上游吸收了相关修复，可考虑缩减 `src/patches` 中的补丁范围。
