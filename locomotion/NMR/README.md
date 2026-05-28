# Neural Motion Retargeting

## 许可证与使用边界

本目录包含非商业和受限许可的第三方代码、依赖、模型和资产，因此 `locomotion/NMR` 作为整体不是 Apache-2.0-only 项目。除非文件头另有说明，本项目自研代码和 NMR 修改部分按 Apache-2.0 发布；文件级版权声明、SPDX 标识、第三方许可证以及 `THIRD_PARTY_LICENSES.md` 中列出的条款优先于目录级默认许可证。

使用者需要自行确认并遵守相关第三方代码、模型、数据和资产的许可条件。非商业或受限组件不得被理解为由本仓库授予商业使用或自由再分发权利。

特别地，`smplx` 以及本项目运行所需的 SMPL-X 模型文件受 SMPL-X / SMPLify-X 官方许可约束，通常仅限非商业科研用途。`smplx` 不作为默认依赖自动安装；相关模型、数据和软件不得被默认理解为可自由再分发；任何商业使用都需要由使用者向权利方另行取得授权。

将人类 SMPL-X 动作重定向到 Unitree G1 人形机器人的神经网络方法，面向昇腾引擎训练与部署。

## 架构概述

**LLaMA Transformer**（`RetargetTransformerPredMotionNoSMPLVQ`）

- 条件输入：SMPL-X → encoder → MLP → 嵌入 `(B, T/4, 512)`
- 输出：预测 G1 向量（包含根节点朝向，根节点速度，关节点位置，关节点速度，关节点自由度）
- 损失：Smooth L1 损失函数

## 配置

### 一、依赖下载

#### 1.GMR

```bash
# 在项目根目录执行
cd tools
git clone https://github.com/YanjieZe/GMR.git
```

执行完成后，目录应为：

```text
NMR/
└── tools/
    └── GMR/
        ├── README.md
        ├── assets/
        ├── gmr/
        └── ...
```

下载完成后可用以下命令快速校验：

```bash
ls tools/GMR
git -C tools/GMR remote -v
```

若第二条命令能看到 `https://github.com/YanjieZe/GMR.git`，说明来源正确。

#### 2.数据文件

**数据文件获取方式**：

以下文件为项目方为复现实验提前训练的辅助文件或统计文件，并非第三方源码、第三方模型文件或 SMPL-X 官方模型参数。若使用者基于其他数据重新训练，可自行重新生成对应文件。为便于复现，已上传至 [Google Drive](https://drive.google.com/drive/folders/1rcGPIPyPsklFnQF40l0vIWJjZQFtY_wF?usp=sharing)。请按如下方式放置：

| 文件                                     | 目标路径                                                       |
| ---------------------------------------- | -------------------------------------------------------------- |
| betas.npy                                | checkpoints/humanoid_model/g1/betas.npy                        |
| gmr_mean.npy                             | data/gmr_data/gmr_mean.npy                                     |
| gmr_std.npy                              | data/gmr_data/gmr_std.npy                                      |
| smplx_mean.npy                           | data/smplx_data/smplx_mean.npy                                 |
| smplx_std.npy                            | data/smplx_data/smplx_std.npy                                  |

#### 3.三维网格模型

[PHUMA 的 asset 目录](https://github.com/DAVIAN-Robotics/PHUMA/tree/main/asset/) 可作为 `human_model` 和 `humanoid_model` 资产准备方式的参考。请不要将 PHUMA 的 `asset/human_model` 理解为 SMPL-X 模型文件的再分发来源；其中涉及 SMPL-X 的模型文件应由用户自行从 SMPL-X 官方渠道获取。

`humanoid_model` 相关 G1 文件请按 PHUMA 仓库说明自行获取或生成，并放置到本项目的 `checkpoints/humanoid_model/` 目录下。

`SMPLX_NEUTRAL.npz` 需用户自行从 [SMPL-X 官方下载页](https://smpl-x.is.tue.mpg.de/download.php) 获取并遵守其许可条款。登录后选择 **SMPL-X 2020：The neutral SMPL-X model with the FLAME 2020 expression blendshapes** 对应的下载项；若下载得到的文件名为 `SMPLX_NEUTRAL_2020.npz`，可将其重命名为 `SMPLX_NEUTRAL.npz`，并放置到 `checkpoints/human_model/SMPLX_NEUTRAL.npz`。

SMPL-X / SMPLify-X 相关软件、模型和数据通常按非商业科研用途授权。本仓库不分发 `SMPLX_NEUTRAL.npz`，也不授予任何超出 SMPL-X 官方许可的权利；商业用途需要用户自行联系权利方取得单独授权。

`g1_custom_collision_with_fixed_hand.urdf` 请从 [YanjieZe/TWIST 的对应文件](https://github.com/YanjieZe/TWIST/blob/master/assets/g1/g1_custom_collision_with_fixed_hand.urdf) 获取，并放置到下表所示路径。

| 文件                                     | 目标路径                                                       |
| ---------------------------------------- | -------------------------------------------------------------- |
| SMPLX_NEUTRAL.npz                        | checkpoints/human_model/SMPLX_NEUTRAL.npz                      |
| g1_custom_collision_with_fixed_hand.urdf | src/metrics/assets/g1/g1_custom_collision_with_fixed_hand.urdf |

放置完成后，目录结构应与当前仓库里的 `checkpoints/` 一致，也就是：

```text
NMR/
└── checkpoints/
    ├── human_model/
    └── humanoid_model/
```

### 二、运行环境配置（昇腾）

本项目不限定具体云平台，支持在可用昇腾训练环境中运行（含云上与本地集群）。

支持的产品型号：

- Atlas A3 系列产品

必要版本约束：

- 框架：PyTorch 2.7.1
- CANN：8.3.rc1
- Python：3.11
- 系统架构：aarch64

Python 基础依赖可通过以下命令安装：

```bash
pip install -e .
```

`smplx` 受 SMPL-X / SMPLify-X 官方许可约束，通常仅限非商业科研用途，因此不作为默认依赖自动安装。如需运行涉及 SMPL-X 模型加载或处理的功能，请先确认已阅读并同意其官方许可条款，再手动安装：

```bash
pip install smplx
```

推荐环境标识（示例）：

- `Ascend-Powered-Engine` / `pytorch_2.7.1-cann_8.3.rc1-py_3.11-euler_2.10.11-aarch64-snt9b`

部署步骤：

1. 准备运行目录并上传项目代码（目录名建议保持为 `NMR`）。
2. 按“预训练模型文件”章节放置必要权重与模型文件，并将 [configs/retarget_fwd.py](configs/retarget_fwd.py) 中第 23-25 行的 `TRAIN_SPLIT_FILE` 和 `TEST_SPLIT_FILE` 路径，修改为训练集和测试集对应的数据路径。
3. 配置训练入口为 `tools/dist_train.py`，并设置输出环境变量 `train_out_path`指向预计输出目录。
4. 在 [configs/retarget_fwd.py](configs/retarget_fwd.py) 的第 154-175 行进行对应修改即可，其中第 161 行 `lr` 为学习率，第 169 行 `train_cfg` 控制训练周期。
5. 提交训练作业并在平台日志中观察启动与收敛状态。

### 三、后训练代码修改

1. 将 [configs/retarget_fwd.py](configs/retarget_fwd.py) 中第 23-25 行 `TRAIN_SPLIT_FILE` 和 `TEST_SPLIT_FILE` 后的文件路径改为训练和测试使用的数据。

2. 将 [configs/retarget_fwd.py](configs/retarget_fwd.py) 中第 33 行 `LOAD_FROM` 修改为当前训练产出的 checkpoint 路径；若不进行后训练，可保持为 `None`。

3. 在 [configs/retarget_fwd.py](configs/retarget_fwd.py) 的第 154-175 行进行对应修改即可，其中第 161 行 `lr` 为学习率，第 169 行 `train_cfg` 控制训练周期。

## 项目结构

```
NMR/
├── configs/                        # 训练配置
│   ├── default_runtime.py          # 基础运行时设置
│   ├── retarget_fwd.py             # 主训练配置（Transformer）
├── src/                            # 核心代码
│   ├── models/
│   │   ├── tokenizers/             # 运动 Tokenizer（仅使用其中的运动编码器）
│   │   └── transformers/           # LLaMA Transformer 重定向模型
│   ├── datasets/                   # 数据加载
│   ├── metrics/                    # 评估指标（MPJPE 等）
│   └── utils/                      # 旋转转换等工具
├── tools/                          # 脚本
│   ├── train.py                    # 训练入口
│   ├── nmr_inference.py            # 测试/评估入口
│   ├── dist_train.sh               # 分布式训练启动
│   ├── dist_train.py               # 分布式启动器（用于启用.sh文件）
│   ├── inference.py                # 端到端推理
│   └── GMR/                        # General Motion Retargeting 子模块
└── work_dirs/                      # 训练输出（权重、日志）
```

## 训练

### 昇腾引擎训练方式

项目默认提供分布式训练脚本，可直接在昇腾引擎上运行。脚本会自动识别可见卡数，并在外部分布式启动器已注入 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 时复用外部环境，否则自动拉起多卡训练。

如果需要指定单机多卡，可先设置可见设备，例如：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6
```

### 训练 Transformer

```bash
# 昇腾引擎分布式训练（默认自动识别可见卡）
bash tools/dist_train.sh

# 单卡训练（调试）
PYTHONPATH=. python tools/train.py configs/retarget_fwd.py \
    --work-dir work_dirs/debug

# 恢复训练
PYTHONPATH=. python tools/train.py configs/retarget_fwd.py \
    --work-dir work_dirs/xxx --resume auto

# 覆盖配置参数
PYTHONPATH=. python tools/train.py configs/retarget_fwd.py \
    --cfg-options train_cfg.max_epochs=500
```

默认训练参数：

- Batch size: 64，Epochs: 1000
- 优化器: AdamW, lr=2e-4
- 学习率: 线性 warmup 500 步 → MultiStep 衰减
- 多卡启动时可通过 `MASTER_ADDR`、`PORT`、`ASCEND_RT_VISIBLE_DEVICES` 调整通信地址与可见设备

### 数据长度约束（重要）

- `RetargetDataset` 在 `window_size > 0` 时会固定采样长度为 `window_size`。
- 若样本时间长度 `T < window_size`，会在数据加载阶段直接抛出 `ValueError`（不是自动补齐）。
- 建议保证数据集中每条样本满足 `T >= window_size`，或将 `window_size` 设为不大于最短样本长度。
- 当前实现中会在数据集初始化阶段按 `min_motion_length` 过滤样本，仅保留 `T >= min_motion_length` 的序列。

## 评估

```bash
# 单卡测试/评估
PYTHONPATH=. python tools/nmr_inference.py <config_path> <checkpoint_path>
```

评估指标：

| 指标     | 说明                      |
| -------- | ------------------------- |
| MPJPE    | 平均关节位置误差 (mm)     |
| PA-MPJPE | Procrustes 对齐后的 MPJPE |
| ACCEL    | 平均关节加速度误差        |
| DOF_ERR  | DOF 角度误差              |

## 推理

在昇腾引擎上将 SMPL-X 动作文件转换为 G1 机器人控制指令：

```bash
# 单文件推理
PYTHONPATH=. python tools/inference.py <config> <checkpoint> \
    --src input.npz --output-dir output/

# 批量推理（目录下所有 npz/pkl 文件）
PYTHONPATH=. python tools/inference.py <config> <checkpoint> \
    --src input_dir/ --output-dir output/

# 禁用低通滤波
PYTHONPATH=. python tools/inference.py <config> <checkpoint> \
    --src input.npz --no-filter
```

**输入格式**：

- NPZ：含 `global_orient`(T,3), `body_pose`(T,63), `transl`(T,3)
- PKL：含 `fullpose`(T,72), `trans`(T,3)（SMPL 格式，自动从 120 FPS 降采样到 30 FPS）

**输出格式**（PKL）：

- `dof`: `(T, 29)` — 关节角度
- `root_trans`: `(T, 3)` — 根节点位置
- `root_rot_quat`: `(T, 4)` — 根节点旋转（wxyz 四元数）

长序列自动按 4 秒分段推理并线性混合拼接。

## Citation

```bibtex
@article{zhao2026maketrackingeasy,
      title = {Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control},
      author = {Zhao, Qingrui and Yang, Kaiyue and Wang, Xiyu and Zhao, Shiqi and Lu, Yi and Zhang, Xinfang and Yin, Wei and Shen, Qiu and Long, Xiao-Xiao and Cao, Xun},
      journal = {arXiv preprint arXiv:2603.22201},
      year = {2026},
      eprint = {2603.22201},
      archivePrefix = {arXiv},
      url = {https://arxiv.org/abs/2603.22201}
}
```
