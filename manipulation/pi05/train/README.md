# LeRobot 框架具身 VLA 模型昇腾训推实践

下表展示了在昇腾 A2 NPU 平台上，基于 LeRobot 框架运行多项 LIBERO 典型任务的推理效果。通过必要的框架适配与环境配置，昇腾平台成功实现了 VLA 模型在多任务场景下的端到端推理，验证了训推方案的可用性和一致性。
<table border="1" cellpadding="6" cellspacing="0"
  style="border-collapse: collapse; 
         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         width: 95%; 
         margin: auto; 
         table-layout: fixed;">
  <colgroup>
    <col style="width: 25%;">
    <col style="width: 25%;">
    <col style="width: 25%;">
    <col style="width: 25%;">
  </colgroup>
  <thead>
    <tr>
      <th style="background: #4a90e2; color: white; text-align: center; vertical-align: middle; padding: 8px; font-size: 13px; word-wrap: break-word;">
        libero_spatial:<br>pick up the black bowl from table center and place it on the plate
      </th>
      <th style="background: #4a90e2; color: white; text-align: center; vertical-align: middle; padding: 8px; font-size: 13px; word-wrap: break-word;">
        libero_object:<br>pick up the salad dressing and place it in the basket
      </th>
      <th style="background: #4a90e2; color: white; text-align: center; vertical-align: middle; padding: 8px; font-size: 13px; word-wrap: break-word;">
        libero_goal:<br>put the wine bottle on top of the cabinet
      </th>
      <th style="background: #4a90e2; color: white; text-align: center; vertical-align: middle; padding: 8px; font-size: 13px; word-wrap: break-word;">
        libero_10:<br>turn on the stove and put the moka pot on it
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center; padding: 10px; border: 1px solid #ddd; vertical-align: middle;">
        <img src="https://raw.gitcode.com/zengzixuan/pic/raw/main/A2-1.GIF" alt="A2 - libero_spatial" width="140" height="140" style="object-fit: cover; border: 1px solid #ccc; display: block; margin: 0 auto;">
      </td>
      <td style="text-align: center; padding: 10px; border: 1px solid #ddd; vertical-align: middle;">
        <img src="https://raw.gitcode.com/zengzixuan/pic/raw/main/A2-2.GIF" alt="A2 - libero_object" width="140" height="140" style="object-fit: cover; border: 1px solid #ccc; display: block; margin: 0 auto;">
      </td>
      <td style="text-align: center; padding: 10px; border: 1px solid #ddd; vertical-align: middle;">
        <img src="https://raw.gitcode.com/zengzixuan/pic/raw/main/A2-3.GIF" alt="A2 - libero_goal" width="140" height="140" style="object-fit: cover; border: 1px solid #ccc; display: block; margin: 0 auto;">
      </td>
      <td style="text-align: center; padding: 10px; border: 1px solid #ddd; vertical-align: middle;">
        <img src="https://raw.gitcode.com/zengzixuan/pic/raw/main/A2-4.GIF" alt="A2 - libero_10" width="140" height="140" style="object-fit: cover; border: 1px solid #ccc; display: block; margin: 0 auto;">
      </td>
    </tr>
  </tbody>
</table>

围绕 LeRobot 在昇腾平台上的训推实践，本文主要介绍环境准备、代码同步、训练、profiling 和评估的使用方式。关于 `pi05` 在昇腾上的性能优化细节、保留优化项和量化收益，会在 [doc/README.md](./doc/README.md) 中展开。

## 项目背景

**LeRobot** 是一个面向真实世界机器人应用的前沿机器学习框架，致力于为模仿学习与强化学习提供高质量的模型、数据集和工具链。

- LeRobot 基于 PyTorch 构建，旨在降低机器人学习的入门门槛，推动社区在真实场景中的模型共享、数据协作与算法复用。
- 框架集成了多种经验证可在真实机器人系统中有效部署的最先进方法，并已发布一系列预训练模型、人工采集的演示数据集以及仿真实验环境，便于研究者和开发者快速上手。
- 所有模型与数据均托管于 [LeRobot Hugging Face 页面](https://huggingface.co/lerobot)，支持一键加载与复现。

在本案例中，我们在昇腾 A2 AI 加速器上完成了 LeRobot 框架的迁移适配，成功运行了多种视觉-语言-动作（Vision-Language-Action, VLA）模型，在多个机器人基准测试任务上实现了端到端的训练与推理。此外，还围绕 π₀.₅ 模型在 LIBERO 上的模仿学习训练做了针对昇腾平台的性能分析与优化。

## 环境准备与代码同步

### CANN 相关环境配置

安装 CANN 软件包。本样例的编译执行依赖 CANN 开发套件包（`cann-toolkit`）与 CANN 二进制算子包（`cann-kernels`），支持的 CANN 软件版本为 `8.3.RC1`。请从软件包下载地址下载 `Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run` 与 `Ascend-cann-kernels-910b_8.3.RC1_linux-aarch64.run`，并参考 CANN 安装文档依次进行安装。

### LeRobot 相关环境配置与安装

完成 CANN 环境配置后，可通过以下步骤获取样例代码并初始化环境。`setup.sh` 作为总入口，会编排 `setup_lerobot.sh`、`setup_cmake.sh`、`setup_deps.sh` 三个模块，完成 LeRobot 指定版本同步、conda 环境准备、基础依赖安装，以及 PI05 所需额外依赖和脚本覆盖。

```bash
cd <your-workdir>
git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git
chmod +x cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts/setup.sh
./cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts/setup.sh
```

网络受限环境可先指定 CMake 国内镜像备用地址（默认已内置清华镜像，可按需覆盖）：

```bash
export CMAKE3_MIRROR_URL="https://mirrors.tuna.tsinghua.edu.cn/kitware/cmake/v3.28/cmake-3.28.3.tar.gz"
./cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts/setup.sh
```

默认目录关系如下：

```bash
<your-workdir>/
├── cann-recipes-embodied-intelligence/
└── lerobot/
```

当前样例默认锁定到以下经过 Python 3.10 / CANN 8.3 实测兼容的版本组合，避免后续 LeRobot 或 PyTorch 继续更新导致适配失效：

- LeRobot version：`v0.4.4`
- LeRobot commit：`8fff0fde7c79f23a93d845d1a50e985de01f8b8a`
- Python：`3.10`
- torch：`2.8.0`
- torchvision：`0.23.0`
- torch_npu：`2.8.0.post2`
- torch_npu 运行时依赖：`pyyaml attrs psutil decorator cloudpickle scipy tornado ml-dtypes`
- torch_npu 可复用依赖：`absl-py`；若运行环境中已存在系统预装且只读的 `absl-py`，脚本会检测到后直接复用，不再强制升级
- torchcodec：固定为 `0.7.0`（对应 git tag `v0.7.0`），aarch64 下需要源码安装，当前 PI05/transformers 导入链必需
- torchcodec 源码获取优先级：已安装的 `torchcodec==0.7.0` -> `TORCHCODEC_LOCAL_DIR` / 默认 `${ROOT_DIR}/torchcodec` 本地缓存 -> `TORCHCODEC_GIT_PRIMARY_URL`（默认 GitCode 镜像）-> GitHub 官方仓库
- torchcodec 构建前置：在 conda 环境中执行 `conda install -y -c conda-forge ffmpeg=7.1.1 pkg-config`
- torchcodec 构建参数：设置 `TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=1`，避免 `ffmpeg 7.1.1` 的弃用告警导致 `v0.7.0` 编译失败
- torchcodec 库发现：脚本会导出 `PKG_CONFIG_PATH=${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig:${PKG_CONFIG_PATH}` 并检查 `libavdevice libavfilter libavformat libavcodec libavutil libswresample libswscale`
- CMake 源码下载：默认先尝试官方地址，失败后自动回退到 `CMAKE3_MIRROR_URL`（默认清华镜像）
- CANN：`8.3.RC1`

如需切换 LeRobot 代码版本，请显式传入 `--lerobot-ref <commit|tag|branch>`；若仅需要同步源码到指定目录，可传入 `--sync-only --lerobot-dir <path>`；若确认要覆盖目标 `lerobot` 目录中的本地修改，可追加 `--force`。

如需单独手动安装 `torchcodec`，推荐在 `lerobot` 环境中执行以下命令：

```bash
conda activate lerobot
conda install -y -c conda-forge ffmpeg=7.1.1 pkg-config
cd /home/ma-user/work/torchcodec
git fetch --tags --force
git checkout -f v0.7.0
export TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=1
export CMAKE=/home/ma-user/.local/cmake3/bin/cmake
export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir):/home/ma-user/.local/cmake3:${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"
export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig:${PKG_CONFIG_PATH}"
python -m pip install -e . --no-build-isolation -v
```

完成上述操作之后，对应的项目文件结构请参考附录部分[正确安装后项目文件结构](#正确安装后项目文件结构)。

## 分布式训练与 Profiling

本目录下训练与 profiling 的入口脚本分别为：

- 训练脚本：`manipulation/pi05/train/src/scripts/run_train.sh`
- profiling 脚本：`manipulation/pi05/train/src/scripts/run_profiling.sh`

如果你只是想先把流程跑通，用这里的命令即可；如果你还想知道当前默认训练路径里已经包含了哪些优化、这些优化为什么有效，以及 profiling 结果应该怎么看，再去看 [doc/README.md](./doc/README.md)。

从零开始复现时，建议按下面的顺序执行：

1. 先执行 `src/scripts/setup.sh` 完成 LeRobot 代码同步、conda 环境准备和依赖安装。模块化入口分别为：`src/scripts/setup_lerobot.sh`、`src/scripts/setup_cmake.sh`、`src/scripts/setup_deps.sh`（默认由 `setup.sh` 自动编排）。
2. 激活 `lerobot` 环境后进入 `src/scripts/` 目录。
3. 训练使用 `./run_train.sh pi05 --nproc 2 --disable-outer-suffix-checkpoint`。
4. profiling 使用 `./run_profiling.sh pi05 --nproc 2 --foreground --disable-outer-suffix-checkpoint --profile-wait 1 --profile-warmup 1 --profile-active 2 --profile-repeat 1`。
5. 如模型和数据已缓存到本地，可再打开 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1` 缩短启动等待时间。

推荐训练命令：

```bash
cd cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts
./run_train.sh pi05 --nproc 2 --disable-outer-suffix-checkpoint
```

推荐 profiling 命令：

```bash
cd cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts
./run_profiling.sh pi05 \
  --nproc 2 \
  --foreground \
  --disable-outer-suffix-checkpoint \
  --profile-wait 1 \
  --profile-warmup 1 \
  --profile-active 2 \
  --profile-repeat 1
```

补充说明：

- `--disable-outer-suffix-checkpoint` 是当前建议显式保留的启动参数，是性能优化收益最大的一项。
- 其他已经验证保留的优化，例如 `sample_beta` fast path、`PaliGemma QKV fusion`、`npu_fusion_attention`、Gemma MLP grouped GEMM 以及 PI05 的 DDP 默认配置，已经吸收到当前脚本和模型实现中，不需要额外手工打开。

## 评估脚本使用说明

本案例提供了一个通用的评估启动脚本 `run_eval.sh`，用于在昇腾单机环境下启动 VLA 模型的评估任务。该脚本对以下功能进行了统一封装：

- 自动配置无头渲染环境（Xvfb + OSMesa）
- 支持多种评估环境（`libero` / `aloha` / `pusht` 等）
- 自动加载评估配置并执行推理
- 输出目录与日志自动管理

### 快速开始

确保已激活包含 `lerobot` 和 PyTorch 的 conda 环境，并完成 Ascend 驱动与 CANN 环境配置后，执行以下命令：

```bash
cd cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts
chmod +x run_eval.sh
./run_eval.sh
```

### 支持的命令行选项

脚本支持两种使用方式：

1. 默认评估：无参数时，运行官方 `smolvla` 的 `libero` 评估任务。
2. 自定义评估：传入参数时，直接透传给 `lerobot_eval` 脚本，支持所有 `lerobot_eval` 的命令行选项。

常用 `lerobot_eval` 选项包括：

| 选项 | 说明 |
| --- | --- |
| `--policy.path <path>` | 指定模型策略路径，例如 Hugging Face 模型或本地 checkpoint |
| `--env.type <type>` | 评估环境类型，例如 `libero`、`aloha`、`pusht` |
| `--env.task <task>` | 具体任务名称，例如 `libero_object`、`libero_spatial` |
| `--eval.batch_size <N>` | 评估批次大小，默认 `1` |
| `--eval.n_episodes <N>` | 评估回合数，默认 `1` |
| `--policy.device <device>` | 指定设备，例如 `npu`、`cuda` |
| `--output_dir <path>` | 输出目录路径 |
| `-h`, `--help` | 显示 `lerobot_eval` 的帮助信息 |

### 使用示例

```bash
# 示例 1：运行默认评估任务（smolvla libero_object）
./run_eval.sh

# 示例 2：评估自定义模型在 libero_spatial 任务上
./run_eval.sh --policy.path="/path/to/your/model" --env.type="libero" --env.task="libero_spatial" --eval.n_episodes=10 --output_dir="./my_eval_output"

# 示例 3：评估 aloha 环境中的任务
./run_eval.sh --policy.path="HuggingFaceVLA/smolvla_aloha" --env.type="aloha" --env.task="aloha_mobile" --policy.device=npu

# 示例 4：查看所有可用选项
./run_eval.sh --help
```

补充说明：

- 脚本会自动启动虚拟显示（Xvfb）并配置 OSMesa 无头渲染，确保在无 GUI 环境下正常运行。
- 评估结果会保存在指定的 `output_dir` 中，包括日志、视频和性能指标。
- 如需保存控制台日志，可使用 `./run_eval.sh > eval.log 2>&1` 的方式重定向。
- 在昇腾环境上执行时，请确认 `--policy.device=npu` 已正确设置。

## Citation

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

## 附录

### 正确安装后项目文件结构

```bash
├── cann-recipes-embodied-intelligence/     # CANN 具身智能案例仓库主目录
│   └── manipulation/
│       └── pi05/
│           └── train/                      # pi05 模型昇腾训推项目目录
│               ├── assets/                 # 示例图片资源
│               ├── doc/
│               │   └── README.md           # 性能优化、profiling 与最佳实践统一说明
│               ├── src/
│               │   ├── scripts/            # 一键部署与运行脚本
│               │   │   ├── setup.sh        # 总入口：调用模块脚本完成初始化与依赖安装
│               │   │   ├── setup_lerobot.sh # LeRobot 同步与基础环境准备
│               │   │   ├── setup_cmake.sh  # CMake 3.x 检查、下载与安装
│               │   │   ├── setup_deps.sh   # PI05/Libero/torchcodec/torch_npu 依赖安装
│               │   │   ├── run_train.sh    # 分布式训练启动脚本
│               │   │   ├── run_eval.sh     # 模型评估启动脚本
│               │   │   └── run_profiling.sh # 分布式训练性能采集脚本
│               └── README.md               # LeRobot 昇腾训推项目使用说明
│
└── lerobot/                                # LeRobot 智能机器人基础框架
    ├── configs/
    │   ├── pi05.yaml                       # Pi05 模型训练配置文件
    │   └── xxx.yaml                        # 其他模型配置文件
    └── src/
        └── lerobot/
            ├── policies/
            │   └── pi05/
            │       └── modeling_pi05.py    # Pi05 模型架构代码
            ├── scripts/
            │   ├── lerobot_train.py        # LeRobot 通用训练入口
            │   ├── lerobot_eval.py         # LeRobot 通用评估入口
            │   └── lerobot_train_profiling.py  # 支持 Ascend Profiling 的训练脚本
            └── utils/
                └── utils.py                # 工具函数（适配昇腾NPU）
```
