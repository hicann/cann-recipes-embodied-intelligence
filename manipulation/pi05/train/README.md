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

## 背景介绍

**LeRobot** 是一个面向真实世界机器人应用的前沿机器学习框架，致力于为模仿学习与强化学习提供高质量的模型、数据集和工具链。

- LeRobot 基于 PyTorch 构建，旨在降低机器人学习的入门门槛，推动社区在真实场景中的模型共享、数据协作与算法复用。  
- 框架集成了多种经验证可在真实机器人系统中有效部署的最先进方法，并已发布一系列预训练模型、人工采集的演示数据集以及仿真实验环境，便于研究者和开发者快速上手。  
- 所有模型与数据均托管于 [LeRobot Hugging Face 页面](https://huggingface.co/lerobot)，支持一键加载与复现。

在本案例中，我们在昇腾 A2 AI 加速器上完成了 LeRobot 框架的迁移适配，成功运行了多种视觉-语言-动作（Vision-Language-Action, VLA）模型，在多个机器人基准测试任务上实现了端到端的训练与推理。此外，还初步探索了π₀.₅模型在 LIBERO 上的模仿学习训练和性能优化。

## 模型、数据集下载与运行环境配置

### CANN 相关环境配置
安装CANN软件包。本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为CANN 8.3.RC1。 请从软件包下载地址下载Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run与Ascend-cann-kernels-910b_8.3.RC1_linux-aarch64.run软件包，并参考CANN安装文档依次进行安装。

### LeRobot 相关环境配置与安装
完成 CANN 环境配置后，可通过以下步骤获取样例代码并初始化环境：
``` bash
# 进入需要放置代码仓的本地xxx目录下：
cd xxx
git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git
chmod +x cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts/setup.sh
./cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts/setup.sh
```
完成上述操作之后对应的项目文件结构请参考附录部分[正确安装后项目文件结构](#正确安装后项目文件结构)。

## 分布式训练启动脚本使用说明
<figure>
  <img src="https://raw.gitcode.com/zengzixuan/pic/raw/main/train_pi05_log.png" alt="train log">
  <figcaption style="font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 14px; color: #555; text-align: center; margin-top: 8px;">
    pi05 模型的训练过程可视化（LIBERO上训练40k steps）
  </figcaption>
</figure>

本案例提供了一个通用的分布式训练启动脚本 `run_train.sh`，用于在昇腾单机多卡环境下启动VLA模型的训练任务。该脚本对以下功能进行了统一封装：

- 自动加载模型配置文件（YAML）
- 多进程（`torchrun`/`accelerate`）管理
- 混合精度训练（fp16 / bf16）
- 断点恢复（resume）
- 输出目录与日志自动管理

---

### 快速开始

确保已激活包含 `lerobot`、`accelerate` 和 PyTorch 的 Conda 环境，并完成 Ascend 驱动与 CANN 环境配置后，执行以下命令：

```bash
# 进入脚本所在目录
cd cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts

# 添加执行权限（首次运行时）
chmod +x run_train.sh

# 启动训练（以 pi05 模型为例）
./run_train.sh pi05
```

### 支持的命令行选项

| 选项                                                   | 说明                                                             |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| `<model_type>`                                         | 模型名称（如 `pi0`），脚本将自动加载 `configs/<model_type>.yaml` |
| `--config <path>`                                      | 指定自定义 YAML 配置文件路径（相对于项目根目录）                 |
| `--nproc <N>`                                          | 指定使用的 NPU 卡数（默认：8）                                   |
| `--port <PORT>`                                        | 指定分布式通信端口（默认：29500）                                |
| `--resume`                                             | 启用断点恢复（从 `output_dir` 中的最新 checkpoint 恢复）         |
| `--mix`, `--mixed`, `--mixed_precision` [fp16 \| bf16] | 启用混合精度训练（默认为 fp16；若不指定精度类型，则使用 fp16）   |
| `-h`, `--help`                                         | 显示帮助信息                                                     |

---

### 使用示例

```bash
# 示例 1：训练 pi0 模型（使用 configs/pi0.yaml）
./run_train.sh pi0

# 示例 2：训练 smolvla 模型并启用断点恢复
./run_train.sh smolvla --resume

# 示例 3：使用 bf16 混合精度训练
./run_train.sh pi0 --mix bf16

# 示例 4：使用自定义配置文件
./run_train.sh --config configs/my_custom_vla.yaml --mix fp16 --nproc 4
```

**tips：**
- 输出目录：若未启用 --resume 且原 output_dir 已存在，脚本会自动在路径后追加时间戳（如 outputs/pi0_20251225_143022），避免覆盖。
- 日志文件：训练日志将保存在 ckpt/logs/ 目录下，格式为 train_<model>_<timestamp>.log。
- 训练监控：本项目使用wandb进行训练过程监控，使用前请先在终端输入`wandb login`进行登陆并配置正确的API key，并在config文件（即configs/<model_type>.yaml）中正确设置`project`和`entity`参数。若不需要训练监控，则将config文件（即configs/<model_type>.yaml）中`wandb`项目的`enable`设置为`false`。
- 后台运行：脚本默认使用 nohup ... & 在后台启动训练，可通过日志文件或 jobs/ps 查看进程。

## 评估脚本使用说明

本案例提供了一个通用的评估启动脚本 `run_eval.sh`，用于在昇腾单机环境下启动VLA模型的评估任务。该脚本对以下功能进行了统一封装：

- 自动配置无头渲染环境（Xvfb + OSMesa）
- 支持多种评估环境（libero / aloha / pusht 等）
- 自动加载评估配置并执行推理
- 输出目录与日志自动管理

---

### 快速开始

确保已激活包含 `lerobot` 和 PyTorch 的 Conda 环境，并完成 Ascend 驱动与 CANN 环境配置后，执行以下命令：

```bash
# 进入脚本所在目录
cd cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts

# 添加执行权限（首次运行时）
chmod +x run_eval.sh

# 启动评估（默认运行 smolvla libero 任务）
./run_eval.sh
```

### 支持的命令行选项

脚本支持两种使用方式：

1. **默认评估**：无参数时，运行官方 smolvla libero 评估任务。
2. **自定义评估**：传入参数时，直接透传给 `lerobot_eval` 脚本，支持所有 `lerobot_eval` 的命令行选项。

常用 `lerobot_eval` 选项包括：

| 选项                       | 说明                                                     |
| -------------------------- | -------------------------------------------------------- |
| `--policy.path <path>`     | 指定模型策略路径（如 HuggingFace 模型或本地 checkpoint） |
| `--env.type <type>`        | 评估环境类型（如 `libero`、`aloha`、`pusht`）            |
| `--env.task <task>`        | 具体任务名称（如 `libero_object`、`libero_spatial`）     |
| `--eval.batch_size <N>`    | 评估批次大小（默认：1）                                  |
| `--eval.n_episodes <N>`    | 评估回合数（默认：1）                                    |
| `--policy.device <device>` | 指定设备（如 `npu`、`cuda`）                             |
| `--output_dir <path>`      | 输出目录路径                                             |
| `-h`, `--help`             | 显示 `lerobot_eval` 的帮助信息                           |

---

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

**说明：**
- 环境依赖：脚本会自动启动虚拟显示（Xvfb）和配置 OSMesa 无头渲染，确保在无 GUI 环境下正常运行。
- 输出目录：评估结果将保存在指定的 `output_dir` 中，包括日志、视频和性能指标。
- 日志文件：评估日志将输出到控制台，可通过重定向保存到文件（如 `./run_eval.sh > eval.log 2>&1`）。
- 设备指定：确保 `--policy.device=npu` 以使用昇腾 NPU 进行推理。

## Citation
```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

## π₀.₅ 模型模仿学习分布式训练调优实践
见 [性能优化与最佳实践](doc/README.md)

## 附录
### 正确安装后项目文件结构

``` bash
├── cann-recipes-embodied-intelligence/     # CANN 具身智能案例仓库主目录
│   └── manipulation/
│       └── pi05/    
│           └── train/                     # pi05 模型昇腾训推项目目录                       
│               ├── assets/                     # 示例图片资源
│               ├── doc/
│               │   └── profiling.md            # 性能分析与训练监控说明文档
│               ├── src/
│               │   ├── scripts/                # 一键部署与运行脚本
│               │   │   ├── setup.sh            # 环境初始化与依赖安装脚本
│               │   │   ├── run_train.sh        # 分布式训练启动脚本
│               │   │   ├── run_eval.sh         # 模型评估启动脚本
│               │   │   └── run_profiling.sh    # 分布式训练性能采集脚本
│               └── README.md                   # LeRobot 昇腾训推项目使用说明
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