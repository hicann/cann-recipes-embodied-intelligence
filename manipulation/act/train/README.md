# ACT 在昇腾 Atlas A2 上的训练样例

本目录提供 ACT 训练样例，完成 ALOHA `transfer_cube` 任务的模型训练，以及方便拓展到其他任务上。

当前样例遵循以下原则：
- `cann-recipes` 仓库中仅保存训练样例目录、配置、脚本、文档和补丁；
- `lerobot` 作为外部依赖仓单独 clone；
- `setup.sh` 固定 `lerobot` commit id，并对已验证的通用 Ascend 训练补丁执行 `git apply`；

## 1. 适用场景
- 硬件：昇腾 Atlas A2
- CANN：8.3.0 及以上
- 任务：`AlohaTransferCube-v0`
- 数据集：`lerobot/aloha_sim_transfer_cube_human`
- 外部训练框架：`huggingface/lerobot`

## 2. 外部依赖与固定版本
本样例不内嵌 `lerobot` 源码，默认使用如下 commit：

```text
58f70b6bd370864139a3795ac3497a9eae8c42d5
```

## 3. 目录说明

```text
manipulation/act/train/
├── README.md
├── doc/
│   └── README.md
└── src/
    ├── configs/
    │   ├── act_aloha.yaml
    │   └── act_aloha_smoke.yaml
    ├── patches/
    │   └── lerobot_ascend_train_common.patch
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
chmod +x manipulation/act/train/src/scripts/setup.sh
./manipulation/act/train/src/scripts/setup.sh
```

该脚本会：
- 在 `cann-recipes` 同级目录下准备 `lerobot` 代码仓；
- checkout 到固定 commit `58f70b6bd370864139a3795ac3497a9eae8c42d5`；
- 应用当前已验证的 Ascend 训练补丁；
- 安装 ACT 所需的 LeRobot 通用 Python 依赖与 `gym-aloha`；
- 默认复用当前已激活环境中的 `torch` / `torch_npu`；
- 如需在新环境中执行，可通过参数创建 conda 环境，并通过本地 wheel 注入平台相关的 `torch` / `torchvision` / `torch_npu`。

常见用法：

```bash
# 查看脚本帮助
./manipulation/act/train/src/scripts/setup.sh --help

# 用当前已准备好的 Ascend 环境
./manipulation/act/train/src/scripts/setup.sh

# 创建新 conda 环境，并从本地 wheel 安装平台栈
./manipulation/act/train/src/scripts/setup.sh \
  --create-conda \
  --env-name lerobot-act \
  --python-version 3.10 \
  --torch-wheel /path/to/torch.whl \
  --torchvision-wheel /path/to/torchvision.whl \
  --torch-npu-wheel /path/to/torch_npu.whl
```

说明：
- 之所以不在脚本中硬编码 `torch_npu` 下载链接，是因为有效的 wheel 组合依赖于宿主机架构、CANN 版本和 Ascend 软件栈；
- 这部分平台依赖建议由已有训练环境复用，或由使用者自行提供本地 wheel。
- 如已提前确认平台栈可用，也可以追加 `--skip-torch-check` 跳过末尾导入校验。

### 4.3 数据集路径
当前配置默认使用工作区相对路径：

```text
../dataset/lerobot/aloha_sim_transfer_cube_human
```

如需调整，请修改：
- `src/configs/act_aloha.yaml`
- `src/configs/act_aloha_smoke.yaml`

这些相对路径默认相对于 `lerobot` 根目录解析，推荐工作区布局如下：

```text
<workspace>/
├── cann-recipes-embodied-intelligence/
├── lerobot/
├── dataset/
│   └── lerobot/
│       └── aloha_sim_transfer_cube_human/
└── ckpt/
```

要求：`root` 必须直接指向包含 `data/`、`meta/` 的数据集根目录。

### 4.4 ResNet18 权重缓存
ACT 默认使用：

```yaml
pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
```

首次训练或评测时，PyTorch 可能会尝试下载 `resnet18-f37072fd.pth`。在无外网环境中，建议提前将该文件放到当前用户的 PyTorch 权重缓存目录，例如：

```text
~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

可在有外网的机器上从 PyTorch 官方地址下载：

```bash
wget -O resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
```

也可以使用：

```bash
curl -L https://download.pytorch.org/models/resnet18-f37072fd.pth -o resnet18-f37072fd.pth
```

下载后，将文件拷贝到目标机器的 PyTorch 权重缓存目录：

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
cp resnet18-f37072fd.pth ~/.cache/torch/hub/checkpoints/
```

如果设置了 `TORCH_HOME`，则实际缓存目录为 `$TORCH_HOME/hub/checkpoints/`。可以通过以下命令确认当前环境的缓存根目录：

```bash
python -c "import torch; print(torch.hub.get_dir())"
```

如果服务器无法联网，又没有提前缓存，ACT 会在模型构建阶段失败。

## 5. 训练配置
### 5.1 smoke 配置
- 配置文件：`src/configs/act_aloha_smoke.yaml`
- 作用：快速验证环境、数据、依赖和多卡训练链路
- 关键参数：
  - `steps: 20`
  - `wandb.enable: false`

启动：

```bash
./manipulation/act/train/src/scripts/run_train.sh act_aloha_smoke --port 29510
```

### 5.2 长训配置
- 配置文件：`src/configs/act_aloha.yaml`
- 关键参数：
  - `steps: 100000`
  - `batch_size: 8`
  - `num_workers: 4`
  - `wandb.enable: true`

启动：

```bash
./manipulation/act/train/src/scripts/run_train.sh act_aloha --port 29510
```

## 6. 评测说明
`run_eval.sh` 只是对 `lerobot-eval` 的轻量封装，参数直接透传。

- **在线评测建议将 MuJoCo 仿真与渲染放在 CPU 侧执行，policy 推理继续使用 NPU**；
- 原因见 [doc/README.md](doc/README.md)。

示例：

```bash
export MUJOCO_GL=osmesa
./manipulation/act/train/src/scripts/run_eval.sh \
  --policy.path=/path/to/pretrained_model \
  --policy.device=npu \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=100 \
  --eval.batch_size=20 \
  --output_dir=/path/to/eval_out
```

说明：
- `MUJOCO_GL=osmesa` 表示 MuJoCo 使用 CPU 软件渲染；
- `--policy.device=npu` 表示模型前向推理继续放在 NPU；
- 这种方式对应“仿真在 CPU，推理在 NPU”。

## 7. 已验证结果摘要
当前已验证的一组参考结果：
- 训练任务：ACT on `lerobot/aloha_sim_transfer_cube_human`
- 任务环境：`AlohaTransferCube-v0`
- 数据规模：`50` episodes，`20000` frames
- 训练硬件：昇腾 Atlas A2 `8` 卡
- 训练步数：`100000`
- 训练 batch 配置：`batch_size: 8`，全局 batch size `64`
- 训练吞吐参考：
  - 统计区间：W&B `train/steps = 5000 ~ 20000`
  - 统计口径：`samples/s = global_batch / (train/update_s + train/dataloading_s)`
  - 平均吞吐：`221.18 samples/s`
- 评测方式：`5 x 100` episodes
- 评测总成功率：`68.0%`

更详细的环境、日志、checkpoint 路径和评测说明见：
- [doc/README.md](doc/README.md)

说明：
- 上述吞吐为 8 卡总吞吐，不是单卡吞吐；
- `samples/s` 使用 W&B 完整 history 计算，包含 dataloader 开销；
- 若只看纯计算阶段，`global_batch / train/update_s` 的去极值均值约为 `265.51 samples/s`。

## 8. W&B 记录占位
![image.png](https://raw.gitcode.com/user-images/assets/8784759/fba46207-e73a-49af-84e9-cc39db1224de/image.png 'wandb日志图')

## 9. 常用命令
### 查看训练日志

```bash
cd ../lerobot
tail -f ../ckpt/logs/train_act_aloha_*.log
```

### resume 训练

```bash
./manipulation/act/train/src/scripts/run_train.sh act_aloha --resume --port 29510
```

## 10. 相关说明
- 本样例目录不包含 `lerobot` 源码；
- 若后续需要扩展到其他 ALOHA 数据集，可新增新的 YAML。
- 样例参考 https://gitcode.com/cann/cann-recipes-embodied-intelligence/blob/master/manipulation/pi05/train/README.md
