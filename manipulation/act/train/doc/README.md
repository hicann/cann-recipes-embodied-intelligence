# ACT ALOHA Transfer Cube 昇腾训练说明

## 1. 训练与评测结论
### 1.1 Atlas A2 训练
- 环境：Python 3.10 + 已验证的 Ascend PyTorch 训练环境
- 数据集：`../dataset/lerobot/aloha_sim_transfer_cube_human`（相对 `lerobot` 根目录）
- 长训配置：`act_aloha.yaml`
- 输出目录：`../ckpt/act_aloha_<timestamp>`（相对 `lerobot` 根目录）
- 训练步数：`100000`
- 默认视频后端：`torchcodec`
- 结论：可在昇腾 Atlas A2 集群上 8 卡并行进行 ACT 模型训练

### 1.2 在线评测
- 评测方式：CPU 执行 MuJoCo 仿真与渲染，NPU 执行 policy 推理
- 环境变量：`MUJOCO_GL=osmesa`
- 500 episode 聚合成功率：`68.0%`

### 1.3 `torchcodec` 快速吞吐对比
- 任务：`ACT`, `8 cards`, `per-device batch_size=64`, `100 steps`
- 当前参考最佳结果如下：

| 配置         | 统计区间      | mean_updt_s | mean_data_s | end-to-end samples/s |
| ------------ | ------------- | ----------: | ----------: | -------------------: |
| `torchcodec` | `step 10~100` |    `0.3191` |    `0.3544` |             `760.24` |

## 3. 为什么推荐仿真在 CPU、推理在 NPU
在线 ALOHA 评测依赖 MuJoCo 的无头渲染环境。实践中更稳妥的做法是：
- MuJoCo 仿真与渲染走 CPU；
- policy 前向推理继续使用 NPU。

推荐组合：
- `MUJOCO_GL=osmesa`
- `--policy.device=npu`

## 4. 当前样例依赖的 lerobot commit
固定版本：

```text
58f70b6bd370864139a3795ac3497a9eae8c42d5
```

说明：
- 本样例以该 commit 作为基线；
- patch 文件也是基于该 commit 提取的；
- 若使用不同commit的版本，可以参考示例patch文件进行修改。

## 5. 关键配置解释
### 5.1 `act_aloha.yaml`
关键项：
- `policy.type: act`
- `policy.device: npu`
- `dataset.root`: 相对 `lerobot` 根目录的数据集路径
- `pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1`
- `steps: 100000`
- `wandb.enable: true`

### 5.2 `act_aloha_smoke.yaml`
适合快速验证：
- `steps: 20`
- `save_freq: 20`
- `wandb.enable: false`

## 6. 常见问题
### 6.1 `gym-aloha` / ALOHA 依赖没有装好
现象：
- `env.type=aloha` 构建失败
- 训练还没开始就报模块缺失

处理：
- 重新执行样例提供的 `setup.sh`，脚本会安装 ACT 所需的通用 Python 依赖和 `gym-aloha`：

```bash
./manipulation/act/train/src/scripts/setup.sh
```

### 6.2 ResNet18 权重下载失败
现象：
- 模型构建阶段尝试访问外网
- 无网环境下报权重下载失败

处理：
- 提前缓存 `resnet18-f37072fd.pth`
- 放到：

```text
~/.cache/torch/hub/checkpoints/
```

### 6.3 wandb 可用性
现状：
- 昇腾 Atlas A2 已验证可以使用官方 wandb
- 如遇 token / 版本问题，优先升级 wandb 再重新登录

### 6.4 `torch_npu` / 平台栈没有准备好
现象：
- `setup.sh` 末尾提示无法 `import torch, torch_npu`
- 或训练启动时直接报 NPU 侧依赖缺失

原因：
- `torch` / `torchvision` / `torch_npu` 的有效组合依赖于 Ascend 软件栈、CANN 版本和机器架构；
- 因此样例不在脚本中硬编码某个固定下载链接。

处理：
- 方案 1：先激活一个已验证可用的 Ascend 训练环境，再执行 `setup.sh`
- 方案 2：执行 `setup.sh` 时显式传入本地 wheel 路径，例如：

```bash
./manipulation/act/train/src/scripts/setup.sh --help

./manipulation/act/train/src/scripts/setup.sh \
  --create-conda \
  --env-name lerobot-act \
  --python-version 3.10 \
  --torch-wheel /path/to/torch.whl \
  --torchvision-wheel /path/to/torchvision.whl \
  --torch-npu-wheel /path/to/torch_npu.whl
```

如已提前确认平台栈可用，也可以追加 `--skip-torch-check` 跳过末尾导入校验。

## 7. 推荐启动方式
### 7.1 smoke

```bash
source /path/to/conda.sh
conda activate <your-ascend-train-env>
source /path/to/Ascend/set_env.sh
./manipulation/act/train/src/scripts/run_train.sh act_aloha_smoke --port 29510
```

### 7.2 正式训练

```bash
source /path/to/conda.sh
conda activate <your-ascend-train-env>
source /path/to/Ascend/set_env.sh
./manipulation/act/train/src/scripts/run_train.sh act_aloha --port 29510
```

### 7.3 resume

```bash
./manipulation/act/train/src/scripts/run_train.sh act_aloha --resume --port 29510
```
