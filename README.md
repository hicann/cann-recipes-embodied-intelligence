# cann-recipes-embodied-intelligence

## 🚀Latest News
- [2026/04] PI0模型支持在昇腾Atlas A2上训练，样例已开源。
- [2026/04] SmolVLA模型支持在昇腾Atlas A2上训练，样例已开源。
- [2026/04] ACT模型支持在昇腾Atlas A2上训练，样例已开源。
- [2026/04] Pi0.5模型在昇腾Ascend 310P上已支持OM静态图推理部署，样例已开源。
- [2026/03] LQC模型在昇腾 A2上已支持训练和推理，样例已开源。
- [2026/03] Pi0.5模型在昇腾Ascend 310P上已支持在线推理部署，样例已开源。
- [2026/02] Isaac-GR00T N1.6模型在昇腾Atlas A3上已支持推理，样例已开源。
- [2026/02] Cosmos-Predict2.5-2B世界模型在昇腾Atlas A3上已支持推理，样例已开源。
- [2026/02] Cosmos-Transfer2.5-2B世界模型在昇腾Atlas A3上已支持推理，样例已开源。
- [2026/02] Alpamayo-R1智驾模型在昇腾Atlas A2上已支持推理，样例已开源。
- [2026/01] Spirit v1.5模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/12] Pi0模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/12] OpenVLA模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/12] DiffusionPolicy模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/12] Action Chunking with Transformers (ACT) 模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/11] Pi0模型在昇腾Atlas A2系列上已支持推理，代码已开源。

## 🎉概述
cann-recipes-embodied-intelligence仓库旨在针对具身智能业务中的典型模型、加速算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地基于CANN平台使用具身智能模型。


## ✨样例列表

### 操作类模型 (Manipulation)

| 实践 | 平台 | 框架 | 简介 | 性能参考 |
|----|----|----|----|----|
| [Pi0](manipulation/pi0/infer_with_torch/README.md) | Atlas A2 | torch | 基于LeRobot库，在Atlas A2环境适配Pi0模型，通过使能融合算子、图模式、计算逻辑优化等手段，实现了较低的推理时延。 | **80 ms** |
| [Pi0](manipulation/pi0/train/README.md) | Atlas A2 | torch | 在Atlas A2环境进行 PI0 模型训练，支持 8 卡分布式训练与评测，并默认集成已验证的训练优化。 | **81.77 samples/s** (优化后) |
| [Pi0](manipulation/pi0/infer_with_om/README.md) | Ascend 310P | OM | 基于LeRobot库，在310P环境OM离线推理的Pi0模型，实现了较低的推理时延。 | **~270 ms** (OrangePi AI Station) |
| [Pi0.5](manipulation/pi05/infer_with_torch/README.md) | Ascend 310P | torch | 在310P环境基于PyTorch直接进行Pi0.5模型在线推理。 | **~862 ms** |
| [Pi0.5](manipulation/pi05/infer_with_om/README.md) | Ascend 310P | OM | 在310P环境OM离线推理的Pi0.5模型，实现了较低的推理时延。 | **~410 ms** |
| [Pi0.5](manipulation/pi05/train/README.md) | Atlas A2 | torch | 在Atlas A2环境进行Pi0.5模型的训练，精度正常，性能达到较优的水平。 | **69.19 samples/s** (优化后) |
| [ACT](manipulation/act/train/README.md) | Atlas A2 | torch | 在Atlas A2环境进行 ACT 模型训练，支持 8 卡分布式训练与评测。 | **220.84 samples/s** (8卡，稳定阶段) |
| [SmolVLA](manipulation/smolvla/train/README.md) | Atlas A2 | torch | 在Atlas A2环境进行 SmolVLA 模型训练，支持 LIBERO 数据集的多卡训练与评测。 | **233~244 samples/s** (8卡，稳定阶段) |
| [ACT](manipulation/act/infer_with_om/README.md) | Ascend 310P | OM | 在310P环境OM离线推理的ACT模型，实现了较低的推理时延。 | **~200 ms** (OrangePi AI Station) |
| [DiffusionPolicy](manipulation/diffusion-policy/infer_with_om/README.md) | Ascend 310P | OM | 在310P环境OM离线推理的DiffusionPolicy模型，实现了较低的推理时延。 | - |
| [OpenVLA](manipulation/openvla/infer_with_om/README.md) | Ascend 310P | OM | 在310P环境OM离线推理的OpenVLA 7B模型，实现了较低的推理时延。 | - |
| [Isaac-GR00T N1.6](manipulation/Isaac-GR00T/README.md) | Atlas A3 | torch | 通用人形机器人基础模型，适配昇腾A3平台。 | - |
| [Spirit v1.5](manipulation/spirit-v1.5/infer_with_torch/README.md) | Ascend 310P | torch | 千寻智能自研的具身智能模型，RoboChallenge评测综合排名第一(截至2026.1.12)。 | - |

### 世界模型 (World Model)

| 实践 | 平台 | 框架 | 简介 | 性能参考 |
|----|----|----|----|----|
| [Cosmos-Predict2.5-2B](world_model/cosmos-predict2.5/README.md) | Atlas A3 | torch | Cosmos世界基础模型，支持文本/图像生成世界(Text2World/Image2World)，生成物理一致的视频。 | **~920 s** (生成5.8s视频) |
| [Cosmos-Transfer2.5-2B](world_model/cosmos-transfer2.5/README.md) | Atlas A3 | torch | Cosmos世界基础模型，支持多控制信号(深度图/语义分割/边缘检测等)的视频风格转换。 | - |

### 导航模型 (Navigation)

| 实践 | 平台 | 框架 | 简介 | 性能参考 |
|----|----|----|----|----|
| [Alpamayo-R1](navigation/alpamayo-r1/README.md) | Atlas A2 | torch | 面向L4/L5级智能驾驶的VLA大模型(10B)，支持因果思维链推理。 | **~7.32 s** (生成10条预测轨迹) |

### 运动控制模型 (Locomotion)

| 实践 | 平台 | 框架 | 简介 | 性能参考 |
|----|----|----|----|----|
| [LQC](locomotion/LQC/README.md) | Atlas A2 | torch | 足式机器人的强化学习运动控制器，适用于G1、GO2等主流机器人型号。 | - |


## 📖目录结构说明
```
├─CONTRIBUTION.md
├─DISCLAIMER.md
├─LICENSE
├─README.md
├─Third_Party_Open_Source_Software_Notice
├─docs                                          # 文档目录
│   └─manipulation                              # 对应模型文档目录
│       └─pi0                                   # Pi0相关文档
│           └─infer_with_torch
├─manipulation                                  # 操作类模型目录
│   ├─act                                       # Action Chunking with Transformers模型样例
│   │   ├─infer_with_om                         # ACT模型om离线推理样例
│   │   └─train                                 # ACT模型训练样例
│   ├─diffusion-policy                          # DiffusionPolicy模型样例
│   │   └─infer_with_om                         # DiffusionPolicy模型om离线推理样例
│   ├─openvla                                   # OpenVLA模型样例
│   │   └─infer_with_om
│   ├─pi0                                       # Pi0模型样例
│   │   ├─infer_with_om
│   │   ├─infer_with_torch                      # Pi0模型torch推理样例
│   │   └─train                                 # Pi0模型训练样例
│   ├─pi05                                      # Pi0.5模型样例
│   │   ├─infer_with_torch                      # Pi0.5模型在线推理样例
│   │   └─train                                 # Pi0.5模型训练样例
│   ├─smolvla                                   # SmolVLA模型样例
│   │   └─train                                 # SmolVLA模型训练样例
│   ├─Isaac-GR00T                               # Isaac-GR00T N1.6模型样例
│   ├─alpamayo-r1                               # Alpamayo-R1智驾模型样例
│   └─spirit-v1.5                               # Spirit v1.5模型样例
│       └─infer_with_torch
└─world_model                                   # 世界模型目录
│   ├─cosmos-predict2.5                         # Cosmos-Predict2.5-2B世界模型
│   └─cosmos-transfer2.5                        # Cosmos-Transfer2.5-2B世界模型
└─locomotion                                    # 运动控制模型目录
    └─LQC                                       # Learning-based Quadruped Robot Controller运动控制模型
```

## 📝相关信息

- [贡献指南](./CONTRIBUTION.md)
- [许可证]

    cann-recipes-embodied-intelligence仓涉及的模型，如模型目录下存在License的以该License为准。如模型目录下不存在License的，遵循Apache 2.0许可证，对应协议文本可查阅[LICENSE](./LICENSE)
- [免责声明](./DISCLAIMER.md)
