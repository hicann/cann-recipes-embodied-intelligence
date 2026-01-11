# cann-recipes-embodied-intelligence

## 🚀Latest News
- [2026/01] Spirit v1.5模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/12] Pi0模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/12] OpenVLA模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/12] DiffusionPolicy模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/12] Action Chunking with Transformers (ACT) 模型在昇腾Ascend 310P上已支持推理，样例已开源。
- [2025/11] Pi0模型在昇腾Atlas A2系列上已支持推理，代码已开源。

## 🎉概述
cann-recipes-embodied-intelligence仓库旨在针对具身智能业务中的典型模型、加速算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地基于CANN平台使用具身智能模型。


## ✨样例列表
| 实践 | 平台 | 框架 | 简介                                                               |
|----|----|----|----|
| [Pi0](manipulation/pi0/infer_with_torch/README.md) | Atlas A2 | torch | 基于LeRobot库，在Atlas A2环境适配Pi0模型，通过使能融合算子、图模式、计算逻辑优化等手段，实现了较低的推理时延。 | 
| [Pi0](manipulation/pi0/infer_with_om/README.md) | Ascend 310P | / | 基于LeRobot库，在310P环境OM离线推理的Pi0模型，实现了较低的推理时延。 |
| [ACT](manipulation/act/infer_with_om/README.md) | Ascend 310P | / | 在310P环境OM离线推理的ACT模型，实现了较低的推理时延。 |
| [DiffusionPolicy](manipulation/diffusion-policy/infer_with_om/README.md) | Ascend 310P | / | 在310P环境OM离线推理的DiffusionPolicy模型，实现了较低的推理时延。|
| [OpenVLA](manipulation/openvla/infer_with_om/README.md) | Ascend 310P | / | 在310P环境OM离线推理的OpenVLA模型，实现了较低的推理时延。 |
| [Pi0.5](manipulation/pi05/train/README.md) | Atlas A2 | torch | 在Atlas A2环境进行Pi0.5模型的训练，精度正常，性能达到较优的水平。 |
| [Spirit v1.5](manipulation/spirit-v1.5/infer_with_torch/README.md) | Ascend 310P | torch | 在310P环境基于PyTorch直接进行Spirit v1.5模型推理。 |



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
└─manipulation                                  # 操作类模型目录
    ├─act                                       # Action Chunking with Transformers模型样例
    │   └─infer_with_om                         # ACT模型om离线推理样例
    ├─diffusion-policy                          # DiffusionPolicy模型样例
    │   └─infer_with_om                         # DiffusionPolicy模型om离线推理样例
    ├─openvla                                   # OpenVLA模型样例
    │   └─infer_with_om
    └─pi0                                       # Pi0模型样例
    |   ├─infer_with_om
    |   └─infer_with_torch                      # Pi0模型torch推理样例
    └─pi05
    |   └─train                                 # Pi0.5模型训练样例
    └─spirit-v1.5
        └─infer_with_torch                      # Spirit v1.5模型推理样例
```

## 📝相关信息

- [贡献指南](./CONTRIBUTION.md)
- [许可证]

    cann-recipes-embodied-intelligence仓涉及的模型，如模型目录下存在License的以该License为准。如模型目录下不存在License的，遵循Apache 2.0许可证，对应协议文本可查阅[LICENSE](./LICENSE)
- [免责声明](./DISCLAIMER.md)
