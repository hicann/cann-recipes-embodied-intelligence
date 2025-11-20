# cann-recipes-embodied-intelligence

## 🚀Latest News
- [2025/11] Pi0模型在昇腾Atlas A2系列上已支持推理，代码已开源。

## 🎉概述
cann-recipes-embodied-intelligence仓库旨在针对具身智能业务中的典型模型、加速算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地基于CANN平台使用具身智能模型。


## ✨样例列表
|实践|简介|
|-----|-----|
|[Pi0](models/pi0/README.md)|基于LeRobot库，在Atlas A2环境适配Pi0模型，通过使能融合算子、图模式、计算逻辑优化等手段，实现了较低的推理时延。


## 📖目录结构说明
```
├── docs                                        # 文档目录
|  └── models                                   # 模型文档目录
|     ├── pi0                                   # Pi0相关文档
|     └── ...
├── models                                      # 模型脚本目录
|  ├── pi0                                      # Pi0的模型脚本及执行配置
│  └── ...
└── contrib                                     # 社区用户贡献的模型与文档目录
|  ├── README.md
│  └── ...
└── CONTRIBUTION.md
└── DISCLAIMER.md
└── LICENSE
└── README.md
└── ...
```

## 📝相关信息

- [贡献指南](./CONTRIBUTION.md)
- [许可证]

    cann-recipes-embodied-intelligence仓涉及的模型，如模型目录下存在License的以该License为准。如模型目录下不存在License的，遵循Apache 2.0许可证，对应协议文本可查阅[LICENSE](./LICENSE)
- [免责声明](./DISCLAIMER.md)
