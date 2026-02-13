# 概述

GR00T N1.6是2025年12月发布的通用机器人基础模型，旨在突破传统VLA模型在长程具身操作任务中的泛化瓶颈，解决小样本场景下动作生成的时序一致性问题；通过升级 VLM 模块与动作预测范式，实现从短程静态桌面操作到动态长程具身任务的能力提升，降低通用人形机器人在真实场景落地的工程化门槛。

# 支持的产品型号

Atlas A3 系列产品

# 环境准备

1. 安装CANN环境
   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.3.RC1`，请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha002)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Atlas-A3-cann-kernels_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。
   
   * `${version}`表示CANN包版本号，如8.3.RC1。
   * `${arch}`表示CPU架构，如aarch64、x86_64。

2. 克隆仓库
   GR00T依赖某些依赖的子模块。克隆时需要包含它们：
   
   ```
   git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
   cd Isaac-GR00T
   git checkout e29d8fc50b0e4745120ae3fb72447986fe638aa6
   cd ..
   
   git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git
   ```
   制和替换cann-recipes-embodied-intelligence代码仓manipulation/Isaac-GR00T中所有文件到官方代码仓Isaac-GR00T中

   ```
   cp -rf cann-recipes-embodied-intelligence/manipulation/Isaac-GR00T/* Isaac-GR00T
   cd Isaac-GR00T
   ```

   
3. 环境设置
   
   GR00T使用[uv](https://github.com/astral-sh/uv)实现快速且可重复的依赖管理。
   
   > 说明：解析`pyproject.toml`中的`[tool.uv.extra-build-dependencies]`  需要 uv v0.8.4+。
   
   
   创建环境并安装GR00T：
   
   ```
   uv sync --python 3.10
   ```
   安装环境所需的ffmpeg和decord库
   ```
   chmod +x setup.sh
   sh ./setup.sh
   export LD_LIBRARY_PATH=$(pwd)/.venv/lib:$LD_LIBRARY_PATH
   ```
   激活uv环境：

   ```
   source .venv/bin/activate
   ```

# 推理执行

1. 数据准备
   * 本样例已在`demo_data`下提供样例数据集。
   * 您也可以自行准备数据集，详情请参阅[数据准备指南 ](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)。
2. 快速推理
   准备好数据后，GR00T模型可使用以下脚本生成输出动作：
   ```
   uv run python scripts/deployment/standalone_inference_script.py \
     --model-path nvidia/GR00T-N1.6-3B \
     --dataset-path demo_data/gr1.PickNPlace \
     --embodiment-tag GR1 \
     --traj-ids 0 1 2 \
     --video-backend decord \
     --seed 42 \
     --action-horizon 8
   ```
   运行上述命令后，模型开始推理并打印出机器人left_arm、right_arm、left_hand、right_hand、waist等部位的运动指令。

# Citation
```
@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint     = {2503.14734},
  title      = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author     = {NVIDIA and Johan Bjorck and Fernando Castañeda, Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
  month      = {March},
  year       = {2025},
  booktitle  = {ArXiv Preprint},
}
```

# 附录
## 根目录相关代码目录树
检查整体代码目录树，经过上述的复制操作，GR00T N1.6适配昇腾的根目录中的最终相关代码目录树如下所示：
```bash
.
├── adaptor_patches
│   ├── dit_patch.py
│   ├── gr00t_n1d6_patch.py
│   ├── gr00t_policy_patch.py
│   └── modeling_siglip2_patch.py
├── pyproject.toml
├── README.md
├── scripts
│   └── deployment
│       ├── model_adaptor.py
│       └── standalone_inference_script.py
└── setup.sh
```
