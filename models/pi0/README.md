# pi0机器人VLA大模型昇腾使用指南
<br>


## pi0整体介绍
论文题目：π0: A Vision-Language-Action Flow Model for General Robot Control

中文译文：π0: 一个用于通用机器人控制的视觉-语言-动作流匹配模型

### 功能介绍

pi0是一个视觉-语言-动作（VLA）模型，专为通用机器人控制而设计。它基于预训练的视觉语言模型（VLM），结合流匹配（flow matching）机制，能够生成高频连续动作，实现对复杂、灵巧机器人任务的精准控制。整合OXE开源数据集和自有数据集，总计超过10,000小时机器人操作数据。在叠衣服、桌面清理、装盒等复杂任务上表现优异，零样本和微调设置下均显著优于现有基线方法（OpenVLA、Octo、ACT等）。成功完成5-20分钟的长时序多阶段任务，展现出强大的鲁棒性和泛化能力。

<br>


## pi0的相关代码仓拉取、模型和数据集下载
### 代码仓拉取
基于lerobot机器人通用数据采集、模型训练-推理框架，进行pi0的数据集采集、训练、推理。lerobot代码仓拉取步骤如下：

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
# 为了能够对齐昇腾迁移过程中的代码版本，避免代码仓更新带来的差异，需要执行下述操作，将代码仓回退到指定老版本(Date: Tue Mar 4 10:53:01 2025 +0100)：
git reset --hard a27411022dd5f3ce6ebb75b460376cb844699df8

# 拉取昇腾开源项目代码仓中pi0相关文件,cann-recipes-embodied-intelligence代码仓根目录与lerobot代码仓根目录属于同级别目录。
cd ../  # 回到lerobot根目录下
git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git

# 一次性执行下面的所有指令,将上一步拉取下来的cann-recipes-embodied-intelligence文件夹中的文件，替换和复制到lerobot代码仓的对应目录中。
cp cann-recipes-embodied-intelligence/models/pi0/{modeling_pi0.py,paligemma_with_expert.py} lerobot/lerobot/common/policies/pi0/ && cp cann-recipes-embodied-intelligence/models/pi0/{pyproject.toml,run_pi0_inference.sh,test_pi0_on_ascend.py} lerobot/
```
完成上述操作之后，最终lerobot根目录中代码目录树详见[附录：lerobot根目录代码目录树](#lerobot根目录代码目录树)

<br>


### pi0模型和koch_test数据集下载
- pi0模型权重的pytorch版本已经开源，下载[pi0模型权重文件](https://modelscope.cn/models/lerobot/pi0/files/01189b1ffb1c9f2f9622c3b1ae773cd884bfd84f)至pi0_model文件夹中。文件较大，可以参考下面的指令进行下载：
  ```bash
  cd lerobot  # 首先进入lerobot根目录下
  sudo apt install git-lfs -y
  git lfs install --skip-smudge
  git clone https://www.modelscope.cn/models/lerobot/pi0.git pi0_model
  cd pi0_model
  git lfs install --force
  GIT_LFS_SKIP_SMUDGE=1 git reset --hard 01189b1ffb1c9f2f9622c3b1ae773cd884bfd84f
  git lfs pull
  ```

- koch_test数据集为pi0模型对应的数据集之一，是基于koch-v1.1六自由度机械臂采集的真机数据集，执行的任务为抓取桌子上的方块到盒子中。在lerobot代码仓根目录下新建koch_test文件夹，下载[koch_test数据集](https://huggingface.co/datasets/danaaubakirova/koch_test/tree/main)网站中所有的文件至koch_test文件夹中，并注意本地数据集目录树与网站上进行对应。文件较大，可以手动下载，也可以参考下面的指令进行下载：
  ```bash
  cd lerobot  # 进入lerobot根目录下
  git lfs install --force
  git clone https://huggingface.co/datasets/danaaubakirova/koch_test
  ```
<br>


## pi0在昇腾A2上的运行环境配置
### 与昇腾服务器无关的环境配置
```bash
# 创建运行环境
conda create -y -n lerobot python=3.10
conda activate lerobot

# 回到lerobot根目录，安装lerobot。
cd lerobot
pip install -e .
```

### 与昇腾平台相关的环境配置
安装CANN软件包。本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.3.RC1`。
请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1)下载`Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run`与`Ascend-cann-kernels-910b_8.3.RC1_linux-aarch64.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)依次进行安装。

```bash
# xxxx为CANN包的实际安装目录，注意每次新建终端时，source一下setenv.bash。
source xxxx/ascend-toolkit/setenv.bash

# 在上述运行环境中继续安装对应版本torch-npu
pip install torch-npu==2.1.0.post12
```

<br>


### pi0在昇腾上的推理步骤
运行下面的代码，即可自动加载koch机械臂数据集，进行pi0模型推理，打印推理性能及机器人动作
```bash
cd lerobot
conda activate lerobot
source xxxx/ascend-toolkit/setenv.bash
chmod +x run_pi0_inference.sh
./run_pi0_inference.sh koch_test pi0_model 10 100
```

基于上述运行过程，得到pi0的单次推理时间及结果如下所示（详细的优化过程介绍见 [pi0 优化说明文档](docs/models/pi0/README.md)）：
- 推理性能：单次推理时间下降至80 ms，达到了预期的推理时间性能优化目标。
- 推理结果：单次推理结果为50组机械臂关节角度序列，shape为[50,6]。
<br>


## pi0在昇腾上的精度验证步骤
### 基于koch机械臂末端位姿的ATE(绝对误差)来验证昇腾的推理精度
- 为了能够以固定变量法进行昇腾平台的推理精度测试，需将pi0推理中action_expert中的高斯噪声采样部分进行固定噪声文件加载(即使用同样的高斯噪声采样数据)。
- 基于pi0模型推理得到的整段轨迹六关节角度序列(维度:50x6)，通过获取koch机械臂的物理DH参数，执行koch机械臂正运动学运算，得到koch机械臂末端执行器中心的实际位姿（位置x-y-z + 姿态r-p-y），然后通过ATE(absolute error)方法进行二范数计算，得到昇腾平台上koch机械臂末端位姿的误差参数，误差参考范围如下所示：
    - 位置ATE误差参考范围：[0, +0.03]m
    - 姿态ATE误差参考范围：[0, +0.2 ]rad
<br>


## Citation
```
@misc{black2024pi0,
      title={$\pi$0: A Vision-Language-Action Flow Model for General Robot Control}, 
      author={Kevin Black and Noah Brown and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Lucy Xiaoyang Shi and James Tanner and Quan Vuong and Anna Walling and Haohuan Wang and Ury Zhilinsky},
      year={2024},
      eprint={2410.24164},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.24164}
}
```

<br>


## 附录
### lerobot根目录代码目录树
- 检查整体代码目录树，经过上述的复制及替换操作，pi0适配昇腾的lerobot根目录中的最终相关代码目录树如下所示
```bash
├── koch_test                                 # koch机械臂抓取任务数据集,符合lerobot数据集格式
├── lerobot                                   # pi0模型训练及推理框架
|   ├── common
|   |   ├── policies
|   |   |   ├── pi0
|   |   |   |   ├── modeling_pi0.py           # pi0的模型训练及推理代码
|   |   |   |   ├── paligemma_with_expert.py  # pi0的模型训练及推理代码
├── pi0_model                                 # koch机械臂抓取任务预训练pi0模型
└── pyproject.toml                            # 运行环境第三方包的安装版本
└── README.md                                 # 昇腾上运行pi0推理的环境配置及操作指导
└── run_pi0_inference.sh                      # 昇腾上运行pi0推理过程一键启动脚本
└── test_pi0_on_ascend.py                     # 昇腾上运行pi0推理过程
```