# pi0机器人VLA大模型昇腾使用指南
<br>


## pi0整体介绍
论文题目：π0: A Vision-Language-Action Flow Model for General Robot Control

中文译文：π0: 一个用于通用机器人控制的视觉-语言-动作流匹配模型

### 项目介绍

pi0是一个视觉-语言-动作（VLA）模型，专为通用机器人控制而设计。它基于预训练的视觉语言模型（VLM），结合流匹配（flow matching）机制，能够生成高频连续动作，实现对复杂、灵巧机器人任务的精准控制。整合OXE开源数据集和自有数据集，总计超过10,000小时机器人操作数据。在叠衣服、桌面清理、装盒等复杂任务上表现优异，零样本和微调设置下均显著优于现有基线方法（OpenVLA、Octo、ACT等）。成功完成5-20分钟的长时序多阶段任务，展现出强大的鲁棒性和泛化能力。
<br>


## pi0的模型、数据集及代码仓拉取
### pi0模型下载
pi0模型已经在huggingface上进行开源，进入下面的网站，进行pi0模型相关文件的下载,在lerobot代码仓根目录下新建pi0_model文件夹，下载网站中对应的文件至pi0_model文件夹中：
```
https://huggingface.co/zabphd/pi0_model_for_koch_v1_1/tree/main

模型文件目录树如下所示：
├── pi0_model                               # koch机械臂抓取任务预训练模型
|   ├── added_tokens.json                   # 新增特殊 token 列表（<pad>、<eos>、动作占位符等）
|   ├── config.json                         # pi0 整体配置（VLM 主干 + Action Expert 参数量、流匹配步数、动作维度等）
|   ├── model-00001-of-00003.safetensors    # 第 1 个权重分片
|   ├── model-00002-of-00003.safetensors    # 第 2 个权重分片
|   ├── model-00003-of-00003.safetensors    # 第 3 个权重分片
|   ├── model.safetensors                   # 完整权重文件
|   ├── model.safetensors.index.json        # 权重分片索引（>2 GB 时才会出现，指向若干 shard 文件）
|   ├── README.md                           # 模型使用说明、输入输出格式、微调示例
|   ├── special_tokens_map.json             # 特殊 token → token_id 的映射表
|   ├── tokenizer_config.json               # 分词器超参（最大长度、是否添加前缀空格、token 类型等）
|   ├── tokenizer.json                      # 快速分词器二进制表（BPE/WordPiece 词汇表 + 合并规则）
|   ├── tokenizer.model                     # 旧格式词汇表（与 tokenizer.json 二选一，兼容 Transformers 加载）

```
<br>


### koch机械臂抓取任务数据集下载
pi0模型的可选数据集之一为基于koch-v1.1六自由度机械臂的真机数据集danaaubakirova/koch_test，执行的任务为抓取桌子上的方块到盒子中。该koch数据集已经开源至huggingface上，进入下面的网站，进行koch数据集相关文件的下载，并注意本地数据集目录树与网站上进行对应。在lerobot代码仓根目录下新建koch_test文件夹，下载网站中对应的文件至koch_test文件夹中：

```bash
# danaaubakirova/koch_test数据集可视化网址链接如下：
https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fdanaaubakirova%2Fkoch_test%2Fepisode_0

# danaaubakirova/koch_test数据集文件下载网址链接如下：
https://huggingface.co/datasets/danaaubakirova/koch_test/tree/main

数据集目录树如下所示：
├── koch_test
|   ├── data                                # 包含机械臂的本体状态及时间戳等信息
|   |   ├── chunk-000                       # 包含51个完整机械臂抓取方块回合
|   ├── meta                                # 包含数据集中的一些机械臂关节配置、数据格式、数据维度、多回合任务配置等
|   ├── videos                              # 包含多视角相机拍摄的视频
|   |   ├── chunk-000                       # 包含laptop及phone这两个视角的完整机械臂回合视频
|   |   |   ├── observation.images.laptop   # 包含laptop视角的机械臂任务场景视频，共51个回合视频，每秒30帧
|   |   |   ├── observation.images.phone    # 包含phone视角的机械臂任务场景视频，共51个回合视频，每秒30帧
|   ├── README.md                           # 包含数据集结构等基础介绍
```

### 代码仓拉取
基于lerobot机器人通用数据采集、模型训练-推理框架，进行pi0的数据集采集、训练、推理。lerobot代码仓拉取步骤如下：

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
# 为了能够对齐昇腾迁移过程中的代码版本，避免代码仓更新带来的差异，需要执行下述操作，将代码仓回退到指定老版本(Date: Tue Mar 4 10:53:01 2025 +0100)：
git reset --hard a27411022dd5f3ce6ebb75b460376cb844699df8


# 拉取昇腾开源项目代码仓中pi0相关文件,cann-recipes-embodied-intelligence代码仓根目录与lerobot代码仓根目录属于同级别目录
cd ../
git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git
```
<br>


## pi0在昇腾A2上的运行环境配置

### 与昇腾服务器无关的环境配置
基于conda虚拟环境进行环境配置。
```bash
# 创建conda虚拟环境
conda create -y -n lerobot python=3.10
conda activate lerobot

# 修改lerobot代码仓pyproject.toml文件中的datasets、transformers、torch、torchvision版本，避免代码运行中出错
datasets==3.3.2
transformers==4.49.0
torch==2.1.0
torchvision==0.16.0

# 安装lerobot
pip install -e .
```

### 与昇腾服务器相关的环境配置
涉及到昇腾CANN安装包（本案例使用CANN的版本为8.3.RC1）及对应的pytorch-npu版本安装。其中，CANN相关工具包的安装及环境配置，请参考 [CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) 安装对应的驱动、固件和软件包。在安装深度学习框架时，选择安装PyTorch框架，并安装配套的torch_npu包。

需要注意每次终端创建时，注意source一下CANN安装路径的setenv.bash。其他的，需要注意transformers、torch、torch-npu、torchvision对应cuda/npu版本的pip安装，具体版本如下：
```bash
# xxxx为CANN包的实际安装目录
source xxxx/Ascend/latest/bin/setenv.bash

# 需要注意的一些pip安装的npu相关packages版本如下：
torch-npu==2.1.0.post12  # 网址为https://pypi.org/project/torch-npu/2.1.0.post12/
torchair==0.1  # 安装方式网址（适配torch-npu==2.1.0.post12）：https://gitee.com/ascend/torchair/blob/master/README.md
```

### 整体环境包版本验证及保证
在安装环境的最后环节，执行下面的指令，来保证关键包的存在及版本正确。
```bash
pip install numpy==1.26.4
pip install pytest==8.4.2
```
<br>


## pi0在昇腾上的运行步骤

- 将pi0_model模型文件夹及koch_test数据集文件夹放置到lerobot文件夹下,如下所示：
```bash
├── koch_test                               # koch机械臂抓取任务数据集,符合lerobot数据集格式
├── pi0_model                               # koch机械臂抓取任务预训练pi0模型

```

- 拉取本项目代码仓中pi0文件夹，将其中的文件复制到lerobot文件夹中（同名文件需要进行替换，如modeling_pi0.py、paligemma_with_expert.py、README.md）。其中，昇腾适配代码仓中的pi0文件中代码目录树如下所示：
```bash
# 查看cann-recipes-embodied-intelligence/models/pi0文件夹中代码目录树情况
├── pi0
|   ├── modeling_pi0.py
|   ├── paligemma_with_expert.py
|   ├── README.md
|   ├── run_pi0_inference.sh
|   ├── test_pi0_on_ascend.py
```


- 替换lerobot代码仓中的文件来适配昇腾
```bash
# 替换modeling_pi0.py和paligemma_with_expert.py文件，以适配昇腾平台pi0推理，并支持进行固定高斯噪声加载进行不同处理器平台精度对比。文件在原始lerobot代码仓目录树中的位置如下所示：
├── lerobot
|   ├── common
|   |   ├── policies
|   |   |   ├── pi0 
|   |   |   |   ├── modeling_pi0.py
|   |   |   |   ├── paligemma_with_expert.py
...
├── README.md
```


- 检查整体代码目录树，经过上述的复制及替换操作，pi0适配昇腾的最终代码目录树如下所示
```bash
├── examples
├── koch_test                                 # koch机械臂抓取任务数据集,符合lerobot数据集格式
├── lerobot                                   # pi0模型训练及推理框架
|   ├── common
|   |   ├── policies
|   |   |   ├── pi0 
|   |   |   |   ├── modeling_pi0.py           # pi0的模型训练及推理代码
|   |   |   |   ├── paligemma_with_expert.py  # pi0的模型训练及推理代码
├── pi0_model                                 # koch机械臂抓取任务预训练pi0模型
├── outputs
└── tests
└── README.md                                 # 昇腾上运行pi0推理的环境配置及操作指导
└── run_pi0_inference.sh                      # 昇腾上运行pi0推理过程一键启动脚本
└── test_pi0_on_ascend.py                     # 昇腾上运行pi0推理过程
```


### pi0在昇腾上的推理步骤
```bash
# 将run_pi0_inference.sh及test_pi0_on_ascend.py放至主目录下，运行下面的代码，即可自动进行torch-npu的加载及转换，自动加载koch机械臂数据集，进行模型推理，得到推理结果(koch机械臂整段轨迹六关节角度序列(维度:50x6))并进行保存。
cd lerobot
conda activate lerobot
# 设置 Ascend CANN 环境变量，xxxx为CANN包的实际安装目录
source xxxx/Ascend/latest/bin/setenv.bash
chmod +x run_pi0_inference.sh
./run_pi0_inference.sh koch_test pi0_model 10 100
```

其中，基于上述过程，得到昇腾上pi0的单次推理时间及结果如下所示（详细的优化过程介绍见cann-recipes-embodied-intelligence/docs/models/pi0/README.md）：
- 推理性能：从开箱性能的670 ms，下降至80 ms，性能达到开箱性能的8.3倍，达到了预期的推理时间性能优化目标。
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