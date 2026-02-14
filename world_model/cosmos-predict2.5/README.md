# Cosmos-predict2.5-2B视频生成具身智能世界模型昇腾使用指南
<br>


## Cosmos-predict2.5-2B整体介绍
论文题目：World Simulation with Video Foundation Models for Physical AI

中文译文：基于视频基础模型的物理智能世界仿真

### 功能介绍

Cosmos-Predict2.5-2B 是 NVIDIA 推出的最新一代世界基础模型（World Foundation Model），专为物理 AI（Physical AI）场景下的世界模拟而设计。它基于流匹配（Flow Matching）架构，结合 Cosmos-Reason1 物理 AI 视觉语言模型作为文本编码器，能够生成高质量、物理一致的视频世界，实现对机器人操作、自动驾驶等复杂动态场景的精准模拟与预测。

该模型整合了 2亿条 精选视频片段的预训练数据，涵盖机器人操作、自动驾驶、智能空间、人体动态及物理现象等多个领域，并通过强化学习后训练进一步优化视频质量与指令对齐能力。在 PAI-Bench 物理 AI 基准测试中，Cosmos-Predict2.5-2B 在文本生成世界（Text2World）和图像生成世界（Image2World）任务上均达到业界领先水平，整体评分与参数量大 7 倍 的 Wan2.2-27B-A14B 模型相当。

支持文本、图像、视频三种条件输入模式，可灵活应用于合成数据生成、策略评估、闭环仿真及多视角相机控制等任务。在机器人策略学习、自动驾驶仿真、VLA 模型训练数据合成等场景中表现优异，模型生成的单段视频长度约为 5.8 秒，通过逐段生成并拼接，成功实现长达 30-120 秒 的长时序一致视频生成，展现出强大的时序连贯性和物理合理性。

<br>


## Cosmos-Predict2.5-2B的相关代码仓拉取
如果已经安装安装git-lfs，可以跳过git-lfs安装这一步。如果昇腾A3环境中没有安装git-lfs，可以安装[git-lfs-linux-arm64-v3.7.1](https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-arm64-v3.7.1.tar.gz)，执行下面的步骤进行安装：
```bash
# 下载压缩包并解压
wget --no-check-certificate https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-arm64-v3.7.1.tar.gz

tar -xzf git-lfs-linux-arm64-v3.7.1.tar.gz

cd git-lfs-3.7.1

# 赋权限，然后进行安装
chmod +x install.sh

./install.sh

# 验证git-lfs的安装版本
git lfs --version
```

```bash
# 进入需要放置代码仓的本地目录下,执行下面的指令进行代码拉取替换：
git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git

git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git && cd cosmos-predict2.5 && git checkout 173b0fe980ab572f17c13258b775271f897b90ed

# 下载模型输入所需的json及视频到assets文件夹中
git lfs pull

# 回退到放置代码仓的本地目录
cd ../

# 复制和替换cann-recipes-embodied-intelligence代码仓cosmos-predict2.5中所有文件到官方代码仓cosmos-predict2.5中
\cp -rf cann-recipes-embodied-intelligence/world_model/cosmos-predict2.5/* ./cosmos-predict2.5
```
完成上述操作之后，最终cosmos-predict2.5根目录中相关代码目录树详见[附录：根目录相关代码目录树](#根目录相关代码目录树)。

<br>


## Cosmos-Predict2.5-2B在昇腾A3上的运行环境配置
### 与昇腾平台相关的环境配置
安装CANN软件包。本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.3.RC1`。

请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1)下载`Ascend-cann-toolkit_${version}_linux-${aarch}.run`与`Atlas-A3-cann-kernels_${version}_linux-${aarch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)依次进行安装。

- ${version}表示CANN包版本号，如 8.3.RC1
- ${aarch}表示CPU架构，如aarch64、x86_64

```bash
# ${cann_install_path}为CANN包的实际安装目录，注意每次新建终端时，首先source一下set_env.sh。
# 方式1：默认路径安装，以root用户为例
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 方式2：指定路径进行安装
source ${cann_install_path}/ascend-toolkit/set_env.sh
```

<br>

### uv环境管理工具安装（可选，如果当前环境已经安装uv，可以跳过）
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

### ffmpeg多媒体处理工具安装(版本4.4.2)
```bash
# 进入cosmos-predict2.5代码仓根目录
cd cosmos-predict2.5

# 安装ffmpeg==4.4.2到cosmos-predict2.5目录下的uv环境中
chmod +x ffmpeg_install.sh
./ffmpeg_install.sh
```


### decord(版本0.6.0)安装，注意和ffmpeg放置在同一文件夹下
```bash
# 如果已经在cosmos-predict2.5代码仓根目录，跳过这一步。否则，需要返回cosmos-predict2.5代码仓根目录
cd cosmos-predict2.5

# 安装decord==0.6.0到cosmos-predict2.5目录下的uv环境中
chmod +x decord_install.sh
./decord_install.sh
```


### 与昇腾服务器无关的其他环境配置
```bash
# 创建运行环境
uv sync
```

<br>


### Cosmos-Predict2.5-2B在昇腾上的推理步骤
运行下面的代码，即可自动下载多个关联模型，然后进行模型推理。
```bash
# 如果已经在cosmos-predict2.5代码仓根目录，跳过这一步。否则，需要返回cosmos-predict2.5代码仓根目录
cd cosmos-predict2.5

# 激活uv虚拟环境
source .venv/bin/activate

# 执行下面的推理命令进行视频生成，会自动下载huggingface上的模型到~/.cache/huggingface/hub/文件夹中
chmod +x inference_npu.sh
uv run bash inference_npu.sh
```

基于上述运行过程，耗时为920秒左右，得到预测的机器人倒水视频文件，保存在./outputs/base_video2world文件夹中，为robot_pouring.mp4视频文件。

下面是输入的语言指令、输入视频及输出预测视频的情况：

<details><summary>Input prompt</summary>
A robotic arm, primarily white with black joints and cables, is shown in a clean, modern indoor setting with a white tabletop. The arm, equipped with a gripper holding a small, light green pitcher, is positioned above a clear glass containing a reddish-brown liquid and a spoon. The robotic arm is in the process of pouring a transparent liquid into the glass. To the left of the pitcher, there is an opened jar with a similar reddish-brown substance visible through its transparent body. In the background, a vase with white flowers and a brown couch are partially visible, adding to the contemporary ambiance. The lighting is bright, casting soft shadows on the table. The robotic arm's movements are smooth and controlled, demonstrating precision in its task. As the video progresses, the robotic arm completes the pour, leaving the glass half-filled with the reddish-brown liquid. The jar remains untouched throughout the sequence, and the spoon inside the glass remains stationary. The other robotic arm on the right side also stays stationary throughout the video. The final frame captures the robotic arm with the pitcher finishing the pour, with the glass now filled to a higher level, while the pitcher is slightly tilted but still held securely by the gripper.
</details>

| Input Video | Output Video
| --- | --- |
| ![robot_pouring_input.gif](https://raw.gitcode.com/user-images/assets/7380116/5fd2d9ca-03a9-42fd-a373-205e2d6b47be/robot_pouring_input.gif 'robot_pouring_input.gif') | ![robot_pouring_output.gif](https://raw.gitcode.com/user-images/assets/7380116/4eb608b9-2ea5-4682-947a-2a9dec4004d6/robot_pouring_output.gif 'robot_pouring_output.gif') |


<br>


## Cosmos-Predict2.5-2B在昇腾上的精度验证步骤
### 基于视频观察和PAI-BENCH测试方法来验证其在昇腾A3上的推理精度
- 和NVIDIA平台生成的预测视频进行对比，观察A3环境生成视频是否一致。

- PAI-BENCH测试方案
  - 参考[PAI-BENCH测试框架](https://github.com/SHI-Labs/physical-ai-bench.git)中的[PAI-Bench-G (Video Generation)](https://github.com/SHI-Labs/physical-ai-bench/tree/main/generation#dataset)部分进行预测视频的质量评估。
  - 执行PAI-BENCH测试脚本之后，会在/physical-ai-bench/generation/evaluation_results/文件夹中生成一个results_xxxxxxxx_xxxxxx_eval_results.json文件,根据Cosmos-Predict2.5-2B模型代码仓/cosmos-predict2.5/assets/base/robot_pouring.json生成的robot_57.mp4（robot_pouring.mp4进行重命名），评估的质量分数中每一项得分参考下面的数值范围即可：
  <br>
  (1) aesthetic_quality（美学质量）             [0.63, 0.66]
  <br>
  (2) background_consistency（背景一致性）      [0.97, 1.00]
  <br>
  (3) imaging_quality（成像质量）               [0.70, 0.73]
  <br>
  (4) motion_smoothness（运动平滑度）           [0.98, 1.00]
  <br>	
  (5) subject_consistency（主体一致性）         [0.98, 1.00]
<br>


## Citation
```
@misc{nvidia2025worldsimulationvideofoundation,
      title={World Simulation with Video Foundation Models for Physical AI}, 
      author={NVIDIA and : and Arslan Ali and Junjie Bai and Maciej Bala and Yogesh Balaji and Aaron Blakeman and Tiffany Cai and Jiaxin Cao and Tianshi Cao and Elizabeth Cha and Yu-Wei Chao and Prithvijit Chattopadhyay and Mike Chen and Yongxin Chen and Yu Chen and Shuai Cheng and Yin Cui and Jenna Diamond and Yifan Ding and Jiaojiao Fan and Linxi Fan and Liang Feng and Francesco Ferroni and Sanja Fidler and Xiao Fu and Ruiyuan Gao and Yunhao Ge and Jinwei Gu and Aryaman Gupta and Siddharth Gururani and Imad El Hanafi and Ali Hassani and Zekun Hao and Jacob Huffman and Joel Jang and Pooya Jannaty and Jan Kautz and Grace Lam and Xuan Li and Zhaoshuo Li and Maosheng Liao and Chen-Hsuan Lin and Tsung-Yi Lin and Yen-Chen Lin and Huan Ling and Ming-Yu Liu and Xian Liu and Yifan Lu and Alice Luo and Qianli Ma and Hanzi Mao and Kaichun Mo and Seungjun Nah and Yashraj Narang and Abhijeet Panaskar and Lindsey Pavao and Trung Pham and Morteza Ramezanali and Fitsum Reda and Scott Reed and Xuanchi Ren and Haonan Shao and Yue Shen and Stella Shi and Shuran Song and Bartosz Stefaniak and Shangkun Sun and Shitao Tang and Sameena Tasmeen and Lyne Tchapmi and Wei-Cheng Tseng and Jibin Varghese and Andrew Z. Wang and Hao Wang and Haoxiang Wang and Heng Wang and Ting-Chun Wang and Fangyin Wei and Jiashu Xu and Dinghao Yang and Xiaodong Yang and Haotian Ye and Seonghyeon Ye and Xiaohui Zeng and Jing Zhang and Qinsheng Zhang and Kaiwen Zheng and Andrew Zhu and Yuke Zhu},
      year={2025},
      eprint={2511.00062},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.00062}, 
}

@misc{zhou2025paibenchcomprehensivebenchmarkphysical,
      title={PAI-Bench: A Comprehensive Benchmark For Physical AI}, 
      author={Fengzhe Zhou and Jiannan Huang and Jialuo Li and Deva Ramanan and Humphrey Shi},
      year={2025},
      eprint={2512.01989},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.01989}, 
}
```

<br>


## 附录
### 根目录相关代码目录树
- 检查整体代码目录树，经过上述的复制及替换操作，cosmos-predict2.5适配昇腾的根目录中的最终相关代码目录树如下所示：
```bash
cosmos-predict2.5
├── packages
|   ├── cosmos-oss
|   |   ├── pyproject.toml
└── decord_install.sh
└── ffmpeg_install.sh
└── inference_npu.sh
└── pyproject.toml
└── README.md
```