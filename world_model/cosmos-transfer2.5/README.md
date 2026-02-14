# cosmos-transfer2.5-2B视频风格转换具身智能世界模型昇腾使用指南
<br>


## cosmos-transfer2.5-2B整体介绍
论文题目：World Simulation with Video Foundation Models for Physical AI

中文译文：基于视频基础模型的物理智能世界仿真

### 功能介绍

Cosmos-Transfer2.5是NVIDIA Cosmos平台的世界基础模型(World Foundation Models)之一,专为物理AI应用设计的多控制网络模型。该模型支持视频风格转换功能,可基于多种控制信号(深度图、语义分割、边缘检测等)实现视频到视频的风格迁移与内容转换,将输入视频转换为符合特定视觉风格或场景需求的输出视频。模型支持接收多种视频模态的结构化输入,包括RGB、深度图、语义分割、边缘检测等,可用于自动驾驶场景模拟、机器人视觉仿真、视频内容创作等真实物理系统场景。本样例基于NVIDIA Cosmos-Transfer2.5-2B模型完成NPU适配优化,实现在昇腾NPU上的高效推理。

<br>


## Cosmos-transfer2.5-2B的相关代码仓拉取
如果昇腾A3环境中没有安装git-lfs，可以安装[git-lfs-linux-arm64-v3.7.1](https://github.com/git-lfs/git-lfs/releases/download/v3.7.1/git-lfs-linux-arm64-v3.7.1.tar.gz),解压缩之后执行下面的步骤进行安装

```
# 赋权限
chmod +x install.sh
./install.sh

# 验证git-lfs的安装版本
git lfs --version
```
```bash
# 进入需要放置代码仓的本地目录下,执行下面的指令进行代码拉取替换：
git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git

git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git && cd cosmos-transfer2.5 && git checkout 8b0e6af4b3bed40408c5762e528cb4e2a233f278

git lfs pull

# 回退到放置代码仓的本地目录
cd ../

cp -rf cann-recipes-embodied-intelligence/world_model/cosmos-transfer2.5/* ./cosmos-transfer2.5
```
完成上述操作之后，最终cosmos-transfer2.5根目录中相关代码目录树详见[附录：根目录相关代码目录树](#根目录相关代码目录树)。
<br>


## Cosmos-transfer2.5-2B在昇腾A3上的运行环境配置
### 与昇腾平台相关的环境配置
安装CANN软件包。本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.3.RC1`。
请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1)下载`Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run`与`Atlas-A3-cann-kernels_8.3.RC1_linux-aarch64.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)依次进行安装。

```bash
# cann_path为CANN包的实际安装目录，注意每次新建终端时，首先source一下set_env.sh
export cann_path=/usr/local/Ascend/ascend-toolkit  # cann包安装路径
source ${cann_path}/set_env.sh
```
<br>

### uv环境管理工具安装（可选，如果当前环境已经安装uv，可以跳过）
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

### ffmpeg多媒体处理工具安装(版本4.4.2)
```
# 进入cosmos-transfer2.5代码仓根目录
cd cosmos-transfer2.5

chmod +x ffmpeg_install.sh
./ffmpeg_install.sh
```


### decord(版本0.6.0)安装，注意和ffmpeg放置在同一文件夹下
```
# 再次返回到cosmos-transfer2.5代码仓根目录
cd cosmos-transfer2.5

# 执行下面的指令进行decord安装
chmod +x decord_install.sh
./decord_install.sh
```
### 安装libGL
```
yum install -y libGL libGLU libEGL libX11-devel
```

### 与昇腾服务器无关的其他环境配置
```bash
# 创建运行环境
uv sync
```


<br>


### Cosmos-Transfer2.5-2B在昇腾上的推理步骤


运行下面的代码，即可自动下载多个关联模型，然后进行模型推理。
```bash
# 进入cosmos-transfer2.5代码仓根目录
cd cosmos-transfer2.5

#移除GPU库
chmod +x nvidia_remove.sh 
./nvidia_remove.sh

# 激活uv虚拟环境
source .venv/bin/activate

# 执行下面的推理命令进行视频生成，会自动下载huggingface上的模型到~/.cache/huggingface/hub/文件夹中
python examples/inference.py \
  -i assets/robot_example/multicontrol/robot_multicontrol_spec.json \
  -o outputs/multicontrol \
  --disable-guardrails 
```

## Cosmos-transfer2.5-2B在昇腾上的精度验证步骤
### 基于视频观察和PAI-BENCH测试方法来验证其在昇腾A3上的推理精度
- 和NVIDIA平台生成的预测视频进行对比，观察A3环境生成视频是否一致。
- PAI-BENCH测试方案
  - 参考[PAI-BENCH测试框架](https://github.com/SHI-Labs/physical-ai-bench.git)中的[PAI-Bench-C(Conditional Video Generation)](https://github.com/SHI-Labs/physical-ai-bench/tree/main/conditional_generation)部分进行预测视频的质量评估。
  - 执行PAI-BENCH测试脚本之后，会在路径下生成一个json文件，其中包含了dover_tech_score、blur_ssim、canny_f1_score、seg_m_iou等多项精度指标。


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
- 检查整体代码目录树，经过上述的复制及替换操作，cosmos-transfer2.5适配昇腾的根目录中的最终相关代码目录树如下所示：
```bash
cosmos-transfer2.5
├── adaptor_patches
│   ├── graph_patch.py
│   ├── minimal_v4_dit_patch.py
│   ├── minimal_v4_lvg_dit_control_vace_patch.py
│   └── qwen2_5_vl_patch.py
├── cosmos_transfer2
│   └── __init__.py
├── decord_install.sh
├── examples
│   ├── inference.py
│   └── model_adaptor.py
├── ffmpeg_install.sh
├── nvidia_remove.sh
├── packages
│   └── cosmos-oss
│       ├── cosmos_oss
│       │   └── __init__.py
│       └── pyproject.toml
├── pyproject.toml
└── README.md
```