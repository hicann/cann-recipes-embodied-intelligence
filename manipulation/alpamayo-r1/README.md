# Alpamayo-R1 智能驾驶VLA大模型昇腾A2平台使用指南
<br>
本目录介绍在 A2 上如何对Alpamayo-R1（下文简称 AR-1）模型进行推理，并附带精度验证及输出示例。

## AR-1整体介绍
AR-1是英伟达提出的面向L4/L5级智能驾驶的视觉-语言-动作（VLA）大模型，旨在解决自动驾驶在长尾场景中因黑箱决策导致的脆弱性与不可解释性问题。该模型通过融合视觉感知与自然语言指令，构建结构化的场景理解，并引入显式的因果思维链（Chain-of-Causal-Thinking, CoC）机制，在推理过程中模拟人类驾驶员“观察—分析因果—预判后果—决策”的认知流程，从而生成更稳健、可验证的轨迹规划，同时输出自然语言形式的决策解释，提升系统透明度与人机协同效率。
<br>


## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.3.RC1.alpha001`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha001)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Atlas-A2-cann-kernels_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。
      - `${version}`表示CANN包版本号，如8.3.RC1.alpha001。
      - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装 uv (如果没有uv)
    ``` bash
    curl -LsSf https://astral.sh/uv/install.sh | sh    
    export PATH="$HOME/.local/bin:$PATH"
    ```
3. 克隆代码
    ```bash
    # 克隆cann-recipes-embodied-intelligence项目代码
    git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git
    
    # 克隆alpamayo原仓库代码
    git clone https://github.com/NVlabs/alpamayo.git
    cd alpamayo
    git reset --hard e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073
    cd ..

    # 应用适配昇腾A2的patch
    patch -p0 < cann-recipes-embodied-intelligence/manipulation/alpamayo-r1/patches/ar-1.patch

    # 将原仓库alpamayo的代码复制到本项目目录下
    cp -rn alpamayo/* cann-recipes-embodied-intelligence/manipulation/alpamayo-r1/
    cd cann-recipes-embodied-intelligence/manipulation/alpamayo-r1
    ```
4. 创建虚拟环境
    ```bash
    uv venv ar1_venv --python python3.11
    source ar1_venv/bin/activate
    ```
5. 安装环境依赖
    ```bash
    # 安装alpamayo-r1依赖
    uv sync --active
    ```
6. 安装physical-ai-av（physical-ai-av依赖python312，需手动修改依赖后安装，再修改语法）
    ```bash
    #手动下载tar包：
    wget https://files.pythonhosted.org/packages/25/20/730fab2dc243e720fa107520c187c7a3a928f471c40d367d55980032daac/physical_ai_av-0.1.0.tar.gz
    #解压
    tar -zxvf physical_ai_av-0.1.0.tar.gz
    #应用patch
    patch -p0 < patches/physical_ai_av-0.1.0-python311.patch
    # 安装
    cd physical_ai_av-0.1.0    
    uv pip install .
    cd ..
    ```

## 模型权重下载
```bash
    hf download nvidia/Alpamayo-R1-10B --local-dir model_ckpt
```

## 模型推理
```bash
    # 执行推理
    cd ./src/alpamayo_r1
    python test_inference.py
```

## AR-1在昇腾A2上的精度验证步骤
### 基于minADE指标来验证昇腾的推理精度
- 分别试验了多种输出预测轨迹，得到Physical-AI-AV-Dataset的chunk_0000共计100个clips的数据在昇腾A2平台上AR-1预测的轨迹路线minADE值，轨迹预测结果如表所示：

    |预测轨迹数|平台|minADE（越低越好）|
    |:---:|:---:|:---:|
    |1|昇腾A2|2.2026|
    |3|昇腾A2|0.9498|
    |5|昇腾A2|0.8514|
    |10|昇腾A2|0.5676|
<br>


## Citation
```
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```

<br>


## 附录
### lerobot根目录相关代码目录树
- 检查整体代码目录树，经过上述的复制及替换操作，AR-1适配昇腾的根目录中的最终相关代码目录树如下所示：
```bash
alpamayo/

├── patches/
├── src/
│   └── alpamayo_r1/
│       ├── action_space/
│       │   └── ...                      # Action space definitions
│       ├── diffusion/
│       │   └── ...                      # Diffusion model components
│       ├── geometry/
│       │   └── ...                      # Geometry utilities and modules
│       ├── models/
│       │   ├── ...                      # Model components and utils functions
│       ├── __init__.py                  # Package marker
│       ├── config.py                    # Model and experiment configuration
│       ├── helper.py                    # Utility functions
│       ├── load_physical_aiavdataset.py # Dataset loader
│       ├── test_inference.py            # Inference test script
├── pyproject.toml                       # Project 
├── README.md                       
```