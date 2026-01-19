# Pi0 具身智能VLA大模型昇腾使用指南

本目录介绍在 310P 上如何对 Pi0 模型进行离线模型转换及推理，附带精度验证步骤。

### Pi0整体介绍

在 π0: A Vision-Language-Action Flow Model for General Robot Control 论文中提出了 Pi0 模型，原仓库地址为：https://github.com/Physical-Intelligence/openpi

Lerobot 基于Openpi的代码进行了迁移，并在v0.4.0后重构了pi0和pi0.5代码结构。基于老Lerobot版本代码训练的模型在新版本代码推理上会出现诸多config类报错，
因此我们在本样例下，主要适配的基于‘577cd10974b84bea1f06b6472eb9e5e74e07f77a’ commit的Lerbot代码。

#### 功能介绍

Pi0 是一种典型的基于 Diffusion 的模仿学习控制策略，该模型将机器人策略表示为一个条件去噪扩散过程，在叠衣服、整理桌面等具有挑战性的任务上表现良好，为主流VLA模型。

### Pi0的相关代码仓拉取、仿真测试集和模型下载

本样例使用的示例模型为：https://huggingface.co/BrunoM42/pi0_aloha_transfer_cube

该模型是一个在MuJoCo仿真环境中经过微调的模型，任务为双臂递方块。

#### 模型输入输出

| **输入数据名**                        | **数据类型(dtype)** | **数据大小(shape)** |
| ------------------------------------- | ------------------- | ------------------- |
| observation.images_xxx （摄像头图像） | Float32             | $[1, N_c, 3, H, W]$ |
| observation.state （机器人位姿）      | Float32             | $[1, N_k]$          |

| **输出数据名** | **数据大小(shape)**     |
| -------------- | ----------------------- |
| actions        | $[1, chunk\_size, N_k]$ |

**参数符号说明：**

- $N_c$ ：摄像头个数（图片个数），训练时决定，通常为一路摄像头～三路摄像头。
- $N_k$ ：自由度，机器人构型决定。
- $H, W$： 摄像头拍摄rgb图像分辨率。
- $chunk\_size$：机器人执行步数个数，定义为 $chunk\_size$ 。

输入输出格式定义在模型文件中的config.json中，样例示例为一路摄像头，内容如下：

```
"input_features": {
        "observation.images.top": {
            "type": "VISUAL",
            "shape": [
                3,
                480,
                640
            ]
        },
        "observation.state": {
            "type": "STATE",
            "shape": [
                14
            ]
        }
    },
    "output_features": {
        "action": {
            "type": "ACTION",
            "shape": [
                14
            ]
        }
    },
"chunk_size": 50,
```

### Pi0在昇腾310P上的运行配置

#### 与昇腾平台相关的环境配置

.om模型转化及运行需要安装CANN软件包。

本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.0.0-8.2.RC1`。 请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)下载对应架构软件包，例如` Ascend-cann-toolkit_8.2.RC1_linux-x86_64.run`与` Ascend-cann-kernels-310p_8.2.RC1_linux-x86_64.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)依次进行安装。

```
# xxxx为CANN包的实际安装目录，注意每次新建终端时，激活一下setenv.bash
source xxxx/ascend-toolkit/setenv.bash
```

#### 与昇腾服务器无关的环境配置

```bash
# 1) 拉取 GitCode 本 PI0 使用仓库并切换指定分支
git clone https://gitcode.com/CANN/cann-recipes-embodied-intelligence.git

# 2) 拉取 lerobot 仓库并切到指定 **commit**
cd cann-recipes-embodied-intelligence/manipulation
git clone https://github.com/huggingface/lerobot.git  # **如果contrib下没有lerobot目录**
cd lerobot
git checkout 577cd10974b84bea1f06b6472eb9e5e74e07f77a

# 3) 创建运行环境
conda create -y -n lerobot python=3.10 # **如果之前没有创建过该环境**
conda activate lerobot

# 4) 安装 lerobot（本地可编辑安装）
pip install -e . # **如果使用不同版本的 lerobot，请重新执行pip install -e .进行更新**

# 5) 仿真依赖（Aloha 仿真需要 gym_aloha / gym-aloha）
# 推荐用 extra 一次性安装：
pip install -e ".[aloha]"

# 6) 固定 numpy 版本（如你的环境需要）
pip install numpy==1.26.0 # 使用atc工具时推荐使用该版本numpy以避免潜在兼容性问题
pip install transformers==4.55.4 # 根据使用的模型训练时所用的transformers版本进行安装，本样例模型使用该版本
pip install decorator==5.2.1

# 7) 库文件替换
export LEROBOT=/path/to/manipulation/lerobot
export INFER_WITH_OM=/path/to/pi0/infer_with_om
cd /path/to/conda/envs/pi0/lib/python3.10/site-packages/transformers/models/gemma # 可以通过pip show transformers  查看transformers安装路径
git apply --check -p1 $INFER_WITH_OM/lib/modeling_gemma.patch # patch补丁文件在当前infer_with_om的lib文件夹内，没有报错则表示可以应用补丁
git apply -p1 $INFER_WITH_OM/lib/modeling_gemma.patch

# 8) 模型分部文件添加
cd contrib/pi0_on_310p
cp $INFER_WITH_OM/lerobot_modify/modeling_pi0_vlm.py           $LEROBOT/src/lerobot/policies/pi0/
cp $INFER_WITH_OM/lerobot_modify/modeling_pi0_action_expert.py $LEROBOT/src/lerobot/policies/pi0/
cp $INFER_WITH_OM/lerobot_modify/paligemma_with_expert_fp16.py $LEROBOT/src/lerobot/policies/pi0/
# Copy normalization fix
cp $INFER_WITH_OM/lerobot_modify/normalize.py                  $LEROBOT/src/lerobot/policies/
```

#### Pi0在昇腾310P上的推理步骤

Pi0分为VLM的PaliGemma和动作专家的Gemma两个模块，在推理流上是解耦的。且Gemma这个模块由于是扩散执行十轮，如果直接编译成离线图会占据大量内存和时间，因此我们单独编译了part1的Paligemma和part2的Gemma,分别转化为两个OM文件，在推理时分两步加载执行。Part2依赖于Part1运行时保存的中间张量，并且需多次调用(Pi0模型中动作专家需执行10次)。

Pi0架构图：![Pi0 arch](https://raw.gitcode.com/user-images/assets/7380116/cda2b3e3-dd28-4eba-96bb-3c3f3f660bf1/image.png)

本节介绍离线推理模式（通过昇腾亲和的OM文件）的部署参考，更多使用参数可参考[atc工具使用文档](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/atctool/atlasatc_16_0003.html)。

<img src="https://raw.gitcode.com/user-images/assets/7380116/c583e4bd-fddf-4d44-bc65-7ad69d84ab02/om_compile_workflow.png" style="zoom:50%;" />

下面给出一条推荐的单机器链路：

1) 在 310P 宿主机导出 ONNX（使用Host CPU）
2) 使用 ATC 将 ONNX 转为 OM（在 310P 上）
3) 使用 OM-backend sim-evaluator 在仿真环境评测（在 310P 上）

需要在转化onnx的机器上额外安装onnx runtime依赖:
```bash
pip install onnx pytest onnxscript
# 基于Host CPU转换onnx请安装(310P 宿主机执行):
pip install onnxruntime
# 基于Host GPU转换onnx请安装：
pip install onnxruntime-gpu
```

#### 1) 导出 ONNX

在 Host（CPU 或 GPU）上执行：

```bash
# 以本地目录为例：pi0_model/ 里包含 config.json 等文件
# 也可以先用 huggingface-cli 下载到 pi0_model/:
#   pip install -U huggingface_hub
#   huggingface-cli download BrunoM42/pi0_aloha_transfer_cube --local-dir pi0_model
cd $INFER_WITH_OM
mkdir runtime_save # 用于保存中间运行时张量
export PYTHONPATH=/path/to/manipulation/lerobot/src:$PYTHONPATH # 需要增加PYTHONPATH环境变量指向lerobot路径保证成功import
./run_pi0_export.sh --pretrained-policy-path ./pi0_model # 替换为你的模型目录路径
```

说明：
- 输入 schema（包含多个摄像头 key 时也支持）严格来自 `config.json.input_features`。
- 默认会用 ONNXRuntime CPU 对比 PyTorch 输出，打印 max/mean abs diff；如需跳过可加 `--no-validate`。


#### 2) ATC 将 ONNX 转为 OM

在 310P 上（已安装并 `source` CANN 环境）执行 ATC：

```bash
# 推荐直接用 atc（路径按你的实际文件位置修改）
atc --model=outputs/onnx/pi0-vlm.onnx \
        --framework=5 \
        --output=outputs/om/pi0_vlm \
        --soc_version=Ascend310P1 \
        --precision_mode_v2=origin

atc --model=outputs/onnx/pi0_action_expert.onnx \
        --framework=5 \
        --output=outputs/om/pi0_action_expert \
        --soc_version=Ascend310P1 \
        --precision_mode_v2=origin
```
soc_version 需要根据 'npu-smi info' 得到的Name Device中芯片型号填写soc_version，比如以下为"310P1"，那么soc_version则填写Ascend310P1。

![npu-smi info](https://raw.gitcode.com/user-images/assets/7380116/91e63da3-7c42-4b70-b7d5-47fa358beeed/npu.jpg)
若报错 `[ERROR] Execute model failed for acl.mdl.execute error 507011`，检查 `npu-smi info` 显示的信息是否与 ATC 转化时使用的 `--soc_version` 相同

当模型转换完成后，当前目录应当存在名为 `pi0_vlm.om` 和 `pi0_action_expert.om`(或者类似`pi0_action_expert_linux_x86_64.om`等包含适配系统信息的格式) 的模型（或者 `--output` 参数指定目录下）,在终端中有输出“ATC run success, welcome to the next use”，后续使用基于生成的 om 对应文件名。

##### 基于mock的数据输入，CPU/GPU与原始Pytorch输出相似度对比

构造输入数据测试Pytorch CPU/GPU和OM 310P NPU的输出余弦相似度对比：

```bash
# 在 310P 上执行（需要 ACL/AclLite Python 依赖）
# https://gitee.com/ascend/ACLLite#%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B 可参考这个进行安装安装后需要增加PYTHONPATH环境变量指向acllite的python包路径
# 如 export PYTHONPATH=/path/to/Ascend/ascend-toolkit/8.2.RC1/thirdpart/python/:$PYTHONPATH 具体路径按照实际安装路径修改（可以通过sudo find / -name "acllite_utils.py" 来查找，设置环境变量为该文件所在目录的上一级目录）
python3 verify_om_onnx_vlm.py \
    --onnx-model-path outputs/onnx/pi0-vlm.onnx \
    --om-model-path outputs/om/pi0-vlm.om

python3 verify_om_onnx_action_expert.py \
    --onnx-model-path outputs/onnx/pi0_action_expert.onnx \
    --om-model-path outputs/om/pi0_action_expert.om 
```
该脚本会根据 config.json 生成确定性的 dummy 输入（支持多摄像头输入），并对比 ONNXRuntime(CPU) vs OM(NPU)

##### 端到端 Pi0 模型推理

Pi0 模型推理需要分两步调用，先调用 part1 的 VLM 模块，保存中间张量后再多次调用 part2 （10次）的动作专家模块，最后将动作输出反归一化映射到原来的动作空间。

```bash
cd ../pi0/infer_with_om

python ./run_om_e2e.py \
    --vlm-model-path ./outputs/om/pi0_vlm.om \
    --action-expert-model-path ./outputs/om/pi0_action_expert.om

# 反归一化的参数默认值为示例模型的参数若需要自定义模型的反归一化参数，若修改模型后需要获取反归一化参数，可以先运行以下命令获取mean.pt和std.pt文件：
# --policy.path 为你的safetensors模型路径
# 注意，若仅运行示例模型，无需获取mean.pt和std.pt文件，直接使用默认值即可

# 此处可以按照你的实际渲染环境配置MUJOCO_GL变量，若无GPU渲染需求，推荐使用osmesa渲染，可先安装osmesa库：
# sudo apt-get install libosmesa6 libosmesa6-dev
# 若未配置代理则使用 huggingface 镜像网站
cd ../../lerobot
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=osmesa
python ./src/lerobot/scripts/eval.py  --policy.path=/path/to/your/safetensors/file  --env.type=aloha  --env.task=AlohaTransferCube-v0  --env.episode_length=10

cd ../pi0/infer_with_om

# 获取到 mean.pt 和 std.pt 文件后，运行端到端OM推理：
python ./run_om_e2e.py \
    --mean-path ../../lerobot/mean.pt \
    --std-path ../../lerobot/std.pt \
    --vlm-model-path ./outputs/om/pi0_vlm.om \
    --action-expert-model-path ./outputs/om/pi0_action_expert.om
```

#### 仿真环境 Pi0 评测
```bash
# 此处policy.path替换为你的safetensors模型路径，实际仅加载config.json用于环境配置，可以不包含权重文件
export MUJOCO_GL=osmesa
python ./eval_pi0_ascend.py \
    --policy.path=/path/to/models/pi0_aloha_model \
    --env.type=aloha --env.task=AlohaTransferCube-v0  \
    --env.episode_length=600  \
    --eval.n_episodes=1  \
    --eval.batch_size=1  \
    --seed=42 \
    --vlm-model-path /path/to/pi0_vlm.om \
    --action-expert-model-path /path/to/pi0_action_expert.om
```

仿真示例效果：
![pi0 sim-demo](https://raw.gitcode.com/user-images/assets/7380116/666ba75f-5bb0-45d3-9c2b-564912cad9d7/pi0.gif)

微调后的Jaka 实机示例效果：
![pi0 real-demo](https://raw.gitcode.com/user-images/assets/7380116/a95eee94-2b4a-4905-b80f-7f85d24cbe36/pi0-pick.gif)

### 可能遇到的问题
1. 运行lerobot/scripts/eval.py时，若使用网络环境下载`google/paligemma-3b-pt-224`模型，需提前取模型对应的 huggingface 页面请求访问 Access，参考详见`https://huggingface.co/docs/huggingface_hub/main/cn/quick-start`和`https://huggingface.co/docs/hub/models-gated` 
2. 若网络环境下载huggingface模型较慢，遇到下载`google/paligemma-3b-pt-224`卡顿，可手动下载模型到本地路径，再修改`lerobot/src/lerobot/policies/pi0/modeling_pi0.py`中对应247行、`lerobot/src/lerobot/policies/pi0/modeling_pi0_vlm.py`中对应306行、`lerobot/src/lerobot/policies/pi0/modeling_pi0_vlm.py`中对应350行处：`google/paligemma-3b-pt-224`为本地路径。
```
@misc{black2024pi0visionlanguageactionflowmodel,
      title={$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control},
      author={Kevin Black and Noah Brown and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Lucy Xiaoyang Shi and James Tanner and Quan Vuong and Anna Walling and Haohuan Wang and Ury Zhilinsky},
      year={2024},
      eprint={2410.24164},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.24164},
}

@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

### 附录

#### lerobot 根目录相关代码目录树

- 检查整体代码目录树，经过上述的操作，pi0适配昇腾的lerobot根目录中的最终相关代码目录树如下所示：

````text
|-- manipulation/pi0/infer_with_om/               # 本目录
├── README.md                                     # 使用指南（本文件）
├── convert_verify_onnx_action_expert.py          # PyTorch -> ONNX（动作专家）转换与验证
├── convert_verify_onnx_vlm.py                    # PyTorch -> ONNX（VLM部分）转换与验证
├── eval_pi0_ascend.py                           # Pi0 昇腾仿真评测脚本
├── run_om_e2e.py                                 # Pi0 昇腾端到端推理脚本
├── verify_om_onnx_action_expert.py               # ONNXRuntime(CPU) vs OM(NPU) 误差对比（动作专家）
├── verify_om_onnx_vlm.py                         # ONNXRuntime(CPU) vs OM(NPU) 误差对比（VLM）
├── run_pi0_export.sh                             # 示例导出脚本
├── lerobot                                       # lerobot代码文件夹
└── lib/
├   └── modeling_gemma.patch                      # Gemma 相关补丁
└── outputs/                                    # 输出结果
    └── onnx/
        ├── pi0-vlm.onnx
        └── pi0_action_expert.onnx
    └── om/
        ├── pi0-vlm.om
        └── pi0_action_expert.om
```
````
