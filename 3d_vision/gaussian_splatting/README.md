# 基于昇腾平台的3D Gausssian Spaltting的训推优化实践
3D Gausssian Spaltting(3DGS)是2023年由法国和德国研究人员提出的一种可微渲染算法，以3D高斯椭球为核心图元，通过可微光栅化完成三维场景的重建与渲染，打破传统建模“速度与保真度不可兼得”的瓶颈。本项目旨在提供3DGS的昇腾适配版本。

本项目基于NPU主要实现了以下优化点，具体内容可至[NPU 3DGS训推优化实践](../../docs/3d_vision/gaussian_splatting/)查看：
- Alpha-blending优化算法及融合算子实现优化；
- 视锥剔除融合算子优化；
- Gaussian负载均衡优化；
- Precise Intersection融合算子优化。
## 执行样例
本样例支持昇腾Atlas A2环境的单卡训练和推理。
### CANN环境准备
1. 本样例的执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），目前使用CANN软件版本为CANN 8.2.RC1。 请从[CANN软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)进行安装。

2. 本样例依赖的**torch与torch_npu版本分别为2.1.0和2.1.0.post12**，请从[Ascend Extension for PyTorch插件](https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html)下载并安装torch与torch_npu安装包。
```
conda create -n 3dgs python=3.8
conda activate 3dgs
```
### 自定义npu算子编译安装
本项目需要将自定义npu算子编译安装至conda环境中。
1. 推荐使用gcc 10.2版本编译，如需安装依赖，执行以下指令：
```
pip install numpy==1.23 decorator sympy scipy attrs cloudpickle psutil synr==0.5.0 tornado cmake pyyaml expecttest protobuf
```
2. 克隆原始仓:
```
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
```
3. 编译自定义npu算子:
```
cd ops/ascendc
bash build.sh --python=3.8
```
参数`--python`指定编译使用的python版本，支持3.8及以上版本。编译成功后会在`ops/ascendc`目录下生成`build`, `meta_gauss_render.egg-info`, `dist`文件夹，生成的whl包在`dist`目录下。

4. 安装npu算子

在`ops/ascendc`路径下执行：
```
pip install dist/*.whl --force-reinstall
```
### python外部依赖安装
```
cd ../../3d_vision/gaussian_splatting
pip install -r requirements.txt
```
### 数据集准备

在`3d_vision/gaussian_splatting`路径下执行：
```
python datasets/download_dataset.py
```
数据集会被下载并解压到`3d_vision/gaussian_splatting/data/360_v2`路径下。
### 快速启动
本样例准备了单卡环境下的训练和推理脚本。执行脚本前，参考[Ascend社区](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)中的CANN安装软件教程配置环境变量：
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
**运行训练脚本：**
```
ASCEND_RT_VISIBLE_DEVICES=1 python train.py
```
`ASCEND_RT_VISIBLE_DEVICES=1`指定npu设备1卡，不指定则默认启动0卡。模型训推的配置文件为`rasterization/config.py`，`--data_dir`指定训练数据集路径，默认为`data/360_v2/garden`，`--result_dir`指定训练结果输出路径，默认为`results/garden`。
```
python train.py --data_dir data/360_v2/bonsai --result_dir results/bonsai
```


**运行推理脚本：**
```
python train.py --data_dir data/360_v2/bonsai --result_dir results/bonsai --ckpt path_to_ckpt.pt
```
推理命令传入`--ckpt`参数指定训练保存的模型，如`results/bonsai/ckpts/ckpt_29999_rank0.pt`，对输入数据集进行渲染与评估。数据集与评估模型的场景类别需要保持一致，推理和评估结果保存在`result_dir`目录下。

**训练所有场景数据集：**

`train_all.sh`按脚本中指定顺序执行数据集中所有场景的训练和推理。
```
bash train_all.sh
```

## 目录结构介绍
```
3d_vision
├── gaussian_splatting
│  ├── datasets             # 数据集下载及解析模块
|  ├── gsplat               # gsplat框架提供的工具函数和分布式训练模块
|  ├── rasterization        # 算法主要实现
|     ├── config.py         # 模型训练、推理配置文件
|     ├── utils.py          # 渲染和模型训练相关工具函数
|     ├── runner.py         # 训练和推理引擎
|     ├── rasterizer.py     # 3dgs渲染流程实现，内部调用并串联各自定义算子
|  ├── README.md            # 项目说明
|  ├── requirements.txt     # 外部依赖
|  ├── train.py             # 训练和推理启动脚本
|  ├── train_all.sh         # 所有场景的训推拉起脚本
└── ...

```

## Citation
```
@article{kerbl3Dgaussians,
    author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal={ACM Transactions on Graphics},
    number={4},
    volume={42},
    month={July},
    year= {2023},
    url={https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

@article{ye2025gsplat,
    title={gsplat: An open-source library for Gaussian splatting},
    author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
    journal={Journal of Machine Learning Research},
    volume={26},
    number={34},
    pages={1--17},
    year={2025}
}
```
