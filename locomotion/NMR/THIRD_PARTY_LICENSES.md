Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.

SPDX-License-Identifier: Apache-2.0

# 第三方组件及许可证说明

本文件用于补充说明本项目中涉及的第三方源码引用、改写代码、工具脚本模板以及 Python 包依赖。它是对根目录 `LICENSE` 的补充，便于逐项确认每个外部组件的来源、版本、引入方式和开源协议。

说明：

- “源码引用/改写”表示本项目文件中存在来自上游仓库的代码片段、结构设计、工具脚本模板或较高相似度实现。
- “包管理依赖”表示通过 `setup.py` 安装或运行时依赖的第三方包。
- `betas.npy`、`gmr_mean.npy`、`gmr_std.npy`、`smplx_mean.npy`、`smplx_std.npy` 等为项目方为复现实验提前训练的辅助文件或统计文件，并非第三方源码、第三方模型文件或 SMPL-X 官方模型参数；若使用者基于其他数据重新训练，可自行重新生成对应文件。它们不作为第三方组件列入下表。
- 本文件不构成法律意见，仅用于工程合规梳理。

## 源码引用与改写

| 本地文件 / 说明 | 组件名称 | 仓库地址 | 依赖版本 | 引入方式 | 开源协议 | 是否允许商用 |
| --- | --- | --- | --- | --- | --- | --- |
| `src/models/tokenizers/encdoc/resnet.py`；扫描命中 `models/resnet.py`，文件头保留 Shanghai AI Laboratory 版权声明。 | MotionMillion-Codes ResNet blocks | https://github.com/VankouF/MotionMillion-Codes | `8a2a7df` / `8a2a7dfa66ecb6a1533d3d9cb49c743a697e1e1c` | 源码引用 / 改写 | Apache-2.0 | 是，按 Apache-2.0 条款 |
| `src/models/transformers/llama_ar.py`；扫描命中 `models/lit_llama/model_hf.py`，文件头保留 Shanghai AI Laboratory 版权声明。 | MotionMillion-Codes LLaMA / lit-llama reference | https://github.com/VankouF/MotionMillion-Codes | `8a2a7df` / `8a2a7dfa66ecb6a1533d3d9cb49c743a697e1e1c` | 源码引用 / 改写 | Apache-2.0 | 是，按 Apache-2.0 条款 |
| `src/utils/rotation_conversions.py`；扫描命中上游文件 src/common/geometry.py，文件保留 ankile 版权声明。 | robust-rearrangement geometry utilities | https://github.com/ankile/robust-rearrangement | `cf80d12d4acae9e4751a49407a96f85f07b342e3` | 源码引用 / 改写 | MIT | 是，按 MIT 条款 |
| `src/models/tokenizers/encdoc/encdoc_attn.py`；文件内 `# Ref:` 指向 `model/cnn_networks.py#L69`；非商业许可组件，不得默认理解为允许商用；相关代码片段仅可按 SnapMoGen 上游 Non-Commercial License 使用和分发。 | SnapMoGen | https://github.com/snap-research/SnapMoGen | `main` 分支，按文件内 `# Ref:` 链接确认 | 源码引用 / 改写 | Snap Inc. Non-Commercial License | 否，仅限非商业用途 |
| `src/models/transformers/llama_ar.py`；文件内说明 based on nanoGPT，文件头已保留 Andrej Karpathy 版权声明。 | nanoGPT | https://github.com/karpathy/nanoGPT | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | 源码引用 / 参考实现 | MIT | 是，按 MIT 条款 |
| `src/utils/rotation_conversions.py`；文件保留 Meta / Facebook 版权声明。 | PyTorch3D rotation conversions | https://github.com/facebookresearch/pytorch3d | `c307c64c7000cd370ff379be421bd92f6dec577b` | 源码引用 / 改写 | BSD-3-Clause | 是，按 BSD-3-Clause 条款 |
| `tools/nmr_inference.py`、`tools/train.py`；文件保留 OpenMMLab 版权声明 | OpenMMLab / MMEngine tool entrypoints | https://github.com/open-mmlab/mmengine | `mmengine>=0.10.0` / `508cb3268f25029787ed8dacb855809da0dca56a` | 源码引用 / 工具脚本改写 | Apache-2.0 | 是，按 Apache-2.0 条款 |
| `src/metrics/assets/g1/g1_custom_collision_with_fixed_hand.urdf`；来源为 TWIST 仓库 `assets/g1/g1_custom_collision_with_fixed_hand.urdf`。 | TWIST G1 URDF asset | https://github.com/YanjieZe/TWIST | `master` 分支 | 资产文件引用 | MIT | 是，按 MIT 条款 |

特别说明：SnapMoGen 相关代码片段采用 Snap Inc. Non-Commercial License，仅限非商业用途，不得默认理解为允许商用。涉及 `src/models/tokenizers/encdoc/encdoc_attn.py` 的 SnapMoGen 来源部分，其使用、修改和分发均需遵守 SnapMoGen 上游许可；本项目仅对 NMR 自研修改部分按 Apache-2.0 授权。

## 包管理依赖

以下依赖来自 `setup.py`。实际发布前建议在目标环境中使用 `pip-licenses` 导出精确版本和许可证信息。

| 组件名称 | 仓库地址 | 依赖版本 | 引入方式 | 开源协议 | 说明 |
| --- | --- | --- | --- | --- | --- |
| PyTorch | https://github.com/pytorch/pytorch | `torch>=2.0` | 包管理依赖 | BSD-3-Clause | `setup.py` 中声明。 |
| MMEngine | https://github.com/open-mmlab/mmengine | `mmengine>=0.10.0` | 包管理依赖 | Apache-2.0 | `setup.py` 中声明。 |
| smplx | https://github.com/vchoutas/smplx | 手动安装，未锁定版本 | 可选包管理依赖 | Software Copyright License for non-commercial scientific research purposes | 不在 `setup.py` 默认依赖中声明；如需运行 SMPL-X 相关功能，用户需确认并遵守 SMPL-X / SMPLify-X 官方许可后手动安装。该依赖及相关模型、数据和软件限制为非商业科研等用途，不能被默认理解为可自由再分发；商业使用需单独授权。 |
| GMR | https://github.com/YanjieZe/GMR | `master` 分支外部依赖 | 外部依赖 | MIT | 运行和数据准备流程中依赖的 General Motion Retargeting 项目；GitHub 页面声明 MIT License。 |
| NumPy | https://github.com/numpy/numpy | `numpy>=1.23,<2.0` | 包管理依赖 | BSD-3-Clause | `setup.py` 中声明。 |
| SciPy | https://github.com/scipy/scipy | `setup.py` 未锁定版本 | 包管理依赖 | BSD-3-Clause | `setup.py` 中声明。 |
| joblib | https://github.com/joblib/joblib | `setup.py` 未锁定版本 | 包管理依赖 | BSD-3-Clause | `setup.py` 中声明。 |
| Pinocchio | https://github.com/stack-of-tasks/pinocchio | `pin`，`setup.py` 未锁定版本 | 包管理依赖 | BSD-2-Clause | `setup.py` 中以包名 `pin` 声明。 |
| imageio | https://github.com/imageio/imageio | `imageio[ffmpeg]`，`setup.py` 未锁定版本 | 包管理依赖 | BSD-2-Clause | `setup.py` 中声明；`ffmpeg` extra 会按 Python 包管理规则引入传递依赖。 |
| OpenCV Python | https://github.com/opencv/opencv-python | `setup.py` 未锁定版本 | 包管理依赖 | Apache-2.0 | `setup.py` 中声明。 |
| Matplotlib | https://github.com/matplotlib/matplotlib | `setup.py` 未锁定版本 | 包管理依赖 | PSF-based License | `setup.py` 中声明。 |
| tqdm | https://github.com/tqdm/tqdm | `setup.py` 未锁定版本 | 包管理依赖 | MIT / MPL-2.0 | `setup.py` 中声明。 |
