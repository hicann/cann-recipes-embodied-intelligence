# 昇腾部署通用排查手册

本文档面向在昇腾平台上部署本仓库样例的用户,提取**多个样例共同出现**的部署问题与解决方法,作为通用排查手册使用。

- 只收录在 **2 个及以上样例**中反复出现的共性问题;仅在单个样例出现的特有问题,请查阅该样例自己的 README。
- 不含设计说明与性能优化技巧,只聚焦"装不上 / 跑不起来 / 结果不对"这类部署阻塞问题。

## 部署通用流程

本仓库各样例的部署大体遵循同一条主线,本文档即按此流程的四个阶段组织:

```
① 环境准备          ② 模型获取          ③ 模型转换/导出        ④ 运行/评测
  CANN + 依赖   →    权重下载/本地化  →   ONNX→OM(仅OM样例)  →   推理 / 仿真评测
```

遇到问题时,先判断卡在哪个阶段,再到对应章节查找。

---

## ① 环境准备阶段

### 1.1 torch_npu / 昇腾软件栈未就绪

**阶段背景**:act、pi0、smolvla 等训练样例(`*/train/`)都提供 `setup.sh` 作为环境准备入口,负责创建 conda 环境并安装 `torch` / `torchvision` / `torch_npu` 等依赖。

**现象**:`setup.sh` 末尾的 `import torch, torch_npu` 校验失败,或训练启动即报 NPU 侧依赖缺失。

**根因**:`torch` / `torchvision` / `torch_npu` 的有效组合依赖 CANN 版本与机器架构,样例不硬编码下载链接,环境不匹配时装不上。

**解决**:
- 方案一:先激活一个已验证可用的 Ascend 训练环境,再执行 `setup.sh`;
- 方案二:执行时显式传入本地 wheel:`--torch-wheel` / `--torchvision-wheel` / `--torch-npu-wheel`;
- 已确认平台栈可用时,可追加 `--skip-torch-check` 跳过末尾导入校验。

**出现于**:`manipulation/act/train`、`manipulation/pi0/train`、`manipulation/smolvla/train`。

### 1.2 `set_env.sh` 未 source,CANN 环境变量缺失

**阶段背景**:依赖 CANN 工具链的样例(ATC 转换、acl 推理、torch_npu 适配)都要求先 source CANN 的 `set_env.sh`,否则后续命令找不到工具或库。

**现象**:ATC 编译直接失败;或运行 NPU 适配脚本报环境变量/路径错误。

**根因**:CANN 的环境变量未生效。该操作不会跨终端持久化,**每开一个新终端都要重新 source**。

**解决**:
```bash
source ${cann_install_path}/ascend-toolkit/set_env.sh
```
确认 CANN 已按样例要求的版本安装(各样例 README 标注了支持的 CANN 版本)。

**出现于**:`manipulation/pi0/infer_with_om`、`manipulation/pi05/infer_with_om`、`world_model/cosmos-predict2.5`、`world_model/cosmos-transfer2.5` 等所有依赖 CANN 工具链的样例。

---

## ② 模型获取阶段

### 2.1 HuggingFace 模型下载失败 / gated model 未授权 / 离线仍联网

**阶段背景**:多数 VLA 样例依赖从 HuggingFace 拉取基础权重,如 `google/paligemma-3b-pt-224`、`pi0_base`、`smolvla_base` 等,其中 `paligemma` 为 gated model。

**现象**:下载超时、403、长时间卡住;或在离线环境启动时仍尝试访问 HuggingFace。

**根因**:gated model 需先授权;无网/弱网环境无法在线拉取;部分代码中模型名是硬编码的在线 repo id。

**解决**(优先尝试镜像):
- **首选**:设置国内镜像后再拉取,通常即可解决下载慢/卡住:
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- gated model 仍需提前在 HuggingFace 模型页面完成访问授权;
- 完全离线时:先在有网机器下载,再把模型目录拷贝到运行环境;
- 若代码中硬编码了在线 repo id,把它改为**本地路径**,具体位置各样例 README 已列出:
  - pi0:`modeling_pi0.py:247`、`modeling_pi0_vlm.py:306`、`modeling_pi0_vlm.py:350`
  - pi05:`processor_pi05.py:145`

**出现于**:`manipulation/pi0/*`、`manipulation/pi05/*`、`manipulation/smolvla/train`、`manipulation/Isaac-GR00T`。

---

## ③ 模型转换/导出阶段(仅 OM 推理样例)

> 本阶段问题仅出现在走 "PyTorch → ONNX → ATC → OM" 链路的 `infer_with_om` 样例(pi0、pi05、act、diffusion-policy)。

### 3.1 ATC 编译失败 / `--soc_version` 与芯片不符

**现象**:ATC 报算子、shape、dtype 或 `soc_version` 相关错误;OM 生成后执行报 `acl.mdl.execute error 507011`。

**根因**:CANN 环境未生效,或 `--soc_version` 与实际芯片型号不一致,或复用了旧 OM。

**解决**:
- 确认已 `source set_env.sh`(见 1.2);
- `npu-smi info` 查芯片型号,`--soc_version` 与之一致(如 310P1 用 `Ascend310P1`);
- 保持样例 README 的精度配置(如 pi0 的 `--precision_mode_v2=origin`);
- 更换模型 / 输入分辨率 / 动作维度后,必须重导 ONNX 并重转 OM,**不要复用旧 OM**。

**出现于**:`manipulation/pi0/infer_with_om`、`manipulation/pi05/infer_with_om`。

### 3.2 `import acl` / ACLLite 模块找不到

**现象**:`ModuleNotFoundError: acllite_*`、`import acl` 失败,OM 校验脚本启动即失败。

**根因**:ACLLite 路径未加入 `PYTHONPATH`。

**解决**:
```bash
source /path/to/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/path/to/Ascend/ascend-toolkit/latest/thirdpart/python/:$PYTHONPATH
# 路径不确定时:find /path/to/Ascend -name acllite_utils.py
```

**出现于**:`manipulation/pi0/infer_with_om`、`manipulation/pi05/infer_with_om`。

### 3.3 输入 shape / dtype / 顺序不匹配

**现象**:OM 推理报输入 shape 不匹配;或 ONNX 与 OM 输出精度对不齐、动作输出异常。

**根因**:测试输入张量的 key、图像分辨率、`chunk_size`、动作维度与模型配置不一致;或 OM 输入顺序、dtype 与 ONNX 导出时不一致(NPU/ATC 对 dtype 组合较敏感)。

**解决**:
- 确认输入张量(如 `start_obs_0.pt`)的 key 与脚本读取 key 完全一致;
- 确认图像分辨率、`chunk_size`、动作维度与模型 `config.json` 一致;
- 确认 OM 输入顺序与 ONNX `input order` 一致,各输入 dtype 与导出时一致;
- 建议分段单独验证 ONNX vs OM 精度,缩小定位范围。

**出现于**:`manipulation/pi0/infer_with_om`、`manipulation/pi05/infer_with_om`。

### 3.4 归一化处理器(processor)丢失

**现象**:OM 能跑通,但输出 action 无法正确落回环境动作空间。

**根因**:训练时对 observation/action 做了归一化,推理需要 `policy_preprocessor` / `policy_postprocessor` 才能正确预处理和反归一化;部分下载的模型只含 `model.safetensors + config.json`,缺这两组产物。

**解决**:
- 先检查模型目录是否自带 `policy_preprocessor.json` / `policy_postprocessor.json` 及对应 safetensors;
- 缺失则在 Host 侧运行迁移脚本生成 `<模型目录>_migrated/`:
  ```bash
  python3 .../lerobot/processor/migrate_policy_normalization.py --pretrained-path <模型目录>
  ```
- 评测脚本会优先在 `--policy.path` 目录查找,找不到则自动回退到同级 `<dir>_migrated/`。

**出现于**:`manipulation/act/infer_with_om`、`manipulation/diffusion-policy/infer_with_om`。

---

## ④ 运行/评测阶段

### 4.1 无头渲染缺少 `libOSMesa.so.0`

**阶段背景**:基于 MuJoCo 的仿真评测(Aloha/Libero)在无显示环境需走 OSMesa 软件渲染。

**现象**:评测/无头渲染报 `Failed to load library ('libOSMesa.so.0')`。

**根因**:系统缺少 OSMesa 运行库。

**解决**:
```bash
ldconfig -p | grep OSMesa          # 先确认是否已安装
# Ubuntu/Debian:
sudo apt-get install -y libosmesa6 libosmesa6-dev
# CentOS/RHEL/openEuler/EulerOS:
sudo yum install -y mesa-libOSMesa
export MUJOCO_GL=osmesa             # 让仿真走 CPU 软件渲染
```

**出现于**:`manipulation/pi0/*`、`manipulation/pi05/infer_with_om`、`manipulation/smolvla/train`。

### 4.2 无外网环境下 wandb 联网导致启动报错

**阶段背景**:训练样例默认可能尝试上传 wandb 在线日志。

**现象**:无外网权限时,训练/评测因联网日志上传失败而报错或卡住。

**根因**:wandb 默认在线模式需要外网。

**解决**:
- 关闭联网日志:`export WANDB_MODE=disabled`(或在配置中 `wandb.enable: false`);
- 如需使用 wandb 且遇 token/版本问题,先升级 wandb 再重新登录。

**出现于**:`manipulation/act/train`、`manipulation/smolvla/train`、`locomotion/LQC`。

---

## 速查表

| 阶段 | 关键报错/现象 | 跳转 |
|---|---|---|
| 环境准备 | `import torch_npu` 失败、NPU 依赖缺失 | [1.1](#11-torch_npu--昇腾软件栈未就绪) |
| 环境准备 | ATC 失败、环境变量缺失 | [1.2](#12-set_envsh-未-sourcecann-环境变量缺失) |
| 模型获取 | paligemma 403 / 下载卡住 / 离线联网 | [2.1](#21-huggingface-模型下载失败--gated-model-未授权--离线仍联网) |
| 转换/导出 | ATC 报错、`507011`、soc_version | [3.1](#31-atc-编译失败----soc_version-与芯片不符) |
| 转换/导出 | `ModuleNotFoundError: acllite_*` | [3.2](#32-import-acl--acllite-模块找不到) |
| 转换/导出 | 输入 shape/dtype/顺序不匹配 | [3.3](#33-输入-shape--dtype--顺序不匹配) |
| 转换/导出 | OM 动作落不回动作空间 | [3.4](#34-归一化处理器processor丢失) |
| 运行/评测 | `libOSMesa.so.0` 缺失 | [4.1](#41-无头渲染缺少-libosmesaso0) |
| 运行/评测 | wandb 联网启动报错 | [4.2](#42-无外网环境下-wandb-联网导致启动报错) |

> 单个样例特有的问题(如显存与 `num_envs` 上限、数据长度约束、特定依赖的虚假告警、第三方库手动 patch 等)请查阅对应样例自己的 README。
