# π₀.₅ 模型训练昇腾迁移与性能优化说明

## 背景介绍

本案例在昇腾平台上使用 π₀.₅ 开源模型进行 LIBERO 基准的训练和评估，并对训练过程的性能进行了深入分析和优化。以下内容将简要介绍 π₀.₅ 模型和 LIBERO 基准的背景信息，并详细介绍在昇腾平台上训练时的性能分析与优化策略，以及无 GPU 的仿真渲染方案。

## 模型训练性能分析与优化

在昇腾平台上训练 `pi05` 时，profiling 最初暴露出四类比较典型的开销：

1. 时间采样路径上的小算子与概率分布采样开销；
2. `PaliGemma` 主干中 attention 前后的投影与算子调度碎片化；
3. 训练图反向阶段的重复recomputation与 DDP 同步成本；
4. Gemma MLP 中的大 `matmul`。

### 优化项

下表列出了当前代码中已经通过端到端训练验证的优化项。并已经在 `modeling_pi05.py`、`run_train.sh`、`run_profiling.sh` 和训练脚本默认配置中。

| 优化项 | 默认状态 | 生效位置 | 作用原理 | 当前实测结论 |
| --- | --- | --- | --- | --- |
| `sample_beta` | 默认开启 | `sample_beta` | 针对 `Beta(1.5, 1.0)` 默认时间采样，避免通用分布采样实现带来的额外构造与调度开销 | 保留，对整体性能影响较小 |
| `PI05_FUSE_PALIGEMMA_QKV=1` | 默认开启 | `PaliGemma` attention 输入投影 | 将 `q/k/v` 三个投影合并到一次线性层计算与拆分中，减少 kernel 启动与中间张量调度 | 保留，attention 计算明显下降 |
| `PI05_USE_NPU_FUSION_ATTENTION=1` | 默认开启 | `PaliGemma` attention | 用 `torch_npu.npu_fusion_attention` 替换 eager attention，减少 attention 主干中的算子碎片化 | 保留，训练性能提升约 `5%+` |
| `PI05_USE_NPU_GROUPED_GEMMA_INPROJ=1` | 默认开启 | Gemma MLP `gate_proj/up_proj` | 对共享同一输入的两段前向 matmul 使用 `npu_grouped_matmul` 合并执行，反向保持标准线性层梯度公式 | 保留，训练级性能提升约 `1.0%~1.2%` |
| `find_unused_parameters=False`、`static_graph=True`、`gradient_as_bucket_view=True` | `pi05` 默认开启 | DDP 包装层 | 关闭无用参数探测，固定训练图，减少 bucket 重建和梯度通信额外开销 | 保留，但不是主要瓶颈 |
| `--disable-outer-suffix-checkpoint` | 推荐显式开启 | `run_train.sh` / `run_profiling.sh` | 关闭位置不合适的外层大 checkpoint，避免反向阶段整段 suffix 被重复重算 | 保留，当前提升最大的单项性能优化 |

第一类优化针对时间采样。`pi05` 的默认时间分布是 `Beta(1.5, 1.0)`，原始实现走的是更通用的概率分布采样路径。对于训练中高频、参数固定的这一特例，直接提供 `sample_beta` 可以减少不必要的分布构造和调度开销，也避免在昇腾上退回到cpu进行计算。

第二类优化针对 transformer 主干。`PaliGemma` 侧的 `q/k/v` 投影默认改为融合执行，同时训练阶段默认启用 `torch_npu.npu_fusion_attention`。这样做的核心收益不是改变模型结构，而是把原本多段 eager 投影、转置、attention 调度压缩成更适合昇腾执行的计算路径，减少 kernel 启动次数和中间张量搬运。经过这一轮处理后，attention 已经不再是训练的首要瓶颈。

第三类优化来自训练图本身。对于 `pi05`，DDP 默认采用 `find_unused_parameters=False`、`static_graph=True` 和 `gradient_as_bucket_view=True`，目的是减少动态图额外探测、稳定 bucket 组织并降低梯度通信侧的额外开销。除此之外，训练命令建议显式加上 `--disable-outer-suffix-checkpoint`。原因在于 suffix 分支外层那一层大 checkpoint 会把整段后缀在反向时重新执行一遍，而模型内部本来已经保留了更细粒度的 checkpoint；两者叠加后，反向阶段会出现位置不合适的重复重算。关闭这层外包式 checkpoint 后，显存策略仍然可控，但反向计算明显变快，因此这是当前收益最直接的一项优化。

第四类优化直接针对 profiling 中占比最高的 MLP 主热点。经过前几轮处理后，训练热点已经集中到 Gemma MLP 的 `gate_proj/up_proj/down_proj` 三段矩阵乘法，其中前两段具有完全相同的输入张量。当前版本在 NPU 上默认启用 `torch_npu.npu_grouped_matmul`，把 `gate_proj` 和 `up_proj` 合并成一次 grouped GEMM 前向计算；由于该算子当前没有现成训练 autograd，反向梯度采用显式手写的标准线性层梯度公式回传，因此数值语义保持与原始 `GemmaMLP` 完全一致。这个优化之所以有效，是因为 `pi05` 训练默认开启 gradient checkpointing，MLP 前向在反向阶段还会被重算一遍；因此只要把这两段前向 matmul 压缩成一次 grouped kernel，就能在一个训练 step 内同时吃到“正向一次 + 反向重算一次”的收益。

### 推荐训练与 profiling 方式

下面给出当前保留优化对应的推荐训练与 profiling 命令。更完整的环境准备和脚本使用说明仍以上层 [../README.md](../README.md) 为准。

训练推荐命令如下：

```bash
cd cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts
./run_train.sh pi05 --nproc 2 --disable-outer-suffix-checkpoint
```

如果需要重新采样 profiling，推荐使用如下短窗口命令，能够较快拿到稳定阶段的热点分布：

```bash
cd cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts
./run_profiling.sh pi05 \
  --nproc 2 \
  --foreground \
  --disable-outer-suffix-checkpoint \
  --profile-wait 1 \
  --profile-warmup 1 \
  --profile-active 2 \
  --profile-repeat 1
```

### 从零复现推荐流程

下面给出当前版本推荐的完整复现流程。该流程对应本仓库现有脚本默认设置。

1. 初始化代码与环境。

```bash
cd <your-workdir>
git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git
chmod +x cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts/setup.sh
./cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts/setup.sh
```

2. 激活环境并进入脚本目录。

```bash
conda activate lerobot
cd <your-workdir>/cann-recipes-embodied-intelligence/manipulation/pi05/train/src/scripts
```

3. 首次训练前，如模型权重和数据集已经缓存到本地，可显式启用离线模式，避免远端探测影响启动时间；如果尚未缓存，则不要打开这三个环境变量。

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

4. 启动推荐训练命令。

```bash
./run_train.sh pi05 --nproc 2 --disable-outer-suffix-checkpoint
```

5. 如需快速采样 profiling，使用下面的短窗口命令。

```bash
./run_profiling.sh pi05 \
  --nproc 2 \
  --foreground \
  --disable-outer-suffix-checkpoint \
  --profile-wait 1 \
  --profile-warmup 1 \
  --profile-active 2 \
  --profile-repeat 1
```

6. 训练与 profiling 结果检查。

- 训练日志位于 `${PROJECT_ROOT}/ckpt/logs/train_<model>_<timestamp>.log`。
- profiling 日志位于 `${PROJECT_ROOT}/ckpt/logs/profiling_<model>_<timestamp>.log`。
- profiler 主目录位于本次训练 `output_dir` 下的 `profiling/`。
- 多卡训练产生 `profiling/rank0/...`、`profiling/rank1/...` 是正常现象。
- 当前默认配置下， `pi05` 的稳定训练区间应大致落在 `5.75 ~ 5.85 s/it`。

### Profiling 输出位置与结果解读

profiling 日志位于 `${PROJECT_ROOT}/ckpt/logs/profiling_<model>_<timestamp>.log`，profiler 主目录位于本次训练 `output_dir` 下的 `profiling/`。多卡训练时看到 `profiling/rank0/...` 和 `profiling/rank1/...` 是正常现象，表示每张卡各自产出一份 profiling 结果；常用数据库文件位于 `profiling/rank*/.../ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db`。


## 仿真实时渲染

由于 NPU 不支持 OpenGL 渲染，因此在使用 NPU 进行仿真训练时，需要将仿真环境的渲染模式切换为离屏渲染（Offscreen Rendering）。在使用 MuJoCo 仿真环境时，可以通过设置环境变量来实现离屏渲染。具体设置如下：

```bash
Xvfb :1 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 &
export DISPLAY=:1
export LIBGL_ALWAYS_SOFTWARE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libOSMesa.so
export LD_LIBRARY_PATH=/lib/aarch64-linux-gnu/:$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
```

![alt text](https://raw.gitcode.com/zengzixuan/pic/raw/main/simulator.png)

- **MuJoCo CPU 渲染**：调用软件渲染（OSMesa / EGL fallback），在 CPU 上完成图像生成。
- **Xvfb 虚拟显示**：创建一个虚拟显存帧缓冲，模拟显示器环境，让渲染画面有“输出窗口”。
- **VNC 转发**：将虚拟显示画面通过 VNC Server 编码并转发，用户可在本地 VNC 客户端实时查看。

## Citation

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
