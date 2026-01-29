
# π₀.₅ 模型训练昇腾迁移与性能优化说明

## 背景介绍

本案例在昇腾平台上使用 π₀.₅ 开源模型进行 LIBERO 基准的训练和评估，并对训练过程的性能进行了深入分析和优化。以下内容将简要介绍 π₀.₅ 模型和 LIBERO 基准的背景信息，并详细介绍在昇腾平台上训练时的性能分析与优化策略，以及无GPU的仿真渲染方案。

### π₀.₅ 模型概述

π₀.₅（pi05）是 Physical Intelligence 团队在 π₀ 模型基础上提出的升级版本，其设计目标并非单纯提升特定任务的成功率，而是系统性增强机器人在**开放世界场景**下的泛化能力（Open-world Generalization）。

具体而言，π₀.₅ 试图应对以下关键挑战：  
- 训练分布之外的新环境与新物体  
- 未见过的语言指令组合与操作目标  
- 多模态输入条件变化带来的不确定性  

因此，π₀.₅ 更强调语义理解、物理推理与行为迁移能力的协同建模，而非仅在固定基准上的性能最优化。

### LIBERO：面向终身学习的具身智能基准

LIBERO 是一个面向**机器人终身学习（Lifelong Learning in Decision Making, LLDM）**的标准化评测基准，旨在评估机器人在长期任务序列中持续学习与知识迁移的能力。与传统一次性训练–评测范式不同，LIBERO 强调机器人是否能够将已有经验有效迁移至新任务、新物体与新场景中，从而更真实地刻画人机长期交互中的学习过程。这一设计使其成为当前具身智能领域研究终身学习与泛化能力的重要评测平台。

为系统性评估不同层次的知识迁移能力，LIBERO 构建了覆盖短时序与长时序操作的多种任务套件，共包含 130 项任务，具体包括：

- **LIBERO-Spatial**（`libero_spatial`）：要求推理空间关系的任务；
- **LIBERO-Object**（`libero_object`）：围绕不同物体操作的核心任务；
- **LIBERO-Goal**（`libero_goal`）：目标条件可变的任务，要求机器人动态调整行为；
- **LIBERO-90**（`libero_90`）：源自 LIBERO-100 的 90 项短时序任务；
- **LIBERO-Long**（`libero_10`）：源自 LIBERO-100 的 10 项长时序、多阶段任务。

LIBERO 被设计为一个可扩展的开放基准，旨在成为具身智能社区测试与改进终身学习算法的共享平台。

## π₀.₅ 昇腾迁移适配
在模型迁移初期，采用 `torch_npu` 提供的自动迁移机制，对原始 PyTorch CUDA 代码进行快速适配。通过在代码入口引入如下接口，可在动态图模式下自动替换大部分 CUDA 相关算子与接口，实现功能级别的无侵入迁移：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

在训练过程中发现，`aten::_sample_dirichlet` 算子在当前 NPU 平台上尚未支持，运行时会 fallback 至 CPU 执行，其执行性能相较 GPU 约存在 6 倍差距。

需要指出的是，该算子在单步训练中仅调用一次，因此对整体训练吞吐影响有限。但在 flow model、diffusion model 及时序建模等场景中，该类概率分布采样算子具有较高使用频率，后续在更复杂模型中仍需重点关注其算子支持与性能表现。

![image](https://raw.gitcode.com/zengzixuan/pic/raw/main/dirichlet.png)

## 模型训练性能分析与优化

### profiling 说明
使用 `torch_npu.profiler` 及 MindStudio Insight 平台对训练时的性能数据进行采集，再进行可视化查看流水图、算子耗时统计图等。采集训练性能数据的方法可参考[分布式训练说明](../README.md#分布式训练启动脚本使用说明)，只需要把执行的 `run_train.sh` 换成 `run_profiling.sh`。具体性能数据采集设置如下：

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None,
)
...
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiling"),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
    with_modules=True,
    with_flops=True,
    experimental_config=experimental_config
) as prof:
```

在性能数据采集过程中，关闭了调用栈采集（`with_stack=False`）。这是因为堆栈信息会引入显著的额外插桩开销，在长时间训练场景下可能干扰真实性能特征的判断。

本次 profiling 重点关注算子级执行时间、内存占用与计算/访存比例，因此优先保证时间轴与算子统计数据的准确性。

### 性能数据分析

训练基本配置为：

- 分布式节点数：single-node, 8-device
- 总训练步数：40000 (40K)
- dataset.num_frames：273465 (273K)
- dataset.num_episodes：1693
- Effective batch size：16 x 8 = 128
- 可学习参数量：3616757520 (4B)

得到迁移后的原始性能数据如下：

![alt text](https://raw.gitcode.com/zengzixuan/pic/raw/main/original_performance.png)

#### 优化保留内存过高

将采集的性能数据文件导入 MindStudio Insight 平台进行分析，观察到在训练过程中显存的保留内存（Reserved Memory）远高于实际持有内存（Allocated Memory），如下图所示：
![alt text](https://raw.gitcode.com/zengzixuan/pic/raw/main/memory.png)
性能分析首先暴露出一个关键问题：训练过程中显存的保留内存（Reserved Memory）显著高于实际持有内存（Allocated Memory），两者差距可达十余 GB。这一现象直接限制了 batch size 的进一步提升（batch_size只能设置为不超过16），甚至在未实际耗尽显存的情况下触发 OOM。

经分析，该问题主要源于 PyTorch NPU 内存分配器的默认内存管理策略。在动态计算图或输入维度变化频繁的场景（如 VLA 模型中变长语言指令、多分辨率视觉输入等）下，分配器倾向于申请多个独立的大块内存段（segments）以满足不同尺寸的张量分配需求。由于这些内存段默认不可扩展且复用率低，导致大量内存被保留但未被有效利用，形成显著的内存碎片。

参考 [昇腾文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/ptmoddevg/trainingmigrguide/performance_tuning_0048.html) 提到的内存优化建议，设置 `export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"` 后，启用可扩展内存段功能，使得内存分配器能够动态调整已分配内存段的大小以适应不同张量的需求，从而提升内存利用率，减少保留内存与实际持有内存之间的差距。经过该优化后，显存保留内存与实际持有内存的差距大幅缩小。

#### 优化计算性能
为了更好地分析性能瓶颈并识别优化机会，我们使用 msprof-analyze（MindStudio Profiler Analyze）工具对采集到的性能数据进行分析。首先，通过源码编译进行安装：

```bash
pip3 install wheel  # 编译前需要安装 wheel
git clone https://gitee.com/ascend/mstt.git  # 下载源码
cd mstt/profiler/msprof_analyze
pip3 install -r requirements.txt && python3 setup.py bdist_wheel  # 编译 whl 包
# 以上命令执行完成后，在 mstt/profiler/msprof_analyze/dist 目录下生成性能工具 whl 安装包 msprof_analyze-{version}-py3-none-any.whl
# 执行如下命令进行性能工具安装
cd dist
pip3 install ./msprof_analyze-{version}-py3-none-any.whl
```
使用 msprof-analyze 工具分析昇腾 NPU 的性能数据，命令如下：

```bash
# 基础用法：分析昇腾 NPU 性能数据
msprof-analyze advisor all -d $HOME/profiling_data/
```

打开./analyze_output目录，可以找到性能数据分析文件，发现影响整体训练性能的最主要瓶颈集中在 **Cube 上执行的 MatMul 类算子**。该类算子在端到端训练时延中占比超过一半，是当前最具优化价值的关键算子。因此，对这一部分算子的优化将对整体训练性能提升产生显著影响，产生最大的性能收益。此外，NPU Cube 的理论计算能力本应更高，但实际测得的性能数据与之不符，这表明在 π₀.₅ 模型中，Matmul 算子的执行在 NPU 上不够亲和，需要进行针对性的优化。

进一步分析，我们想了解哪些 Matmul 算子的执行速度显著最慢，对整体性能的影响最大。通过查看算子执行时间分布图，可以发现 MatMulV3 算子的耗时是最多的，达到了 45.39%，平均耗时 2168.580 μs，最大耗时达到了 4330.870 μs。
![alt text](https://raw.gitcode.com/zengzixuan/pic/raw/main/operator.png)
在算子执行流水图中找到对应的 MatMulV3 算子，如下图：
![alt text](https://raw.gitcode.com/zengzixuan/pic/raw/main/matmul.png)为了了解这个算子在 NPU 上执行缓慢的原因，我们需要进一步查看算子内部执行的流水图。

为此，我们编写了一个简单的 Matmul 算子测试脚本 `test_matmul.py`，代码如下：
```bash
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

device = torch.device('npu:0' if torch.npu.is_available() else 'cpu')

A_3d = torch.randn(16, 712, 2048, device=device, dtype=torch.bfloat16)
B_3d = torch.randn(16, 712, 16384, device=device, dtype=torch.bfloat16)

A = A_3d.view(-1, 2048)   # (11392, 2048)
B = B_3d.view(-1, 16384)  # (11392, 16384)

output = torch.matmul(B.t(), A)

print("A reshaped shape :", A.shape)    # (11392, 2048)
print("B reshaped shape :", B.shape)    # (11392, 16384)
print("Output shape     :", output.shape)  # (16384, 2048)
```
使用 msprof 工具对该脚本进行单算子场景下的性能数据采集，命令如下：
```bash
msprof op simulator --soc-version=Ascend910B2 --output=./matmul_profiling/ python ./test_matmul.py
```
将得到的性能数据导入 MindStudio Insight 平台进行分析，查看该算子的执行流水图，可以发现：
![alt text](https://raw.gitcode.com/zengzixuan/pic/raw/main/matmul_profiling.png)
算子执行的大部分时间花费在了数据搬运上，mte2 执行的时间很长。于是，我们猜测矩阵乘数据分块不合理，导致 memory bound，数据回写现象严重，整体拖慢算子性能。为此，我们查看了 ops-nn 开源库，研究 MatMulV3 算子分块策略，在 `ops-nn/matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp` 中找到了如下关键函数：
```cpp
bool MatmulV3BaseTiling::IsSupportSingleCoreSplitK() const
{
    bool isMKNsmallK = IsSupportSingleCoreSplitSmallK(args_.nValue, args_.mValue);
    bool isNKMsmallK = IsSupportSingleCoreSplitSmallK(args_.mValue, args_.nValue);
    if (isMKNsmallK || isNKMsmallK) {
        OP_LOGI(args_.opName, "Hit mat_mul_v3 single-core-splitk (K=1536) MKN/NKM channel.\n");
        return true;
    }
    // n非对齐为fixpipe bound, 走单核切K，由于存在串行的前vector处理，非对齐场景可能性能更差，维持原4M限制不变
    if (args_.isHf32 && !n256Align_ && args_.mValue * args_.nValue < DETER_THRES_OUT_SIZE * MB_SIZE) {
        return false;
    }
    if (args_.kValue >= SPLIT_K_THRES) {
        OP_LOGD(args_.opName, "K >= SPLIT_K_THRES, enable SingleCoreSplitK.");
        return true;
    }
    ...
}
```

在`ops-nn/matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_common.h`中可以找到`SPLIT_K_THRES`的默认值为`constexpr uint64_t SPLIT_K_THRES = 27392;`，在当前算子实例中，K 维度的值为 `16*712=11952`，小于该阈值，因此没有进行切 K 优化，走的是 base 模板，只会切 m、n 轴，没有切 k 轴，会导致 L2 cache 很容易放满，数据回写现象严重，整体拖慢算子性能。通过调整 tiling 策略，从 base 模板切换到单核切 k 模板，会切 m、n、k 轴，分块 size 更小了，缓解了 L2 cache 回写问题，将这个 MatmulV3 算子的执行时间从 4.3 ms 降低到了 2.5 ms，性能提升显著。在模型的整网训练中，由于之前解决了保留内存过高的问题，保证不出现 OOM 错误的前提下，通过调整 batch size 使 K 维度超过 `SPLIT_K_THRES` 阈值，MatMulV3 算子自动切换至单核切 K 模板，显著改善了数据分块与 L2 Cache 利用效率。在该条件下，算子执行时间由原先的约 4.3 ms 降低至 2.5 ms。

结合前述显存优化手段，在保证训练稳定性的前提下整体训练吞吐得到显著提升。

通过以上优化，基于如下配置，最终 π₀.₅ 在昇腾 A2 上的训练性能达到了如下水平：

![alt text](https://raw.gitcode.com/zengzixuan/pic/raw/main/final_performance.png)
可以看出，通过引入内存管理策略与计算调度改进，在显著提升批处理规模（从128增至512）的同时，成功将峰值显存占用从58.62GiB降至51.02GiB，降幅达13%，实现了“更大批量、更低内存”。尽管单步迭代耗时有所增加（由2.45秒升至7.4秒），但由于批大小扩大带来的数据吞吐量提升，整体训练周期缩短了25%（从27小时17分压缩至20小时33分），单位时间吞吐量从52.24 samples/s 增长到69.19 samples/s，处理效率实际提升了约32%。

## 仿真实时渲染
由于 NPU 不支持 OpenGL 渲染，因此在使用 NPU 进行仿真训练时，需要将仿真环境的渲染模式切换为离屏渲染（Offscreen Rendering）。在使用 MuJoCo 仿真环境时，可以通过设置环境变量来实现离屏渲染。具体设置如下：
```bash
Xvfb:1-screen01024x768x24>/tmp/xvfb.log2>&1&
exportDISPLAY=:1
exportLIBGL_ALWAYS_SOFTWARE=1
exportLD_PRELOAD=/usr/lib/aarch64-linux-gnu/libOSMesa.so
exportLD_LIBRARY_PATH=/lib/aarch64-linux-gnu/:$LD_LIBRARY_PATH
exportMUJOCO_GL=osmesa
```
![alt text](https://raw.gitcode.com/zengzixuan/pic/raw/main/simulator.png)
- **MuJoCo CPU 渲染**：调用软件渲染（OSMesa/EGL fallback），在 CPU 上完成图像生成。
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
@misc{intelligence2025pi05visionlanguageactionmodelopenworld,
      title={$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization}, 
      author={Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Manuel Y. Galliker and Dibya Ghosh and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and James Tanner and Quan Vuong and Homer Walke and Anna Walling and Haohuan Wang and Lili Yu and Ury Zhilinsky},
      year={2025},
      eprint={2504.16054},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.16054}, 
}
```