# NPU Hunyuan3D 模型推理优化实践
本文主要介绍Hunyuan3D模型基于NPU的推理优化策略，其中包括以下优化点：

shapegen部分：
- 使能融合算子与高性能计算算子
- PFA算子适配
- torchair图模式适配
  - DIT 图模式适配
  - VAE-decoder 图模式适配
- CPU-NPU搬运优化
- DIT-Cache step-level适配（step-level指跳过/预测step范式）

texgen部分：
- 多线程并行光栅化
- aicpu算子迁移
- Inpaint计算优化

## shapegen性能优化介绍
### 融合算子优化
**优化原因：** RmsNorm算子常见于LLaMA、LLaMA2、Baichuan等LLM模型中，由于torch侧没有提供RmsNorm算子的接口，因此在模型中通常是以自定义类的形式出现，在forward函数下定义计算逻辑，需要的运行算子比较多，计算公式为
```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

**优化方式：** 将`hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`文件中torch.rsqrt函数替换如下
```python
'''替换部分
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

'''
#替换后
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        return torch_npu.npu_rms_norm(x, self.scale, epsilon=self.eps)[0]

```

### 替代高性能算子
#### 替代高性能算子
- **优化原因：** 在神经网络中，GELU是一个重要的激活函数，其灵感来源于relu和dropout，在激活中引入了随机正则的思想。为了提升GELU算子在NPU上的运行性能，业界提出了FastGelu等版本。本接口FasterGelu是针对FastGelu的化简版本，可以大幅度提升计算性能。原理是使用泰勒展开逼近计算结果。
- **优化方式：** 修改`hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`文件中GELU函数，使用`torch_npu.npu_fast_gelu`替换`nn.functional.gelu`的算子实现。
```python
'''替换部分
class GELU(nn.Module):
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.gelu(x, approximate=self.approximate)
'''
#替换后
class GELU(nn.Module):
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return torch_npu.npu_fast_gelu(x)
```

#### FIA算子适配和优化
- **优化原因：** 官方使用的`scaled_dot_product_attention`对应npu算子是`npu_fusion_attention`，`npu_fusion_attention`不支持图模式，因此如果将`npu_fusion_attention`做静态图编译时，会导致`npu_fusion_attention`编译为三个attention核心算子进行运算，带来性能劣化。
    - `BatchMatMul`
    - `SoftmaxV2`
    - `BatchMatMul`
- **优化方式：** 使用`torch_npu.fused_infer_attention_score`代替`torch_npu.npu_fusion_attention`，FIA同时适配增量&全量推理场景的FlashAttention算子，能够进行图编译，并且同时解决了`Softmax`运算溢出问题以及内存连续问题。进行如下优化：
```python
'''替换部分
scaled_dot_product_attention = nn.functional.scaled_dot_product_attention

def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    x = scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x
'''
# 替换后
def npu_fia(q, k, v, scale):
    attn_mask = None
    batch_size, num_head, seq_len, head_dim = q.shape
    out = torch_npu.torch_npu.fused_infer_attention_score(
        q, k, v, num_heads=num_head, input_layout="BNSD", scale=scale, atten_mask=attn_mask
    )[0]
    return out

def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    x = npu_fia(q, k, v, scale=(1 / math.sqrt(head_dim)))
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

```
### torchair图模式适配
- **优化原因：**  在 PyTorch 的单算子执行模式下，算子不仅可能面临主机端下发的host bound，还可能错过未被识别的算子融合机会而未能充分发挥性能。相较于单算子模式，图模式可以通过计算图优化、多流并行、内存复用和模型下沉等技术手段，加速模型执行效率，减少模型内存占用。因此可通过 torch.compile 启用图模式以实现性能优化。[TorchAir（即 Torch Ascend Intermediate Representation）](https://www.hiascend.com/document/detail/zh/Pytorch/720/modthirdparty/torchairuseguide/torchair_00003.html)作为 Ascend Extension for PyTorch（torch_npu）的图模式功能拓展工具库，提供了昇腾设备亲和的torch.compile图模式后端。它能够助力 PyTorch 网络在昇腾 NPU 上开启图模式推理，进而实现运算加速与性能层面的优化提升。
-  **优化方式：** 在模型推理场景下，使能DIT model模块以及VAE decoder模块图编译可以避免标量入图导致的图编译失败，从而提高推理性能。同时启用tiling图下沉优化。
```python
#增加部分
if args.full_graph:
    import torchair
    import torch._dynamo
    config = torchair.CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = torchair.get_npu_backend(compiler_config=config)
#编译入图部分     
if args.full_graph:
    pipeline.model = torch.compile(pipeline.model, dynamic=False, backend=npu_backend, fullgraph=True)
    pipeline.vae.geo_decoder = torch.compile(pipeline.vae.geo_decoder, dynamic=False, backend=npu_backend, fullgraph=True)
#图模式指令      
python minimal_demo_npu.py --full_graph 
```      
### CPU-NPU搬运优化
- **优化原因：** 从CPU搬运到NPU存在搬运耗时，频繁的搬运会导致性能劣化，在生成时间步的步骤尤其明显，
- **优化方式：** 直接在NPU上生成时间步
```python
'''替换部分
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )
'''
#替换后
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32,device=t.device) / half)
```

### DIT-Cache step-level适配
- **基本原理：** DIT-Cache作为扩散模型推理加速的缓存框架，通过复用/预测已有的结果，减少冗余前向计算。其加速逻辑可清晰的分为Step-level和Block-level范式，Step-level通过判断不同采样步数step间的特定特征差异，通过阈值比较，决定是否跳过完整的step计算，直接复用或者预测缓存结果。
- **Step-level典型方法：** ** 在Step-level加速范畴内，FBCache与TeaCache是典型的cache方法。[TeaCache](https://arxiv.org/pdf/2411.19108) 利用模型输入与输出的强相关性，通过Timestep Emebdding（输入）来估计输出差异：先利用该输入粗估输出变化，再通过多项式拟合修正缩放偏差，最终以累积差异作为判断标准，动态决定是否复用上一步被Cache的输出，避免冗余计算。[FBCache](https://arxiv.org/pdf/2411.19108)的原理是基于First Block L1误差，比较第一个Block输出残差与上一步的第一个Block输出残差之间的差异，如果首块输出误差与上一轮首块输出误差差异小于指定阈值，就跳过当前步计算，复用残差，对当前步的输出进行估计。
- **block-level典型方法：** ** 在Block-level加速范畴内，Taylorseer是典型的以预测范式代替缓存的方法。[Taylorseer](https://github.com/Shenyi-Z/TaylorSeer) 解决了Step-level中因扩散模型中的特征相似性显著下降，导致的特征缓存引入的错误显著增加，损害生成质量。Taylorseer方法基于未来时间步的扩散模型特征可以基于前一时间步的值进行预测这一理论基础和特征在时间步间缓慢且连续变化的现象，采用微分方法近似特征的高阶导数，并应用Taylor级数预测后续特征。与直接重用缓存特征的方法不同，Taylorseer利用特征变化的连续性来预测未来特征，使扩散模型能够实现无训练且高比例的加速，而不会显著降低生成质量。 
- **FBCache优化效果：** 通过63张图像生成任务，使用UNI3D指标：

    |阈值|跳过率|采样耗时（s）|Uni3d-I|加速比|
    |:---:|:---:|:---:|:---:|:---:|
    |baseline|0|19.28|0.3748|x1.00|
    |0.04|50%|10.19|0.3704|x1.89|
    |0.05|63%|9.62|0.3687|x2.08|
    |0.06|66%|7.80|0.3690|x2.47|
- **TeaCache优化效果：** 通过63张图像生成任务，使用UNI3D/ULIP指标，结果如下，在跳过率为75%精度损失大于1%，因此推荐使用阈值0.1：
    |阈值|跳过率|采样耗时（s）|Uni3d-I|ULIP-I|加速比|
    |:---:|:---:|:---:|:---:|:---:|:---:|
    |baseline|0|19.28|0.3179|0.2157|x1.00|
    |0.1|48%|10.32|0.3090|0.2050|x1.86|
    |0.2|75%|5.83|0.23908|0.1879|x3.30|
- **Taylorseer性能优化效果：** 在推理步数为100，预热步数为3，截断步数为0，进行多轮生成，取每次生成时间的平均值：
   |导数阶数 |跳过间隔|0|1|2|3|4|5|6|
     |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    ||attention次数|4800|2496|1728|1344|1104|960|864|
    ||MLP次数|6400|3328|2304|1792|1472|1280|1152|
    ||预测步数|0|48|64|72|77|80|83|
    |0|总时间|19.9s|10.24s|7.66s|6.17s|5.14s|4.58s|3.98s|
    |0|加速比|1.00|1.85|2.60|3.23|3.87|4.35|5.00|
    |0|性能提升|0%|46.0%|61.5%|69.0%|74.2%|77.0%|79.9%|
    |1|总时间|20.44s|11.14s|8.02s|6.44s|5.51s|4.92s|4.35s|
    |1|加速比|1.00|1.84|2.55|3.17|3.71|4.15|4.70|
    |1|性能提升|0%|45.5%|60.8%|68.8%|72.9%|75.9%|78.7%|
    |2|总时间|20.72s|11.47s|8.30s|6.67s|5.82s|5,23s|4.64s|
    |2|加速比|1.00|1.81|2.50|3.11|3.56|3.96|4.47|
    |2|性能提升|0%|44.7%|60.0%|67.9%|71.9%|74.8%|77.6%|


    |导数/阶数|跳过率|加速比|Dit耗时|性能提升|Uni3d-I|ULIP-I|
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |baseline|0|1.00|19.26s|0%|0.374|0.221|
    |O1N1|48|1.84|11.14s|45.5%|0.374|0.222|
    |O1N3|72|3.17|6.4s|68.6%|0.372|0.220|
    |O1N5|78|4.15|4.92s|75.9%|0.371|0.217|
    |O2N1|48|1.81|11.47s|44.7%|0.374|0.222|
    |O2N3|72|3.11|6.67s|67.9%|0.373|0.221|
    |O2N5|78|3.96|5.23s|74.8%|0.368|0.217|
- **启动方式：** 本代码模块通过修改cache_config.json文件决定是否使用Cache，Cache范式，Cache相关参数均在[3d_vision\Hunyuan3D\hy3dgen\cache\cache_config.json](../../../3d_vision/Hunyuan3D/hy3dgen/cache/cache_config.json) 中直接修改，同时，使用如下指令可以自定义cache_config.json位置
```python
python minimal_demo_npu.py  --cache_config './hy3dgen/cache/cache_config.json'  #cache_config.json位置
```      
其中参数意义如下
```python
{
        "cache_forward": "NoCache", #直接设置Cache方案，目前支持FBCache/TeaCache/Taylorseer,默认启动NoCache，也就是无ditcache方法，只需按照下面的提示代替NoCache即可启动
        "comment": "choose from FBCache/TeaCache/Taylorseer, otherwise use NoCache",
        "enable_separate_cfg": false, #CFG串行开关，适配wan2.2等CFG串行的项目，开启双分支TeaCache/FBcache,不支持Taylorseer，同时在非CFG串行项目保持关闭，Hunyuan3D为非CFG串行，因此关闭
        "FBCache":{
                "cache_name": "FBCache",
                "rel_l1_thresh": 0.05, #FBCache阈值，阈值越大跳过越多，精度损失越大，需要平衡性能和精度
                "latent": "latent",
                "judge_input": "cache_latent"
        },
        "TeaCache":{
                "cache_name" : "TeaCache",
                "rel_l1_thresh": 0.1, #TeaCache阈值，阈值越大跳过越多，精度损失越大，需要平衡性能和精度
                "coefficients": [733.226126,-401.131952,67.5869174,-3.149879,0.0961237896], #TeaCache多项式拟合，通过输入输出进行拟合
                "warmup": 2, #预热步数，强制前面几步进行计算
                "latent": "latent",
                "judge_input": "modulated_inp"
        },
        "Taylorseer":{
                "cache_name" : "Taylorseer",
                "n_derivatives": 3, #Taylorseer导数设置，导数阶数越大，理论上对原特征拟合越优秀，但是过大的导数可能会出现过拟合现象，且占用显存会更大
                "skip_interval_steps": 4, #Taylorseer计算间隔设置，注意 计算间隔 = 跳过计算步数+1 跳过计算间隔越大，对性能提升越明显，但是精度会下降
                "warmup": 1, #预热步数，强制前面几步进行计算，避免前期特征变化明显的时候出现跳过计算
                "cutoff_steps": 1, #截断步数，强制最后几步进行计算
                "offload": false #offload开关，开启后可以降低显存占用
        },
        "NoCache":{
            "cache_name" : "NoCache"
    }
    }

```
- **框架位置：** 使用dit_cache作为自定义库，dit_cache方案插入逻辑如下：
```  
    cann-recipes-embodied-intelligence
        +--- module #公共优化模块
        |   +--- dit_cache #Dit_Cache通用方案核心位置
        |       +--- __init__.py #初始化
        |       +--- cache_method.py #Dit_Cache方案选择和方案设计
        +--- 3d_vision
            +--- Hunyuan3D
                +--- set_env.sh #激活样例环境，能够导入 module/dit_cache
                +--- minimal_demo_npu.py #顶层推理入口，进行cache方案选择，模型层数传入和前向推理替换
                +--- hy3dgen
                    +--- cache #cache适配模型口
                        +--- cache_block.py #Dit_Cache替换模块位置
                        +--- cache_config.json #默认cache参数位置
                    +--- models
                        +--- denoisers
                            +--- hunyuan3ddit #Cache适配模型核心位置
 ```
## texgen性能优化介绍
### 多线程并行光栅化
**优化原因：** CPU侧执行光栅化会带来巨大时延，Hunyuan3D texgen在6个不同视角分别进行法向信息、位姿信息、纹理信息投影，共进行18次光栅化。由于光栅化相互之间与delighting来自不同数据通路，没有数据依赖关系，可以采用并行计算的方式，隐藏光栅化时延。

**优化方法：** 将法向计算和位姿计算，以及delighting过程中相互间没有依赖关系的计算过程施加并行，提升计算效率。对hy3dgen/texgen/pipelines.py中的代码做如下替换：

```python
'''替换部分
images_prompt = [self.models['delight_model'](image_prompt) for image_prompt in images_prompt]
...
normal_maps = self.render_normal_multiview(
selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
position_maps = self.render_position_multiview(
selected_camera_elevs, selected_camera_azims)
...
texture, mask = self.bake_from_multiview(multiviews,
                                         selected_camera_elevs,selected_camera_azims,selected_view_weights,
                                         method=self.config.merge_method)
'''

#替换后

def _delighting_render_multiview(self,images_prompt,selected_camera_elevs,selected_camera_azims
    ,use_abs_coor=True):
    total_tasks = len(images_prompt) + 2 * len(selected_camera_elevs)
    max_workers = min(20, total_tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        normal_futures = {}
        position_futures = {}

        for idx,(elev,azim) in enumerate(zip(selected_camera_elevs, selected_camera_azims)):
            normal_future = executor.submit(
                self.render.render_normal, elev, azim,
                use_abs_coor = use_abs_coor,return_type = 'pl'
            )
            normal_futures[normal_future] = idx

            position_future = executor.submit(
                self.render.render_position, elev, azim,
                use_abs_coor=use_abs_coor, return_type='pl'
            )
            position_futures[position_future] = idx

    rast_out_maps = [None] * len(selected_camera_elevs)
    normal_maps = [None] * len(selected_camera_elevs)
    position_maps = [None] * len(selected_camera_elevs)
    images_prompt = [self.moels['delight_model'](image_prompt) for image_prompt in images_prompt]

    for future in as_completed(list(normal_futures.keys()) + list(position_futures.keys())):
        try:
            result = future.result()
            if future in normal_futures:
                idx = normal_futures[future]
                normal_maps[idx] = normal
            elif future in position_futures:
                idx = position_futures[future]
                position_maps[idx] = result
        except Exception as e:
            print(f"Task failed with error: {e}")
    
    return images_prompt, normal_maps, position_maps

def _bake_from_multiview(self, view, camera_elevs,
                         camera_azim, view_weights,method='graphcut'):
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for idx,(view,camera_elev, camera_azim, weight) in enumerate(zip(
            views, camera_elevs, camera_azims, view_weights)):
            future = executor.submit(self.render.back_project,
                                     view, camera_elev, camera_azim)
            futures.append(future)
        
        project_textures = []
        project_weighted_cos_maps = []
        project_boundary_maps = []

        for future in as_completed(futures):
            project_texture, project_weighted_cos_map, project_boundary_map = future.result()
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_weighted_cos_map)
            project_boundary_maps.append(project_boundary_map)

def __call__(self,mesh,image):
    ...
    # 对应光栅化和delighting部分
    images_prompt, normal_maps, position_maps = self._delighting_render_multiview(images_prompt, selected_camera_elevs,
                                  selected_camera_azims, use_abs_coor=True)
    ...
    # 对应multiview部分
    texture, mask = self._bake_from_multiview(multiviews, selected_camera_elevs
                                              selected_camera_azims, selected_view_weights,
                                              method = self.config.merge_method)
```

### aicpu算子迁移
**优化原因：** FP32数据格式会导致torch.mean、torch.std算子运行在aicpu上，性能下降。

**优化方法：** 将FP32类型的数据手动进行转换，使其运行在aicore上。

```python
# hy3dgen/texgen/pipelines.py
'''替换部分
texture = torch.tensor(texture_np / 255).float().to(texture.device)
'''

#替换为
texture = torch.tensor(texture_np / 255).to(torch.float16).to(texture.device)
```

```python
# hy3dgen/texgen/utils/dehighlight_utils.py
'''替换部分
src_mean, src_stddev = torch.mean(src_flat[:, i].to(torch.float32)), torch.std(src_flat[:, i].to(torch.float32))
target_mean, target_stddev = torch.mean(target_flat[:, i].to(torch.float32)), torch.std(target_flat[:, i].to(torch.float32))
'''

#替换为
src_mean, src_stddev = torch.mean(src_flat[:, i].to(torch.float16)), torch.std(src_flat[:, i].to(torch.float16))
target_mean, target_stddev = torch.mean(target_flat[:, i].to(torch.float16)), torch.std(target_flat[:, i].to(torch.float16))

'''替换部分
image_tensor = torch.tensor(np.array(image)/255.0).to(self.device)
'''

#替换为
image_tensor = torch.tensor(np.array(image) / 255.0).to(torch.float16).to(self.device)
```

### Inpaint计算优化
**优化原因：** 生成的3Dmesh纹理仍存在部分未上色点，需要通过对其平滑化，补全未上色点。源码实现了python代码和c++代码，默认使用python代码进行着色点平滑，但是c++具有更好的性能，需要手动进行替换。

**优化方法：** 手动替换为c++代码。
```python
# hy3dgen/texgen/differentiable_renderer/mesh_render.py
'''替换部分
from .mesh_processor import meshVerticeInpaint
'''

#替换为
import mesh_processor

'''替换部分
texture_np, mask = meshVerticeInpaint(
                   texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
'''

#替换为
texture_np, mask = mesh_processor.meshVerticeInpaint(
                   texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
```

### 光栅化结果复用优化
**优化原因：** 源码的18次光栅化计算中，只有6种不同输入数据，其余12次计算为冗余计算，造成性能损失。

**优化方法：** 将第一次计算的光栅化过程进行保存，每当输入相同的相机位姿，算法就从缓存中直接读取计算结果。
```python
'''替换部分
rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)
'''
#替换为
if (self.save_render and 
            (elev, azim, camera_distance) in self.render_result.keys()):
            rast_out = self.render_result[(elev, azim, camera_distance)]

...

self.render_result[(elev, azim, camera_distance)] = rast_out
```
### 光栅化过程npu迁移
**优化原因：** Hunyuan3D源码光栅化没有可在NPU执行的版本，会将光栅化过程迁移至CPU侧执行，不必要的内存搬运和串行计算会带来巨大时延。为了适配NPU架构，提升光栅化性能，本仓库引入render_npu，将光栅化的计算过程迁移至NPU执行。

**优化方法：** 修改光栅化算法，将原先采用遍历mesh顶点的方式，改为对二维平面分块，每个区域中的三角网格合并成一个矩阵计算深度、重心坐标等参数，最终判断每个像素点对应的三角面片。由NPU侧执行小算子计算不能同时进行光栅化并行。
```python
'''替换部分
rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)
'''
#替换为
if self.use_render_npu:
    rast_out, _ = render_npu_rasterize(
                    resolution, self.pos_idx, pos_clip
    )
else:
    rast_out, _ = self.raster_rasterize(
                    pos_clip, self.pos_idx, resolution=resolution
    )
```
## 性能优化指标
### shapegen性能优化指标
本方案使用1卡Atlas 800I A2推理产品，输入hunyuan3D 2.0 提供的样例数据(`assets/example_images/004.png`)，在扩散步数为num_inference_steps=100（minimal_demo_npu.py第89行与111行）情况下,性能指标如下
|使能方法|DIT 扩散耗时（s）|VAE 生成耗时（s）|
|:---:|:---:|:---:|
|baseline|21.63|17.98|
|算子优化＋搬运优化|20.07|18.71|
|算子优化+图模式编译|19.00|16.68|

### texgen性能优化指标
本方案使用1卡Atlas 800I A2推理产品，输入hunyuan3D 2.0 提供的样例数据(`assets/example_images/004.png`)，输入mesh网格面数量为20000，开启减少面数量（--face_reduce）性能指标如下
|使能方法|texgen运行时长（s）|
|:---:|:---:|
|baseline|59.83|
|光栅化并行|35.43|
|光栅化并行+算子优化|30.46|
|NPU光栅化+算子优化+Inpaint优化+光栅化结果复用|32.86|
|光栅化并行+算子优化+Inpaint优化+光栅化结果复用|26.52|
