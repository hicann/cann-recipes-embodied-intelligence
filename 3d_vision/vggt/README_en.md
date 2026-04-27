# VGGT inference on Ascend Atlas A2
## CANN Environment Preparaton
1. The inference of VGGT depends on the CANN development kit package (`cann-toolkit`) and the CANN binaray operator package(`cann-kernels`). The supported CANN software version is CANN 8.5.0.
   
    Download the `Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Ascend-cann-${chip_type}-ops_linux-${arch}.run` packages from the [CANN Software Package Download Page](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0) and install them by referring to the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html).

2. The required versions of torch and torch_npu are 2.7.1 and 2.7.1.post2.

    Download the binary package from [Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) and install torch and torch_npu.
    ```shell
    conda create -n vggt python==3.11.13
    conda activate vggt
    pip3 install torch==2.7.1
    pip3 install torch-npu==2.7.1.post2
    ```

## VGGT Model Preparation
1. Download the open-source [VGGT network code](https://github.com/facebookresearch/vggt/tree/main) from the github repo.
   ```shell
    git clone https://github.com/facebookresearch/vggt.git
   ```
2. Download the code of this repository:
   ```shell
   git clone https://gitcode.com/cann/cann-recipes-embodied-intelligence.git
   ```
3. Copy the code from the VGGT repository to this project directory in non-overwrite mode:
    ```shell
    cp -r vggt/examples cann-recipes-embodied-intelligence/3d_vision/vggt/
    cp -rn vggt/vggt/dependency cann-recipes-embodied-intelligence/3d_vision/vggt/vggt/dependency
    cp -rn vggt/vggt/heads cann-recipes-embodied-intelligence/3d_vision/vggt/vggt/
    cp -rn vggt/vggt/layers cann-recipes-embodied-intelligence/3d_vision/vggt/vggt/
    cp -rn vggt/vggt/utils cann-recipes-embodied-intelligence/3d_vision/vggt/vggt/ 
    ```
4. Install Python dependencies:
    ```shell
    pip3 install -r requirements.txt
    ```
5. Download [VGGT model weights](https://huggingface.co/spaces/facebook/vggt) and copy it to the local path `ckpt`.
    ```
    VGGT
        +--- examples
        +--- demo_infer.py
        +--- eval
        +--- ckpt
            +--- model.pt
        +--- quant
        +--- vggt
            +--- dependency
            +--- heads
            +--- layers
            +--- models
            +--- utils
            +--- sp
    ```
## Performance Measurement
This repo provides script to test the functionality and the performance of VGGT model on NPU.
1. Before executing the test scripts, refer to the Ascend Community CANN installation tutorial to set environment variables:
    ```shell 
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    ```
2. Run the inference script and the output presents the average inference time of vggt bf16 model.
   ```shell
    python demo_infer.py --ckpt "ckpt/model.pt"
   ```
  
3. Run the inference script and the output presents the average inference time of vggt bf16_sp model.
   ```shell
    bash infer_test.sh
   ```
   Parameter description for multi NPU inference:
   ```
    torchrun --nproc_per_node=1 demo_infer.py \
      --ckpt ${model_base} \
      --images_path examples/kitchen/images \
      --enable_sp \
      --ulysses_degree 1 \
      --ring_degree 1
      
   # nproc_per_node：The torchrun parameter, the number of processes started by each node, needs to be equal to the number of NPU cards used；
   # ckpt：Model checkpoint file path；
   # images_path：Enter the directory where the image sequence is located；
   # enable_sp：Whether to enable sequence parallelism, default value: False, with the prerequisite that nproc_per_node>1；
   # ulysses_degree：Ulysses parallelism, constraint Ulysses_degree × ring_degree=nproc_per_node; Num_ attention heads must be divisible by Ulysses_degree；
   # ring_degree：Ring parallelism, constraint Ulysses_degree × ring_degree=nproc_per_node

4. To perform vggt int8 model inference, you first need to build the vggt int8 model:
   ```shell
    python demo_infer.py --ckpt "ckpt/model.pt" --buildW8A8
   ```
   The vggt int8 model will be built in the current path, and then used for inference:
   ```shell
    python demo_infer.py --ckpt VGGT_model_W8A8.pt --enableW8A8
   ```
## Accurancy Benchmark
This repo provides accurancy benchmark to evaluate the VGGT model on NPU. The full benchmark include three programs to test the accurancy of VGGT on Pose Evaluation, Point Map Evaluation and Depth Evaluation. 

Since the full dataste of benchmark is large, we can initially test the accurancy of VGGT model in Pose Evaluation with the subset of the full Co3DV2 dataset.

### Dataset Preparation:
1. Download data `CO3D_apple.zip` and data `CO3D_backpack.zip` from [CO3D website](https://ai.meta.com/datasets/co3d-downloads/) and unzip them to `datasets/co3d/co3d_data/`.
    ```
    VGGT
        +--- datasets
            +--- co3d
                +--- co3d_data
                        +--- apple
                        +--- backpack
                    ...
    ```
2. Prepare metadata of the dataset:
    ```shell
   export VGGT_DIR=$(pwd)
   cd eval/pose_evaluation/dataset_prepare
   python preprocess_co3d.py --category all --co3d_v2_dir $VGGT_DIR/datasets/co3d/co3d_data/ --output_dir $VGGT_DIR/datasets/co3d/co3d_anno/ 
    ```
### Accurancy Measurement
- Execute the benchmark program:
-   * Use vggt bf16 model:
    ```shell
    export VGGT_DIR=$(pwd)
    cd eval/pose_evaluation
    python eval_co3d.py --co3d_dir $VGGT_DIR/datasets/co3d/co3d_data/
        --co3d_anno_dir $VGGT_DIR/datasets/co3d/co3d_anno/  --ckpt $VGGT_DIR/ckpt/model.pt
    ```
    * Currently, the bf16 model measurement accurancy is about 0.911.
    
-   * Use vggt int8 model:
    ```shell
    export VGGT_DIR=$(pwd)
    cd eval/pose_evaluation
    python eval_co3d.py --co3d_dir $VGGT_DIR/datasets/co3d/co3d_data/
        --co3d_anno_dir $VGGT_DIR/datasets/co3d/co3d_anno/  --ckpt VGGT_model_W8A8.pt --enableW8A8
    ```
    * Currently, the int8 model measurement accurancy is about 0.907.