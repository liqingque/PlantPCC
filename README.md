# PlantPCC: Dual Sampling and Multi-level Geometry-aware Contrastive Regularization for Plant Point Cloud Completion
The official implementation of the paper：
PlantPCC: Plant Point Cloud Completion Based on Edge-Uniform Dual Sampling and Hierarchical Geometry-aware Contrastive Regularization

Contact: xiaomengli@cau.edu.cn Any questions or discussion are welcome!

-----
+ [2024.09.11] We have uploaded the dataset, which can be downloaded from the following link: https://drive.google.com/file/d/1uJoQVBfs39XP36-uaCVHlGGvPRQN9Qou/view?usp=drive_link.

+ [2024.09.05] We have initialized the repo. The related resources will be released after the manuscript is accepted.


<img src="assets/PlantPCC.png" alt="Dataset" width="800" height="600">




## Abstract
Plant point cloud completion is essential for tasks like segmentation and surface reconstruction in plant phenotyping. Unlike the relatively simpler Computer-Aided Design models found in datasets like ShapeNet, plant point clouds are characterized by their rich geometric shapes and intricate edge features, making the task of completion significantly more challenging. In response to this, we propose a learnable uniform-edge dual sampling feature extractor that efficiently captures complex geometric features in plant point clouds by ensuring comprehensive coverage of overall morphology while focusing on edge regions with critical geometric details. Additionally, we propose a multi-level geometry-aware contrastive regularization method to improve the alignment between the predicted point clouds and the missing regions, enhancing their distributional similarity to the ground truth. Experiments on the PlantPCC dataset show our model outperforms state-of-the-art methods, improving Chamfer Distance by 6.4\% over the second-best model. 

## Contributions
1. We present the first high-quality dataset for plant point cloud completion, featuring three distinct plant categories with diverse shapes. This dataset also serves as a benchmark for evaluating algorithms that complete geometrically complex point clouds.
   
2. We proposed UEFE, a novel feature extraction module utilizing learnable uniform-edge dual sampling, designed to effectively capture both global structural shapes and intricate local geometric features.   

3. We designed MGCR, a multi-level  geometric-aware contrastive regularization method that uses upsampled partial point clouds as negative samples to improve completion performance by focusing on missing regions.

`

## Usage

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
```
The solution for a common bug in chamfer distance installation can be found in Issue [#6](https://github.com/yuxumin/PoinTr/issues/6)
```
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Note: If you still get `ModuleNotFoundError: No module named 'gridding'` or something similar then run these steps

```
    1. cd into extensions/Module (eg extensions/gridding)
    2. run `python setup.py install`
```

That will fix the `ModuleNotFoundError`.




### Inference

To inference sample(s) with pretrained model

```
python tools/inference.py \
${POINTR_CONFIG_FILE} ${POINTR_CHECKPOINT_FILE} \
[--pc_root <path> or --pc <file>] \
[--save_vis_img] \
[--out_pc_root <dir>] \
```


### Evaluation

To evaluate a pre-trained PoinTr model on the Three Dataset with single GPU, run:

```
bash ./scripts/test.sh <GPU_IDS>  \
    --ckpts <path> \
    --config <config> \
    --exp_name <name> \
    [--mode <easy/median/hard>]
```


### Training

To train a point cloud completion model from scratch, run:

```
# Use DistributedDataParallel (DDP)
bash ./scripts/dist_train.sh <NUM_GPU> <port> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
# or just use DataParallel (DP)
bash ./scripts/train.sh <GPUIDS> \
    --config <config> \
    --exp_name <name> \
    [--resume] \
    [--start_ckpts <path>] \
    [--val_freq <int>]
```
## Acknowledgement
A large part of the code is borrowed from [Anchorformer](https://github.com/chenzhik/AnchorFormer), [PoinTr](https://github.com/ifzhang/ByteTrack),  Thanks for their wonderful works!

## Citation
The related resources will be released after the manuscript is accepted. 
