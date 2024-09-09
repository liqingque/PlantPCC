# PlantPCC: Plant Point Cloud Completion Based on Edge-Uniform Dual Sampling and Hierarchical Geometry-aware Contrastive Regularization
The official implementation of the paper：
PlantPCC: Plant Point Cloud Completion Based on Edge-Uniform Dual Sampling and Hierarchical Geometry-aware Contrastive Regularization

Contact: xiaomengli@cau.edu.cn Any questions or discussion are welcome!

-----


+ [2024.09.05] We have initialized the repo. The related resources will be released after the manuscript is accepted.

##DataSet

<img src="assets/dataset.png" alt="Dataset" width="800" height="600">




## Abstract
Completing plant point clouds is crucial for applications such as segmentation and surface reconstruction in plant phenotyping. Unlike the relatively simpler CAD models found in datasets like ShapeNet, plant point clouds present increased complexity and detail, making their completion particularly challenging. In response to this, we propose a novel model specifically designed for plant point cloud completion. Our approach incorporates a learnable edge-uniform dual sampling strategy to capture intricate geometric features better. Furthermore, we introduce a geometry-aware contrastive regularization method to enhance the differentiation between the predicted point cloud and the input incomplete point cloud, thereby ensuring that the reconstructed point cloud more closely aligns with the ground truth. Experimental results on the newly introduced PlantPCC dataset demonstrate that our model achieves state-of-the-art performance, with a 5\% improvement over the next best model. Additionally, our model exhibits competitive performance on the general object completion dataset PCN, indicating robust generalization capabilities. 

## Contributions
1. **Plant Point Cloud Completion Model**: We propose a novel model specifically designed for plant point cloud completion, addressing the increased complexity and detail compared to simpler CAD models found in datasets like ShapeNet. This model is crucial for applications such as segmentation and surface reconstruction in plant phenotyping.

2. **Innovative Techniques for Enhanced Performance**: Our approach incorporates a learnable edge-uniform dual sampling strategy to better capture intricate geometric features. Additionally, we introduce a geometry-aware contrastive regularization method to enhance the differentiation between the predicted point cloud and the input incomplete point cloud, ensuring that the reconstructed point cloud more closely aligns with the ground truth.

3. **State-of-the-Art Results and Robust Generalization**: Experimental results on the newly introduced PlantPCC dataset demonstrate that our model achieves state-of-the-art performance, with a 5% improvement over the next best model. Furthermore, our model exhibits competitive performance on the general object completion dataset PCN, indicating robust generalization capabilities. Relevant datasets and codes are available for further research and validation.

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
