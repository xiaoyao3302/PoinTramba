This is the official PyTorch implementation of our paper:

> **[PoinTramba: A Hybrid Transformer-Mamba Framework for Point Cloud Analysis](https://arxiv.org/abs/2405.15463)**



> **Abstract.** 
> Point cloud analysis has seen substantial advancements due to deep learning, although previous Transformer-based methods excel at modeling long-range dependencies on this task, their computational demands are substantial. Conversely, the Mamba offers greater efficiency but shows limited potential compared with Transformer-based methods. In this study, we introduce PoinTramba, a pioneering hybrid framework that synergies the analytical power of Transformer with the remarkable computational efficiency of Mamba for enhanced point cloud analysis. Specifically, our approach first segments point clouds into groups, where the Transformer meticulously captures intricate intra-group dependencies and produces group embeddings, whose inter-group relationships will be simultaneously and adeptly captured by efficient Mamba architecture, ensuring comprehensive analysis. Unlike previous Mamba approaches, we introduce a bi-directional importance-aware ordering (BIO) strategy to tackle the challenges of random ordering effects. This innovative strategy intelligently reorders group embeddings based on their calculated importance scores, significantly enhancing Mamba's performance and optimizing the overall analytical process. Our framework achieves a superior balance between computational efficiency and analytical performance by seamlessly integrating these advanced techniques, marking a substantial leap forward in point cloud analysis. Extensive experiments on datasets such as ScanObjectNN, ModelNet40, and ShapeNetPart demonstrate the effectiveness of our approach, establishing a new state-of-the-art analysis benchmark on point cloud recognition. For the first time, this paradigm leverages the combined strengths of both Transformer and Mamba architectures, facilitating a new standard in the field.


## Installation

```
# Create virtual env and install PyTorch
$ conda create -n PoinTramba python=3.9
$ conda activate PoinTramba

(PoinTramba) $ pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

(PoinTramba) $ pip install -r requirements.txt

(PoinTramba) $ cd ./extensions/chamfer_dist && python setup.py install --user
(PoinTramba) $ cd ./extensions/emd && python setup.py install --user

(PoinTramba) $ pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

(PoinTramba) $ pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

(PoinTramba) $ pip install causal-conv1d==1.1.1
(PoinTramba) $ pip install mamba-ssm==1.1.1

(PoinTramba) $ pip install -r requirements.txt
```


## Dataset:

### ScanObjNN
```
wget http://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip
```

### ModelNet40
```
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
```

### ShapeNetPart
```
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

Please modify the dataset path in configuration files (dataset_configs). 
```
├── data
    ├── ScanObjNN
        ├── h5_files
            ├── main_split
            └── ....
    ├── ModeNet40
        └── modelnet40_ply_hdf5_2048
            ├── ply_data_train0.h5
            └── ....
    └── ShapeNetPart
        └── shapenetcore_partanno_segmentation_benchmark_v0_normal
            ├── 02691156
            └── ....


```


## Usage

To run our code, you can directly run the sh files like:

```
bash run.sh
```


To run with different settings, please modify the args settings, including

```
--attention_depth
--mode_group
--type_pooling
--type_weighting 
--mode_sort 
--seed
etc
```

Note to modify the 'NAME' of the 'model' in the config files.
PointMambaFormer includes our alignment loss and our importance loss
BasePointMamba is the pure framework 

## Citation

If you find these projects useful, please consider citing our paper.

## Note

We will further improve our PoinTramba on PartSegmentation and we wil release the segmentation code soon.



## Acknowledgement

We thank [PointMamba](https://github.com/LMD0311/PointMamba), [PointCloudMamba](https://github.com/SkyworkAI/PointCloudMamba) [PointBERT](https://github.com/lulutang0608/Point-BERT), [Mamba](https://github.com/state-spaces/mamba) and other relevant works for their amazing open-sourced projects!
