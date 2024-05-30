# HortiMapping

This repository contains the codes for the IROS 2023 paper "Panoptic Mapping with Fruit Completion and Pose Estimation for Horticultural Robots". 

[**video**](https://youtu.be/fSyHBhskjqA?si=01ff9t4c4qVzRdmL) | [**paper**](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2023iros.pdf)

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/256343660-51b38d98-03ff-423d-8ab9-f6809b3529b2.png" width="100%" />
</p>

----

Panoptic Mapping | Fruit Completion |
:-: | :-: |
<video src='https://github.com/PRBonn/HortiMapping/assets/34207278/9c39c1f6-30af-4da7-bfe1-eb51ff2b339c'> | <video src='https://github.com/PRBonn/HortiMapping/assets/34207278/10714f5a-7d1d-4de4-9528-c9150ca0c9a3'> |


----
## Abstract
Monitoring plants and fruits at high resolution play a key role in the future of agriculture. Accurate 3D information can pave the way to a diverse number of robotic applications in agriculture ranging from autonomous harvesting to precise yield estimation. Obtaining such 3D information is non-trivial as agricultural environments are often repetitive and cluttered, and one has to account for the partial observability of fruit and plants.
In this paper, we address the problem of jointly estimating complete 3D shapes of fruit and their pose in a 3D multi-resolution map built by a mobile robot. 
To this end, we propose an online multi-resolution panoptic mapping system where regions of interest are represented with a higher resolution. We exploit data to learn a general fruit shape representation that we use at inference time together with an occlusion-aware differentiable rendering pipeline to complete partial fruit observations and estimate the 7 DoF pose of each fruit in the map.
The experiments presented in this paper, evaluated both in the controlled environment and in a commercial greenhouse, show that our novel algorithm yields higher completion and pose estimation accuracy than existing methods, with an improvement of 41% in completion accuracy and 52% in pose estimation accuracy while keeping a low inference time of 0.6s in average.

----

## Install

### 1. Set up conda environment

```
conda create --name homa python=3.8
conda activate homa
```

### 2. Install the key requirement PyTorch

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia 
```

The commands depend on your CUDA version. You may check the instructions [here](https://pytorch.org/get-started/previous-versions/).

### 3. Install other dependency

```
pip3 install open3d==0.17 opencv-python scikit-image wandb tqdm plyfile
```

----
## Instructions

### Clone the repository

```
git clone git@github.com:PRBonn/HortiMapping.git
cd HortiMapping
```

### Panoptic mapping

For the multi-resolution panoptic mapping part, we use our previous work [Voxfield Panmap](https://github.com/VIS4ROB-lab/voxfield-panmap).

### Fruit shape completion and pose estimation

We provide an [example data sequence](https://uni-bonn.sciebo.de/s/ovg3hIXHOeHdht6) generated from the public [BUP20 sweet pepper dataset](http://agrobotics.uni-bonn.de/sweet_pepper_dataset/) using multi-resolution panoptic mapping.

You can download this example data by:
```
sh scripts/download_bup_example.sh 
```

You can then test the shape completion and pose estimation using the example data sequence after setting the path by:

```
python test_wild_completion.py -c ./configs/wild_pepper.yaml 
```

You will see a visualizer showing the optimization process. You can then check the ```submaps_complete``` and ```submaps_pose``` folder in the example data folder for the completed mesh and pose for each fruit.

----

## Citation
If you use the repository for any academic work, please cite our paper.
```
@inproceedings{pan2023iros,
  author = {Y. Pan and F. Magistri and T. L\"abe and E. Marks and C. Smitt and C.S. McCool and J. Behley and C. Stachniss},
  title = {Panoptic Mapping with Fruit Completion and Pose Estimation for Horticultural Robots},
  booktitle={Proceedings of the IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```



