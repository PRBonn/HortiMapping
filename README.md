# HortiMapping

This repository contains the codes for the IROS 2023 paper "Panoptic Mapping with Fruit Completion and Pose Estimation for Horticultural Robots". The codes will be released soon.

<p align="center">
  <img src="https://user-images.githubusercontent.com/34207278/256343660-51b38d98-03ff-423d-8ab9-f6809b3529b2.png" width="100%" />
</p>


----
## Abstract
Monitoring plants and fruits at high resolution play a key role in the future of agriculture. Accurate 3D information can pave the way to a diverse number of robotic applications in agriculture ranging from autonomous harvesting to precise yield estimation. Obtaining such 3D information is non-trivial as agricultural environments are often repetitive and cluttered, and one has to account for the partial observability of fruit and plants.
In this paper, we address the problem of jointly estimating complete 3D shapes of fruit and their pose in a 3D multi-resolution map built by a mobile robot. 
To this end, we propose an online multi-resolution panoptic mapping system where regions of interest are represented with a higher resolution. We exploit data to learn a general fruit shape representation that we use at inference time together with an occlusion-aware differentiable rendering pipeline to complete partial fruit observations and estimate the 7 DoF pose of each fruit in the map.
The experiments presented in this paper, evaluated both in the controlled environment and in a commercial greenhouse, show that our novel algorithm yields higher completion and pose estimation accuracy than existing methods, with an improvement of 41% in completion accuracy and 52% in pose estimation accuracy while keeping a low inference time of 0.6s in average.

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


