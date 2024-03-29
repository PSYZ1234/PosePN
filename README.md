# PosePN
LiDAR-based localization using universal encoding and memory-aware regression

## Environment

- python

- pytorch

- MinkowskiEngine

## Data

We support the Oxford Radar RobotCar, vReLoc, and NCLT datasets right now.
```
Oxford data_root
├── 2019-01-11-14-02-26-radar-oxford-10k
│   ├── velodyne_left
│       ├── xxx.bin
├── pose_stats.txt
├── pose_max_min.txt
├── train_split.txt
├── test_split.txt
```

## Run
### Oxford, vReLoc, NCLT

- train -- 1 GPU
```
python train.py
```

- test  -- 1 GPU
```
python eval.py
```

## Acknowledgement

 We appreciate the code of PointNet, PointNet++, SOE-Net, MinkLoc3D, and AtLoc they shared.

## Citation

```
@article{YU2022108685,
title = {LiDAR-based localization using universal encoding and memory-aware regression},
journal = {Pattern Recognition},
volume = {128},
pages = {108685},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.108685},
author = {Shangshu Yu and Cheng Wang and Chenglu Wen and Ming Cheng and Minghao Liu and Zhihong Zhang and Xin Li}
}
```
