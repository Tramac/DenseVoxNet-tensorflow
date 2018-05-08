# DenseVoxNet-tensorflow
An implementation of DenseVoxNet introduced in TensorFlow.

Link to the original paper: [Automatic 3D Cardiovascular MR Segmentation with Densely-Connected Volumetric ConvNets, MICCAI 2017](https://arxiv.org/abs/1708.00573)

## Introduction
This repository includes the code (training and testing) for DenseVoxNet. The code is based on 3D-CNN for volumetric segmentation.

## Requirements
  ```
  python 2.7.x
  tensorflow >= 1.4.0
  ```

## Usage
1. Download [HVSMR](http://segchd.csail.mit.edu/data.html) dataset (phase 2) and put them in folder ``data``.
2. Prepare the hdf5 data to train the model.
  ```shell
  cd DenseVoxNet-tensorflow
  #modify parameters in prepare_h5_data.py file
  python prepare_h5_data.py
  ```
3. Train the model
  ```shell
  #the parameter of --mode in train.py need to be "train"
  python train.py
  ```
4. Test the model
  ```shell
  #the parameter of --mode in train.py need to be "test"
  python train.py
  ```
  
## Reference
  ```
@article{yu2017automatic,
    author = {Yu, Lequan and Cheng,Jie-Zhi and Dou, Qi and Yang, Xin and Chen, Hao and Qin, Jing and Heng, Pheng-Ann},
    title = {Automatic 3D Cardiovascular MR Segmentation with Densely-Connected Volumetric ConvNets},
    Journal = {MICCAI},
    year = {2017}
  }
  ```
