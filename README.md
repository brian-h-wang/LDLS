# LDLS: Label Diffusion Lidar Segmentation

![LDLS](readme_assets/kitti_example.png)

LDLS performs 3D instance segmentation of lidar point clouds, by using a pretrained Mask-RCNN model to perform 2D segmentation of an aligned camera image, constructing a graph that connects 2D pixels to 3D lidar points, and then performing label diffusion to output final lidar point labels.

LDLS requires no annotated 3D training data, and is capable of performing instance segmentation of any object class that the 2D image segmentation model is trained to recognize.

For details on the algorithm, please see our paper, ["LDLS: 3-D Object Segmentation Through Label Diffusion From 2-D Images"](https://ieeexplore.ieee.org/document/8735751), published in the IEEE Robotics and Automation Letters (to be presented at IROS 2019 in Macau).
 
## Citation

From IEEE Xplore:
```
@ARTICLE{8735751,
author={B. H. {Wang} and W. {Chao} and Y. {Wang} and B. {Hariharan} and K. Q. {Weinberger} and M. {Campbell}},
journal={IEEE Robotics and Automation Letters},
title={LDLS: 3-D Object Segmentation Through Label Diffusion From 2-D Images},
year={2019},
volume={4},
number={3},
pages={2902-2909},
keywords={Three-dimensional displays;Two dimensional displays;Image segmentation;Laser radar;Sensors;Cameras;Task analysis;Object detection;segmentation and categorization;RGB-D perception},
doi={10.1109/LRA.2019.2922582},
ISSN={},
month={July},}
```

## Installation

Requires Python 3.6+

Depends on
* [Matterport Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN)
* [CuPy](https://cupy.chainer.org/) for sparse matrix multiplication on the GPU
* [Numba](https://numba.pydata.org/numba-doc/dev/user/installing.html) for speeding up graph construction using the GPU.

Installing dependencies using conda is recommended, in particular this makes it easier to install Numba with CUDA GPU support. Use the included `environment.yml` file:

``conda env create -f environment.yml``

The [Point Processing Toolkit](https://github.com/heremaps/pptk) can also be useful for visualizing KITTI lidar point clouds:

``pip install pptk``

## Usage

See the `demo.ipynb` Jupyter notebook for an example of how to use LDLS.

If you used conda to install dependencies, activate your ldls conda environment, then run the following command to create an iPython kernel which you can use with the Jupyter notebook:

``ipython kernel install --user --name=LDLS``


## Results

See the following videos for demonstrations of LDLS in action:

* [KITTI 64 Residential Road Sequence](https://youtu.be/XlXneiGB5NU)
* [KITTI 91 Urban Sequence](https://youtu.be/EtLl4KnuM-s)
* [Campus Mobile Robot Data Collection](https://youtu.be/4azvaDHEcQU)


## Evaluation Data

The manually-labeled ground truth KITTI instance segmentation data used in our experiments as available [here](https://drive.google.com/drive/folders/11rD0Nm65YwvR_unVxxZ--5j00qR8xO_H?usp=sharing).

The Python annotation tool used to label the data is also available at https://github.com/brian-h-wang/kitti-3d-annotator .

Please consider citing our paper if these are useful to you.
