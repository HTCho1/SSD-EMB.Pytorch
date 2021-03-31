# SSD-EMB: An Improved SSD using Enhanced Feature Map Block for Object Detection
This is implementtation of SSD-EMB from Hong-Tae Choi, Ho-Jun Lee, Hoon Kang, Sungwook Yu, and Ho-Hyun Park.

This code is heavily depend on [here](https://github.com/amdegroot/ssd.pytorch).

Thanks deGroot.
## Environment
Python 3.x

PyTorch 1.3+

Numpy

OpenCV

...
## Datasets
To make things easy, we provide bash scripts to handle the dataset downloads and setup for you.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).

### COCO
Microsoft COCO: Common Objects in Context

##### Download COCO train2017 and test2015
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
cd data/scripts
sh COCO2017.sh
```

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
cd data/scripts
sh VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
cd data/scripts
sh VOC2012.sh # <directory>
```
## Train
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:
  https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```
