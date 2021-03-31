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
