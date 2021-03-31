# SSD-EMB: An Improved SSD using Enhanced Feature Map Block for Object Detection
This is implementtation of SSD-EMB from Hong-Tae Choi, Ho-Jun Lee, Hoon Kang, Sungwook Yu, and Ho-Hyun Park.
This code is heavily depend on [here](https://github.com/amdegroot/ssd.pytorch). We make some changes, fix some bugs, and give out our code.
Thank you to deGroot and his team.
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
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
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

- Training Parameter Options: 

```Python
parser = argparse.ArgumentParser(
    description='SSD-EMB Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--input', default=300, type=int, choices=[300, 512],
                    help='SSD input size, currently support ssd300 and ssd512')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.

<img src= "https://github.com/HTCho1/SSD-EMB.Pytorch/blob/main/doc/comparison_of_bboxes.PNG">


## Performance

#### VOC 2007 test set

##### mAP

| SSD300-EMB | SSD512-EMB |
|:-:|:-:|
| 78.4 % | 80.4 % |

##### FPS
**RTX 2080Ti:** ~30 FPS

***Note:*** The current code is not neatly organized, so we will keep updating it. The final code will be uploaded soon.

## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original SSD Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](http://www.webyclip.com) for their help in finishing the data augmentation portion.
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo):
  * [Chainer](https://github.com/Hakuyume/chainer-ssd), [Keras](https://github.com/rykov8/ssd_keras), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow)
