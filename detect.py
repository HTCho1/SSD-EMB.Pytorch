import os
import sys
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from data import VOC_CLASSES as labels
from ssd import build_ssd
from tqdm import tqdm

weight_path = './weights/ssd300_VOC_b32_mAP_78.4.pth'
voc_path = './data/VOCdevkit/VOC2007'
model_input = 300

with open(os.path.join(voc_path, 'ImageSets/Main/test.txt')) as f:
    ids = f.read().splitlines()
    train_images=[]
    for id in ids:
        # Parse annotation's XML file
        # objects = pares_annotation(os.path.join(voc_path, 'Annotations', id + '.xml'), state='train')
        # if len(objects) == 0:
        #     continue
        # n_objects += len(objects)
        # train_objects.append(objects)
        train_images.append(os.path.join(voc_path, 'JPEGImages', id + '.jpg'))

count=0

net = build_ssd('test', model_input, 21)
net.load_state_dict(torch.load(weight_path))
tqdm.write('State is loaded!')
#net.load_weights(weight_path)

for img_path in train_images[0:]:
    file_id = img_path[36:-4]
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (model_input, model_input)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))   # [1, 3, 300, 300]

    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    top_k=10
    detections = y.data

    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.2:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s'%(label_name)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            j += 1
            image = cv2.rectangle(image,(pt[0],pt[1]),(pt[2],pt[3]),(255,0,0), 2)
            image = cv2.putText(image, display_txt, (pt[0],pt[1]), cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,0),True)

    cv2.imwrite('./SSD300_EMB_detect_VOC2007test/{}.jpg'.format(file_id), image)
    print(('Image {} Saved'.format(file_id)))
print('All images Saved!')
