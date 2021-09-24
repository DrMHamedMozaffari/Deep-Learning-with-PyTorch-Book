#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:37:39 2021

@author: hamed
"""

''' How to use pretrained models '''

from torchvision import models
from torchvision import transforms
from PIL import Image
import torch

# name of all models available in PyTorch folder
# print(dir(models))

# using AlexNet as an example
alexnet = models.AlexNet()

# Resnet model with pretrained data
resnet = models.resnet101(pretrained=True)

# now is the time to use the models
# first preprocessing pipeline

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])])

img = Image.open('../repos/dlwpt-code/data/p1ch2/bobby.jpg')
img.show()

img_t = preprocess(img)

batch_t = torch.unsqueeze(img_t, 0)

# evaluate the network internal settings like batchnormalization and dropputs off
resnet.eval()

out = resnet(batch_t)

with open('../repos/dlwpt-code/data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
    
_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# print(labels[index[0]], percentage[index[0]].item())

# to find the second best
_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])

print('test')



