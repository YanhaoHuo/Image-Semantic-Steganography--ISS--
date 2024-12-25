from torchvision import transforms
import numpy as np
import torch
import copy
from PIL import Image
import random
from torchvision import transforms


def data_tr_1(x):
    x = x.resize((64, 64))
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


def data_transform_1(x):
    a = random.randint(1, 6)
    if a == 1:
        x = transforms.RandomHorizontalFlip(p=1)(x)
    if a == 2:
        x = transforms.RandomVerticalFlip(p=1)(x)
    if a == 3:
        x = transforms.RandomRotation(degrees=(45, 45), expand=True)(x)
    if a == 4:
        x = transforms.CenterCrop(28)(x)
    if a == 5:
        x = transforms.RandomCrop(28)(x)
    if a == 6:
        x = transforms.RandomCrop(64, pad_if_needed=True, fill=0,padding_mode='constant')(x)
    x = x.resize((64, 64))
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

