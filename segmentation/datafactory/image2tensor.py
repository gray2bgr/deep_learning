#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image,ImageFilter
import random
import io
import os
import os.path
import numpy as np
import numbers
import cv2

class Dataset(data.Dataset):
    def __init__(self, txt_list):

        self.data_list = self.read(txt_list)

    def __getitem__(self, index):
        image_path,mask_path = self.data_list[index]
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path,0)
        input_tensor = torch.from_numpy(image).float().div(255);
        target_tensor = torch.from_numpy((mask>128).astype(np.uint8));
        return input_tensor, target_tensor

    def __len__(self):
        return len(self.image_list)

    def read(self, txt_list):
        dataset_list = []
        with open(txt_list,'r') as f:
            for line in f.readlines():
                image_name,mask_name = line.strip().split(' ')
                dataset_list.append((image_name,mask_name))
        return dataset_list

