"""Semseg challenge Dataloader"""

import os
import numpy as np

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import cv2
import json

from .luv2rgb import convert_luv_to_rgb


class Semseg(data.Dataset):

    NUM_CLASS = 8

    def __init__(self, image_paths, label_paths=None, image_height=512, image_width=1664, luv=False, small_sample=False):
        self.image_paths = image_paths
        if small_sample:
            self.image_paths = image_paths[:20]

        self.label_paths = label_paths
        
        if label_paths and small_sample:
            self.label_paths = label_paths[:20]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_height = image_height
        self.image_width = image_width
        self.luv = luv

        self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7]
        self._key = [0, 3, 0, 3, 3, 3, 3, 0, 3, 3, 0, 0,
                    3, 3, 3, 3, 3, 0, 0, 5, 0, 0, -1, 4,
                    0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 7, 6, 0,
                    1, 2, 1, 1, 2, 1, 6, 1, 2, 2, 2, 0, 2,
                    1, 1, 2, 0, 0, 0, 1, 1, 1, 0, 0, -1, -1]
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    
    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        for i in range(len(index)):
            index[i] = self._key[index[i]]
        return index.reshape(mask.shape)

    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = cv2.imread(image_path, -1)

        if self.luv:
            img = torch.squeeze(convert_luv_to_rgb(img[:,:,0], img[:,:,[1,2]]))
        
        img = np.clip(img, 0, 255)
        img = self.transform(img)

        image_height = img.shape[1]
        if image_height > self.image_height:
            img = img[:,image_height - self.image_height:, :]

        label_path = self.label_paths[idx]
        label = cv2.imread(label_path, -1)
        label = np.array(label).astype('int32')

        label_height = label.shape[0]
        if label_height > self.image_height:
            label = label[image_height - self.image_height:, :]

        target = self._class_to_index(np.array(label).astype('int32'))
        target = torch.LongTensor(np.array(target).astype('int32'))

        return img, target


    def __len__(self):
        return len(self.image_paths)


    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
