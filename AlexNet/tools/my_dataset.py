# -*- coding: utf-8 -*-
"""
# @file name  : my_dataset.py
"""
import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)

class CatDogDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=620, transform=None):
        """
        :param data_dir:
        :param mode:
        :param split_n:
        :param rng_seed:
        :param transform:
        """
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        self.split_n = split_n
        self.data_info = self._get_img_info()
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir: {} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def _get_img_info(self):

        img_names = os.listdir(self.data_dir)
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

        random.seed(self.rng_seed)
        random.shuffle(img_names)

        img_labels = [0 if x.startswith('cat') else 1 for x in img_names]

        split_idx = int(len(img_labels) * self.split_n)
        if self.mode == "train":
            img_set = img_names[:split_idx]   # 数据集90%训练
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别，仅支持(train, valid)")

        path_img_set = [os.path.join(self.data_dir, n) for n in img_set]
        data_info = [(n, l) for n, l in zip(path_img_set, label_set)]

        return data_info
