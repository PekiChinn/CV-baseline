# -*- coding: utf-8 -*-
"""
# @file name  : my_dataset.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-02-14
# @brief      : NFCM数据集读取：https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
class_name = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]

class NCFMDataSet(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=620, transform=None):
        """
        鱼类分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        self.split_n = split_n
        self.data_info = self._get_img_info()  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def _get_img_info(self):

        img_path = []
        for root, dirs, files in os.walk(self.data_dir):
            for name in files:
                if name.endswith(".jpg"):
                    img_path.append(os.path.join(root, name))

        random.seed(self.rng_seed)
        random.shuffle(img_path)

        img_labels = [class_name.index(os.path.basename(os.path.dirname(p))) for p in img_path]

        split_idx = int(len(img_labels) * self.split_n)
        # split_idx = int(100 * self.split_n)
        if self.mode == "train":
            img_set = img_path[:split_idx]     # 数据集90%训练
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_path[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别，仅支持(train, valid)")

        path_img_set = [os.path.join(self.data_dir, n) for n in img_set]
        data_info = [(n, l) for n, l in zip(path_img_set, label_set)]

        return data_info
