# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : 通用函数
"""
import numpy as np
import torch
import torch.nn as nn
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.models as models
from Paper.GoogleNet_V4.tools.inceptionv4 import inceptionv4
from Paper.GoogleNet_V4.tools.inceptionresnetv2 import inceptionresnetv2


def get_inception_v4(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = inceptionv4(num_classes=1001, pretrained=None)
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)
        model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 299, 299), device="cpu")

    model.to(device)
    return model


def get_inception_resnet_v2(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = inceptionresnetv2(num_classes=1001, pretrained=False)
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)
        model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 299, 299), device="cpu")

    model.to(device)
    return model




