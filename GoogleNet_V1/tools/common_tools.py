# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-02-14
# @brief      : 通用函数
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models


def get_googlenet(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.googlenet(init_weights=False)
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)
        model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model
