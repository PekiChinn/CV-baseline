# -*- coding: utf-8 -*-
"""
# @file name   : common_tools
# @author      : PeiJi
# @date        : 2021/12/7
# @brief       :
"""
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models


def get_vgg16(path_state_dict, device, vis_model=False):
    """
    :param path_state_dict:
    :param device:
    :param vis_model:
    :return:
    """
    model = models.vgg16()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model