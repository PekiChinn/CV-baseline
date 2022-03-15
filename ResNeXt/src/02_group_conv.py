# -*- coding: utf-8 -*-
"""
# @file name  : 02_group_conv.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-07-17
# @brief      : 分组卷积
"""
import torch
import torch.nn as nn

if __name__ == '__main__':

    in_channels = 15
    groups = 3
    fake_img = torch.randn(1, in_channels, 17, 17)
    group_conv_layer = nn.Conv2d(in_channels, 4, 3, padding=1, groups=groups)
    f_map = group_conv_layer(fake_img)

    print(f_map.shape)





