# -*- coding: utf-8 -*-
"""
# @file name  : alexnet_visualization.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-14
# @brief      : alexnet_visualization
"""
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.models as models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    log_dir = os.path.join(BASE_DIR, "..", "results")
    # ----------------------------------- kernel visualization -----------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_kernel")

    # m1
    # alexnet = models.alexnet(pretrained=True)

    # m2
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet-owt-4df8aa71.pth")
    alexnet = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    alexnet.load_state_dict(pretrained_state_dict)

    kernel_num = -1
    vis_max = 1
    for sub_module in alexnet.modules():
        if not isinstance(sub_module, nn.Conv2d):
            continue
        kernel_num += 1
        if kernel_num > vis_max:
            break

        kernels = sub_module.weight
        c_out, c_int, k_h, k_w = tuple(kernels.shape)

        # 拆分channel
        for o_idx in range(c_out):
            kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)  # 获得(3, h, w), 但是make_grid需要 BCHW，这里拓展C维度变为（3， 1， h, w）
            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
            writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)

        kernel_all = kernels.view(-1, 3, k_h, k_w)  # 3, h, w
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
        writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=620)

        print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))

    # ----------------------------------- feature map visualization -----------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_feature map")

    # 数据
    path_img = os.path.join(BASE_DIR, "..", "data", "tiger cat.jpg")  # your path to image
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert('RGB')
    img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw

    # 模型
    # alexnet = models.alexnet(pretrained=True)

    # forward
    convlayer1 = alexnet.features[0]
    fmap_1 = convlayer1(img_tensor)

    # 预处理
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('feature map in conv1', fmap_1_grid, global_step=620)
    writer.close()
