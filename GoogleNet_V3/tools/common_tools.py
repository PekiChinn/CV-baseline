import numpy as np
import torch
import torch.nn as nn
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models


def get_googlenet_v3(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.inception_v3()
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)
        model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 299, 299), device="cpu")

    model.to(device)
    return model


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    """
    def __init__(self, eps=0.001):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps

    def forward(self, x, target):
        # CE(q, p) = - sigma(q_i * log(p_i))
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)  # 实现  log(p_i)

        # H(q, p)
        H_pq = -log_probs.gather(dim=-1, index=target.unsqueeze(1))  # 只需要q_i == 1的地方， 此时已经得到CE
        H_pq = H_pq.squeeze(1)

        # H(u, p)
        H_uq = -log_probs.mean()  # 由于u是均匀分布，等价于求均值

        loss = (1 - self.eps) * H_pq + self.eps * H_uq
        return loss.mean()

