# -*- coding: utf-8 -*-
"""
# @file name  : train_googlenet.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-01
# @brief      : googlenet-v3 traning
"""

import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision.models as models
from lesson.C_GoogLeNet.tools.my_dataset import NCFMDataSet
from lesson.E_GoogLeNet_v3.tools.common_tools import get_googlenet_v3, LabelSmoothingCrossEntropy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    # config
    data_dir = os.path.join(BASE_DIR, "..", "..", "Data", "NCFM", "train")
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "inception_v3_google-1a9a5a14.pth")
    # path_state_dict = False
    num_classes = 8

    MAX_EPOCH = 3
    BATCH_SIZE = 8
    LR = 0.001
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    lr_decay_step = 1

    # ============================ step 1/5 数据 ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((342)),  # 299 / (224/256) = 342
        transforms.CenterCrop(299),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    normalizes = transforms.Normalize(norm_mean, norm_std)
    valid_transform = transforms.Compose([
        transforms.Resize((342, 342)),
        transforms.TenCrop(299, vertical_flip=False),
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops])),
    ])

    # 构建MyDataset实例
    train_data = NCFMDataSet(data_dir=data_dir, mode="train", transform=train_transform)
    valid_data = NCFMDataSet(data_dir=data_dir, mode="valid", transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4)

    # ============================ step 2/5 模型 ============================
    googlenet_v3_model = get_googlenet_v3(path_state_dict, device, False)

    num_ftrs = googlenet_v3_model.fc.in_features
    googlenet_v3_model.fc = nn.Linear(num_ftrs, num_classes)

    num_ftrs_1 = googlenet_v3_model.AuxLogits.fc.in_features
    googlenet_v3_model.AuxLogits.fc = nn.Linear(num_ftrs_1, num_classes)

    googlenet_v3_model.to(device)
    # ============================ step 3/5 损失函数 ============================
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy(eps=0.001)
    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    flag = 0
    # flag = 1
    if flag:
        fc_params_id = list(map(id, googlenet_v3_model.classifier.parameters()))  # 返回的是parameters的 内存地址
        base_params = filter(lambda p: id(p) not in fc_params_id, googlenet_v3_model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': LR * 0.1},  # 0
            {'params': googlenet_v3_model.classifier.parameters(), 'lr': LR}], momentum=0.9)

    else:
        optimizer = optim.SGD(googlenet_v3_model.parameters(), lr=LR, momentum=0.9)  # 选择优化器

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)  # 设置学习率下降策略
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=5)

# ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        googlenet_v3_model.train()
        for i, data in enumerate(train_loader):

            # 1. forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = googlenet_v3_model(inputs)

            # 2. backward
            optimizer.zero_grad()
            loss_main, aug_loss1 = criterion(outputs[0], labels), criterion(outputs[1], labels)
            loss = loss_main + (0.3 * aug_loss1)
            loss.backward()

            # 3. update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # 打印训练信息
            loss_mean += loss_main.item()
            train_curve.append(loss_main.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss_main:{:.4f} Acc:{:.2%} lr:{}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total, scheduler.get_last_lr()))
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            googlenet_v3_model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    bs, ncrops, c, h, w = inputs.size()
                    outputs = googlenet_v3_model(inputs.view(-1, c, h, w))
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                    loss = criterion(outputs_avg, labels)

                    _, predicted = torch.max(outputs_avg.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val/len(valid_loader)
                valid_curve.append(loss_val_mean)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_mean, correct_val / total_val))
            googlenet_v3_model.train()

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()





