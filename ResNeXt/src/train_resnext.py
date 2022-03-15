import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from Paper.ResNeXt.tools.cifar10_dataset import CifarDataset
from Paper.ResNeXt.tools.resnext_29 import ResNeXt29_2x64d
from Paper.ResNeXt.tools.common_tools import ModelTrainer, show_confMat, plot_line

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    # config
    train_dir = os.path.join(BASE_DIR, "..", "..", "Data", "cifar-10",  "cifar10_train")
    test_dir = os.path.join(BASE_DIR, "..", "..", "Data", "cifar-10", "cifar10_test")

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_classes = 10

    MAX_EPOCH = 300     # 182     # 64000 / (45000 / 128) = 182 epochs
    BATCH_SIZE = 128   # 1660ti 最大适合96，大家根据自己的设备情况进行设置batchsize
    LR = 0.1
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    milestones = [150, 225]  #

    # ============================ step 1/5 数据 ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建MyDataset实例
    train_data = CifarDataset(data_dir=train_dir, transform=train_transform)
    valid_data = CifarDataset(data_dir=test_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=32, num_workers=2)

    # ============================ step 2/5 模型 ============================
    resnet_model = ResNeXt29_2x64d()

    resnet_model.to(device)
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    optimizer = optim.SGD(resnet_model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)  # 选择优化器

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)

# ============================ step 5/5 训练 ============================
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, resnet_model, criterion, optimizer, epoch, device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, resnet_model, criterion, device)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
            epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        scheduler.step()  # 更新学习率

        # 绘图
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == MAX_EPOCH-1)
        show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == MAX_EPOCH-1)

        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if epoch > (MAX_EPOCH/2) and best_acc < acc_valid:
            best_acc = acc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": resnet_model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                      best_acc, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)






