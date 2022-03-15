# -*- coding: utf-8 -*-
"""
# @file name  : BN_visual.py
# @brief      : BN层对数据分布对影响
"""

import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from Paper.BatchNormalization.tools.common_tools import generate_data, Net

torch.manual_seed(1)  # reproducible
np.random.seed(1)


def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):

    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0:
            p_range = (-7, 10)
            the_range = (-7, 10)
        else:
            p_range = (-4, 4)
            the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)
        ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359')
        ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]:
            a.set_yticks(())
            a.set_xticks(())

        ax_pa_bn.set_xticks(p_range)
        ax_bn.set_xticks(the_range)

        axs[0, 0].set_ylabel('PreAct')
        axs[1, 0].set_ylabel('BN PreAct')
        axs[2, 0].set_ylabel('Act')
        axs[3, 0].set_ylabel('BN Act')

    plt.suptitle("Activation:{} epoch:{}/{}".format(act_name, epoch, EPOCH))
    plt.pause(0.05)
    # plt.savefig("{}.png".format(epoch))


if __name__ == "__main__":

    act_name = "ReLU"
    # act_name = "Tanh"
    # act_name = "Sigmoid"
    activations = {"ReLU": torch.relu, "Tanh": torch.tanh, "Sigmoid": torch.sigmoid}
    ACTIVATION = activations[act_name]
    EPOCH = 12
    LR = 0.03
    N_HIDDEN = 8
    N_SAMPLES = 2000
    BATCH_SIZE = 64
    B_INIT = -0.2  # use a bad bias constant initializer

    # 1. 生成虚假数据
    train_x, train_y, test_x, test_y = generate_data(N_SAMPLES)
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # show data
    plt.scatter(train_x.numpy(), train_y.numpy(), c='#FF9359', s=50, alpha=0.2, label='train')
    plt.legend(loc='upper left')

    # 2. 创建网络/loss/优化器
    nets = [Net(N_HIDDEN, ACTIVATION, batch_normalization=False), Net(N_HIDDEN, ACTIVATION, batch_normalization=True)]
    loss_func = torch.nn.MSELoss()
    opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]

    # 3. 训练，绘图
    f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
    plt.ion()  # something about plotting
    plt.show()

    losses = [[], []]  # recode loss for two networks
    for epoch in range(EPOCH):
        print('Epoch: {}/{}'.format(epoch, EPOCH))

        # 记录数据
        layer_inputs, pre_acts = [], []
        for net, l in zip(nets, losses):
            net.eval()              # set eval mode to fix moving_mean and moving_var
            pred, layer_input, pre_act = net(test_x)
            l.append(loss_func(pred, test_y).data.item())
            layer_inputs.append(layer_input)
            pre_acts.append(pre_act)
            net.train()             # free moving_mean and moving_var
        plot_histogram(*layer_inputs, *pre_acts)     # plot histogram

        # 训练更新模型
        for step, (b_x, b_y) in enumerate(train_loader):
            for net, opt in zip(nets, opts):     # train for each network
                pred, _, _ = net(b_x)
                loss = loss_func(pred, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()    # it will also learns the parameters in Batch Normalization

    plt.ioff()

    # plot training loss
    plt.figure(2)
    plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
    plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.ylim((0, 2000))
    plt.legend(loc='best')

    # evaluation
    # set net to eval mode to freeze the parameters in batch normalization layers
    [net.eval() for net in nets]    # set eval mode to fix moving_mean and moving_var
    preds = [net(test_x)[0] for net in nets]
    plt.figure(3)
    plt.plot(test_x.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
    plt.plot(test_x.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
    plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
    plt.legend(loc='best')
    plt.show()












