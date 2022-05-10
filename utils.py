from __future__ import print_function  # 打印
import argparse  # 超参数
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F  # 有包括卷积在内的各种函数
import torch.optim as optim  # 优化器
from torchvision import datasets, transforms  # 数据集和图像变换
from torch.optim.lr_scheduler import StepLR  # 不断改变的学习率
import torchvision
import numpy as np

# vgg16 = torchvision.models.vgg16(pretrained=False)
# resnet34 = torchvision.models.resnet34(pretrained=False)


# 如果需要预训练模型就让pretrained=True
# train_data = torchvision.datasets.CIFAR10('./data', train=True,
#                                           transform=torchvision.transforms.ToTensor(), download=True)
# print(vgg16)


# vgg16.classifier.add_module('Linear',nn.Linear(1000,10))

# MLP
# 建立一个四层感知机网络
# class MLP(nn.Module):  # 继承 torch 的 Module
#     def __init__(self):
#         super(MLP, self).__init__()  #
#         # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
#         self.fc1 = nn.Linear(784, 512)  # 第一个隐含层
#         self.fc2 = nn.Linear(512, 128)  # 第二个隐含层
#         self.fc3 = nn.Linear(128, 3)  # 输出层
#         pass
#
#     def forward(self, din):
#         # 前向传播， 输入值：din, 返回值 out
#         din = din.view(-1, 28 * 28)  # 将一个多行的Tensor,拼接成一行
#         out = F.relu(self.fc1(din))  # 使用 relu 激活函数
#         out = F.relu(self.fc2(out))
#         out = F.softmax(self.fc3(out), dim=1)  # 输出层使用 softmax 激活函数
#         # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
#         return out
#         # pass
#
#
# # U-Net
# class DownsampleLayer(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DownsampleLayer, self).__init__()
#         self.Conv_BN_ReLU_2 = nn.Sequential(
#             nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )
#         self.downsample = nn.Sequential(
#             nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         """
#         :param x:
#         :return: out输出到深层，out_2输入到下一层，
#         """
#         out = self.Conv_BN_ReLU_2(x)
#         out_2 = self.downsample(out)
#         return out, out_2
#
#
# class UpSampleLayer(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         # 512-1024-512
#         # 1024-512-256
#         # 512-256-128
#         # 256-128-64
#         super(UpSampleLayer, self).__init__()
#         self.Conv_BN_ReLU_2 = nn.Sequential(
#             nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(out_ch * 2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
#                       padding=(1, 1)),
#             nn.BatchNorm2d(out_ch * 2),
#             nn.ReLU()
#         )
#         self.upsample = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
#                                padding=(1, 1),
#                                output_padding=(1, 1)),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )
#
#     def forward(self, x, out):
#         '''
#         :param x: 输入卷积层
#         :param out:与上采样层进行cat
#         :return:
#         '''
#         x_out = self.Conv_BN_ReLU_2(x)
#         x_out = self.upsample(x_out)
#         cat_out = torch.cat((x_out, out), dim=1)
#         return cat_out


# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]
#         # 下采样
#         self.d1 = DownsampleLayer(3, out_channels[0])  # 3-64
#         self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
#         self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
#         self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
#         # 上采样
#         self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
#         self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
#         self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
#         self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
#         # 输出
#         self.o = nn.Sequential(
#             nn.Conv2d(out_channels[1], out_channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(out_channels[0]),
#             nn.ReLU(),
#             nn.Conv2d(out_channels[0], out_channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.BatchNorm2d(out_channels[0]),
#             nn.ReLU(),
#             nn.Conv2d(out_channels[0], 3, (3, 3), (1, 1), (1, 1)),
#             nn.Sigmoid(),
#             # BCELoss
#         )
#
#     def forward(self, x):
#         out_1, out1 = self.d1(x)
#         out_2, out2 = self.d2(out1)
#         out_3, out3 = self.d3(out2)
#         out_4, out4 = self.d4(out3)
#         out5 = self.u1(out4, out_4)
#         out6 = self.u2(out5, out_3)
#         out7 = self.u3(out6, out_2)
#         out8 = self.u4(out7, out_1)
#         out = self.o(out8)
#         return out


# class classification(nn.Module):  # 继承 torch 的 Module
#     def __init__(self):
#         super(classification, self).__init__()  #
#         # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
#         self.fc1 = nn.Linear(784, 512)  # 第一个隐含层
#         self.fc2 = nn.Linear(512, 128)  # 第二个隐含层
#         self.fc3 = nn.Linear(128, 10)  # 输出层
#         pass
#
#     def forward(self, din):
#         # 前向传播， 输入值：din, 返回值 dout
#         din = din.view(-1, 28 * 28)  # 将一个多行的Tensor,拼接成一行
#         dout = F.relu(self.fc1(din))  # 使用 relu 激活函数
#         dout = F.relu(self.fc2(dout))
#         dout = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
#         # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
#         return dout
#         pass


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
