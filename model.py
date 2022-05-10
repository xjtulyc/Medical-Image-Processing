# import torch
# import torchvision
# import utils
# import torch.nn as nn
# import os
# import torch
# import warnings
# import numpy as np
# from visdom import Visdom
# import torch.optim as optim
# import torchxrayvision as xrv
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from sklearn.metrics import roc_auc_score
# from torch.optim.lr_scheduler import StepLR
#
# dense_net = xrv.models.DenseNet(num_classes=4, in_channels=3).to('cpu')  # DenseNet 模型，二分类
#
#
# # vgg16 = utils.vgg16
# # print(vgg16)
#
# class CancerClassificationNet(nn.Module):
#     def __init__(self):
#         super(CancerClassificationNet, self).__init__()
#         self.backbone = torchvision.models.resnet50(pretrained=False)
#         self.fc2 = nn.Linear(1000, 512)
#         self.fc3 = nn.Linear(512, 3)
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x
#
#
# class CancerDetectionNet(nn.Module):
#     def __init__(self):
#         super(CancerDetectionNet, self).__init__()
#         self.vgg16 = utils.vgg16
#         self.mlp = utils.MLP()
#         self.classification_double = utils.classification()
#         self.classification_single = utils.classification()
#         pass
#
#     def forward(self, x):
#         x1, x2 = x
#         if x2 is None:
#             """
#             无手工图像特征输入
#             """
#             out = self.vgg16(x1)
#             out = self.classification_single
#             return out
#         else:
#             out1 = self.vgg16(x1)
#             out2 = self.MLP(x2)
#             out = torch.cat([out1, out2], dim=1)
#             out = self.classification_double(out)
#             return out
#         pass
#
#
# class CancerSegmentationNet(nn.Module):
#     """
#     U-Net
#     """
#
#     def __init__(self):
#         super(CancerSegmentationNet, self).__init__()
#         out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]
#         # 下采样
#         self.d1 = utils.DownsampleLayer(3, out_channels[0])  # 3-64
#         self.d2 = utils.DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
#         self.d3 = utils.DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
#         self.d4 = utils.DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
#         # 上采样
#         self.u1 = utils.UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
#         self.u2 = utils.UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
#         self.u3 = utils.UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
#         self.u4 = utils.UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
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
