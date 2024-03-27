import torch
from torch import nn
import torch.nn.functional as F


# ChannelGate类是一个自定义的通道注意力模块，它通过压缩空间维度来学习通道之间的依赖关系。
class ChannelGate(nn.Module):
    # 构造函数
    def __init__(self, channel, reduction=16):
        super().__init__()  # 调用父类nn.Module的初始化函数
        # 定义一个自适应平均池化层，将输入特征图的空间维度压缩到1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 定义一个多层感知机(MLP)，包含两个线性层和一个ReLU激活函数
        # 第一个线性层将通道数减少到原来的1/reduction，第二个线性层恢复到原来的通道数
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),  # 使用原地激活，节省存储空间
            nn.Linear(channel // reduction, channel)
        )
        # 定义一个批量归一化层，作用于通道维度
        self.bn = nn.BatchNorm1d(channel)

        # 前向传播函数

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入特征图的形状：批大小、通道数、高度、宽度
        # 使用自适应平均池化将特征图的空间维度压缩到1x1，然后展平为[b, c]的形状
        y = self.avgpool(x).view(b, c)
        # 通过MLP学习通道之间的依赖关系，并输出与输入相同通道数的权重向量
        y = self.mlp(y)
        # 对权重向量进行批量归一化，并重新整形为[b, c, 1, 1]，以便后续与输入特征图进行广播操作
        y = self.bn(y).view(b, c, 1, 1)
        # 使用expand_as方法将权重向量广播到与输入特征图相同的形状，并返回结果
        return y.expand_as(x)


import torch.nn as nn


# SpatialGate类是一个自定义的空间注意力模块，它通过卷积操作来学习空间依赖关系。
class SpatialGate(nn.Module):
    # 构造函数
    def __init__(self, channel, reduction=16, kernel_size=3, dilation_val=4):
        super().__init__()  # 调用父类nn.Module的初始化函数
        # 定义第一个卷积层，输入通道数和输出通道数都是channel // reduction
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        # 定义一个卷积序列，包含两个卷积层，批标准化和ReLU激活函数
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),  # 批标准化层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),  # 批标准化层
            nn.ReLU(inplace=True)  # ReLU激活函数
        )
        # 定义第三个卷积层，输出通道数为1
        self.conv3 = nn.Conv2d(channel // reduction, 1, kernel_size=1)
        # 定义批标准化层，作用于通道维度
        self.bn = nn.BatchNorm2d(1)

        # 前向传播函数

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入特征图的形状：批大小、通道数、高度、宽度
        y = self.conv1(x)  # 通过第一个卷积层
        y = self.conv2(y)  # 通过第二个卷积序列
        y = self.conv3(y)  # 通过第三个卷积层
        y = self.bn(y)  # 通过批标准化层
        return y.expand_as(x)  # 将输出形状扩展为与输入相同，并返回结果


# 定义BAM类，继承自nn.Module
class BAM(nn.Module):
    def __init__(self, channel):  # 初始化函数，接收通道数作为参数
        super(BAM, self).__init__()  # 调用父类nn.Module的初始化函数

        # 定义通道注意力模块
        self.channel_attn = ChannelGate(channel)

        # 定义空间注意力模块
        self.spatial_attn = SpatialGate(channel)

        # 前向传播函数

    def forward(self, x):  # 接收输入特征图x
        # 计算通道注意力和空间注意力的加权和，并应用sigmoid激活函数得到注意力权重
        attn = torch.sigmoid(self.channel_attn(x) + self.spatial_attn(x))

        # 将注意力权重与输入特征图相乘，并加上原始特征图，实现注意力机制的效果
        return x + x * attn
