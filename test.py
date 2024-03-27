from ultralytics import YOLO


# if __name__ == '__main__':
#     # Load a model
#     # model = YOLO('yolov8m.pt')  # load an official model
#     model = YOLO('runs/detect/train12/weights/best.pt')  # load a custom model
#     results = model.predict(source="ultralytics/assets", device='0',save=True)  # predict on an image
#     print(results)
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.bn = nn.BatchNorm1d(256)  # 初始化批归一化层

    def forward(self, x):
        if self.training and x.size(0) == 1:
            # 如果是训练且只有一个样本，则不应用批归一化
            return x.view(1, 256, 1, 1)
        else:
            x = self.bn(x)  # 对卷积层的输出应用批归一化
            x = x.view(1, 256, 1, 1)  # 展平特征图以输入全连接层
            return x

# 实例化网络并传入数据
net = SimpleNet()
input_data = torch.randn(1, 256)  # 假设有16张3通道、大小为28x28的图像
output_data = net(input_data)