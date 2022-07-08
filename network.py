import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块结构
    """
    def __init__(self, input_channels, output_channels, conv_x1=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=strides),
            nn.BatchNorm2d(output_channels),
        )
        if conv_x1:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        else:
            self.conv3 = None

    def forward(self, x):
        #  第一次卷积
        out = self.conv1(x)
        # 第二次卷积
        out = self.conv2(out)

        if self.conv3:
            x = self.conv3(x)

        out = F.leaky_relu_(x + out, 0.1)
        return out


class FanNet(nn.Module):
    """基于残差块的卷积神经网络 run
    """
    def __init__(self):
        super(FanNet, self).__init__()

        # 卷积层1 input 1x28x28  output 16x12x12
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16, affine=True),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
        )
        # 卷积层2 input 16x12x12  output 32x4x4
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            # nn.BatchNorm2d(32, affine=True),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
        )

        # 全连接层
        self.fc1 = nn.Linear(32 * 4 * 4, 84)
        self.fc2 = nn.Linear(84, 10)

        self.dropout = nn.Dropout(0.25)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten层
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)

        return out
