"""
Authors:
Nawras Abbas    315085043
Michael Bikman  317920999
This module contains our ResNet18 implementation
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2)
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class BasicBlockDownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlockDownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2)
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet18(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlockDownSample(64, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlockDownSample(128, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlockDownSample(256, 512),
            BasicBlock(512, 512)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):  # (200,3,84,84)
        x = self.conv1(x)  # (200,64,42,42)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)  # (200,64,21,21)

        x = self.layer1(x)  # out (200,64,21,21)
        x = self.layer2(x)  # out (200,128,11,11)
        x = self.layer3(x)  # out (200,256,6,6)
        x = self.layer4(x)  # out (200,512,3,3)
        x = self.classifier(x)  # out (200, 4608)

        return x


if __name__ == '__main__':
    # import torchvision.models as models
    # m = models.resnet18(False)
    # print(m)
    model = ResNet18()
    print(model)

    input_batch = torch.randn(200, 3, 84, 84)
    model.forward(input_batch)
