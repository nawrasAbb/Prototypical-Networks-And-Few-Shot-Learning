"""
Authors:
Nawras Abbas    315085043
Michael Bikman  317920999

Insprired by the implementation from:
https://github.com/kjunelee/MetaOptNet/tree/7a8e2ae25ef47cfe75a6fe8bc7920dc9fd29191f

This ResNet network was designed following the practice of the following papers:
'TADAM: Task dependent adaptive metric for improved few-shot learning' (Oreshkin et al., in NIPS 2018),
'A Simple Neural Attentive Meta-Learner' (Mishra et al., in ICLR 2018).
"""
import torch
from torch import nn
from torch.distributions import Bernoulli
import torch.nn.functional as F

from utils import DEVICE


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).to(DEVICE)
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().to(DEVICE)
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).to(DEVICE).long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, down_sample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.max_pool = nn.MaxPool2d(stride)
        self.down_sample = down_sample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        out = self.max_pool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * self.num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)

            # else:
            #     out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, drop_rate, drop_block_size=5):
        self.in_planes = 3
        super(ResNet, self).__init__()

        # --- choose feature map configuration here --- #
        filters = [64, 64, 64, 64]
        # filters = [32,64,128,256]
        # filters = [64,160,320,640]
        # filters = [64, 128, 128, 256]
        # --------------------------------------------- #
        self.layer1 = self._make_layer(block, filters[0], stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, filters[1], stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, filters[2], stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=drop_block_size)
        self.layer4 = self._make_layer(block, filters[3], stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=drop_block_size)

        self.drop_rate = drop_rate
        self.flatten = nn.Sequential(
            nn.Flatten(start_dim=1)
        )
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.in_planes, planes, stride, down_sample, drop_rate, drop_block, block_size)]
        self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avg_pool(x)
        x = self.flatten(x)
        return x


def resnet12(drop_rate):
    """
    Constructs a ResNet-12 model
    """
    return ResNet(BasicBlock, drop_rate)


if __name__ == '__main__':
    model = resnet12(drop_rate=0.1)
    print(model)
