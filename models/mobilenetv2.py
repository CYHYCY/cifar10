import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_planes, output_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, output_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(output_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != output_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, output_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(output_planes))

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        output = output + self.shortcut(x) if self.stride == 1 else output
        return output


class MobileNetV2(nn.Module):
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, output_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, output_planes, expansion, stride))
                in_planes = output_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.layers(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output


if __name__ == '__main__':
    net = MobileNetV2()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
