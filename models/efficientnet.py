import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(input):
    return input * input.sigmoid()


def drop_connect(input, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([input.shape[0], 1, 1, 1], dtype=input.dtype, device=input.device)
    mask.bernoulli_(keep_ratio)
    input.div_(keep_ratio)
    input.mul_(mask)
    return input


class SE(nn.Module):
    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, input):
        output = F.adaptive_avg_pool2d(input, (1, 1))
        output = swish(self.se1(output))
        output = self.se2(output).sigmoid()
        output = input * output
        return output


class Block(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, stride, expand_ratio=1, se_ratio=0., drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                               padding=(1 if kernel_size == 3 else 2), groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        self.conv3 = nn.Conv2d(channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)

        self.has_skip = (stride == 1) and (in_channels == output_channels)

    def forward(self, input):
        output = input if self.expand_ratio == 1 else swish(self.bn1(self.conv1(input)))
        output = swish(self.bn2(self.conv2(output)))
        output = self.se(output)
        output = self.bn3(self.conv3(output))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                output = drop_connect(output, self.drop_rate)
            output = output + input
        return output


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg['output_channels'][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'output_channels', 'num_blocks', 'kernel_size', 'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, output_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_rate'] * b / blocks
                layers.append(
                    Block(in_channels, output_channels, kernel_size, stride, expansion, se_ratio=0.25, drop_rate=drop_rate))
                in_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, input):
        output = swish(self.bn1(self.conv1(input)))
        output = self.layers(output)
        output = F.adaptive_avg_pool2d(output, 1)
        output = output.view(output.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            output = F.dropout(output, p=dropout_rate)
        output = self.linear(output)
        return output


def EfficientNetB0():
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'output_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_rate': 0.2,
    }
    return EfficientNet(cfg)


if __name__ == '__main__':
    net = EfficientNetB0()
    input = torch.randn(2, 3, 32, 32)
    y = net(input)
    print(y.shape)
