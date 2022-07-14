import torch
from torch import nn
import consts


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.layer(input)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, output_dim):
        super(ResNet, self).__init__()

        layers = [2, 2, 2]

        self.inplanes = 64
        super(ResNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            self._make_layer(64, layers[0]),
            self._make_layer(128, layers[1], stride=2),
            self._make_layer(256, layers[2], stride=2),
            # self._make_layer(512, layers[3], stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(self.inplanes, output_dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.inplanes, consts.hidden_size),
        #     nn.LayerNorm(consts.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(consts.hidden_size, 3),
        # )

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, x, X2=None):
        if self.is_cuda():
            x = x.cuda()
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        dim = planes * BottleNeck.expansion
        if stride != 1 or self.inplanes != dim:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, dim,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(dim),
            )

        layers = [BottleNeck(self.inplanes, planes, stride, downsample)]
        self.inplanes = dim
        for i in range(1, blocks):
            layers.append(BottleNeck(self.inplanes, planes))

        return nn.Sequential(*layers)
