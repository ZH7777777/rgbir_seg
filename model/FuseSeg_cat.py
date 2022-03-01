from numpy.lib.function_base import delete
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.nn.modules.linear import Identity
from torch.nn.modules.upsampling import Upsample
import torchvision
import matplotlib.pylab as plt
import numpy as np


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class UpSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampler, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.convt(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FuseSeg_cat(nn.Module):
    def __init__(self, n_class):
        super(FuseSeg_cat, self).__init__()
        c = [96, 192, 384, 1056, 1104]
        densenet = torchvision.models.densenet161(pretrained=True)
        self.encoder_conv0 = nn.Conv2d(4, 96, kernel_size=7, stride=2, padding=3, bias=False)
        self.rgb_features1 = nn.Sequential(
            self.encoder_conv0,
            densenet.features.norm0,
            densenet.features.relu0,
        )
        self.rgb_features2 = densenet.features.pool0
        self.rgb_features3 = nn.Sequential(
            densenet.features.denseblock1,
            densenet.features.transition1
        )
        self.rgb_features4 = nn.Sequential(
            densenet.features.denseblock2,
            densenet.features.transition2
        )
        self.rgb_features5 = nn.Sequential(
            densenet.features.denseblock3,
            densenet.features.transition3
        )
        self.rgb_features6 = nn.Sequential(
            densenet.features.denseblock4,
            _Transition(c[4] * 2, c[4])
        )

        del densenet

        self.feat_extractor5 = FeatureExtractor(c[3] * 2, c[3])
        self.feat_extractor4 = FeatureExtractor(c[2] * 2, c[2])
        self.feat_extractor3 = FeatureExtractor(c[1] * 2, c[1])
        self.feat_extractor2 = FeatureExtractor(c[0] * 2, c[0])
        self.feat_extractor1 = FeatureExtractor(c[0] * 2, c[0])

        self.upsampler5 = UpSampler(c[4], c[3])
        self.upsampler4 = UpSampler(c[3], c[2])
        self.upsampler3 = UpSampler(c[2], c[1])
        self.upsampler2 = UpSampler(c[1], c[0])
        self.upsampler1 = UpSampler(c[0], c[0])
        self.out_block = nn.ConvTranspose2d(c[0], n_class, kernel_size=2, stride=2)

    def forward(self, x):
        rgb = x[:, :3]
        ir = x[:, 3:]



        x1 = self.rgb_features1(x)
        x2 = self.rgb_features2(x1)
        x3 = self.rgb_features3(x2)
        x4 = self.rgb_features4(x3)
        x5 = self.rgb_features5(x4)
        x6 = self.rgb_features6(x5)

        x = self.upsampler5(x6)
        pad = nn.ConstantPad2d((0, 0, 1, 0), 0)
        x = pad(x)
        x = torch.cat((x, x5), dim=1)
        x = self.feat_extractor5(x)
        x = self.upsampler4(x)
        x = torch.cat((x, x4), dim=1)
        x = self.feat_extractor4(x)
        x = self.upsampler3(x)
        x = torch.cat((x, x3), dim=1)
        x = self.feat_extractor3(x)
        x = self.upsampler2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.feat_extractor2(x)
        x = self.upsampler1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.feat_extractor1(x)
        x = self.out_block(x)
        return x
