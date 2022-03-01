import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.nn.modules.linear import Identity
import torchvision
import matplotlib.pylab as plt
import numpy as np
from torchvision.models import inception
from torchvision.models.resnet import Bottleneck, BasicBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """下·
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """
    #512 512 REDUCTION=64

    def __init__(self, in_dim, out_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(reduction_dim*5, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        size = x_size[2:]
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, size=x_size[2:], mode='bilinear', align_corners=True)
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        out = self.conv_out(out)
        return out

class Resdiv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resdiv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, bias=False, groups=2),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        _, c, _, _ = x.size()
        identity = x
        x = self.conv1(x)
        x = self.conv3(x)
        x = F.relu(x + identity, inplace=True)

        x = self.conv4(x)
        x = self.upconv1(x)
        x = self.conv6(x)
        return x

class FuNNet(nn.Module):
    def __init__(self, n_class):
        super(FuNNet, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        self.rgb_features1 = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu, resnet.maxpool,
                                           resnet.layer1)
        self.rgb_features2 = resnet.layer2
        self.rgb_features3 = resnet.layer3
        self.rgb_features4 = resnet.layer4

        del resnet
        resnet = torchvision.models.resnet34(pretrained=True)
        conv1 = nn.Conv2d(1,
                          64,
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False)
        conv1.weight.data = torch.unsqueeze(torch.mean(
            resnet.conv1.weight.data, dim=1),
                                            dim=1)
        self.ir_features1 = nn.Sequential(
            conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        self.ir_features2 = resnet.layer2
        self.ir_features3 = resnet.layer3
        self.ir_features4 = resnet.layer4
        del resnet

        self.aspp_rgb = _AtrousSpatialPyramidPoolingModule(512, 512, reduction_dim=64)
        self.aspp_ir = _AtrousSpatialPyramidPoolingModule(512, 512, reduction_dim=64)

        self.resdiv5 = Resdiv(1024, 512)
        self.resdiv4 = Resdiv(512, 256)
        self.resdiv3 = Resdiv(256, 128)
        self.resdiv2 = Resdiv(128, 64)
        self.resdiv1 = Resdiv(64, n_class)

    def forward(self, x):
        rgb = x[:, :3]
        ir = x[:, 3:]
        rgb1 = self.rgb_features1(rgb)
        rgb2 = self.rgb_features2(rgb1)
        rgb3 = self.rgb_features3(rgb2)
        rgb4 = self.rgb_features4(rgb3)
        rgb4 = self.aspp_rgb(rgb4)

        ir1 = self.ir_features1(ir)
        ir2 = self.ir_features2(ir1)
        ir3 = self.ir_features3(ir2)
        ir4 = self.ir_features4(ir3)
        ir4 = self.aspp_ir(ir4)

        x = self.resdiv5(torch.cat((rgb4, ir4), dim=1))
        x = self.resdiv4(x + torch.cat((rgb3, ir3), dim=1))
        x = self.resdiv3(x + torch.cat((rgb2, ir2), dim=1))
        x = self.resdiv2(x + torch.cat((rgb1, ir1), dim=1))
        x = self.resdiv1(x)
        # return x, []
        return x
def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = FuNNet(n_class=9)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,9,480,640), 'output shape (2,9,480,640) is expected!'
    print('test ok!')


if __name__ == '__main__':
    unit_test()