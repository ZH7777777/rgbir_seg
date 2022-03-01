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
# from util.bn import InPlaceABNSync as BatchNorm2d
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        # self.bn = BatchNorm2d(out_chan, activation='none')
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            # elif isinstance(module, BatchNorm2d):
            #     nowd_params += list(module.parameters())
        return wd_params, nowd_params
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

class FuseSeg_edge(nn.Module):
    def __init__(self, n_class):
        super(FuseSeg_edge, self).__init__()
        c = [96, 192, 384, 1056, 1104]
        densenet = torchvision.models.densenet161(pretrained=True)
        self.rgb_features1 = nn.Sequential(
            densenet.features.conv0,
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
            _Transition(c[4]*2, c[4])
        )

        del densenet
        densenet = torchvision.models.densenet161(pretrained=True)
        conv1 = nn.Conv2d(1,
                          c[0],
                          kernel_size=7,
                          stride=2,
                          padding=3,
                          bias=False)
        conv1.weight.data = torch.unsqueeze(torch.mean(
            densenet.features.conv0.weight.data, dim=1), dim=1)
        self.ir_features1 = nn.Sequential(
            conv1,
            densenet.features.norm0,
            densenet.features.relu0,
        )
        self.ir_features2 = densenet.features.pool0
        self.ir_features3 = nn.Sequential(
            densenet.features.denseblock1,
            densenet.features.transition1
        )
        self.ir_features4 = nn.Sequential(
            densenet.features.denseblock2,
            densenet.features.transition2
        )
        self.ir_features5 = nn.Sequential(
            densenet.features.denseblock3,
            densenet.features.transition3
        )
        self.ir_features6 = nn.Sequential(
            densenet.features.denseblock4,
            _Transition(c[4]*2, c[4])
        )
        del densenet

        self.feat_extractor5 = FeatureExtractor(c[3]*2, c[3])
        self.feat_extractor4 = FeatureExtractor(c[2]*2, c[2])
        self.feat_extractor3 = FeatureExtractor(c[1]*2, c[1])
        self.feat_extractor2 = FeatureExtractor(c[0]*2, c[0])
        self.feat_extractor1 = FeatureExtractor(c[0]*2, c[0])

        self.upsampler5 = UpSampler(c[4], c[3])
        self.upsampler4 = UpSampler(c[3], c[2])
        self.upsampler3 = UpSampler(c[2], c[1])
        self.upsampler2 = UpSampler(c[1], c[0])
        self.upsampler1 = UpSampler(c[0], c[0])
        self.out_block = nn.ConvTranspose2d(c[0], n_class, kernel_size=2, stride=2)
        self.edgeoutput1=BiSeNetOutput(96,64,1)
        self.edgeoutput2 = BiSeNetOutput(96, 64, 1)
        self.edgeoutput3 = BiSeNetOutput(192, 64, 1)
        self.edgeoutput4 = BiSeNetOutput(384, 64, 1)
        self.edgeoutput5 = BiSeNetOutput(1056, 64, 1)
        self.edgeoutput6 = BiSeNetOutput(1104, 64, 1)
        self.edgeoutput7 = BiSeNetOutput(96, 64, 1)
        self.edgeoutput8 = BiSeNetOutput(96, 64, 1)
        self.edgeoutput9 = BiSeNetOutput(192, 64, 1)
        self.edgeoutput10 = BiSeNetOutput(384, 64, 1)
        self.edgeoutput11 = BiSeNetOutput(1056, 64, 1)
        self.edgeoutput12 = BiSeNetOutput(1104, 64, 1)




    def forward(self, x):
        rgb = x[:, :3]
        ir = x[:, 3:]

        ir1 = self.ir_features1(ir)#1*96*240*320
        e1=self.edgeoutput1(ir1)
        ir2 = self.ir_features2(ir1)#1*96*120*160
        e2 = self.edgeoutput2(ir2)
        ir3 = self.ir_features3(ir2)#1*192*60*80
        e3 = self.edgeoutput3(ir3)
        ir4 = self.ir_features4(ir3)#1*384*30*40
        e4 = self.edgeoutput4(ir4)
        ir5 = self.ir_features5(ir4)#1*1056*15*20
        e5 = self.edgeoutput5(ir5)
        ir6 = self.ir_features6(ir5)#1*1104*7*10
        e6= self.edgeoutput6(ir6)

        x1 = self.rgb_features1(rgb) + ir1
        e7 = self.edgeoutput7(x1)
        x2 = self.rgb_features2(x1) + ir2
        e8 = self.edgeoutput8(x2)
        x3 = self.rgb_features3(x2) + ir3
        e9 = self.edgeoutput9(x3)
        x4 = self.rgb_features4(x3) + ir4
        e10 = self.edgeoutput10(x4)
        x5 = self.rgb_features5(x4) + ir5
        e11 = self.edgeoutput11(x5)
        x6 = self.rgb_features6(x5) + ir6
        e12 = self.edgeoutput12(x6)

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
        return x, [e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12]
def unit_test():
    num_minibatch = 1
    rgb = torch.randn(num_minibatch, 4, 480, 640)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = FuseSeg_edge(9)
    input = rgb
    output, fuse = rtf_net(input)
    for i in fuse:
        print(0)
    print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()