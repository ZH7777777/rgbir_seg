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





class backbonedensenetencoder(nn.Module):
    def __init__(self,dim=2,):
        super(backbonedensenetencoder, self).__init__()
        # c = [96, 192, 384, 1056, 1104]
        densenet = torchvision.models.densenet121(pretrained=True)
        self.encoder_conv0 = nn.Conv2d(dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # AAA=densenet.features.conv0.weight.data
        # A=torch.mean(densenet.features.conv0.weight.data, dim=1)
        # self.encoder_conv0.weight.data = torch.repeat_interleave(torch.mean(densenet.features.conv0.weight.data, dim=1),
        #                                                          dim=1,repeats=4)
        self.rgb_features1 = nn.Sequential(
            self.encoder_conv0,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
            densenet.features.denseblock1
        )
        self.rgb_features2 = nn.Sequential(

            densenet.features.transition1,
            densenet.features.denseblock2

        )
        self.rgb_features3 = nn.Sequential(

            densenet.features.transition2,
            densenet.features.denseblock3
        )
        self.rgb_features4 = nn.Sequential(

            densenet.features.transition3,
            densenet.features.denseblock4


        )



        del densenet


    def forward(self, x):
        f1=self.rgb_features1(x)
        f2=self.rgb_features2(f1)
        f3=self.rgb_features3(f2)
        f4=self.rgb_features4(f3)


        return f1,f2,f3,f4

def unit_test():
    num_minibatch = 1
    rgb = torch.randn(num_minibatch, 4, 480, 640)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = mffenet(9)
    input = rgb
    output, fuse = rtf_net(input)
    for i in fuse:
        print(0)
    print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()