import torch
import torchvision
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


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        # c = [96, 192, 384, 1056, 1104]
        vgg=torchvision.models.vgg16(pretrained=True)
        self.rgb_features1=nn.Sequential(Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True))
        self.rgb_features2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True)),
        self.rgb_features3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True))
        self.rgb_features4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    ,Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    ,nn.ReLU(inplace=True)
    ,Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    ,nn.ReLU(inplace=True),
    Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True))
        self.rgb_features5 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    ,Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    ,nn.ReLU(inplace=True)
    ,Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    ,nn.ReLU(inplace=True)
    ,Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    ,nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

    def forward(self, x):
        rgb = x[:, :3]
        ir = torch.cat((x[:, 3:],x[:, 3:],x[:, 3:]),dim=1)

        rgb1 = self.rgb_features1(rgb)  # 1*256*120*160
        rgb2 = self.rgb_features2(rgb1)  # 1*512*60*80
        rgb3 = self.rgb_features3(rgb2)  # 1*1024*30*40
        rgb4 = self.rgb_features4(rgb3)  # 1*1024*15*20


        ir1 = self.ir_features1(ir)# 1*256*120*160

        fuse1=ir1+rgb1
        ir2 = self.ir_features2(fuse1)# 1*512*60*80
        fuse2=ir2+rgb2
        ir3 = self.ir_features3(fuse2)# 1*1024*30*40
        fuse3=ir3+rgb3
        ir4=self.ir_features4(fuse3)# 1*1024*15*20
        fuse4=ir4+rgb4



        x_after_brc1=self.brc1(fuse1)#1*128*120*160
        x_after_brc2 = self.brc2(fuse2)#1*128*60*80
        x_after_brc3 = self.brc3(fuse3)#1*128*30*40
        x_after_brc4 = self.brc4(fuse4)#1*128*15*20

        x_afterbrc_up4=F.interpolate(x_after_brc4,scale_factor=8,mode='bilinear')
        x_afterbrc_up3 = F.interpolate(x_after_brc3, scale_factor=4, mode='bilinear')
        x_afterbrc_up2 = F.interpolate(x_after_brc2, scale_factor=2, mode='bilinear')
        x_after_caspp=self.caspp(x_after_brc4)
        x_after_caspp_up=F.interpolate(x_after_caspp,scale_factor=8,mode='bilinear')

        F_concate=torch.cat((x_after_brc1,x_afterbrc_up2,x_afterbrc_up3,x_afterbrc_up4,x_after_caspp_up),dim=1)#1*640*120*160
        F_concate=self.samm(F_concate)#1*230*120*160
        F_concate=self.brc5(F_concate)
        F_concate=self.bn2(F_concate)
        F_enhance=self.dropout(F_concate)#1*160*120*160
        F_enhance=F.interpolate(F_enhance,scale_factor=2, mode='bilinear')








        x_semantic1=self.conv1_semantic(F_enhance)
        x_semantic2=self.conv2_semantic(x_semantic1)

        x_semantic_output_final=self.out_block2(x_semantic2)


        return x_semantic_output_final


def unit_test():
    num_minibatch = 1
    rgb = torch.randn(num_minibatch, 4, 480, 640)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = vgg16()
    input = rgb
    output, fuse = rtf_net(input)
    for i in fuse:
        print(0)
    print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()