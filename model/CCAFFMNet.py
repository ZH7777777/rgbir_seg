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


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # mip = max(8, inp // reduction)
        mip = inp//2

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # out = identity * a_w * a_h
        out = identity * a_w +identity* a_h

        return out
class catandconv(nn.Module):
    def __init__(self, featuere1_channel,feature2_channel,out_channel):
        super(catandconv, self).__init__()
        self.conv=nn.Conv2d(in_channels=featuere1_channel+feature2_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU6(inplace=True)

    def forward(self, featuer1,feature2):
        cat_feature=torch.cat((featuer1,feature2),dim=1)
        out_feature=self.conv(cat_feature)
        out_feature=self.bn(out_feature)
        out=self.relu(out_feature)
        return out
class catandupconv4three(nn.Module):
    def __init__(self, featuere1_channel,feature2_channel,feature3_channel,out_channel):
        super(catandupconv4three, self).__init__()
        self.conv=nn.Conv2d(in_channels=featuere1_channel+feature2_channel+feature3_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU6(inplace=True)
        self.up=nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, featuer1,feature2,feature3):
        cat_feature=torch.cat((featuer1,feature2,feature3),dim=1)
        upfeature=self.up(cat_feature)
        out_feature=self.conv(upfeature)
        out_feature=self.bn(out_feature)
        out=self.relu(out_feature)
        return out
class bifpndecoder(nn.Module):
    def __init__(self,inchannel0,inchannel1,inchannel2,inchannel3,inchannel4):
        super(bifpndecoder, self).__init__()
        self.down_0 = nn.Conv2d(in_channels=inchannel0,out_channels=inchannel0,kernel_size=3,stride=2,padding=1)
        self.down_1 = nn.Conv2d(in_channels=inchannel1,out_channels=inchannel1,kernel_size=3,stride=2,padding=1)
        self.down_2 =nn.Conv2d(in_channels=inchannel2,out_channels=inchannel2,kernel_size=3,stride=2,padding=1)
        self.down_3 = nn.Conv2d(in_channels=inchannel3,out_channels=inchannel3,kernel_size=3,stride=2,padding=1)
        self.catandconv10=catandconv(64,256,64)
        self.catandconv20 = catandconv(64, 256, 256)
        self.catandconv21 = catandupconv4three(256, 256,512, 256)
        self.catandconv30 = catandconv(256, 512, 512)
        self.catandconv31 = catandupconv4three(512,512,1024, 512)
        self.catandconv40 = catandconv(512, 1024, 1024)
        self.catandconv41 = catandupconv4three(1024,1024,2048, 1024)
        self.catandconv50 = catandconv(1024, 2048, 2048)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv=nn.Conv2d(in_planes*2, in_planes , 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.up50=nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, fusion0,fusion1,fusion2,fusion3,fusion4):
        down_feature0=self.down_0(fusion0)
        x20 = self.catandconv20(down_feature0, fusion1)
        down_feature1 = self.down_1(x20)
        x30 = self.catandconv30(down_feature1, fusion2)
        down_feature2 = self.down_2(x30)
        x40 = self.catandconv40(down_feature2, fusion3)
        down_feature3 = self.down_3(x40)
        x50 = self.catandconv50(down_feature3, fusion4)
        x50_up=self.up50(x50)


        x41 = self.catandconv41(x40,x50_up, fusion3)
        x31 = self.catandconv31(x30,x41, fusion2)
        x21 = self.catandconv21(x20,x31,fusion1)
        x10 = self.catandconv10(x21, fusion0)


        return x10
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv=nn.Conv2d(in_planes*2, in_planes , 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat((avg_out,max_out),dim=1)
        out=self.conv(out)
        out2=self.sigmoid(out)

        return out2*x
class ccdsab(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ccdsab, self).__init__()
        self.channelpath=ChannelAttention(in_channels)
        self.corordinate=CoordAtt(in_channels,in_channels)
        self.conv=nn.Conv2d(in_channels*2,in_channels,kernel_size=1,padding=0)



    def forward(self, feature):
        feature_channel=self.channelpath(feature)
        feature_coor=self.corordinate(feature)
        feature_out=torch.cat((feature_coor,feature_channel),dim=1)
        out=self.conv(feature_out)





        return out
class ccaff(nn.Module):
    def __init__(self, in_channels):
        super(ccaff, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.ccdsab=ccdsab(in_channels,in_channels)



    def forward(self, rgb,ir):
        catfeature=torch.cat((rgb,ir),dim=1)
        b,c,h,w=catfeature.size()
        mid_feature=self.conv1(catfeature)
        out_feature=self.ccdsab(mid_feature)
        out=mid_feature+out_feature

        return out


class ccaffmnet(nn.Module):
    def __init__(self, n_class):
        super(ccaffmnet, self).__init__()

        resnext = torchvision.models.resnext50_32x4d(pretrained=True)
        # self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnext.conv1.weight.data, dim=1),
        #                                                          dim=1)
        self.ir_stage0 = nn.Sequential(
            # self.encoder_thermal_conv1,
            resnext.conv1,
            resnext.bn1,
            resnext.relu,

        )

        self.ir_stage1 = nn.Sequential(
            resnext.maxpool,
            resnext.layer1

        )
        self.ir_stage1 = nn.Sequential(
            resnext.maxpool,
            resnext.layer1

        )
        self.ir_stage2 = resnext.layer2
        self.ir_stage3 = resnext.layer3
        self.ir_stage4 = resnext.layer4

        del resnext
        resnext = torchvision.models.resnext50_32x4d(pretrained=True)

        self.rgb_stage0 = nn.Sequential(
            resnext.conv1,
            resnext.relu,
            resnext.bn1,


        )
        self.rgb_stage1 = nn.Sequential(
            resnext.maxpool,
            resnext.layer1

        )

        self.rgb_stage2 = resnext.layer2
        self.rgb_stage3 = resnext.layer3
        self.rgb_stage4 = resnext.layer4
        del resnext
        self.ccaffm0=ccaff(in_channels=64)
        self.ccaffm1 = ccaff(in_channels=256)
        self.ccaffm2 = ccaff(in_channels=512)
        self.ccaffm3 = ccaff(in_channels=1024)
        self.ccaffm4 = ccaff(in_channels=2048)

        self.decoder=bifpndecoder(64,256,512,1024,2048)
        self.upfeature=nn.Upsample(scale_factor=2, mode='bilinear')
        self.outblock=nn.ConvTranspose2d(64,9,kernel_size=2,stride=2)


    def forward(self, x):
        rgb = x[:, :3]
        ir = torch.cat((x[:, 3:],x[:, 3:],x[:, 3:]),dim=1)

        ir0=self.ir_stage0(ir)#64*240*320
        ir1 = self.ir_stage1(ir0)#256*120*160
        ir2 = self.ir_stage2(ir1)#512*60*80
        ir3 = self.ir_stage3(ir2)#1024*30*40
        ir4 = self.ir_stage4(ir3)#2048*15*20

        rgb0 = self.rgb_stage0(rgb)
        fusion0=self.ccaffm0(rgb0,ir0)
        rgb1 = self.rgb_stage1(fusion0)
        fusion1 = self.ccaffm1(rgb1, ir1)
        rgb2 = self.rgb_stage2(fusion1)
        fusion2 = self.ccaffm2(rgb2, ir2)
        rgb3 = self.rgb_stage3(fusion2)
        fusion3 = self.ccaffm3(rgb3, ir3)
        rgb4 = self.rgb_stage4(fusion3)
        fusion4 = self.ccaffm4(rgb4, ir4)

        out=self.decoder(fusion0,fusion1,fusion2,fusion3,fusion4)
        # out_up=self.upfeature(out)
        semantic_out=self.outblock(out)

        # rgb0=self.rgb_stage0(rgb)
        # fusion=self.





        return semantic_out


def unit_test():
    num_minibatch = 1
    rgb = torch.randn(num_minibatch, 4, 480, 640)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = ccaffmnet(9)
    input = rgb
    output, fuse = rtf_net(input)
    for i in fuse:
        print(0)
    print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()