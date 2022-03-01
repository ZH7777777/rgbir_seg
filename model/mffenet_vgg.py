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
from .vgg16_c import VGG16_C


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class BRC(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1):
        super(BRC, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        # self.relu=F.relu(inplace=True)
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=1,padding=padding)

    def forward(self, x):
        x = self.bn(x)
        x=F.relu(x,inplace=True)
        x=self.conv(x)
        return x
class BRC_caspp(nn.Module):
    def __init__(self, in_channels, out_channels,dilation=None):
        super(BRC_caspp, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        # self.relu=F.relu(inplace=True)
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=dilation,dilation=dilation)

    def forward(self, x):
        x = self.bn(x)
        x=F.relu(x,inplace=True)
        x=self.conv(x)
        return x
class caspp(nn.Module):
    def __init__(self):
        super(caspp, self).__init__()
        self.brc1=BRC_caspp(128,128,dilation=2)
        self.brc2 = BRC_caspp(128, 128, dilation=4)
        self.brc3 = BRC_caspp(128, 128, dilation=6)
        self.brc=BRC(3*128,128,kernel_size=1,padding=0)





    def forward(self, x):
        x1=self.brc1(x)
        x2=self.brc2(x)
        x3=self.brc3(x)
        x_all=torch.cat((x1,x2,x3),dim=1)
        x_out=self.brc(x_all)
        return x_out
class samm(nn.Module):
    def __init__(self):
        super(samm, self).__init__()

        self.brc1=BRC(640,230,kernel_size=1,padding=0)
        self.brc2 = BRC(640, 160, kernel_size=1, padding=0)
        self.brc3 = BRC(160 ,230, kernel_size=1, padding=0)






    def forward(self, x):
        x1=self.brc1(x)
        x2=self.brc2(x)
        x3=self.brc3(x2)
        # x_all=torch.cat((x1,x2,x3),dim=1)
        x4=F.sigmoid(x3)
        x_out=x4*x1
        return x_out

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


class mffenet_vgg(nn.Module):
    def __init__(self, n_class):
        super(mffenet_vgg, self).__init__()
        # c = [96, 192, 384, 1056, 1104]
        densenet = torchvision.models.densenet121(pretrained=True)
        self.encoder_conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

        self.brc1=BRC(256,128)
        self.brc2 = BRC(512,128)
        self.brc3 = BRC(1024,128)
        self.brc4 = BRC(1024,128)
        self.brc5 = BRC(230, 160)
        self.caspp=caspp()
        self.samm=samm()
        self.bn2=nn.BatchNorm2d(160)
        self.dropout=nn.Dropout2d(p=0.2)




        # self.upsampler5 = UpSampler(c[4], c[3])
        # self.upsampler4 = UpSampler(c[3], c[2])
        # self.upsampler3 = UpSampler(c[2], c[1])
        # self.upsampler2 = UpSampler(c[1], c[0])
        # self.upsampler1 = UpSampler(c[0], c[0])
        self.conv1_salient = nn.Conv2d(256,
                                       256,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.conv2_salient = nn.Conv2d(256,
                                       256,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.conv1_semantic = nn.Conv2d(160,160,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.conv2_semantic = nn.Conv2d(160,160,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.conv1_boundary = nn.Conv2d(512,256,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.conv2_boundary = nn.Conv2d(256,128,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(256)
        # self.batchnorm2 = nn.BatchNorm2d(3 * c[0])
        self.out_block1 = nn.ConvTranspose2d(230,230, kernel_size=2, stride=2)
        self.out_block2 = nn.ConvTranspose2d(160, n_class, kernel_size=2, stride=2)
        self.out_block_reconstruct = nn.ConvTranspose2d(160, 9, kernel_size=2, stride=2)
        self.out_block_salient1 = nn.ConvTranspose2d(256,128, kernel_size=2, stride=2)
        self.out_block_salient2 = nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2)
        self.out_block_boundary1 = nn.ConvTranspose2d(256,256, kernel_size=2, stride=2)
        self.out_block_boundary2 = nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2)
        # self.rgb_vgg=VGG16_C(pretrain='E:\google drive\MFNet-pytorch-master\weights/vgg16.npy')
        # self.ir_vgg=VGG16_C(pretrain='E:\google drive\MFNet-pytorch-master\weights/vgg16.npy')

    def forward(self, x):
        # print(x.shape)



        rgb1 = self.rgb_features1(x)  # 1*256*120*160
        rgb2 = self.rgb_features2(rgb1)  # 1*512*60*80
        rgb3 = self.rgb_features3(rgb2)  # 1*1024*30*40
        rgb4 = self.rgb_features4(rgb3)  # 1*1024*15*20






        x_after_brc1=self.brc1(rgb1)#1*128*120*160
        x_after_brc2 = self.brc2(rgb2)#1*128*60*80
        x_after_brc3 = self.brc3(rgb3)#1*128*30*40
        x_after_brc4 = self.brc4(rgb4)#1*128*15*20

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

        # x_semantic_output_final=self.out_block2(x_semantic2)
        reconstruct=self.out_block_reconstruct(x_semantic2)
        # output=torch.tanh(reconstruct)/2+0.5
        # output = reconstruct


        # return x_semantic_output_final,reconstruct
        # print(reconstruct.shape)
        return reconstruct


def unit_test():
    num_minibatch = 1
    rgb = torch.randn(num_minibatch, 4, 288, 384)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = mffenet_vgg(9)
    input = rgb
    output, reconstruct,rgb_vgg_feature,ir_vgg_feature = rtf_net(input)
    # for i in fuse:
    #     print(0)
    print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()