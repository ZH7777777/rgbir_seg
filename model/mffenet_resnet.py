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
import torchvision.models as models
# def TSE(x, kernel, se_ratio):
#     # x: input feature map [N, C, H, W]
#     # kernel: tile size (Kh, Kw)
#     # se_ratio: SE channel reduction ratio
#     N, C, H, W = x.size()
#     # tiled squeeze
#     sq = nn.AvgPool2d(kernel, stride=kernel, ceil_mode=True)
#     # original se excitation
#     ex = nn.Sequential(
#     nn.Conv2d(C, C // se_ratio, 1),
#     nn.ReLU(inplace=True),
#     nn.Conv2d(C // se_ratio, C, 1),
#     nn.Sigmoid()
#     )
#     y = ex(sq(x))
#     # nearest neighbor interpolation
#     y = torch.repeat_interleave(y, kernel[0], dim=-2)[:,:,:H,:]
#     y = torch.repeat_interleave(y, kernel[1], dim=-1)[:,:,:,:W]
#     return x * y
class TSE_edge(nn.Module):
    def __init__(self,N, C, H, W, kernel, se_ratio):
        super(TSE_edge, self).__init__()
        self.N=N;
        self.C=C;
        self.H = H;
        self.W = W;
        self.se_ratio=se_ratio

        self.kernel=kernel
        self.sq=nn.AvgPool2d(self.kernel, stride=self.kernel, ceil_mode=True)
        self.ex = nn.Sequential(
            nn.Conv2d(self.C, self.C // self.se_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C // self.se_ratio, self.C, 1),
            nn.Sigmoid()
        )


    def forward(self, x,semantic):
        y = self.ex(self.sq(x))
        # nearest neighbor interpolation
        y = torch.repeat_interleave(y, self.kernel[0], dim=-2)[:, :, :self.H, :]
        y = torch.repeat_interleave(y, self.kernel[1], dim=-1)[:, :, :, :self.W]

        return semantic*y
class TSE(nn.Module):
    def __init__(self,N, C, H, W, kernel, se_ratio):
        super(TSE, self).__init__()
        self.N=N;
        self.C=C;
        self.H = H;
        self.W = W;
        self.se_ratio=se_ratio

        self.kernel=kernel
        self.sq=nn.AvgPool2d(self.kernel, stride=self.kernel, ceil_mode=True)
        self.ex = nn.Sequential(
            nn.Conv2d(self.C, self.C // self.se_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C // self.se_ratio, self.C, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.ex(self.sq(x))
        # nearest neighbor interpolation
        y = torch.repeat_interleave(y, self.kernel[0], dim=-2)[:, :, :self.H, :]
        y = torch.repeat_interleave(y, self.kernel[1], dim=-1)[:, :, :, :self.W]

        return x*y

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
        # self.brc1=BRC(768,230,kernel_size=1,padding=0)
        # self.brc2 = BRC(768, 160, kernel_size=1, padding=0)
        # self.brc3 = BRC(160 ,230, kernel_size=1, padding=0)






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

# class Edge(nn.Module):
#     def __init__(self, channels):
#         super(Edge, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, 1, kernel_size=1, bias=True),
#             nn.Sigmoid()
#         )
#         self.recon = nn.Sequential(
#             nn.Conv2d(1, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#         )
#         self.final = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )
#         # v = [[1, 0, -1],
#         #      [2, 0, -2],
#         #      [1, 0, -1]]
#         # h = [[1, 2, 1],
#         #      [0, 0, 0],
#         #      [-1, -2, -1]]
#         # v = torch.FloatTensor(v).unsqueeze(0).unsqueeze(0)
#         # h = torch.FloatTensor(h).unsqueeze(0).unsqueeze(0)
#         # self.v = nn.Parameter(v, requires_grad=False)
#         # self.h = nn.Parameter(h, requires_grad=False)
#
#         # lap = [[-1, -1, -1],
#         #        [-1, 8, -1],
#         #        [-1, -1, -1]]
#         # lap = torch.FloatTensor(lap).unsqueeze(0).unsqueeze(0)
#         # self.lap = nn.Parameter(lap, requires_grad=False)
#
#     def forward(self, x):
#         edge_out = self.conv(x)
#         out=x+x*edge_out
#         # v = F.conv2d(edge_out, self.v, padding=1)
#         # h = F.conv2d(edge_out, self.h, padding=1)
#         # edge_out = torch.sqrt(torch.pow(v, 2) + torch.pow(h, 2) + 1e-6)
#         # x = self.final(torch.cat((x, self.recon(edge_out)), dim=1))
#
#         # lap = F.conv2d(edge_out, self.lap, padding=1)
#         # edge_out = torch.abs(lap)
#         # edge_feat = self.recon(edge_out)
#
#         return edge_out, out
class Edge(nn.Module):
    def __init__(self, channels):
        super(Edge, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            # nn.Sigmoid()
        )
        self.recon = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # self.tse=TSE_edge()
        # v = [[1, 0, -1],
        #      [2, 0, -2],
        #      [1, 0, -1]]
        # h = [[1, 2, 1],
        #      [0, 0, 0],
        #      [-1, -2, -1]]
        # v = torch.FloatTensor(v).unsqueeze(0).unsqueeze(0)
        # h = torch.FloatTensor(h).unsqueeze(0).unsqueeze(0)
        # self.v = nn.Parameter(v, requires_grad=False)
        # self.h = nn.Parameter(h, requires_grad=False)

        # lap = [[-1, -1, -1],
        #        [-1, 8, -1],
        #        [-1, -1, -1]]
        # lap = torch.FloatTensor(lap).unsqueeze(0).unsqueeze(0)
        # self.lap = nn.Parameter(lap, requires_grad=False)

    def forward(self, x,tse_edge):
        edge_feature = self.conv(x)
        edge_tse=tse_edge(edge_feature,x)
        edge_out=self.final(edge_tse)

        out=x+x*edge_out
        # v = F.conv2d(edge_out, self.v, padding=1)
        # h = F.conv2d(edge_out, self.h, padding=1)
        # edge_out = torch.sqrt(torch.pow(v, 2) + torch.pow(h, 2) + 1e-6)
        # x = self.final(torch.cat((x, self.recon(edge_out)), dim=1))

        # lap = F.conv2d(edge_out, self.lap, padding=1)
        # edge_out = torch.abs(lap)
        # edge_feat = self.recon(edge_out)

        return edge_out, out


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
class mffenet_resnet(nn.Module):
    def __init__(self, n_class):
        super(mffenet_resnet, self).__init__()
        # c = [96, 192, 384, 1056, 1104]
        resnet_raw_model1 = models.resnet101(pretrained=True)
        resnet_raw_model2 = models.resnet101(pretrained=True)
        densenet = torchvision.models.densenet121(pretrained=True)
        self.rgb_features1 = nn.Sequential(
            densenet.features.conv0,
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
        densenet = torchvision.models.densenet121(pretrained=True)

        self.ir_features1 = nn.Sequential(
            densenet.features.conv0,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
            densenet.features.denseblock1
        )
        self.ir_features2 = nn.Sequential(

            densenet.features.transition1,
            densenet.features.denseblock2

        )
        self.ir_features3 = nn.Sequential(

            densenet.features.transition2,
            densenet.features.denseblock3
        )
        self.ir_features4 = nn.Sequential(

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
        self.out_block_salient1 = nn.ConvTranspose2d(256,128, kernel_size=2, stride=2)
        self.out_block_salient2 = nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2)
        self.out_block_boundary1 = nn.ConvTranspose2d(256,256, kernel_size=2, stride=2)
        self.out_block_boundary2 = nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2)
        self.edge1=Edge(256)
        self.edge2 = Edge(512)
        self.edge3 = Edge(1024)
        self.edge4 = Edge(1024)
        self.feat_extractor=FeatureExtractor(416,256)
        self.upsample=UpSampler(256,160)
        self.gap1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.tse=TSE(N=4,C=230,H=120,W=160,kernel=[6,8],se_ratio=5)
        self.tse_edge1=TSE_edge(N=4,C=256,H=120,W=160,kernel=[6,8],se_ratio=2)
        self.tse_edge2 = TSE_edge(N=4, C=512, H=60, W=80, kernel=[3, 4], se_ratio=4)
        self.tse_edge3 = TSE_edge(N=4, C=1024, H=30, W=40, kernel=[3, 4], se_ratio=8)
        self.tse_edge4 = TSE_edge(N=4, C=1024, H=15, W=20, kernel=[3, 4], se_ratio=8)

        # self.tse=TSE()

    # def tse(x, kernel, se_ratio):
    #
    #     # x: input feature map [N, C, H, W]
    #     # kernel: tile size (Kh, Kw)
    #     # se_ratio: SE channel reduction ratio
    #     N, C, H, W = x.size()
    #     # tiled squeeze
    #     sq = nn.AvgPool2d(kernel, stride=kernel, ceil_mode=True)
    #     # original se excitation
    #     ex = nn.Sequential(
    #         nn.Conv2d(C, C // se_ratio, 1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(C // se_ratio, C, 1),
    #         nn.Sigmoid()
    #     )
    #     y = ex(sq(x))
    #     # nearest neighbor interpolation
    #     y = torch.repeat_interleave(y, kernel[0], dim=-2)[:, :, :H, :]
    #     y = torch.repeat_interleave(y, kernel[1], dim=-1)[:, :, :, :W]
    #     return x * y
    def forward(self, x):
        rgb = x[:, :3]
        ir = torch.cat((x[:, 3:],x[:, 3:],x[:, 3:]),dim=1)


        rgb1 = self.rgb_features1(rgb)  # 1*256*120*160
        rgb2 = self.rgb_features2(rgb1)  # 1*512*60*80
        rgb3 = self.rgb_features3(rgb2)  # 1*1024*30*40
        rgb4 = self.rgb_features4(rgb3)  # 1*1024*15*20



        ir1 = self.ir_features1(ir)# 1*256*120*160

        fuse1=ir1+rgb1
        e1,fuse_edge=self.edge1(fuse1,self.tse_edge1)

        ir2 = self.ir_features2(fuse_edge)# 1*512*60*80
        fuse2=ir2+rgb2
        e2, fuse_edge2 = self.edge2(fuse2,self.tse_edge2)
        ir3 = self.ir_features3(fuse_edge2)# 1*1024*30*40
        fuse3=ir3+rgb3
        e3, fuse_edge3 = self.edge3(fuse3,self.tse_edge3)
        ir4=self.ir_features4(fuse_edge3)# 1*1024*15*20
        fuse4=ir4+rgb4
        e4, fuse_edge4 = self.edge4(fuse4,self.tse_edge4)



        x_after_brc1=self.brc1(fuse1)#1*128*120*160
        x_after_brc2 = self.brc2(fuse2)#1*128*60*80
        x_after_brc3 = self.brc3(fuse3)#1*128*30*40
        x_after_brc4 = self.brc4(fuse_edge4)#1*128*15*20

        x_afterbrc_up4=F.interpolate(x_after_brc4,scale_factor=8,mode='bilinear')
        x_afterbrc_up3 = F.interpolate(x_after_brc3, scale_factor=4, mode='bilinear')
        x_afterbrc_up2 = F.interpolate(x_after_brc2, scale_factor=2, mode='bilinear')
        x_after_caspp=self.caspp(x_after_brc4)
        # x_after_gap=self.gap1(x_after_caspp)

        x_after_caspp_up=F.interpolate(x_after_caspp,scale_factor=8,mode='bilinear')
        # x_after_gap = x_after_gap.expand_as(x_after_caspp_up)

        F_concate=torch.cat((x_after_brc1,x_afterbrc_up2,x_afterbrc_up3,x_afterbrc_up4,x_after_caspp_up),dim=1)#1*640*120*160
        # F_concate=torch.cat((x_after_brc1,x_afterbrc_up2,x_afterbrc_up3,x_afterbrc_up4,x_after_caspp_up,x_after_gap),dim=1)
        F_concate=self.samm(F_concate)#1*230*120*160
        F_concate=self.tse(F_concate)
        F_concate=self.brc5(F_concate)
        F_concate=self.bn2(F_concate)
        F_enhance=self.dropout(F_concate)#1*160*120*160
        # F_enhance=torch.cat((F_enhance,fuse1),dim=1)#1*416*120*160
        # F_enhance=self.feat_extractor(F_enhance)
        # F_enhance=self.upsample(F_enhance)


        F_enhance=F.interpolate(F_enhance,scale_factor=2, mode='bilinear')

        x_semantic1=self.conv1_semantic(F_enhance)
        x_semantic2=self.conv2_semantic(x_semantic1)

        x_semantic_output_final=self.out_block2(x_semantic2)


        return x_semantic_output_final,[e1,e2,e3,e4]


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