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

class FuseSegwithmultilabel(nn.Module):
    def __init__(self, n_class):
        super(FuseSegwithmultilabel, self).__init__()
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
        self.conv1_salient=nn.Conv2d(c[0],
                          c[0],
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False)
        self.conv2_salient = nn.Conv2d(c[0],
                                       c[0],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.conv1_semantic = nn.Conv2d(3*c[0],
                                       3*c[0],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.conv2_semantic = nn.Conv2d(3*c[0],
                                        3*c[0],
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.conv1_boundary = nn.Conv2d(c[0],
                                        c[0],
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.conv2_boundary = nn.Conv2d(c[0],
                                        c[0],
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.batchnorm=nn.BatchNorm2d(c[0])
        self.batchnorm2 = nn.BatchNorm2d(3*c[0])
        self.out_block1 = nn.ConvTranspose2d(c[0], c[0], kernel_size=2, stride=2)
        self.out_block2 = nn.ConvTranspose2d(3*c[0], n_class, kernel_size=2, stride=2)
        self.out_block_salient1 = nn.ConvTranspose2d(c[0], c[0], kernel_size=2, stride=2)
        self.out_block_salient2 = nn.ConvTranspose2d(c[0], 1, kernel_size=2, stride=2)
        self.out_block_boundary1 = nn.ConvTranspose2d(c[0], c[0], kernel_size=2, stride=2)
        self.out_block_boundary2 = nn.ConvTranspose2d(c[0], 1, kernel_size=2, stride=2)

    def forward(self, x):
        rgb = x[:, :3]
        ir = x[:, 3:]

        ir1 = self.ir_features1(ir)
        ir2 = self.ir_features2(ir1)
        ir3 = self.ir_features3(ir2)
        ir4 = self.ir_features4(ir3)
        ir5 = self.ir_features5(ir4)
        ir6 = self.ir_features6(ir5)

        x1 = self.rgb_features1(rgb) + ir1#1*96*240*320
        x2 = self.rgb_features2(x1) + ir2#1*96*120*160
        x3 = self.rgb_features3(x2) + ir3#1*192*60*80
        x4 = self.rgb_features4(x3) + ir4#1*384*30*40
        x5 = self.rgb_features5(x4) + ir5#1*1056*15*20
        x6 = self.rgb_features6(x5) + ir6#1*1104*7*10

        x = self.upsampler5(x6)#1*1056*14*20
        pad = nn.ConstantPad2d((0, 0, 1, 0), 0)
        x = pad(x)#1*1056*15*20
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
        x = self.feat_extractor2(x)#1*96*120*160
        x = self.upsampler1(x)
        x = torch.cat((x, x1), dim=1)
        x = self.feat_extractor1(x)#1*96*240*320
        #----------multilabel

        x_salient1=self.conv1_salient(x)
        x_salient2= self.conv2_salient(x_salient1)
        # x_salient_output=self.out_block_salient(x_salient2)

        x_salient_attention=self.relu(x_salient2)
        x_salient_attention=self.batchnorm(x_salient_attention)
        x_semantic_withsalientattention=x*x_salient_attention

        x_boundary1=self.conv1_boundary(x)
        x_boundary2=self.conv2_boundary(x_boundary1)
        x_boundary_attention=self.relu(x_boundary2)
        x_boundary_attention=self.batchnorm(x_boundary_attention)
        x_semantic_withboundaryattention=x*x_boundary_attention

        x_semantic_all=torch.cat((x,x_semantic_withboundaryattention,x_semantic_withsalientattention),dim=1)


        # x_semantic_withsalientattention1=self.out_block1(x_semantic_withsalientattention)
        # x_semantic_withsalientattention2=self.conv1_semantic(x_semantic_withsalientattention1)
        # x_semantic_withsalientattention3=self.conv2_semantic(x_semantic_withsalientattention2)
        # # x_semantic_output=self.out_block(x_semantic_withsalientattention2)
        #
        # x_boundary_withsemantic=self.out_block_boundary1(x)
        #
        # x_boundary_withsemantic1=torch.cat((x_boundary_withsemantic,x_semantic_withsalientattention2),dim=1)
        # x_boundary_withsemantic2=self.conv1_boundary(x_boundary_withsemantic1)
        # x_boundary_withsemantic3=self.conv2_boundary(x_boundary_withsemantic2)
        #

        # x_salient_output = self.out_block_salient1(x_salient2)
        x_salient_output = self.out_block_salient2(x_salient2)

        x_semantic_output = self.conv1_semantic(x_semantic_all)
        x_semantic_output=self.relu(x_semantic_output)
        x_semantic_output=self.batchnorm2(x_semantic_output)
        x_semantic_output=x_semantic_output+x_semantic_all
        x_semantic_output2=self.conv2_semantic(x_semantic_output)
        x_semantic_output2 = self.relu(x_semantic_output2)
        x_semantic_output2 = self.batchnorm2(x_semantic_output2)
        x_semantic_output_final=x_semantic_output+x_semantic_output2
        x_semantic_output_final=self.out_block2(x_semantic_output_final)

        x_boundary_output = self.out_block_boundary2(x_boundary2)
        # x_boundary_output=0

#  现在转置卷积作为 classification层，把卷积层作为classification  转置卷积只用来提高分辨率




        # x = self.out_block(x)
        return x_semantic_output_final,x_salient_output,x_boundary_output
def unit_test():
    num_minibatch = 1
    rgb = torch.randn(num_minibatch, 4, 480, 640)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = FuseSegwithmultilabel(9)
    input = rgb
    output,fuse = rtf_net(input)
    for i in fuse:
        print(0)
    print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()