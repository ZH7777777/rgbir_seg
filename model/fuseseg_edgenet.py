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
# import bdcn
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
class fusesegboundaryOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(fusesegboundaryOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        # self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x
class GFT(nn.Module):
    def __init__(self, in_channels, out_channels,scale):
        super(GFT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=scale,kernel_size=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,stride=1,kernel_size=1,padding=0)
        # self.sigmoid=F.sigmoid()
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=scale, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=1, kernel_size=1, padding=0)



    def forward(self, feature,edge):
        edge1=self.conv1(edge)
        edge2=self.conv2(edge1)
        edge_sigmoid=F.sigmoid(edge2)
        edge3=edge2*edge_sigmoid
        # print(feature.shape)
        # print(edge2.shape)
        # print(edge_sigmoid.shape)
        # print(edge3.shape)

        feature_mul=feature*edge3
        edge4=self.conv3(edge)
        edge5=self.conv4(edge4)
        edge_sigmoid2=F.sigmoid(edge5)
        feature_add=edge5*edge_sigmoid2
        feature_out=feature_add+feature_mul+feature
        return feature_out

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


class FuseSeguseedgenet(nn.Module):
    def __init__(self, n_class):
        super(FuseSeguseedgenet, self).__init__()
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
            _Transition(c[4] * 2, c[4])
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

        self.feat_extractor5_ir = FeatureExtractor(c[3] * 2, c[3])
        self.feat_extractor4_ir = FeatureExtractor(c[2] * 2, c[2])
        self.feat_extractor3_ir = FeatureExtractor(c[1] * 2, c[1])
        self.feat_extractor2_ir = FeatureExtractor(c[0] * 2, c[0])
        self.feat_extractor1_ir = FeatureExtractor(c[0] * 2, c[0])
        self.edge_extractor1_ir = FeatureExtractor(c[0] * 2, c[0])
        self.edge_extractor1_rgb = FeatureExtractor(c[0] * 2, c[0])
        self.edge_extractor1_all = FeatureExtractor(c[0] * 4, c[0])


        self.upsampler5_ir = UpSampler(c[4], c[3])
        self.upsampler4_ir = UpSampler(c[3], c[2])
        self.upsampler3_ir = UpSampler(c[2], c[1])
        self.upsampler2_ir = UpSampler(c[1], c[0])
        self.upsampler1_ir = UpSampler(c[0], c[0])
        self.out_block_ir = nn.ConvTranspose2d(c[0], n_class, kernel_size=2, stride=2)
        self.softplus=nn.Softplus()
        self.gft1=GFT(in_channels=1,out_channels=96,scale=2)
        self.gft2 = GFT(in_channels=1, out_channels=96, scale=4)
        self.gft3 = GFT(in_channels=1, out_channels=192, scale=8)
        self.conv_semantic2edge=Conv2d(in_channels=9,out_channels=1,kernel_size=3,padding=1,stride=1)
        self.fusesegoutputboundary1 = fusesegboundaryOutput(9, 9, 1)
        # self.gft4 = GFT(in_channels=384, out_channels=384, scale=2)
        self.out_block_edge = nn.ConvTranspose2d(c[0], 1, kernel_size=2, stride=2)





    def forward(self, x,target,epoch):
        rgb = x[:, :3]
        ir = x[:, 3:]

        # ir1 = self.ir_features1(ir)#2,96,240,320
        # ir1fuseedge=self.gft1(ir1,edge)
        # ir2 = self.ir_features2(ir1fuseedge)#2,96,120,160
        # ir2fuseedge = self.gft2(ir2, edge)
        # ir3 = self.ir_features3(ir2fuseedge)#2,192,60,80
        # ir3fuseedge = self.gft3(ir3, edge)
        # ir4 = self.ir_features4(ir3fuseedge)#2,384,30,40

        ir1 = self.ir_features1(ir)  # 2,96,240,320

        ir2 = self.ir_features2(ir1)  # 2,96,120,160

        ir3 = self.ir_features3(ir2)  # 2,192,60,80

        ir4 = self.ir_features4(ir3)
        ir5 = self.ir_features5(ir4)#2,1056,15,20
        ir6 = self.ir_features6(ir5)#2,1104,7,10

        x1 = self.rgb_features1(rgb)
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
        x_rgb_edge=self.edge_extractor1_rgb(x)
        x_edge2 = self.feat_extractor1(x)
        x = self.out_block(x_edge2)

        x_ir = self.upsampler5_ir(ir6)
        pad = nn.ConstantPad2d((0, 0, 1, 0), 0)
        x_ir = pad(x_ir)
        x_ir = torch.cat((x_ir, ir5), dim=1)
        x_ir = self.feat_extractor5_ir(x_ir)
        x_ir = self.upsampler4_ir(x_ir)
        x_ir = torch.cat((x_ir, ir4), dim=1)
        x_ir = self.feat_extractor4_ir(x_ir)
        x_ir = self.upsampler3_ir(x_ir)
        x_ir = torch.cat((x_ir, ir3), dim=1)
        x_ir = self.feat_extractor3_ir(x_ir)
        x_ir = self.upsampler2_ir(x_ir)
        x_ir = torch.cat((x_ir, ir2), dim=1)
        x_ir = self.feat_extractor2_ir(x_ir)
        x_ir = self.upsampler1_ir(x_ir)#2,96,240,320
        x_ir = torch.cat((x_ir, ir1), dim=1)#2,192,240,320

        x_ir_edge2 = self.feat_extractor1_ir(x_ir)
        x_ir_edge=self.edge_extractor1_ir(x_ir)#2,96,240,320

        #edge___-------------------------------
        x_edge=torch.cat((x_rgb_edge,x_ir_edge,x_ir_edge2,x_edge2),dim=1)
        x_edge=self.edge_extractor1_all(x_edge)
        edge_output=self.out_block_edge(x_edge)
        #----------------
        x_ir = self.out_block_ir(x_ir_edge2)


        semantic_output=x+x_ir
        # edge_input=semantic_output.argmax(1)
        # edge_output=self.fusesegoutputboundary1(semantic_output)#2，1，480，640

        # edge_input=torch.cat((edge_input,edge_input,edge_input),dim=1)#2,3,480,640
        # with torch.no_grad():
        #     edge_output=edgenet(edge_input)
        # edge_out = F.sigmoid(edge_output[-1])



        return semantic_output,edge_output


        # return x, []
def unit_test():
    num_minibatch = 2
    input = torch.randn(num_minibatch, 4, 480, 640)
    label = torch.zeros(num_minibatch,  480, 640)
    # thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = FuseSeguseedgenet(9)
    epoch=1

    evidence_a,loss = rtf_net(input,label,epoch)
    # for i in fuse:
    #     print(0)
    # print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()