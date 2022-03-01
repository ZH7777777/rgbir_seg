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
from model.transformerlikefuse import Attention
from model.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding,LearnedPositionalEncoding2
from model.Transformer import TransformerModel
from model.External_attention import External_attention

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


class fusesegusetrans2(nn.Module):
    def __init__(self, n_class):
        super(fusesegusetrans2, self).__init__()
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
        self.attention=Attention(dim=1104)
        self.out_block = nn.ConvTranspose2d(c[0], n_class, kernel_size=2, stride=2)

        self.upsampler5 = UpSampler(c[4], c[3])
        self.upsampler4 = UpSampler(c[3], c[2])
        self.upsampler3 = UpSampler(c[2], c[1])
        self.upsampler2 = UpSampler(c[1], c[0])
        self.upsampler1 = UpSampler(c[0], c[0])
        # self.out_block = nn.ConvTranspose2d(c[0], n_class, kernel_size=2, stride=2)
        # self.position_encoding = LearnedPositionalEncoding(
        #
        # )
        # self.position_encoding2 = LearnedPositionalEncoding(
        #
        # )
        # self.position_encoding3 = LearnedPositionalEncoding2(
        #
        # )
        # self.position_encoding4 = LearnedPositionalEncoding2(
        #
        # )
        # self.pe_dropout = nn.Dropout(p=0.1)
        # self.transformer = TransformerModel(
        #     dim=1104,
        #     depth=4,
        #     heads=8,
        #     mlp_dim=2048
        #
        #     ,
        #
        #     dropout_rate=0.1,
        #     attn_dropout_rate=0.1,
        # )
        # self.transformer2 = TransformerModel(
        #
        #     dim=1104,
        #     depth=4,
        #     heads=8,
        #     mlp_dim=2048
        #
        #     ,
        #
        #     dropout_rate=0.1,
        #     attn_dropout_rate=0.1,
        # )
        # self.transformer3 = TransformerModel(
        #     dim=70,
        #     depth=6,
        #     heads=7,
        #     mlp_dim=2048
        #
        #     ,
        #
        #     dropout_rate=0.1,
        #     attn_dropout_rate=0.1,
        # )
        # self.transformer4 = TransformerModel(
        #
        #     dim=70,
        #     depth=6,
        #     heads=7,
        #     mlp_dim=2048
        #
        #     ,
        #
        #     dropout_rate=0.1,
        #     attn_dropout_rate=0.1,
        # )
        # self.pe_dropout = nn.Dropout(p=0.1)
        # self.linear_encoding = nn.Linear(1104,1104)
        # self.linear_encoding2 = nn.Linear(1104,1104)
        # self.linear_encoding3 = nn.Linear(70, 70)
        # self.linear_encoding4 = nn.Linear(70, 70)
        # self.pre_head_ln = nn.LayerNorm(1104)
        # self.pre_head_ln2 = nn.LayerNorm(70)
        # [96, 192, 384, 1056, 1104]
        self.exattention=External_attention(1104)
        self.exattention2=External_attention(1104)
        # self.resdiv5 = Resdiv(1056*2, 384*2)
        # self.resdiv4 = Resdiv(384*2, 192*2)
        # self.resdiv3 = Resdiv(192*2, 96*2)
        # self.resdiv2 = Resdiv(96*2, 96*2)
        # self.resdiv1 = Resdiv(96, n_class)

    def forward(self, x):
        rgb = x[:, :3]
        ir = x[:, 3:]

        ir1 = self.ir_features1(ir)
        ir2 = self.ir_features2(ir1)
        ir3 = self.ir_features3(ir2)
        ir4 = self.ir_features4(ir3)
        ir5 = self.ir_features5(ir4)
        ir6 = self.ir_features6(ir5)

        x1 = self.rgb_features1(rgb)   # 1*96*240*320
        x2 = self.rgb_features2(x1)   # 1*96*120*160
        x3 = self.rgb_features3(x2)   # 1*192*60*80
        x4 = self.rgb_features4(x3)   # 1*384*30*40
        x5 = self.rgb_features5(x4)   # 1*1056*15*20
        x6 = self.rgb_features6(x5)  # 1*1104*7*10
        B,C,H,W=x6.shape
        imageattention=self.exattention(x6)
        thermalattention=self.exattention2(ir6)

        # image = x6.permute(0, 2, 3, 1).contiguous()
        # thermal = ir6.permute(0, 2, 3, 1).contiguous()
        # image = image.view(image.size(0), -1, image.size(3))  # 1*70*1104
        # thermal = thermal.view(thermal.size(0), -1, thermal.size(3))
        # image2=image
        # thermal2=thermal#1*70*1104
        # image2 = image2.permute(0, 2, 1).contiguous()
        # thermal2 = thermal2.permute(0, 2, 1).contiguous()#1*1104*70
        # #像素间注意力
        #
        # image2 = self.linear_encoding3(image2)
        # thermal2 = self.linear_encoding4(thermal2)
        #
        # image2 = self.position_encoding(image2)  # 1*1104*70
        # image2 = self.pe_dropout(image2)
        # thermal2 = self.position_encoding2(thermal2)  # 1*1104*70
        # thermal2 = self.pe_dropout(thermal2)
        # image2 = self.transformer3(image2)
        # image2 = self.pre_head_ln2(image2)
        # thermal2 = self.transformer4(thermal2)
        # thermal2 = self.pre_head_ln2(thermal2)#1*1104*70





        #通道间注意力
        # image = image.permute(0, 2, 1).contiguous()
        # thermal = thermal.permute(0, 2, 1).contiguous()  # 1*1104*70
        # image = self.linear_encoding(image)#1*70*1104
        # thermal = self.linear_encoding2(thermal)  # 1*70*1104
        #
        # image = self.position_encoding3(image)  # 1*70*1104
        # image = self.pe_dropout(image)
        # thermal = self.position_encoding4(thermal)  # 1*70*1104
        # thermal = self.pe_dropout(thermal)
        # # image = image.permute(0, 2, 1).contiguous()
        # # thermal = thermal.permute(0, 2, 1).contiguous()
        #
        # image = self.transformer(image)
        # image = self.pre_head_ln(image)
        # thermal = self.transformer2(thermal)
        # thermal = self.pre_head_ln(thermal)#1*70*1104

        # imageaftertrans=decode()(intmd_image)

        # thermalaftertrans=decode2()(intmd_thermal)# 1*300*2048
        # transformer之后的输出已经得到  后续继续进行 selfattetnion 两个模态之间并没有发生关系 可以哪个大取哪个
        # allattention=torch.cat((imageaftertrans,thermalaftertrans),dim=1)#1*600*2048
        # imageattention = image2+image.permute(0, 2, 1).contiguous()
        # thermalattention = thermal2+thermal.permute(0, 2, 1).contiguous()
        # imageattention = image2
        # thermalattention = thermal2
        # imageattention = imageattention.contiguous().view(B, 1104, 7, 10)  # bchw
        # # imageattention=self.Enblock8_1(imageattention)
        # # imageattention=self.Enblock8_2(imageattention)
        #
        # thermalattention = thermalattention.contiguous().view(B, 1104, 7, 10)  # bchw
        # thermalattention = self.Enblock8_3(thermalattention)
        # thermalattention = self.Enblock8_4(thermalattention)
        allattention = self.attention(imageattention, thermalattention)

        # attention=self.attention(x6,ir6)
        x=allattention

        x = self.upsampler5(x)  # 1*1056*14*20
        pad = nn.ConstantPad2d((0, 0, 1, 0), 0)
        x = pad(x)  # 1*1056*15*20
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
        # x = self.resdiv5(torch.cat((x, x5), dim=1))
        # x_2 = self.feat_extractor4(x)
        # x = self.resdiv4(x + torch.cat((x_2, x4), dim=1))
        # x_3 = self.feat_extractor3(x)
        # x = self.resdiv3(x + torch.cat((x_3, x3), dim=1))
        # x_4 = self.feat_extractor2(x)
        # x = self.resdiv2(x + torch.cat((x_4, x2), dim=1))
        # x_5 = self.feat_extractor1(x)
        # x = self.resdiv1(x_5)
        # x = torch.cat((x, x5), dim=1)
        # x = self.feat_extractor5(x)
        # x = self.upsampler4(x)
        # x = torch.cat((x, x4), dim=1)
        # x = self.feat_extractor4(x)
        # x = self.upsampler3(x)
        # x = torch.cat((x, x3), dim=1)
        # x = self.feat_extractor3(x)
        # x = self.upsampler2(x)
        # x = torch.cat((x, x2), dim=1)
        # x = self.feat_extractor2(x)
        # x = self.upsampler1(x)
        # x = torch.cat((x, x1), dim=1)
        # x = self.feat_extractor1(x)
        # x = self.out_block(x)
        return x


def unit_test():
    num_minibatch = 1
    rgb = torch.randn(num_minibatch, 4, 480, 640)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = fusesegusetrans2(9)
    input = rgb
    output, fuse = rtf_net(input)
    for i in fuse:
        print(0)
    print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()