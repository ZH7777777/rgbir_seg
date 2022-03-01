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
class LearnedPositionalEncoding(nn.Module):
    def __init__(self,b,c,hw,external):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(b, hw+external,c)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings
class casppfortransformer(nn.Module):
    def __init__(self,in_channels, out_channels,scale,kernel_size,padding):
        super(casppfortransformer, self).__init__()
        self.brc1=BRC_caspp(in_channels,in_channels,dilation=2)
        self.brc2 = BRC_caspp(in_channels,in_channels, dilation=4)
        self.brc3 = BRC_caspp(in_channels,in_channels, dilation=6)
        self.brc=BRC(3*in_channels,in_channels,kernel_size=kernel_size,padding=padding,stride=scale)






    def forward(self, x):


        x1=self.brc1(x)
        x2=self.brc2(x)
        x3=self.brc3(x)
        x_all=torch.cat((x1,x2,x3),dim=1)
        x_out=self.brc(x_all)

        return x_out
class transformerfusion(nn.Module):
    def __init__(self, in_channels, out_channels,scale,b,c,hw,kernal_size,padding):
        super(transformerfusion, self).__init__()
        self.caspp=casppfortransformer(in_channels,out_channels,scale,kernal_size,padding)
        self.casppforir = casppfortransformer(in_channels, out_channels, scale,kernal_size,padding)
        external_seq_len = 10
        self.hw=hw


        self.position_encoding = LearnedPositionalEncoding(b, c, hw,external_seq_len

                                                           )
        self.position_encodingforir = LearnedPositionalEncoding(b, c, hw, external_seq_len

                                                           )


        self.external_seq1 = nn.Parameter(torch.Tensor(external_seq_len, c))
        self.external_seq2 = nn.Parameter(torch.Tensor(external_seq_len, c))
        self._reset_parameters()
        self.transformer = TransformerModel(
            dim=c,
            depth=2,
            heads=8,
            mlp_dim=1104

            ,

            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )
        self.transformerforir = TransformerModel(
            dim=c,
            depth=2,
            heads=8,
            mlp_dim=1104

            ,

            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )
        self.pre_head_ln = nn.LayerNorm(c)
        self.attention = Attention(dim=c)
        self.up=nn.Upsample(scale_factor=scale,mode='bilinear')
        self.pad = nn.ConstantPad2d((0, 0, 0, 15), 0)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.external_seq1)
        nn.init.xavier_uniform_(self.external_seq2)

    def forward(self, rgb,ir):
        batch_size = rgb.size(0)
        c1=rgb.size(1)
        h1 = rgb.size(2)
        w1= rgb.size(3)
        # hw=h1*w1
        batch_external_seq1 = self.external_seq1.repeat(batch_size, 1, 1)
        batch_external_seq2 = self.external_seq2.repeat(batch_size, 1, 1)
        #rgb--------------------
        x=self.caspp(rgb)#1*64*7*10
        c2 = x.size(1)
        h2 = x.size(2)
        w2 = x.size(3)
        x=x.permute(0,2,3,1).contiguous()#1*7*10*64
        x=x.view(x.size(0),-1,x.size(3))
        x=torch.cat((x,batch_external_seq1),dim=1)#1*80*64
        x=self.position_encoding(x)#1*80*64
        x=self.transformer(x)
        x = self.pre_head_ln(x)
        x_output=x[:, :self.hw, :]
        x_output=x_output.permute(0,2,1).contiguous()
        # print(batch_size)
        # print(c2)
        # print(h2)
        # print(w2)

        # print(x_output.shape)
        x_output=x_output.view(batch_size,c2,h2,w2)#1*64*7*10
        # rgb--------------------
        # ir--------------------
        y = self.casppforir(ir)  # 1*64*7*10
        y = y.permute(0, 2, 3, 1).contiguous()  # 1*7*10*64
        y = y.view(y.size(0), -1, y.size(3))
        y = torch.cat((y, batch_external_seq2),dim=1)  # 1*80*64
        y = self.position_encodingforir(y)  # 1*80*64
        y = self.transformerforir(y)
        y = self.pre_head_ln(y)
        y_output = y[:, :self.hw, :]
        y_output = y_output.permute(0,2,1).contiguous()
        y_output = y_output.view(batch_size, c2, h2, w2)  # 1*64*7*10

        fuse=self.attention(x_output,y_output)
        fuse=self.up(fuse)#padding到 120，160
        # fuse=self.pad(fuse)
        fuse=fuse[:,:,:h1,:]
        output=fuse+rgb+ir





        return output
class BRC(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1,stride=1):
        super(BRC, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        # self.relu=F.relu(inplace=True)
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)

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
        self.brc1=BRC_caspp(512,512,dilation=2)
        self.brc2 = BRC_caspp(512, 512, dilation=4)
        self.brc3 = BRC_caspp(512, 512, dilation=6)
        self.brc=BRC(3*512,512,kernel_size=1,padding=0)





    def forward(self, x):
        x1=self.brc1(x)
        x2=self.brc2(x)
        x3=self.brc3(x)
        x_all=torch.cat((x1,x2,x3),dim=1)
        x_out=self.brc(x_all)
        return x_out
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


class fusesegusetrans_new(nn.Module):
    def __init__(self, n_class):
        super(fusesegusetrans_new, self).__init__()
        c = [96, 192, 384, 1056, 1104]
        densenet = torchvision.models.densenet121(pretrained=True)
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
        # self.rgb_features6 = nn.Sequential(
        #     densenet.features.denseblock4,
        #     _Transition(c[4] * 2, c[4])
        # )

        del densenet
        densenet = torchvision.models.densenet121(pretrained=True)
        conv1 = nn.Conv2d(1,
                          64,
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
        # self.ir_features6 = nn.Sequential(
        #     densenet.features.denseblock4,
        #     _Transition(c[4] * 2, c[4])
        # )
        del densenet

        self.feat_extractor5 = FeatureExtractor(c[3] * 2, c[3])
        self.feat_extractor4 = FeatureExtractor(256, 128)
        self.feat_extractor3 = FeatureExtractor(128, 64)
        self.feat_extractor2 = FeatureExtractor(128, 64)
        self.feat_extractor1 = FeatureExtractor(64, 9)
        self.attention=Attention(dim=1104)

        self.upsampler5 = UpSampler(512, 256)
        self.upsampler4 = UpSampler(256, 128)
        self.upsampler3 = UpSampler(128, 64)
        self.upsampler2 = UpSampler(64, 64)
        self.upsampler1 = UpSampler(c[0], c[0])
        self.out_block = nn.ConvTranspose2d(64, n_class, kernel_size=2, stride=2)
        #pos encoding需要改成80
        # self.transformerfusion1 = transformerfusion(in_channels=64, out_channels=64, scale=32, b=2, c=64, hw=80,
        #                                             kernal_size=3, padding=1)
        # self.transformerfusion2 = transformerfusion(in_channels=64, out_channels=64, scale=16, b=2, c=64, hw=80,
        #                                             kernal_size=3, padding=1)
        # self.transformerfusion3 = transformerfusion(in_channels=128, out_channels=128, scale=8, b=2, c=128, hw=80,
        #                                             kernal_size=3, padding=1)
        # self.transformerfusion4 = transformerfusion(in_channels=256, out_channels=256, scale=4, b=2, c=256, hw=80,
        #                                             kernal_size=3, padding=1)
        self.transformerfusion5 = transformerfusion(in_channels=512, out_channels=512, scale=2, b=2, c=512, hw=80,
                                                    kernal_size=3, padding=1)
        # self.transformerfusion6 = transformerfusion(in_channels=1104, out_channels=1104, scale=1, b=2, c=1104, hw=70,
        #                                             kernal_size=3, padding=1)

        self.pe_dropout = nn.Dropout(p=0.1)

        self.pe_dropout = nn.Dropout(p=0.1)
        # self.linear_encoding = nn.Linear(1104,1104)
        # self.linear_encoding2 = nn.Linear(1104,1104)
        #最好的那次 fusesegusetrans3 需要加上两个encoding3，4
        # self.linear_encoding3 = nn.Linear(80, 80)
        # self.linear_encoding4 = nn.Linear(80, 80)
        #--------------------------------------------------------
        self.pre_head_ln = nn.LayerNorm(1104)
        self.pre_head_ln2 = nn.LayerNorm(80)
        [96, 192, 384, 1056, 1104]
        self.resdiv5 = Resdiv(256*2, 128*2)
        self.resdiv4 = Resdiv(128*2, 64*2)
        self.resdiv3 = Resdiv(64*2, 64*2)
        self.resdiv2 = Resdiv(64*2,64)
        self.resdiv1 = Resdiv(64, n_class)

        # # ---------------- init external sequence --------------
        # external_seq_len = 10
        # # self.external_seq1 = nn.Parameter(torch.Tensor(external_seq_len, 70))
        # # self.external_seq2 = nn.Parameter(torch.Tensor(external_seq_len, 70))
        # self.external_seq1 = nn.Parameter(torch.Tensor(1104, external_seq_len))
        # self.external_seq2 = nn.Parameter(torch.Tensor(1104, external_seq_len))
        # self._reset_parameters()
        # # ------------------------------------------------------

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.external_seq1)
        nn.init.xavier_uniform_(self.external_seq2)

    def forward(self, x):

        # ---------------------------------------------

        rgb = x[:, :3]
        ir = x[:, 3:]

        ir1 = self.ir_features1(ir)
        ir2 = self.ir_features2(ir1)
        ir3 = self.ir_features3(ir2)
        ir4 = self.ir_features4(ir3)
        ir5 = self.ir_features5(ir4)
        # ir6 = self.ir_features6(ir5)

        x1 = self.rgb_features1(rgb) + ir1  # 1*64*240*320
        x2 = self.rgb_features2(x1) + ir2  # 1*64*120*160
        x3 = self.rgb_features3(x2) + ir3  # 1*128*60*80
        x4 = self.rgb_features4(x3) + ir4  # 1*256*30*40
        # x5 = self.rgb_features5(x4) + ir5  # 1*512*15*20
        # x6 = self.rgb_features6(x5) + ir6  # 1*1104*7*10
        # x1 = self.transformerfusion1(self.rgb_features1(rgb),ir1)  # 1*96*240*320
        # x2 = self.transformerfusion2(self.rgb_features2(x1),ir2)  # 1*96*120*160
        # x3 = self.transformerfusion3(self.rgb_features3(x2),ir3)  # 1*192*60*80
        # x4 = self.transformerfusion4(self.rgb_features4(x3),ir4) # 1*384*30*40
        x5 = self.transformerfusion5(self.rgb_features5(x4),ir5) # 1*1056*15*20
        # a=self.rgb_features6(x5)
        # x6 = self.transformerfusion6(self.rgb_features6(x5),ir6)  # 1*1104*7*10


        x = self.upsampler5(x5)  # 1*1056*14*20
        # pad = nn.ConstantPad2d((0, 0, 1, 0), 0)
        # x = pad(x)  # 1*1056*15*20
        x = self.resdiv5(torch.cat((x, x4), dim=1))
        x_2 = self.feat_extractor4(x)
        x = self.resdiv4(x + torch.cat((x_2, x3), dim=1))
        x_3 = self.feat_extractor3(x)
        x = self.resdiv3(x + torch.cat((x_3, x2), dim=1))
        x_4 = self.feat_extractor2(x)
        x = self.resdiv2(x + torch.cat((x_4, x1), dim=1))
        x_5 = self.feat_extractor1(x)
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
        return x_5


def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 4, 480, 640)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = fusesegusetrans_new(9)
    input = rgb
    output, fuse = rtf_net(input)
    for i in fuse:
        print(0)
    print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()