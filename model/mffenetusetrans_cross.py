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
from model.Transformer import TransformerModel
from model.cswinTransformer import cswinTransformerModel
from model.transformerlikefuse import Attention
class Mlp(nn.Module):
    # two mlp, fc-relu-drop-fc-relu-drop
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class LinearProject(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super(LinearProject, self).__init__()
        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads,  attn_drop=0., proj_drop=0.):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim*2, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, complement):

        # x [B, HW, C]
        B_x, N_x, C_x = x.shape

        x_copy = x

        complement = torch.cat([x, complement], 1)

        B_c, N_c, C_c = complement.shape

        # q [B, heads, HW, C//num_heads]
        q = self.to_q(x).reshape(B_x, N_x, self.num_heads, C_x//self.num_heads).permute(0, 2, 1, 3)
        kv = self.to_kv(complement).reshape(B_c, N_c, 2, self.num_heads, C_c//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_x, N_x, C_x)

        x = x + x_copy

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 1., attn_drop=0., proj_drop=0.,drop_path = 0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(CrossTransformerEncoderLayer, self).__init__()
        self.x_norm1 = norm_layer(dim)
        self.c_norm1 = norm_layer(dim)

        self.attn = MultiHeadCrossAttention(dim, num_heads, attn_drop, proj_drop)

        self.x_norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

        self.drop1 = nn.Dropout(drop_path)
        self.drop2 = nn.Dropout(drop_path)

    def forward(self, x, complement):
        x = self.x_norm1(x)
        complement = self.c_norm1(complement)

        x = x + self.drop1(self.attn(x, complement))
        x = x + self.drop2(self.mlp(self.x_norm2(x)))
        return x



class CrossTransformer(nn.Module):
    def __init__(self, x_dim, c_dim, depth, num_heads, mlp_ratio =1., attn_drop=0., proj_drop=0., drop_path =0.):
        super(CrossTransformer, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearProject(x_dim, c_dim, CrossTransformerEncoderLayer(c_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path)),
                LinearProject(c_dim, x_dim, CrossTransformerEncoderLayer(x_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop_path))
            ]))

    def forward(self, x, complement):
        for x_attn_complemnt, complement_attn_x in self.layers:
            x = x_attn_complemnt(x, complement=complement) + x
            complement = complement_attn_x(complement, complement=x) + complement
        return x, complement


class LearnedPositionalEncoding(nn.Module):
    def __init__(self,b,c,hw,external):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(b, c,hw)) #8x

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
    def __init__(self, in_channels, out_channels,scale,b,c,hw,kernal_size,padding,depth=4):
        super(transformerfusion, self).__init__()
        self.caspp=casppfortransformer(in_channels,out_channels,scale,kernal_size,padding)
        self.casppforir = casppfortransformer(in_channels, out_channels, scale,kernal_size,padding)
        external_seq_len = 8


        self.position_encoding = LearnedPositionalEncoding(b, c, hw,external_seq_len

                                                           )
        self.position_encodingforir = LearnedPositionalEncoding(b, c, hw, external_seq_len

                                                           )


        self.external_seq1 = nn.Parameter(torch.Tensor(external_seq_len, c))
        self.external_seq2 = nn.Parameter(torch.Tensor(external_seq_len, c))
        self._reset_parameters()
        self.transformer = TransformerModel(
            dim=hw,
            depth=depth,
            heads=10,
            mlp_dim=1024

            ,

            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )
        self.transformerforir = TransformerModel(
            dim=hw,
            depth=depth,
            heads=10,
            mlp_dim=1024

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
        # batch_external_seq1 = self.external_seq1.repeat(batch_size, 1, 1)
        # batch_external_seq2 = self.external_seq2.repeat(batch_size, 1, 1)
        #rgb--------------------
        # x=self.caspp(rgb)#1*64*7*10
        x=rgb
        c2 = x.size(1)
        h2 = x.size(2)
        w2 = x.size(3)
        x=x.permute(0,2,3,1).contiguous()#1*7*10*64
        x=x.view(x.size(0),-1,x.size(3)).permute(0,2,1).contiguous()#1*512*300 #b,c,hw
        # x=torch.cat((x,batch_external_seq1),dim=1)#1*80*64
        x=self.position_encoding(x)#1*512*300
        x=self.transformer(x)
        x_output = x.permute(0, 2, 1).contiguous()
        x = self.pre_head_ln(x_output)
        # x_output = x.permute(0, 2, 1).contiguous()
        # x_output=x[:, :80, :]
        x_output=x_output.permute(0,2,1).contiguous()
        # print(batch_size)
        # print(c2)
        # print(h2)
        # print(w2)
        #
        # print(x_output.shape)
        x_output=x_output.view(batch_size,c2,h2,w2)#1*64*7*10
        # rgb--------------------
        # ir--------------------
        # y = self.casppforir(ir)  # 1*64*7*10
        y=ir
        y = y.permute(0, 2, 3, 1).contiguous()  # 1*7*10*64
        y = y.view(y.size(0), -1, y.size(3)).permute(0,2,1).contiguous()
        # y = torch.cat((y, batch_external_seq2),dim=1)  # 1*80*64
        y = self.position_encodingforir(y)  # 1*80*64
        y = self.transformerforir(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.pre_head_ln(y)
        # y_output = y[:, :80, :]
        y_output = y.permute(0,2,1).contiguous()
        y_output = y_output.view(batch_size, c2, h2, w2)  # 1*64*7*10

        fuse=self.attention(x_output,y_output)
        # fuse=self.up(fuse)#padding到 120，160
        # fuse=self.pad(fuse)
        # fuse=fuse[:,:,:h1,:]
        output=fuse+rgb+ir





        return output
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

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
class samm(nn.Module):
    def __init__(self):
        super(samm, self).__init__()

        self.brc1=BRC(1472,256,kernel_size=1,padding=0)
        self.brc2 = BRC(1472, 736, kernel_size=1, padding=0)
        self.brc3 = BRC(736, 256, kernel_size=1, padding=0)






    def forward(self, x):
        x1=self.brc1(x)
        x2=self.brc2(x)
        x3=self.brc3(x2)
        # x_all=torch.cat((x1,x2,x3),dim=1)
        x3=F.sigmoid(x3)
        x_out=x3*x1
        return x_out


class mffenetusetrans_cross(nn.Module):
    def __init__(self, n_class):
        super(mffenetusetrans_cross, self).__init__()
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

        del densenet
        self.brc1=BRC(64,64)
        self.brc2 = BRC(128,128)
        self.brc3 = BRC(256,256)
        self.brc4 = BRC(512,512)
        self.brc5 = BRC(256, 256)
        self.caspp=caspp()
        self.samm=samm()
        self.bn2=nn.BatchNorm2d(256)
        self.dropout=nn.Dropout2d(p=0.2)






        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(256)
        self.batchnorm2 = nn.BatchNorm2d(3 * c[0])
        self.out_block1 = nn.ConvTranspose2d(256,256, kernel_size=2, stride=2)
        self.out_block2 = nn.ConvTranspose2d(128, n_class, kernel_size=2, stride=2)
        self.transformerfusion1=transformerfusion(in_channels=64,out_channels=64,scale=16,b=2,c=64,hw=80,kernal_size=17,padding=8)
        self.transformerfusion2 = transformerfusion(in_channels=128,out_channels=128, scale=8, b=2, c=128,hw= 4800,kernal_size=9,padding=4,depth=2)
        self.transformerfusion3 = transformerfusion(in_channels=256, out_channels=256, scale=4, b=2,c= 256, hw=1200,kernal_size=5,padding=2,depth=2)
        self.transformerfusion4 = transformerfusion(in_channels=512,out_channels= 512, scale=2, b=2, c=512, hw=300,kernal_size=3,padding=1)
        # self.transformerfusion1 = transformerfusion(64, 64, 16, 2, 64, 80, kernal_size=1, padding=1)
        # self.transformerfusion2 = transformerfusion(128, 128, 8, 2, 128, 80, kernal_size=1, padding=1)
        # self.transformerfusion3 = transformerfusion(256, 256, 4, 2, 256, 80, kernal_size=1, padding=1)
        # self.transformerfusion4 = transformerfusion(512, 512, 2, 2, 512, 80, kernal_size=1, padding=1)
        self.cross_transformer=CrossTransformer(512,512,4,8)
        self.position_encoding = LearnedPositionalEncoding(2, 512, 300,10)
        self.position_encoding2 = LearnedPositionalEncoding(2, 512, 300, 10)
        self.conv1_semantic = nn.Conv2d(256, 256,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)
        self.conv2_semantic = nn.Conv2d(256, 128,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False)



    def forward(self, x):
        rgb = x[:, :3]
        ir = x[:, 3:]
        # b=rgb.size(0)

        rgb1 = self.rgb_features1(rgb)  # 1*64*240*320
        rgb2 = self.rgb_features2(rgb1)  # 1*64*120*160
        rgb3 = self.rgb_features3(rgb2)  # 1*128*60*80
        rgb4 = self.rgb_features4(rgb3)  # 1*256*30*40
        rgb5 = self.rgb_features5(rgb4)#1*512*15*20

        ir1 = self.ir_features1(ir)#1*64*240*320
        ir2 = self.ir_features2(ir1)#1*64*120*160
        fuse1=rgb2+ir2
        # fuse1=self.transformerfusion1(rgb2,ir2)
        ir3 = self.ir_features3(fuse1)#1*128*60*80
        fuse2 = rgb3 + ir3
        # fuse2=self.transformerfusion2(rgb3,ir3)
        ir4 = self.ir_features4(fuse2)#1*256*30*40
        fuse3 = rgb4 + ir4
        # fuse3=self.transformerfusion3(rgb4,ir4)
        ir5 = self.ir_features5(fuse3)#1*512*15*20
        rgb5_trans=rgb5.view(rgb5.size(0),rgb5.size(1),-1).contiguous()
        ir5_trans = ir5.view(ir5.size(0), ir5.size(1), -1).contiguous()
        rgb5_trans=self.position_encoding(rgb5_trans)
        ir5_trans=self.position_encoding2(ir5_trans)
        rgb5_trans=rgb5_trans.permute(0,2,1)
        ir5_trans=ir5_trans.permute(0,2,1)

        rgb5_aftertrans,ir5_aftertrans=self.cross_transformer(rgb5_trans,ir5_trans)
        rgb5_aftertrans = rgb5_aftertrans.permute(0, 2, 1)
        ir5_aftertrans = ir5_aftertrans.permute(0, 2, 1)
        rgb5_aftertrans=rgb5_aftertrans.view(rgb5.size(0),rgb5.size(1),rgb5.size(2),rgb5.size(3))
        ir5_aftertrans = rgb5_aftertrans.view(rgb5.size(0), rgb5.size(1), rgb5.size(2), rgb5.size(3))
        fuse4=rgb5_aftertrans+ir5_aftertrans




        x_after_brc1=self.brc1(fuse1)
        x_after_brc2 = self.brc2(fuse2)
        x_after_brc3 = self.brc3(fuse3)
        x_after_brc4 = self.brc4(fuse4)

        x_afterbrc_up4=F.interpolate(x_after_brc4,scale_factor=8,mode='bilinear')
        x_afterbrc_up3 = F.interpolate(x_after_brc3, scale_factor=4, mode='bilinear')
        x_afterbrc_up2 = F.interpolate(x_after_brc2, scale_factor=2, mode='bilinear')
        x_after_caspp=self.caspp(x_after_brc4)
        x_after_caspp_up=F.interpolate(x_after_caspp,scale_factor=8,mode='bilinear')

        F_concate=torch.cat((x_after_brc1,x_afterbrc_up2,x_afterbrc_up3,x_afterbrc_up4,x_after_caspp_up),dim=1)#1*1472*120*160
        F_concate=self.samm(F_concate)#1*256*120*160
        F_concate=self.brc5(F_concate)
        F_concate=self.bn2(F_concate)
        F_enhance=self.dropout(F_concate)







        x_semantic=self.out_block1(F_enhance)
        x_semantic1=self.conv1_semantic(x_semantic)
        x_semantic2=self.conv2_semantic(x_semantic1)

        x_semantic_output_final=self.out_block2(x_semantic2)

        x_salient_output=0
        x_boundary_output=0
        return x_semantic_output_final


def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 4, 480, 640)
    thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = mffenetusetrans_cross(9)
    input = rgb
    output, fuse = rtf_net(input)
    for i in fuse:
        print(0)
    print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()