import torch.nn as nn
from util.util import IntermediateSequential


class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.conv1=nn.Conv2d(dim,dim,kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv6 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv7=nn.Conv2d(dim,dim,kernel_size=3,padding=1)
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.attn_drop2 = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm2d(dim)
        self.batchnorm2 = nn.BatchNorm2d(dim)
        self.batchnorm3 = nn.BatchNorm2d(dim)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, rgb,thermal):
        #1*c*h*w
        B,C,H,W = rgb.shape
        rgb=self.batchnorm1(rgb)
        thermal=self.batchnorm2(thermal)
        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # q, k, v = (
        #     qkv[0],
        #     qkv[1],
        #     qkv[2],
        # )  # make torchscript happy (cannot use tensor as tuple)
        q=self.conv1(rgb)
        k=self.conv2(thermal)
        v=self.conv3(thermal)
        q2 = self.conv4(thermal)
        k2 = self.conv5(rgb)
        v2 = self.conv6(rgb)
        b,c,h,w=q.shape
        #rgb=q kv=thermal
        q=q.permute(0,2,3,1).view(b,-1,c).contiguous()#b*(h*w)*c
        k = k.permute(0, 2, 3, 1).view(b,-1,c).contiguous()#b*(h*w)*c
        v = v.permute(0, 2, 3, 1).view(b,-1,c).contiguous()#b*(h*w)*c

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)#b*(hw)*c
        #thermal=q kv=rgb
        # q2 = q2.permute(0, 2, 3, 1).view(b, -1, c).contiguous()  # b*(h*w)*c
        # k2 = k2.permute(0, 2, 3, 1).view(b, -1, c).contiguous()  # b*(h*w)*c
        # v2 = v2.permute(0, 2, 3, 1).view(b, -1, c).contiguous()  # b*(h*w)*c
        #
        # attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        # attn2 = attn2.softmax(dim=-1)
        # attn2 = self.attn_drop2(attn2)
        #
        # x2 = (attn2 @ v2).transpose(1, 2)

        output=x.reshape(B,C,H,W)
        # output2 = x2.reshape(B, C, H, W)
        output=output+rgb+thermal
        # output=self.conv7(output)
        # output=self.batchnorm3(output)
        # output=self.relu(output)


        return output


