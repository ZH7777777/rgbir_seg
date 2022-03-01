import torch.nn as nn
from util.util import IntermediateSequential
import torch

class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0,sw=1,h=1,w=1
    ):
        super().__init__()
        self.sw=sw
        self.h=h
        self.w=w
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_w = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape#1*80*1104


        qkv = (
            self.qkv(x)#1*80*1104*3
            .reshape(B,N, 3, self.num_heads, C // self.num_heads)#1*80*8*1104/8
            .permute(2, 0,3, 1, 4)
        )
        half_head=self.num_heads/2
        qkv_h=qkv[:,:,:4,:,:]
        qkv_v = qkv[:, :, 4:, :, :]
        # print(self.num_heads/2)
        # print(self.h)
        # print(self.w)
        # print(C // self.num_heads)
        qkv_h=qkv_h.view(3,B,4,self.h,self.w, C // self.num_heads)
        qkv_h=qkv_h.view(3,B,4,int(self.h/self.sw),int(self.sw*self.w),C // self.num_heads)

        qkv_v = qkv_v.view(3, B, int(self.num_heads / 2), self.h, self.w, C // self.num_heads)
        qkv_v = qkv_v.view(3, B, int(self.num_heads / 2), int(self.w / self.sw),int( self.sw * self.h), C // self.num_heads)
        q_h, k_h, v_h = (
            qkv_h[0],
            qkv_h[1],
            qkv_h[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        q_v, k_v, v_v = (
            qkv_v[0],
            qkv_v[1],
            qkv_v[2],
        )

        attn_v = (q_v@ k_v.transpose(-2, -1)) * self.scale
        attn_v = attn_v.softmax(dim=-1)
        attn_v = self.attn_drop(attn_v)

        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        attn_h = self.attn_drop(attn_h)
        x_v = (attn_v @ v_v).reshape( B, int(self.num_heads / 2), int(self.w *self.h), C // self.num_heads)
        x_h = (attn_h @ v_h).reshape( B, int(self.num_heads / 2), int(self.w *self.h), C // self.num_heads)
        x_all=torch.cat((x_h,x_v),dim=1)


        x_all = x_all.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_all)
        x_out = self.proj_drop(x_out)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class cswinTransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        sw=1,
        h=1,
        w=1


    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate,sw=sw,h=h,w=w),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)
