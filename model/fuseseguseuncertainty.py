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
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)
# def KL(alpha, c):
#     beta = torch.ones(( 1,c,480,640))#1，10
#     S_alpha = torch.sum(alpha, dim=1, keepdim=True)#2，1
#     S_beta = torch.sum(beta, dim=1, keepdim=True)#1，1
#     lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)#2，1
#     lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)#1，1
#     dg0 = torch.digamma(S_alpha)#2，1
#     dg1 = torch.digamma(alpha)#2，10
#     kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni#2，1
#     kl=torch.sum(torch.flatten(kl,start_dim=2,end_dim=3),dim=2,keepdim=True)/(480*640)
#     return kl#每一个像素点来说需要flatten 全部相加吗  还是说是相加除以总数 还是说直接返回1，480，640
# def KL(alpha, c):
#     beta = torch.ones((1, c)).cuda()
#     S_alpha = torch.sum(alpha, dim=1, keepdim=True)
#     S_beta = torch.sum(beta, dim=1, keepdim=True)
#     lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
#     lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
#     dg0 = torch.digamma(S_alpha)
#     dg1 = torch.digamma(alpha)
#     kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
#     return kl
#
#
# def ce_loss(p, alpha, c, global_step, annealing_step):
#     # a = alpha.squeeze(0)
#     loss=torch.zeros(alpha.size(0),alpha.size(1),1).cuda()
#     for i in range(alpha.size(0)):
#
#         S = torch.sum(alpha[i], dim=1, keepdim=True)
#
#         E = alpha[i] - 1
#         # label = p.view(-1)
#         label = F.one_hot(p[i], num_classes=c)
#         # label = p.view(-1)
#         a=label * (torch.digamma(S) - torch.digamma(alpha[i]))
#         # print(a)
#         A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha[i])), dim=1, keepdim=True)#2*1    #ln(f(x))对x求导
#
#         annealing_coef = min(1, global_step / annealing_step)
#
#         alp = E * (1 - label) + 1#2*10
#         B = annealing_coef * KL(alp, c)
#         B1 = torch.zeros_like(B)
#         loss[i]=A+B1
    # S = torch.sum(alpha, dim=1, keepdim=True) #一个像素点的 九个类别概率加和
    # E = alpha - 1
    # # label = F.one_hot(p, num_classes=c)
    # label = p.unsqueeze(1)
    # # dd=torch.digamma(S) - torch.digamma(alpha)
    # # a = torch.flatten(label * (torch.digamma(S) - torch.digamma(alpha)),start_dim=2,end_dim=3)
    # A = torch.sum(torch.flatten(label * (torch.digamma(S) - torch.digamma(alpha)),start_dim=2,end_dim=3), dim=1, keepdim=True)  # 2*1    #ln(f(x))对x求导
    # A=torch.sum(A,dim=2,keepdim=True)/(480*640)
    # # print(A)
    # annealing_coef = min(1, global_step / annealing_step)
    # # cc=1-label
    #
    # alp = E * (1 - label) + 1  # 2*10
    # B = annealing_coef * KL(alp, c)




    # return loss

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

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


class FuseSeguseuncertainty(nn.Module):
    def __init__(self, n_class):
        super(FuseSeguseuncertainty, self).__init__()
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

        self.upsampler5_ir = UpSampler(c[4], c[3])
        self.upsampler4_ir = UpSampler(c[3], c[2])
        self.upsampler3_ir = UpSampler(c[2], c[1])
        self.upsampler2_ir = UpSampler(c[1], c[0])
        self.upsampler1_ir = UpSampler(c[0], c[0])
        self.out_block_ir = nn.ConvTranspose2d(c[0], n_class, kernel_size=2, stride=2)
        self.softplus=nn.Softplus()

    def DS_Combin(self, alpha):
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = 9 / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, 9, 1), b[1].view(-1, 1, 9))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = 9 / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    #     """
    #     :param alpha: All Dirichlet distribution parameters.
    #     :return: Combined Dirichlet distribution parameters.
    #     """
    #

    #     def DS_Combin_two(alpha1, alpha2):
    #         """
    #         :param alpha1: Dirichlet distribution parameters of view 1
    #         :param alpha2: Dirichlet distribution parameters of view 2
    #         :return: Combined Dirichlet distribution parameters
    #         """
    #         alpha = dict()
    #         alpha[0], alpha[1] = alpha1, alpha2
    #         b, S, E, u = dict(), dict(), dict(), dict()
    #         for v in range(2):
    #             S[v] = torch.sum(alpha[v], dim=2, keepdim=True)
    #             E[v] = alpha[v] - 1
    #             b[v] = E[v] / (S[v].expand(E[v].shape))
    #             u[v] = 9 / S[v]
    #
    #         # b^0 @ b^(0+1)
    #         bb=torch.zeros(2,307200,9,9).cuda()
    #
    #         for batch in range(2):
    #             bb[batch] = torch.bmm(b[0][batch].view( 307200, 9, 1), b[1][batch].view( 307200, 1, 9))
    #
    #
    #
    #         # bb = torch.bmm(b[0].view(2,307200, 9, 1), b[1].view(2,307200, 1, 9))
    #         # b^0 * u^1
    #         uv1_expand = u[1].expand(b[0].shape)
    #         bu = torch.mul(b[0], uv1_expand)
    #         # b^1 * u^0
    #         uv_expand = u[0].expand(b[0].shape)
    #         ub = torch.mul(b[1], uv_expand)
    #         # calculate C
    #         bb_sum = torch.sum(bb, dim=(2, 3), out=None)
    #         bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    #         C = bb_sum - bb_diag
    #         dasd=torch.mul(b[0], b[1])
    #         dasd=(1 - C).view(2,-1, 1)
    #         asdfasf=(1 - C).view(2,-1, 1).expand(b[0].shape)
    #
    #         # calculate b^a
    #         b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(2,-1, 1).expand(b[0].shape))
    #         # calculate u^a
    #         u_a = torch.mul(u[0], u[1]) / ((1 - C).view(2,-1, 1).expand(u[0].shape))
    #
    #         # calculate new S
    #         S_a = 9 / u_a
    #         # calculate new e_k
    #         e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    #         alpha_a = e_a + 1
    #         return alpha_a
    #
    #     for v in range(len(alpha) - 1):
    #         if v == 0:
    #             alpha_a = DS_Combin_two(alpha[0], alpha[1])
    #         else:
    #             alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
    #     return alpha_a
    # def DS_Combin(self, alpha):
    #     """
    #     :param alpha: All Dirichlet distribution parameters.
    #     :return: Combined Dirichlet distribution parameters.
    #     """
    #     def DS_Combin_two(alpha1, alpha2):
    #         """
    #         :param alpha1: Dirichlet distribution parameters of view 1
    #         :param alpha2: Dirichlet distribution parameters of view 2
    #         :return: Combined Dirichlet distribution parameters
    #         """
    #         alpha = dict()
    #         alpha[0], alpha[1] = alpha1, alpha2
    #         b, S, E, u = dict(), dict(), dict(), dict()
    #         for v in range(2):
    #             S[v] = torch.sum(alpha[v], dim=1, keepdim=True)#1,10
    #             E[v] = alpha[v]-1#2,10
    #             b[v] = E[v]/(S[v].expand(E[v].shape))#2,10
    #             u[v] = 9/S[v]#1,10
    #
    #         # b^0 @ b^(0+1)
    #         #b[0]2,10
    #         # bb = torch.bmm(b[0].view(-1, 9, 1), b[1].view(-1, 1, 9))
    #         # aaa=b[0].view(-1, 9, 1)
    #         bb = torch.bmm(b[0].view(-1, 9, 1), b[1].view(-1, 1, 9))
    #         # b^0 * u^1
    #         uv1_expand = u[1].expand(b[0].shape)#2,10
    #         bu = torch.mul(b[0], uv1_expand)#2,10
    #         # b^1 * u^0
    #         uv_expand = u[0].expand(b[0].shape)#2,10
    #         ub = torch.mul(b[1], uv_expand)#2,10
    #         # calculate C
    #         bb_sum = torch.sum(bb, dim=(1, 2), out=None)#1,2
    #         bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)#2,10  然后做sum 以第二维做sum 得到2，1
    #         C = bb_sum - bb_diag
    #         d=(1-C).view(-1, 1).reshape(alpha1.size(0),1,480,640).expand(b[0].shape)
    #         # calculate b^a
    #         b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).reshape(alpha1.size(0),1,480,640).expand(b[0].shape))
    #         # calculate u^a
    #         u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).reshape(alpha1.size(0),1,480,640).expand(u[0].shape))
    #
    #         # calculate new S
    #         S_a = 9/ u_a
    #         # calculate new e_k
    #         e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    #         alpha_a = e_a + 1
    #         return alpha_a
    #
    #     for v in range(len(alpha)-1):
    #         if v==0:
    #             alpha_a = DS_Combin_two(alpha[0], alpha[1])
    #         else:
    #             alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
    #     return alpha_a


    def forward(self, x,target,epoch):
        rgb = x[:, :3]
        ir = x[:, 3:]

        ir1 = self.ir_features1(ir)
        ir2 = self.ir_features2(ir1)
        ir3 = self.ir_features3(ir2)
        ir4 = self.ir_features4(ir3)
        ir5 = self.ir_features5(ir4)
        ir6 = self.ir_features6(ir5)

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
        x = self.feat_extractor1(x)
        x = self.out_block(x)

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
        x_ir = self.upsampler1_ir(x_ir)
        x_ir = torch.cat((x_ir, ir1), dim=1)
        x_ir = self.feat_extractor1_ir(x_ir)
        x_ir = self.out_block_ir(x_ir)
        # output=x+x_ir


        evidence =dict()
        x_output=self.softplus(x)
        x_iroutput=self.softplus(x_ir)
        # evidence[0] = x_output
        # evidence[1] = x_iroutput
        #
        loss = 0
        alpha = dict()
        x_output=x_output.view(x_output.size(0),9,-1).permute(0,2,1).contiguous().view(-1,9).contiguous()#2*(480*640)*9
        x_iroutput = x_iroutput.view(x_iroutput.size(0), 9, -1).permute(0,2, 1).contiguous().view(-1,9).contiguous()
        target = target.view(target.size(0), -1).view(-1)#2*(480*640)*1
        evidence[0] = x_output
        evidence[1] = x_iroutput
        # evidence=torch.zeros_like(x_output)
        # evidence_a=torch.zeros_like(x_output)

        # evidence[0]=x_output[i]
        # evidence[1]=x_iroutput[i]
        # label=target[i]

        for v_num in range(len(evidence)):
        #     # step two
            alpha[v_num] = evidence[v_num] + 1
        #     # step three
        #     # alpha[v_num]=alpha[v_num].view(evidence[v_num].size(0),9,-1).permute(2,0,1).contiguous()#(480*640)*2*9
        #     # target=target.view(target.size(0),-1).permute(1,0).contiguous()#(480*640)*2
        #
        #
        #
        #     loss += ce_loss(target, alpha[v_num], 9, epoch, annealing_step=20)
            # a=torch.mean(loss)
        # step four
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        # evidence
        loss += ce_loss(target, alpha_a, 9, epoch, annealing_step=90)


            # evidence_a=evidence_a.view(1,9,480,640)
        # loss=loss/(480*640)
        # loss = torch.mean(torch.sum(loss,dim=1),dim=0)/(480*640)
        loss = torch.mean(loss)
        evidence_a=evidence_a.view(x.size(0),480,640,9).permute(0,3,1,2)
        return  evidence_a,loss


        # return x, []
def unit_test():
    num_minibatch = 1
    input = torch.randn(num_minibatch, 4, 480, 640)
    label = torch.zeros(num_minibatch,  480, 640)
    # thermal = torch.randn(num_minibatch, 1, 480, 640)
    rtf_net = FuseSeguseuncertainty(9)
    epoch=1

    evidence_a,loss = rtf_net(input,label,epoch)
    # for i in fuse:
    #     print(0)
    # print(output.shape)
    # print('The model: ', rtf_net.modules)


if __name__ == '__main__':
    unit_test()