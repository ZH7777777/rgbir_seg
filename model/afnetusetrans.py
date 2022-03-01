import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from model.backboneforafnet import resnet4thermal, resnet4rgb, Bottleneck
from model.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from model.Transformer import TransformerModel
from model.transformerlikefuse import Attention

# resnet4rgb = resnet4rgb(Bottleneck, [3, 4, 6, 3])
# resnet4thermal = resnet4thermal(Bottleneck, [3, 4, 6, 3])


# from .backbone import build_backbone
class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1
class EnBlock3(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock3, self).__init__()

        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock4(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1
class decode(nn.Module):
    def __init__(self):
        super(decode, self).__init__()
        self.intmd_layers=[1,2,3]
        # self.intmd_x=intmd_x
    def forward(self,x):
        encoder_outputs = {}
        all_keys = []
        for i in self.intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = x[val]
        all_keys.reverse()

        xoutput=encoder_outputs[all_keys[0]]
        return xoutput
class decode2(nn.Module):
    def __init__(self):
        super(decode2, self).__init__()
        self.intmd_layers=[1,2,3]
        # self.intmd_x=intmd_x
    def forward(self,x):
        encoder_outputs = {}
        all_keys = []
        for i in self.intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = x[val]
        all_keys.reverse()

        xoutput=encoder_outputs[all_keys[0]]
        return xoutput


class afnetusetrans(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, n_class):
        super(afnetusetrans, self).__init__()
        """ Initializes the model.

        """
        self.attentionfuse=Attention(dim=2048)
        self.num_resnet_layers = 50

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048
        # self.model4rgb=resnet_raw_model1
        # self.model4thermal=resnet_raw_model2
        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4
        self.embedding_dim=2048
        self.position_encoding = LearnedPositionalEncoding(

        )
        self.position_encoding2 = LearnedPositionalEncoding(

        )
        self.pe_dropout = nn.Dropout(p=0.1)
        self.transformer = TransformerModel(
            dim=2048,
            depth=3,
            heads=8,
            mlp_dim=2048


            ,

            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )
        self.transformer2 = TransformerModel(

            dim=2048,
            depth=3,
            heads=8,
            mlp_dim=2048

            ,

            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)
        self.attentionforrgb=decode()
        self.Enblock8_1 = EnBlock1(in_channels=2048)
        self.Enblock8_2 = EnBlock2(in_channels=1024)
        self.Enblock8_3 = EnBlock3(in_channels=2048)
        self.Enblock8_4 = EnBlock4(in_channels=1024)
        self.linear_encoding = nn.Linear(300,300)
        self.linear_encoding2 = nn.Linear(300, 300)

        # self.n_class=n
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv3 =nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, n_class, kernel_size=3, stride=1, padding=1, dilation=1),

        )
        # self.convforrgbtoken= x = nn.Conv2d(
        #         2048,
        #         512,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1
        #     )
        # 5类

        # resnet50 = models.resnet50(pretrained=True)
        # pretrained_dict = resnet50.state_dict()
        # pretrained_dict2 = resnet50.state_dict()
        # model_dict = resnet4rgb.state_dict()
        # model_dict2 = resnet4thermal.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # pretrained_dict2 = {k: v for k, v in pretrained_dict2.items() if k in model_dict2}
        # model_dict.update(pretrained_dict)
        # model_dict2.update(pretrained_dict2)
        # resnet4rgb.load_state_dict(model_dict)
        # resnet4thermal.load_state_dict(model_dict2)
        # # self.mo
        # # resnet4rgb = resnet4rgb(Bottleneck, [3, 4, 6, 3])
        # # resnet4thermal = resnet4thermal(Bottleneck, [3, 4, 6, 3])
        #
        # self.model4rgb = resnet4rgb
        # self.model4thermal = resnet4thermal

    def forward(self, input):
        rgb = input[:, :3]
        thermal = input[:, 3:]
        B,C,H,W=rgb.size()
        # verbose = False
        rgb = self.encoder_rgb_conv1(rgb)
        # if verbose: print("rgb.size() after conv1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        # if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        # if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)

        thermal = self.encoder_thermal_conv1(thermal)
        # if verbose: print("thermal.size() after conv1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        # if verbose: print("thermal.size() after bn1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        # if verbose: print("thermal.size() after relu: ", thermal.size())  # (240, 320)

        rgb = rgb + thermal

        rgb = self.encoder_rgb_maxpool(rgb)
        # if verbose: print("rgb.size() after maxpool: ", rgb.size())  # (120, 160)

        thermal = self.encoder_thermal_maxpool(thermal)
        # if verbose: print("thermal.size() after maxpool: ", thermal.size())  # (120, 160)

        ######################################################################

        rgb = self.encoder_rgb_layer1(rgb)
        # if verbose: print("rgb.size() after layer1: ", rgb.size())  # (120, 160)
        thermal = self.encoder_thermal_layer1(thermal)
        # if verbose: print("thermal.size() after layer1: ", thermal.size())  # (120, 160)

        rgb = rgb + thermal

        ######################################################################

        rgb = self.encoder_rgb_layer2(rgb)
        # if verbose: print("rgb.size() after layer2: ", rgb.size())  # (60, 80)
        thermal = self.encoder_thermal_layer2(thermal)
        # if verbose: print("thermal.size() after layer2: ", thermal.size())  # (60, 80)

        rgb = rgb + thermal

        ######################################################################

        rgb = self.encoder_rgb_layer3(rgb)
        # if verbose: print("rgb.size() after layer3: ", rgb.size())  # (30, 40)
        thermal = self.encoder_thermal_layer3(thermal)
        # if verbose: print("thermal.size() after layer3: ", thermal.size())  # (30, 40)

        rgb = rgb + thermal

        ######################################################################

        rgb = self.encoder_rgb_layer4(rgb)
        # if verbose: print("rgb.size() after layer4: ", rgb.size())  # (15, 20)
        thermal = self.encoder_thermal_layer4(thermal)
        # if verbose: print("thermal.size() after layer4: ", thermal.size())  # (15, 20)
        fuse = rgb + thermal#1*2048*15*20

        # image = self.model4rgb(rgb)
        # thermal = torch.cat((thermal, thermal, thermal), dim=1)
        # thermal = self.model4thermal(thermal)

        image= fuse.permute(0, 2, 3, 1).contiguous()
        thermal = thermal.permute(0, 2, 3, 1).contiguous()
        image = image.view(image.size(0), -1, image.size(3))#1*300*2048
        thermal = thermal.view(thermal.size(0), -1, thermal.size(3))
        image=image.permute(0,2,1).contiguous()
        thermal = thermal.permute(0, 2, 1).contiguous()#1*2048*300
        image=self.linear_encoding(image)
        thermal=self.linear_encoding2(thermal)#1*2048*300





        image = self.position_encoding(image)  # 1*2048*300
        image = self.pe_dropout(image)
        thermal= self.position_encoding2(thermal)  # 1*2048*300
        thermal = self.pe_dropout(thermal)
        image = image.permute(0, 2, 1).contiguous()
        thermal = thermal.permute(0, 2, 1).contiguous()

        image, intmd_image = self.transformer(image)
        image = self.pre_head_ln(image)
        thermal, intmd_thermal = self.transformer2(thermal)
        thermal = self.pre_head_ln(thermal)

        # imageaftertrans=decode()(intmd_image)

        # thermalaftertrans=decode2()(intmd_thermal)# 1*300*2048
        # transformer之后的输出已经得到  后续继续进行 selfattetnion 两个模态之间并没有发生关系 可以哪个大取哪个
        # allattention=torch.cat((imageaftertrans,thermalaftertrans),dim=1)#1*600*2048
        imageattention=image.permute(0,2,1).contiguous()
        thermalattention=thermal.permute(0,2,1).contiguous()
        imageattention=imageattention.contiguous().view(B,2048,15,20)#bchw
        # imageattention=self.Enblock8_1(imageattention)
        # imageattention=self.Enblock8_2(imageattention)

        thermalattention = thermalattention.contiguous().view(B, 2048, 15, 20)  # bchw
        # thermalattention = self.Enblock8_3(thermalattention)
        # thermalattention = self.Enblock8_4(thermalattention)
        allattention=self.attentionfuse(imageattention,thermalattention)



        # allattention=torch.cat((imageattention,thermalattention),dim=1)




        # b, c, h, w = image.size()
        #
        # image_mat = image.contiguous().view(image.size(0), image.size(1), -1)
        # thermal_mat = thermal.contiguous().view(thermal.size(0), thermal.size(1), -1)  ##
        # image_transpose = torch.transpose(image_mat, 1, 2)
        # thermal_transpose = torch.transpose(thermal_mat, 1, 2)
        #
        # image_similarity = torch.bmm(F.normalize(image_transpose, dim=2), F.normalize(thermal_mat, dim=1))
        # thermal_similarity = torch.bmm(F.normalize(thermal_transpose, dim=2), F.normalize(image_mat, dim=1))
        #
        # rgb_base_attention = F.softmax(image_similarity, dim=2)
        # thermal_base_attention = F.softmax(thermal_similarity, dim=2)
        #
        # rgb_base_enhanced = torch.bmm(thermal_mat, rgb_base_attention.transpose(1, 2))
        # thermal_base_enhanced = torch.bmm(image_mat, thermal_base_attention.transpose(1, 2))
        # rgb_base_enhanced = rgb_base_enhanced.contiguous().view(b, c, h, w)
        # thermal_base_enhanced = thermal_base_enhanced.contiguous().view(b, c, h, w)
        # rgb_enhanced = rgb_base_enhanced + image
        # thermal_enhanced = thermal_base_enhanced + thermal
        # output = rgb_enhanced + thermal_enhanced
        output = self.upsample(allattention)
        output = self.conv1(output)
        output = self.upsample(output)
        output = self.conv2(output)
        output = self.upsample(output)
        output = self.conv3(output)
        output=self.upsample(output)
        output=self.conv4(output)
        output = self.upsample(output)
        output = self.conv5(output)

        return output
        # return 0
if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        # x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        x = torch.rand((1, 4, 480,640))
        model = afnetusetrans(n_class=9)
        model.cpu()
        y = model(x)
        print(y.shape)