
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from .backboneforafnet import resnet4thermal,resnet4rgb,Bottleneck,Bottleneck2
import numpy as np
resnet4rgb = resnet4rgb(Bottleneck, [3, 4, 6, 3])
resnet4thermal = resnet4thermal(Bottleneck2, [3, 4, 6, 3])




# from .backbone import build_backbone

class afnet(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, n_class):
        super(afnet, self).__init__()
        """ Initializes the model.
        
        """

        # self.n_class=n
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.conv1=nn.Conv2d(in_channels=2048,out_channels=512,kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv2=nn.Conv2d(in_channels=512,out_channels=128,kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv3=nn.Conv2d(in_channels=128,out_channels=n_class,kernel_size=3,stride=1,padding=1,dilation=1)
        #5类

        resnet50=models.resnet50(pretrained=True)
        resnet50forthermal=models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        pretrained_dict2 = resnet50forthermal.state_dict()
        model_dict = resnet4rgb.state_dict()
        model_dict2 = resnet4thermal.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dict2 = {k: v for k, v in pretrained_dict2.items() if k in model_dict2}
        model_dict.update(pretrained_dict)
        model_dict2.update(pretrained_dict2)
        resnet4rgb.load_state_dict(model_dict)
        resnet4thermal.load_state_dict(model_dict2)
        # self.mo
        # resnet4rgb = resnet4rgb(Bottleneck, [3, 4, 6, 3])
        # resnet4thermal = resnet4thermal(Bottleneck, [3, 4, 6, 3])





        self.model4rgb = resnet4rgb
        self.model4thermal=resnet4thermal


    def forward(self, input):
        rgb = input[:, :3]
        thermal = input[:, 3:]

        image=self.model4rgb(rgb)
        thermal=torch.cat((thermal,thermal,thermal),dim=1)
        thermal=self.model4thermal(thermal)
        b, c, h, w = image.size()

        image_mat = image.contiguous().view(image.size(0), image.size(1), -1) #c*t
        thermal_mat = thermal.contiguous().view(thermal.size(0), thermal.size(1), -1)## #c*t
        image_transpose = torch.transpose(image_mat,1,2) #t*c
        thermal_transpose = torch.transpose(thermal_mat, 1, 2)#t*c




        image_similarity = torch.bmm(F.normalize(image_transpose, dim=2), F.normalize(thermal_mat, dim=1))
        thermal_similarity = torch.bmm(F.normalize(thermal_transpose, dim=2), F.normalize(image_mat, dim=1))





        rgb_base_attention = F.softmax(image_similarity,dim=2)
        thermal_base_attention = F.softmax(thermal_similarity,dim=2)

        # rgb_base_enhanced = torch.bmm(thermal_mat, rgb_base_attention.transpose(1,2))
        # thermal_base_enhanced = torch.bmm(image_mat, thermal_base_attention.transpose(1,2))
        rgb_base_enhanced = torch.bmm(thermal_mat, rgb_base_attention.transpose(1, 2))
        thermal_base_enhanced = torch.bmm(image_mat, thermal_base_attention.transpose(1, 2))
        rgb_base_enhanced = rgb_base_enhanced.contiguous().view(b, c, h, w)
        thermal_base_enhanced = thermal_base_enhanced.contiguous().view(b, c, h, w)
        rgb_enhanced = rgb_base_enhanced + image
        thermal_enhanced = thermal_base_enhanced + thermal
        output = rgb_enhanced + thermal_enhanced
        output = self.upsample(output)
        output = self.conv1(output)
        output = self.upsample(output)
        output = self.conv2(output)
        output = self.upsample(output)
        output = self.conv3(output)

        return output
