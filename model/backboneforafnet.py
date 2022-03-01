import torchvision.models as models
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

#Bottleneck是一个class 里面定义了使用1*1的卷积核进行降维跟升维的一个残差块，可以在github resnet pytorch上查看
class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,dilation=1,padding=1, downsample=None):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation,padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation=dilation
        self.padding=padding

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,dilation=1,padding=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation,padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation=dilation
        self.padding=padding

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#不做修改的层不能乱取名字，否则预训练的权重参数无法传入
class resnet4thermal(nn.Module):

    def __init__(self, block2, layers, num_classes=9):
        self.inplanes = 64
        super(resnet4thermal, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block2, 64,layers[0],stride=1,padding=1 )
        self.layer2 = self._make_layer(block2, 128, layers[1], stride=2,padding=1)
        self.layer3 = self._make_layer(block2, 256, layers[2], stride=1,dilation=2,padding=2)
        self.layer4 = self._make_layer(block2, 512, layers[3], stride=1,dilation=4,padding=4)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block2, planes, blocks, stride=1,dilation=1,padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block2.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block2.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block2.expansion),
            )

        layers = []
        layers.append(block2(self.inplanes, planes, stride,dilation,padding, downsample))
        self.inplanes = planes * block2.expansion
        for i in range(1, blocks):
            layers.append(block2(self.inplanes, planes))

        return nn.Sequential(*layers)
	#这一步用以设置前向传播的顺序，可以自行调整，前提是合理
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        # 新加层的forward


        return x


class resnet4rgb(nn.Module):

    def __init__(self, block, layers, num_classes=9):
        self.inplanes = 64
        super(resnet4rgb, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,  padding=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,  padding=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, padding=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, padding=4)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation,padding, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # 这一步用以设置前向传播的顺序，可以自行调整，前提是合理
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 新加层的forward

        return x