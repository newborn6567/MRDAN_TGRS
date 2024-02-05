import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Function
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilated=False):#out_channel is 4*planes
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilated = dilated
        self.stride = stride

        self.conv_dilated2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False, dilation=2)##
        self.conv_dilated3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=3, bias=False, dilation=3)##padding==dilation


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dilated == True:##plus 3 layers before relu
            out_L2 = self.conv_dilated2(out)
            out_L2 = self.bn2(out_L2)
            out_L2 = self.relu(out_L2)
            out_L3 = self.conv_dilated3(out)
            out_L3 = self.bn2(out_L3)
            out_L3 = self.relu(out_L3)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.dilated == True:##
            out = out + out_L2 + out_L3##

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):#block==Bottleneck, layers==[3,4,6,3]
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, usedilated=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, usedilated=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, usedilated=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, usedilated=False):#(block=Bottleneck, planes=64, blocks=3, stride=1)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))#64,64,1,
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i==blocks-1 and usedilated==True:
                layers.append(block(self.inplanes, planes, dilated=True))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    else:
        net_dict = model.state_dict()#
        state_dic = torch.load('../pretrain/resnet50.pth')
        state_dict = {k: v for k, v in state_dic.items() if k in net_dict.keys()}#
        net_dict.update(state_dict)#
        model.load_state_dict(net_dict)
    return model

resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}

class AdversarialLayer(Function):
    @staticmethod
    def forward(ctx, x, max_iter,iter_num):
        ctx.max_iter = max_iter
        ctx.iter_num = iter_num
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradOutput):
        # print("---grl---")
        if ctx.iter_num >= ctx.max_iter:
            ctx.iter_num = ctx.max_iter
        coeff = np.float(2.0 / (1.0 + np.exp(-10.0 * ctx.iter_num / ctx.max_iter)) - 1.0)
        output = gradOutput.neg() * coeff
        return output, None, None


class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=31,conv_feature_dim=2048,ad_local_num=7):
        super(ResNetFc, self).__init__()
        self.backbone_net = resnet50(False)
        self.feature_layers = self.backbone_net.feature_layers
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.conv_feature_dim = conv_feature_dim
        if new_cls:
            if self.use_bottleneck:
                bottleneck_linear = nn.Linear(ad_local_num * ad_local_num * conv_feature_dim, bottleneck_dim)
                bottleneck_linear.weight.data.normal_(0, 0.005)
                bottleneck_linear.bias.data.fill_(0.0)
                self.bottleneck = nn.Sequential()
                self.bottleneck.add_module("linear", bottleneck_linear)
                self.bottleneck.add_module("bn", nn.BatchNorm1d(bottleneck_dim))
                self.bottleneck.add_module("relu", nn.ReLU(True))
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            else:
                self.fc = nn.Linear(ad_local_num * ad_local_num * conv_feature_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            self.__in_features = bottleneck_dim
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        return x

    def output_num(self):
        return self.conv_feature_dim

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 256)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.relu1 = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(256)
        self.ad_layer2 = nn.Linear(256, 1)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.bn(x)
        x = self.ad_layer2(x)
        x = self.sigmoid(x)
        return x

    def output_num(self):
        return 1

class AlexNetFc(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000, conv_feature_dim=256,ad_local_num=6):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.feature_layers = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.conv_feature_dim = conv_feature_dim

        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_alexnet.classifier[6].in_features, bottleneck_dim)
                self.bottleneck.weight.data.normal_(0, 0.005)
                self.bottleneck.bias.data.fill_(0.0)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            else:
                self.fc = nn.Linear(model_alexnet.classifier[6].in_features, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            self.__in_features = bottleneck_dim

    def forward(self, x):
        x = self.feature_layers(x)
        return x

    def output_num(self):
        return self.conv_feature_dim
