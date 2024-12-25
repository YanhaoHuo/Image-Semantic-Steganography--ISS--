import torch
from torch import nn
from torch.nn import ModuleList
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from functools import partial
from packaging import version
from collections import namedtuple
from functools import wraps


def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0, eps=1e-3):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_channels, eps)
    )
    return layer


class DenseEncoder(nn.Module):
    def __init__(self, bpp, k1, k2, hidden_size, out_ch):
        super(DenseEncoder, self).__init__()
        self.H = 64
        self.W = 64
        self.hidden_size = hidden_size
        self.out_ch = out_ch
        self.bpp = bpp
        if self.bpp < 1:
            self.data_depth = 1
        else:
            self.data_depth = int(self.bpp)
            self.left = 0
            self.right = 0
        self.k1 = k1
        self.k2 = k2

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.data_depth, out_ch, 3, 1, 1),
            nn.Conv2d(out_ch, out_ch, self.k1, stride=1, padding=0),
            nn.Conv2d(out_ch, out_ch, self.k2, stride=1, padding=0),
            nn.Conv2d(out_ch, self.data_depth, 3, 1, 1),
        )
        self.features = nn.Sequential(
            nn.Conv2d(3, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.hidden_size + self.data_depth, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_size*2 + self.data_depth, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.hidden_size*3 + self.data_depth, 3, 3, 1, 1),
        )
        self.conv5 = nn.Sequential(
            #nn.Conv2d(self.hidden_size * 4, 3, 3, 1, 1),
        )

    def forward(self, image, data):
        data = self.conv1(data)
        x = self.features(image)
        x_list = [x]
        x = torch.cat(x_list + [data], dim=1)
        x = self.conv2(x)
        x_list.append(x)

        x = torch.cat(x_list + [data], dim=1)
        x = self.conv3(x)
        x_list.append(x)

        x = torch.cat(x_list + [data], dim=1)
        x = self.conv4(x)
        x = image + x
        return x


class DenseDecoder(nn.Module):
    def __init__(self, bpp, k1, k2, hidden_size, out_ch):
        super(DenseDecoder, self).__init__()
        self.H = 64
        self.W = 64
        self.hidden_size = hidden_size
        self.out_ch = out_ch
        self.bpp = bpp
        if self.bpp < 1:
            self.data_depth = 1
        else:
            self.data_depth = int(self.bpp)
            self.left = 0
            self.right = 0
        self.k1 = k1
        self.k2 = k2
        self.tconv1 = nn.Sequential(
            nn.Conv2d(3, out_ch, 3, 1, 1),
            nn.ConvTranspose2d(out_ch, out_ch, self.k2, stride=1, padding=0),
            nn.ConvTranspose2d(out_ch, out_ch, self.k1, stride=1, padding=0),
            nn.Conv2d(out_ch, 3, 3, 1, 1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 2, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 3, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 4, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 5, self.data_depth, 3, 1, 1),

        )

        # self.AvgPool = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # )
        # self.Liner = nn.Sequential(
        #     nn.Linear(self.H * self.W, round(self.bpp * self.H * self.W))
        # )

    def forward(self, x):
        x = self.tconv1(x)
        x = self.conv1(x)
        x_list = [x]

        x = torch.cat(x_list, dim=1)
        x = self.conv2(x)
        x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.conv3(x)
        x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.conv4(x)
        x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.conv5(x)
        x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.conv6(x)
        #x_list.append(x)

        #x = torch.cat(x_list, dim=1)
        #x = self.conv7(x)
        return x


class ReDecoder(nn.Module):
    def __init__(self, bpp, hidden_size):
        super(ReDecoder, self).__init__()
        self.H = 64
        self.W = 64
        self.hidden_size = hidden_size
        self.bpp = bpp
        if self.bpp < 1:
            self.data_depth = 1
        else:
            self.data_depth = int(self.bpp)
            self.left = 0
            self.right = 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv2 = nn.Sequential(

            nn.Conv2d(self.hidden_size, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 2, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 3, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 4, self.hidden_size, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 5, self.data_depth, 3, 1, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x_list = [x]

        x = torch.cat(x_list, dim=1)
        x = self.conv2(x)
        x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.conv3(x)
        x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.conv4(x)
        x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.conv5(x)
        x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.conv6(x)
        return x


class LayerType1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LayerType1, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=stride,
                              padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class LayerType2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LayerType2, self).__init__()
        self.type1 = LayerType1(in_channels=in_channels,
                                out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=stride,
                              padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = self.type1(x)
        out = self.bn(self.conv(out))
        out = torch.add(x, out)
        return out


class LayerType3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LayerType3, self).__init__()
        self.type1 = LayerType1(in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride)
        self.conv = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=stride,
                              padding=1)
        self.convs = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.type1(x)
        out = self.pool(self.bn(self.conv(out)))
        res = self.bn(self.convs(x))
        out = torch.add(out, res)
        return out


class LayerType4(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LayerType4, self).__init__()
        self.type1 = LayerType1(in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride)
        self.conv = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.type1(x) # 64,512,4,4
        x = self.conv(x) # 64,512,2,2
        x = self.bn(x) # 64,512,2,2
        x = self.pool(x)
        return x


class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        self.types = {'type1': LayerType1,
                      'type2': LayerType2,
                      'type3': LayerType3,
                      'type4': LayerType4}
        self.layer1 = self._make_layer(types=self.types['type1'], number=2)
        self.layer2 = self._make_layer(types=self.types['type2'], number=5)
        self.layer3 = self._make_layer(types=self.types['type3'], number=4)
        self.layer4 = self._make_layer(types=self.types['type4'], number=1)
        self.ip = nn.Linear(1 * 1 * 512, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.layer1(x) # 64,16,64,64
        x = self.layer2(x) # 64,16,64,64
        x = self.layer3(x) # 64,256,4,4
        x = self.layer4(x) # 64,512,1,1
        x = x.view(x.size(0), -1) # 64,512
        x = self.ip(x) # 64, 2
        return x

    def _make_layer(self, types, number):
        layers = []
        if types == LayerType1:
            print('type = LayerType1')
            out_channels = [64, 16]
            in_channels = [3, 64]
            for i in range(number):
                layers.append(types(in_channels=in_channels[i],
                                    out_channels=out_channels[i]))
        elif types == LayerType2:
            print('type = LayerType2')
            in_channels = 16
            out_channels = 16
            for i in range(number):
                layers.append(types(in_channels=in_channels,
                                    out_channels=out_channels))
        elif types == LayerType3:
            print('type = LayerType3')
            in_channels = [16, 16, 64, 128]
            out_channels = [16, 64, 128, 256]
            for i in range(number):
                layers.append(types(in_channels=in_channels[i],
                                    out_channels=out_channels[i]))
        elif types == LayerType4:
            print('type = LayerType4')
            for i in range(number):
                in_channels = 256
                out_channels = 512
                layers.append(types(in_channels=in_channels,
                                    out_channels=out_channels))
        return nn.Sequential(*layers)

    def reset_parameters(self):
        print('reset_parameters......')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.2)
            if isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()


class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8,
                               kernel_size=5, padding=2, bias=None)
        self.norm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=5, padding=2, bias=None)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=1, bias=None)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=1, bias=None)
        self.norm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=1,  bias=None)
        self.norm5 = nn.BatchNorm2d(128)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        self.glpool = nn.AvgPool2d(kernel_size=4)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.ip = nn.Linear(1*1*128, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.pool(self.tanh(self.norm1(torch.abs(self.conv1(x)))))
        x = self.pool(self.tanh(self.norm2(self.conv2(x))))
        x = self.pool(self.relu(self.norm3(self.conv3(x))))
        x = self.pool(self.relu(self.norm4(self.conv4(x))))
        x = self.glpool(self.relu(self.norm5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.ip(x)
        return x

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, 0, 0.01)
            elif isinstance(mod, nn.BatchNorm2d):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.xavier_uniform(mod.weight)


def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()

