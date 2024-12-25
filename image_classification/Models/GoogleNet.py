import torch
from torch import nn
import torch.nn.functional as F


# define a layer
def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0, eps=1e-3):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels, eps),
        nn.ReLU(True)
    )
    return layer


class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        # the first line
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        # the second line
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        # the third line
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        # the fourth line
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        # forward
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, training):
        super(InceptionAux, self).__init__()
        self.training = training
        self.averagePool = nn.AvgPool2d(3, stride=1) # changed
        self.conv = conv_relu(in_channels, 128, 1)  # output[batch, 128, 4, 4]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # N x 512 x 6 x 6, N x 528 x 6 x 6
        #x = self.averagePool(x) #64
        # N x 512 x 4 x 4, N x 528 x 4 x 4
        x = self.conv(x)    # N x 128 x 6 x 6
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)  # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)  # N x 1024
        x = self.fc2(x)  # N x num_classes
        return x


class googlenet(nn.Module):
    def __init__(self, in_channel, num_classes, aux_logits=True, training=True, init_weights=False, verbose=False):
        super(googlenet, self).__init__()
        self.verbose = verbose
        self.aux_logits = aux_logits
        self.init_weights = init_weights
        self.training = training

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )
        self.block4_1 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64)
        )
        self.block4_2 = nn.Sequential(
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
        )
        self.block4_3 = nn.Sequential(
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128)
        )
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes, self.training)  # InceptionAux类
            self.aux2 = InceptionAux(528, num_classes, self.training)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.verbose: print('input: {}'.format(x.shape))
        x = self.block1(x)
        if self.verbose: print('block 1 output: {}'.format(x.shape))

        x = self.block2(x)
        if self.verbose: print('block 2 output: {}'.format(x.shape))

        x = self.block3(x)
        if self.verbose: print('block 3 output: {}'.format(x.shape))

        x = self.block4_1(x)
        if self.verbose: print('block 4_1 output: {}'.format(x.shape))

        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)
        if self.verbose: print('block aux1 output: {}'.format(aux1.shape))

        x = self.block4_2(x)
        if self.verbose: print('block 4_2 output: {}'.format(x.shape))

        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)
        if self.verbose: print('block aux2 output: {}'.format(aux2.shape))

        x = self.block4_3(x)
        if self.verbose: print('block 4_3 output: {}'.format(x.shape))

        x = self.block5(x)
        if self.verbose: print('block 5 output: {}'.format(x.shape))

        x = self.avgpool(x)  # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)  # N x 1024
        x = self.dropout(x)
        x = self.classifier(x)  # N x 1000 (num_classes)

        if self.training and self.aux_logits:  # eval model不执行该部分
            return x, aux2, aux1

        return x


class googlenet2(nn.Module):
    def __init__(self, in_channel, num_classes, aux_logits=True, training=True, init_weights=False, verbose=False):
        super(googlenet2, self).__init__()
        self.verbose = verbose
        self.aux_logits = aux_logits
        self.init_weights = init_weights
        self.training = training

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channels=16, kernel=3, stride=1, padding=1),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )
        self.block2 = nn.Sequential(
            conv_relu(16, 64, kernel=1),
            conv_relu(64, 32, kernel=3, padding=1),
            nn.MaxPool2d(3, 2, ceil_mode=True)
        )
        self.block3 = nn.Sequential(
            inception(in_channel, 4, 32, 16, 8, 4, 4),
            #inception(256, 128, 128, 192, 32, 96, 64),
            #nn.MaxPool2d(3, 2, ceil_mode=True)
        )
        self.block4_1 = nn.Sequential(
            #inception(480, 192, 96, 208, 16, 48, 64)
        )
        self.block4_2 = nn.Sequential(
            #inception(512, 160, 112, 224, 24, 64, 64),
            #inception(512, 128, 128, 256, 24, 64, 64),
            #inception(512, 112, 144, 288, 32, 64, 64),
        )
        self.block4_3 = nn.Sequential(
            #inception(528, 256, 160, 320, 32, 128, 128),
            #nn.MaxPool2d(3, 2, ceil_mode=True)
        )
        self.block5 = nn.Sequential(
            #inception(832, 256, 160, 320, 32, 128, 128),
            #inception(832, 384, 182, 384, 48, 128, 128)
        )
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes, self.training)  # InceptionAux类
            self.aux2 = InceptionAux(528, num_classes, self.training)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(32, num_classes)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.verbose: print('input: {}'.format(x.shape))
        x = self.block1(x)
        if self.verbose: print('block 1 output: {}'.format(x.shape))

        x = self.block2(x)
        if self.verbose: print('block 2 output: {}'.format(x.shape))

        x = self.block3(x)
        if self.verbose: print('block 3 output: {}'.format(x.shape))

        #x = self.block4_1(x)
        #if self.verbose: print('block 4_1 output: {}'.format(x.shape))

        #if self.training and self.aux_logits:  # eval model lose this layer
        #    aux1 = self.aux1(x)
        #if self.verbose: print('block aux1 output: {}'.format(aux1.shape))

        #x = self.block4_2(x)
        #if self.verbose: print('block 4_2 output: {}'.format(x.shape))

        #if self.training and self.aux_logits:  # eval model lose this layer
        #    aux2 = self.aux2(x)
        #if self.verbose: print('block aux2 output: {}'.format(aux2.shape))

        #x = self.block4_3(x)
        #if self.verbose: print('block 4_3 output: {}'.format(x.shape))

        #x = self.block5(x)
        #if self.verbose: print('block 5 output: {}'.format(x.shape)) #

        x = self.avgpool(x)  # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)  # N x 1024
        #x = self.dropout(x)
        x = self.classifier(x)  # N x 1000 (num_classes)

        #if self.training and self.aux_logits:  # eval model不执行该部分
            #return x, aux2, aux1

        return x