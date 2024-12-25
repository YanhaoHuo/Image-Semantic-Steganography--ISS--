import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, -1, 1)
        output = ((input+1.0)/2.0 * 255.).type(torch.uint8) / 255.*2.0-1.0
        return output.type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


class sc_encoder(nn.Module):
    def __init__(self, k1, k2, out_ch):
        super(sc_encoder, self).__init__()
        self.out_ch = out_ch
        self.k1 = k1
        self.k2 = k2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.out_ch, 3, 1, 1),
            nn.Conv2d(self.out_ch, self.out_ch, self.k1, stride=1, padding=0),
            nn.Conv2d(self.out_ch, self.out_ch, self.k2, stride=1, padding=0),
            nn.Conv2d(self.out_ch, 3, 3, 1, 1),
        )

    def forward(self, x):
        # encoder
        out = self.conv1(x)
        return out


class sc_decoder(nn.Module):
    def __init__(self, k1, k2, out_ch):
        super(sc_decoder, self).__init__()
        self.out_ch = out_ch
        self.k1 = k1
        self.k2 = k2
        self.tconv1 = nn.Sequential(
            nn.Conv2d(3, self.out_ch, 3, 1, 1),
            nn.ConvTranspose2d(self.out_ch, self.out_ch, self.k2, stride=1, padding=0),
            nn.ConvTranspose2d(self.out_ch, self.out_ch, self.k1, stride=1, padding=0),
            nn.Conv2d(self.out_ch, 3, 3, 1, 1),
        )

    def forward(self, x):
        # decoder
        out = self.tconv1(x)
        return out


def transform(tensor, target_range):
    source_min = tensor.min()
    source_max = tensor.max()

    # normalize to [0, 1]
    tensor_target = (tensor - source_min)/(source_max - source_min)
    # move to target range
    tensor_target = tensor_target * (target_range[1] - target_range[0]) + target_range[0]
    return tensor_target



