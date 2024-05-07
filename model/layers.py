import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from kornia.geometry.transform import rotate
import pdb


class DualFourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(DualFourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer_vert = weight_norm(torch.nn.Conv2d(in_channels=2 * in_channels,
                                          out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False))
        self.conv_layer_hor = weight_norm(torch.nn.Conv2d(in_channels=2 * in_channels,
                                          out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False))
        self.relu  = torch.nn.LeakyReLU(negative_slope=0.05, inplace=True)

        self.fft_norm = fft_norm

    def forward(self, x):
        batch,c,height,weight = x.shape # (batch, c, h, w)

        horizontal_ffted = torch.fft.rfft(x, dim=(-1), norm=self.fft_norm) # (batch, c, h, w/2+1)
        horizontal_ffted = torch.cat((horizontal_ffted.real, horizontal_ffted.imag), dim=1)  # (batch, 2c, h, w/2+1)

        vertical_ffted = torch.fft.rfft(x, dim=(-2), norm=self.fft_norm) # (batch,c, h/2+1, w)
        vertical_ffted = torch.cat((vertical_ffted.real, vertical_ffted.imag), dim=1)        # (batch, 2c, h/2+1, w)

        horizontal_ffted = self.conv_layer_hor(horizontal_ffted)  # (batch, c, h, w/2+1)
        horizontal_ffted = self.relu(horizontal_ffted)

        vertical_ffted = self.conv_layer_vert(vertical_ffted)  # (batch, c, h/2+1, w)
        vertical_ffted = self.relu(vertical_ffted)

        horizontal_ffted_real, horizontal_ffted_imag= torch.chunk(horizontal_ffted,2,dim=1)
        horizontal_ffted = torch.complex(horizontal_ffted_real, horizontal_ffted_imag) # (batch, c/2, h, w/2+1)
        horizontal_output = torch.fft.irfft(horizontal_ffted, n=(weight), dim=(-1), norm=self.fft_norm)  # (batch, c/2, h, w)

        vertical_ffted_real, vertical_ffted_imag= torch.chunk(vertical_ffted,2,dim=1)
        vertical_ffted = torch.complex(vertical_ffted_real, vertical_ffted_imag) # (batch,c/2, h/2+1, w)
        vertical_output = torch.fft.irfft(vertical_ffted, n=(height), dim=(-2), norm=self.fft_norm)# (batch, c/2, h, w)

        return torch.cat([horizontal_output,vertical_output],dim=1)


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.stride = stride
        self.conv1 = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=1, bias=False)),
            nn.LeakyReLU(negative_slope=0.05, inplace=True))
        self.fu = DualFourierUnit(
            out_channels // 2, out_channels // 2)
        self.conv2 = weight_norm(torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=1, bias=False))

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        return output


class ShareSepConv(nn.Module):
    "Gated Context Aggregation Network for Image Dehazing and Deraining, WACV, 2019"
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, "kernel size should be odd"
        self.padding = (kernel_size - 1) // 2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1 # center = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(
            inc, 1, self.kernel_size, self.kernel_size
        ).contiguous()
        return F.conv2d(x, expand_weight, None, 1, self.padding, 1, inc)

class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, feats):
        super().__init__()
        self.pre_conv1 = ShareSepConv(1)
        self.pre_conv2 = ShareSepConv(3)
        self.pre_conv4 = ShareSepConv(7)
        self.pre_conv8 = ShareSepConv(15)

        self.conv1 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=1, dilation=1, bias=False,))
        self.conv2 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=2, dilation=2, groups=1, bias=False))
        self.conv4 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=4, dilation=4, groups=1, bias=False))
        self.conv8 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=8, dilation=8, groups=1, bias=False))
        self.conv = nn.Conv2d(feats * 2, feats, 3, 1, padding=1, bias=False)

    def forward(self, x):
        y1 = F.leaky_relu(self.conv1(self.pre_conv1(x)), 0.2)
        y2 = F.leaky_relu(self.conv2(self.pre_conv2(x)), 0.2)
        y4 = F.leaky_relu(self.conv4(self.pre_conv4(x)), 0.2)
        y8 = F.leaky_relu(self.conv8(self.pre_conv8(x)), 0.2)
        y = torch.cat((y1, y2, y4, y8), dim=1)
        y = self.conv(y) + x
        y = F.leaky_relu(y, 0.2)
        return y


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, feats):
        super().__init__()
        self.pre_conv1 = nn.Conv2d(feats, feats, 1, 1, 0, groups=feats)
        self.pre_conv2 = nn.Conv2d(feats, feats, 3, 1, 1, groups=feats)
        self.pre_conv4 = nn.Conv2d(feats, feats, 7, 1, 3, groups=feats)
        self.pre_conv8 = nn.Conv2d(feats, feats, 15, 1, 7, groups=feats)

        self.conv1 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=1, dilation=1, bias=False,))
        self.conv2 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=2, dilation=2, groups=1, bias=False))
        self.conv4 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=4, dilation=4, groups=1, bias=False))
        self.conv8 = weight_norm(nn.Conv2d(feats, feats // 2, 3, 1, padding=8, dilation=8, groups=1, bias=False))
        self.conv = nn.Conv2d(feats * 2, feats, 3, 1, padding=1, bias=False)

    def forward(self, x):
        y1 = F.leaky_relu(self.conv1(self.pre_conv1(x)), 0.2)
        y2 = F.leaky_relu(self.conv2(self.pre_conv2(x)), 0.2)
        y4 = F.leaky_relu(self.conv4(self.pre_conv4(x)), 0.2)
        y8 = F.leaky_relu(self.conv8(self.pre_conv8(x)), 0.2)
        y = torch.cat((y1, y2, y4, y8), dim=1)
        y = self.conv(y) + x
        y = F.leaky_relu(y, 0.2)
        return y


class GateAttn(nn.Module):
    def __init__(self, feats):
        super(GateAttn, self).__init__()
        self.attention_1 = nn.Conv2d(feats*2, 2, 1, 1, padding=0, groups=1, bias=False)
        self.attention_2 = nn.Conv2d(2, 2, kernel_size=(5, 5), stride=1, padding=(2, 2), groups=1, bias=False)


    def forward(self, x):
        h, w = x.size(2),x.size(3)
        attn = self.attention_1(x)
        attn = self.attention_2(attn)
        return attn


class DDB(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1,
                 dilation=1, bias=False, padding_type='reflect', gated=True):
        super(DDB, self).__init__()

        self.convs2s = MultiScaleResidualBlock(feats = in_channels)
        self.convs2f = SpectralTransform(in_channels, out_channels, stride)
        self.convf2s = MultiScaleResidualBlock(feats=in_channels)
        self.convf2f = SpectralTransform(in_channels, out_channels, stride)
        self.gated = gated
        if self.gated:
            self.gate = GateAttn(feats = in_channels)

    def forward(self, x):
        x_s, x_f = x
        if self.gated:
            total_input = torch.cat((x_s,x_f), dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            f2s_gate, s2f_gate = gates.chunk(2, dim=1)
        else:
            f2s_gate, s2f_gate  = 1, 1
        
        out_xs = self.convf2s(x_f) * f2s_gate + self.convs2s(x_s)
        out_xf = self.convs2f(x_s) * s2f_gate + self.convf2f(x_f)
        return out_xs, out_xf



class FFC_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, padding=1, dilation=1, bias=False, activation_layer=nn.ReLU,
                 padding_type='reflect'):
        super(FFC_ACT, self).__init__()
        self.ffc = DDB(in_channels, out_channels, kernel_size, stride, padding, dilation,
                       bias, padding_type=padding_type)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self,  x ):
        x_s, x_f = x
        x_s, x_f = self.ffc((x_s, x_f))
        x_s = self.act(x_s)
        x_f = self.act(x_f)
        return x_s, x_f


class FFCResnetBlock(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.conv1 = FFC_ACT(n_feat, n_feat, kernel_size=3, padding=1, dilation=1,
                                activation_layer=nn.LeakyReLU,
                                padding_type='reflect')
        self.conv2 = FFC_ACT(n_feat, n_feat, kernel_size=3, padding=1, dilation=1,
                                activation_layer=nn.LeakyReLU,
                                padding_type='reflect')

    def forward(self, x):
        x_s, x_f = x
        id_s, id_f = x_s, x_f
        x_s, x_f = self.conv1((x_s, x_f))
        x_s, x_f = self.conv2((x_s, x_f))
        x_s, x_f = id_s + x_s, id_f + x_f
        return x_s, x_f


if __name__ == '__main__':
    model = DualFourierUnit(20,20)
    x = torch.ones(1,20,10,10)
    y = model(x)
    print(y)