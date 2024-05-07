import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .layers import FFCResnetBlock, SmoothDilatedResidualBlock

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def color_affine(rgb, trans_mat):
    c, h, w = rgb.shape
    rgb = rgb.permute(1, 2, 0)
    rgb_vec = torch.reshape(rgb, [-1, c, 1])
    trans_mat = trans_mat.permute(1, 2, 0)
    trans_mat = torch.reshape(trans_mat, [-1, c, c])
    restored = torch.bmm(trans_mat, rgb_vec)
    restored = restored.squeeze()
    restored = torch.reshape(restored, [h, w, c])
    restored = restored.permute(2, 0, 1)
    return restored

def unpixel_shuffle(feature, r: int = 1):
    b, c, h, w = feature.shape
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    feature_view = feature.contiguous().view(b, c, out_h, r, out_w, r)
    feature_prime = (
        feature_view.permute(0, 1, 3, 5, 2, 4)
        .contiguous()
        .view(b, out_channel, out_h, out_w))
    return feature_prime


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))
    def forward(self, x):
        x = self.down(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))
    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats):
        super(Encoder, self).__init__()
        self.encoder_level1 = [SmoothDilatedResidualBlock(n_feat) for _ in range(2)]
        self.encoder_level2 = [SmoothDilatedResidualBlock(n_feat + scale_unetfeats) for _ in range(2)]
        self.encoder_level3 = [SmoothDilatedResidualBlock(n_feat + (scale_unetfeats * 2)) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, scale_feats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [SmoothDilatedResidualBlock(n_feat) for _ in range(2)]
        self.decoder_level2 = [SmoothDilatedResidualBlock(n_feat + scale_feats) for _ in range(2)]
        self.decoder_level3 = [SmoothDilatedResidualBlock(n_feat + (scale_feats * 2)) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = SmoothDilatedResidualBlock(n_feat)
        self.skip_attn2 = SmoothDilatedResidualBlock(n_feat + scale_feats)

        self.up21 = SkipUpSample(n_feat, scale_feats)
        self.up32 = SkipUpSample(n_feat + scale_feats, scale_feats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1

class OriginalResolutionBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, num_blk):
        super(OriginalResolutionBlock, self).__init__()
        modules_body = [FFCResnetBlock(n_feat) for _ in range(num_blk)]
        self.body = nn.Sequential(*modules_body)
        self.body_tail = conv(n_feat*2, n_feat*2, kernel_size)

    def forward(self, x):
        x_s,x_f = x
        res_s,res_f = self.body((x_s, x_f))
        res_s,res_f = torch.chunk(self.body_tail(torch.cat((res_s,res_f),dim=1)),2,dim=1)
        res_s = x_s + res_s
        res_f = x_f + res_f
        return res_s, res_f

class ORSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, num_blk):
        super(ORSNet, self).__init__()

        self.orb1 = OriginalResolutionBlock(n_feat, kernel_size, num_blk)
        self.orb2 = OriginalResolutionBlock(n_feat, kernel_size, num_blk)
        self.orb3 = OriginalResolutionBlock(n_feat, kernel_size, num_blk)

    def forward(self, x):
        x_s,x_f = self.orb1((x,x))
        x_s,x_f  = self.orb2((x_s,x_f))
        x_s,x_f  = self.orb3((x_s,x_f))
        return torch.cat((x_s,x_f),dim=1)   



class UDCDDDBnet(nn.Module):
    def __init__(self, n_feat_high=96, n_feat_low=24, scale_feats=8, num_blk=2, kernel_size=3, bias=True):
        super(UDCDDDBnet, self).__init__()
        self.shallow_feat = nn.Sequential(conv(48, n_feat_high, kernel_size, bias=bias))
        self.orsnet = ORSNet(n_feat_high, kernel_size, num_blk)
        self.tail_beta = conv(n_feat_high*2, 48, kernel_size, bias=bias)
        self.tail_alpha = conv(n_feat_high*2, 48, kernel_size, bias=bias)

        self.shallow_sub = nn.Sequential(conv(3, n_feat_low, kernel_size, bias=bias), SmoothDilatedResidualBlock(n_feat_low))
        self.encoder_sub = Encoder(n_feat_low, 8)
        self.decoder_sub = Decoder(n_feat_low, 8)
        self.tail_sub = conv(n_feat_low, 12, kernel_size, bias=bias)

    def forward(self, x):
        n,c,h,w = x.size()
        x_fre = unpixel_shuffle(x, 4)
        x_fre = self.shallow_feat(x_fre) 
        x_fre = self.orsnet(x_fre)

        x_alpha = self.tail_alpha(x_fre)
        x_beta = self.tail_beta(x_fre)
        x_alpha = F.pixel_shuffle(x_alpha, 4)
        x_beta = F.pixel_shuffle(x_beta, 4)
        x_fre = x * x_alpha + x_beta

        x_low = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=True)
        _, _, h, w = x_low.shape
        x_low = self.shallow_sub(x_low)
        x_low = self.encoder_sub(x_low)
        x_low = self.decoder_sub(x_low)
        x_low = self.tail_sub(x_low)
        x_low = F.interpolate(x_low, scale_factor=4, mode="bilinear", align_corners=True)
        gamma, delta = torch.split(x_low, 9, dim=1)

        x_low_for_test = []
        for i, data in enumerate(zip(x, gamma)):
            x_low_for_test.append(color_affine(data[0], data[1]))
        x_low_for_test = torch.stack(x_low_for_test, dim=0) + delta

        out_final = []
        for i, data in enumerate(zip(x_fre, gamma)):
            out_final.append(color_affine(data[0], data[1]))
        out_final = torch.stack(out_final, dim=0) + delta
        return x_fre, x_low_for_test, out_final


if __name__ == '__main__':
    net = UDCDDDBnet().cuda()
    print_network(net)
    x1 = torch.randn(8, 3, 512, 512).cuda()
    _, _, output = net(x1)

    # 估计模型所需的运行内存
    memory_estimate = torch.cuda.memory_summary(torch.device("cuda"))
    print(memory_estimate)

    # 打印模型的参数量
    print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")
