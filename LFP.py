import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class ConvDWT(nn.Module):
    """Discrete Wavelet Transform: (B,C,H,W) -> (B,4C,H/2,W/2)"""
    def __init__(self, wave='haar', mode='zero'):
        super(ConvDWT, self).__init__()
        self.dwt_forward = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            if x.dtype != torch.float32:
                x = x.float()
            Yl, Yh = self.dwt_forward(x)

        b, c, h, w = x.shape
        Yh = Yh[0].transpose(1, 2).reshape(Yh[0].shape[0], -1, Yh[0].shape[3], Yh[0].shape[4])

        output = torch.cat((Yl, Yh), dim=1)
        output = F.interpolate(output, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
        return output


class ConvIDWT(nn.Module):
    """Inverse Discrete Wavelet Transform"""
    def __init__(self, wave='haar', mode='zero'):
        super(ConvIDWT, self).__init__()
        self.dwt_inverse = DWTInverse(wave=wave, mode=mode)

    def forward(self, low_freqs, high_freqs):
        B, C, H, W = low_freqs.shape
        high_freqs = high_freqs.reshape(B, C, 3, H, W)

        with torch.cuda.amp.autocast(enabled=False):
            reconstruction = self.dwt_inverse((low_freqs, [high_freqs.float()]))
        reconstruction = F.interpolate(reconstruction, size=(2 * H, 2 * W), mode='bilinear', align_corners=False)

        return reconstruction


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7, bn_before_sigmoid=False):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1  # 修复：添加空格
        self.bn_before_sigmoid = bn_before_sigmoid
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        if bn_before_sigmoid:
            self.bn = nn.BatchNorm2d(1)
            self.bn.bias.data.fill_(0)
            self.bn.bias.requires_grad = False

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        if self.bn_before_sigmoid:
            x = self.bn(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class LearnableGaussianFilterBank(nn.Module):
    """Learnable Gaussian Filter Bank"""
    def __init__(self, kernel_size, num_filters, num_channels):
        super(LearnableGaussianFilterBank, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.C = num_channels
        self.padding = kernel_size // 2

        self.sigmas = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0])) for _ in range(num_filters)
        ])

    def forward(self, x):
        weights = [
            self._gaussian_kernel(self.kernel_size, sigma).repeat(self.C, 1, 1, 1)
            for sigma in self.sigmas
        ]
        filtered_outputs = [
            F.conv2d(
                F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='replicate'),
                weight.to(x.device),
                groups=self.C
            ) for weight in weights
        ]
        return torch.cat(filtered_outputs, dim=1)

    def _gaussian_kernel(self, kernel_size, sigma):
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[:, :, i, j] = torch.exp(
                    -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)
                )
        return kernel / kernel.sum()


class LFP(nn.Module):
    """Low-Frequency Guided Feature Purification Module"""
    def __init__(self, in_channels, wave='haar', mode='zero', with_gauss=True, gauss_gate=0.5):
        super(LFP, self).__init__()
        self.dwt = ConvDWT(wave=wave, mode=mode)
        self.idwt = ConvIDWT(wave=wave, mode=mode)
        self.with_gauss = with_gauss
        self.gauss_gate = gauss_gate

        self.attention = SpatialAttention()
        if self.with_gauss:
            self.gaussian_filter = LearnableGaussianFilterBank(
                kernel_size=3,
                num_filters=1,
                num_channels=3 * in_channels
            )

    def forward(self, x):
        B, C, H, W = x.shape
        dwt_out = self.dwt(x)  # (B, 4C, H/2, W/2)

        LL = dwt_out[:, :C, :, :]      # 低频分量
        Yh = dwt_out[:, C:, :, :]      # 高频分量

        # 低频引导高频调制
        att = self.attention(LL)
        Yh = Yh * att

        # 高斯滤波处理高频噪声
        if self.with_gauss:
            Yh_blurred = self.gaussian_filter(Yh)
            mask = (Yh.abs() < self.gauss_gate).float()
            Yh = Yh * (1 - mask) + Yh_blurred * mask

        x_rec = self.idwt(LL, Yh)
        return x_rec


# 为了兼容原代码，添加别名
wav_Enhance = LFP