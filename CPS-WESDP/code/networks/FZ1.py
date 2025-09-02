import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Wave(nn.Module):
    def __init__(self, base_dim=768, wavelet_type='haar'):
        super().__init__()
        self.dim = base_dim
        self.LL, self.LH, self.HL, self.HH = self.get_wav(self.dim, wavelet_type)

    def get_wav(self, in_channels, wavelet_type):
        # Haar wavelet implementation
        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

        harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
        harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
        harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
        harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

        filter_LL = torch.from_numpy(harr_wav_LL).float().unsqueeze(0)
        filter_LH = torch.from_numpy(harr_wav_LH).float().unsqueeze(0)
        filter_HL = torch.from_numpy(harr_wav_HL).float().unsqueeze(0)
        filter_HH = torch.from_numpy(harr_wav_HH).float().unsqueeze(0)

        net = nn.Conv2d
        LL = net(in_channels, in_channels, 2, 2, 0, groups=in_channels, bias=False)
        LH = net(in_channels, in_channels, 2, 2, 0, groups=in_channels, bias=False)
        HL = net(in_channels, in_channels, 2, 2, 0, groups=in_channels, bias=False)
        HH = net(in_channels, in_channels, 2, 2, 0, groups=in_channels, bias=False)

        for conv, w in zip([LL, LH, HL, HH], [filter_LL, filter_LH, filter_HL, filter_HH]):
            conv.weight.data = w.expand(in_channels, -1, -1, -1).clone()
            conv.weight.requires_grad = False

        return LL, LH, HL, HH


class EnhancedSE(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BoundaryAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv_layers(x)


class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 2 * channels, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, z1, z2):
        fusion = torch.cat([z1, z2], dim=1)
        att = self.channel_att(fusion)
        att1, att2 = att.chunk(2, dim=1)
        return att1 * z1 + att2 * z2


class MF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(3 * channels, 3 * channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(3 * channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(3 * channels, 3 * channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(3 * channels),
            nn.ReLU(inplace=True),
        )
        self.se = EnhancedSE(3 * channels)
        self.final_conv = nn.Conv2d(3 * channels, channels, 1)

    def forward(self, x):
        identity = x
        x = self.se(x)
        x = self.process(x)
        return self.final_conv(x + identity)


class ImageEnhancementModule(nn.Module):
    def __init__(self, base_dim=768):
        super().__init__()
        self.wave = Wave(base_dim)
        self.boundary_att = BoundaryAttention(base_dim)
        self.mf = MF(base_dim)
        self.fusion = FeatureFusion(base_dim)

        # Normalization layers
        self.bn_h = nn.BatchNorm2d(3 * base_dim)
        self.bn_l = nn.BatchNorm2d(base_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Reshape input
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(B, C, 7, 7)

        # Wavelet decomposition
        LL = self.wave.LL(x)
        LH = self.wave.LH(x)
        HL = self.wave.HL(x)
        HH = self.wave.HH(x)

        # Process high/low frequency components
        feat_H = torch.cat([LH, HL, HH], dim=1)
        feat_L = LL

        # Normalization and activation
        feat_H = F.relu(self.bn_h(feat_H))
        feat_L = F.relu(self.bn_l(feat_L))

        # Feature enhancement
        z1 = self.boundary_att(feat_L)
        z2 = self.mf(feat_H)

        # Adaptive fusion
        z = self.fusion(z1, z2)

        # Spatial recovery and residual connection
        z = F.interpolate(z, size=(7, 7), mode='bicubic', align_corners=False)
        enhanced = z + x

        # Reshape to original format
        return enhanced.view(B, L, C)
'''
process_image = ImageEnhancementModule()
# Example usage
data = torch.rand(2, 49, 768)
# Process the image and get the enhanced image
enhanced_image = process_image(data)
print("en shape:", enhanced_image.shape)
'''
