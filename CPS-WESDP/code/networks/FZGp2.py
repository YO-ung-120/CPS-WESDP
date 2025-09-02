import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


# 改进后的 Wavelet transform function
def get_wav(in_channels, wavelet_type='haar', pool=True):
    if wavelet_type == 'haar':
        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

        harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
        harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
        harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
        harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

        filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
        filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
        filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
        filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported wavelet type: {wavelet_type}")

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    LH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH


# 改进后的 Wave class
class Wave(nn.Module):
    def __init__(self, base_dim=768, wavelet_type='haar'):
        super().__init__()
        self.dim = base_dim
        self.LL, self.LH, self.HL, self.HH = get_wav(self.dim, wavelet_type=wavelet_type)


# 改进后的 SE_Block class
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16, activation=nn.ReLU(inplace=True), dropout_rate=0.1):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 改进后的 边界注意力模块
class BoundaryAttention(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(BoundaryAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        boundary = self.conv1(x)
        boundary = self.relu1(boundary)
        boundary = self.conv2(boundary)
        boundary = self.relu2(boundary)
        boundary = self.conv3(boundary)
        boundary_mask = self.sigmoid(boundary)
        return x * boundary_mask


# 改进后的 MF class
class MF(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super(MF, self).__init__()
        self.mask_map = nn.Conv2d(3 * channels, 1, 1, 1, 0, bias=True)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(3 * channels, 3 * channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3 * channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(3 * channels, 3 * channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3 * channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.se = SE_Block(3 * channels, 16)
        self.norm = nn.BatchNorm2d(3 * channels)
        self.final_conv = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = x
        x = self.se(x)
        out = self.bottleneck(x)
        out = out * x1
        out = self.norm(out)
        out = self.final_conv(out)
        return out


# 改进后的图像增强模块
class ImageEnhancementModule(nn.Module):
    def __init__(self, base_dim=768):
        super().__init__()
        self.wave = Wave(base_dim)
        self.boundary_attention = BoundaryAttention(base_dim)
        self.mf = MF(base_dim)
        self.weight_z1 = nn.Parameter(torch.ones(1))  # 可学习的权重
        self.weight_z2 = nn.Parameter(torch.ones(1))  # 可学习的权重

        # 添加跳跃连接相关的层
        self.conv_skip = nn.Conv2d(base_dim, base_dim, kernel_size=1, bias=False)
        self.skip_norm = nn.BatchNorm2d(base_dim)

        # 融合多个尺度特征的卷积层
        self.multi_scale_conv = nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1, bias=False)
        self.multi_scale_norm = nn.BatchNorm2d(base_dim)

        # 增加残差连接
        self.residual_conv = nn.Conv2d(base_dim, base_dim, kernel_size=1, bias=False)

    def forward(self, data, skip_data=None):
        """
        输入：
        - data: 输入的图像特征，形状为 (B, L, C)，L为序列长度，C为通道数
        - skip_data: 来自编码器的跳跃连接特征，形状为 (B, C, H, W)
        """
        B, L, C = data.shape
        data = data.permute(0, 2, 1).view(B, C, 7, 7)  # 将输入的形状转换为 (B, C, 7, 7)

        # Wavelet变换
        LL_output = self.wave.LL(data)
        LH_output = self.wave.LH(data)
        HL_output = self.wave.HL(data)
        HH_output = self.wave.HH(data)

        feat_H = torch.cat([LH_output, HL_output, HH_output], dim=1)  # 高频特征
        feat_L = LL_output  # 低频特征

        feat_H = F.relu(nn.BatchNorm2d(feat_H.size(1)).to(data.device)(feat_H))
        feat_L = F.relu(nn.BatchNorm2d(feat_L.size(1)).to(data.device)(feat_L))

        # 进行Boundary Attention和MF模块操作
        z1 = self.boundary_attention(feat_L)  # 注意力模块处理低频特征
        z2 = self.mf(feat_H)  # 特征融合模块处理高频特征

        # 使用可学习的权重进行特征融合
        z = self.weight_z1 * z1 + self.weight_z2 * z2

        # 融合来自编码器的跳跃连接特征
        if skip_data is not None:
            skip_data = self.conv_skip(skip_data)
            skip_data = self.skip_norm(skip_data)
            z = z + skip_data  # 跳跃连接增强特征

        # 融合多尺度特征
        z = torch.cat([z, feat_L], dim=1)  # 融合低频和增强特征
        z = self.multi_scale_conv(z)  # 通过卷积增强多尺度特征
        z = self.multi_scale_norm(z)  # 正则化

        # 增加残差连接
        residual = self.residual_conv(data)
        z = z + residual

        # 最终的图像增强
        enhanced_image = F.interpolate(z, size=(7, 7), mode='bicubic') + data
        enhanced_image = enhanced_image.view(B, C, L).permute(0, 2, 1)
        enhanced_image = enhanced_image.view(B, L, -1)

        return enhanced_image


if __name__ == "__main__":
    process_image = ImageEnhancementModule()
    # Example usage
    data = torch.rand(2, 49, 768)
    # Process the image and get the enhanced image
    enhanced_image = process_image(data)
    print("en shape:", enhanced_image.shape)

