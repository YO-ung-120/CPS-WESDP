#GPT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# 定义Sobel滤波器
def get_kernel_sobel(kernel_size, in_channels=64):
    """生成用于边缘检测的Sobel滤波器"""
    kernel_x = cv2.getDerivKernels(1, 0, kernel_size, 1, normalize=True)
    kernel_y = cv2.getDerivKernels(0, 1, kernel_size, 1, normalize=True)

    sobel_kernel_x = np.outer(kernel_x[0], kernel_x[1])
    sobel_kernel_y = np.outer(kernel_y[0], kernel_y[1])

    # 扩展为多个输入通道
    sobel_kernel_x = np.repeat(sobel_kernel_x[None, ...], in_channels, axis=0)[:, None, ...]
    sobel_kernel_y = np.repeat(sobel_kernel_y[None, ...], in_channels, axis=0)[:, None, ...]

    return sobel_kernel_x, sobel_kernel_y

# Boundary Attention模块
def boundary_attention(x_hf_agg, dim=2304, kernel_shape=3, L_num=3):
    """
    Perform multi-scale processing with Sobel filters, followed by a 3D convolution and a linear upsampling.

    Args:
    - x_hf_agg (torch.Tensor): Input tensor with shape [B, C, H, W]
    - dim (int): Number of input channels (default: 9)
    - kernel_shape (int): Kernel size for Sobel filters (default: 3)
    - L_num (int): Number of levels in the pyramid (default: 3)

    Returns:
    - torch.Tensor: Processed tensor with shape [B, 3, H, W]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate Sobel kernels for edge detection
    sobel_kernel_x, sobel_kernel_y = get_kernel_sobel(kernel_size=kernel_shape, in_channels=dim)

    sobel_kernel_x = torch.from_numpy(sobel_kernel_x).float().to(device)
    sobel_kernel_y = torch.from_numpy(sobel_kernel_y).float().to(device)

    # Define the 3D convolution layer and linear upsampling layer
    boundary_lvl_agg = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(L_num, 1, 1), bias=False,
                                 groups=dim).to(device)
    linear_upsample = nn.Linear(dim, 768).to(device)  # Output 3 channels

    # Multi-scale processing
    G0 = x_hf_agg
    L0 = G0[:, :, None, ...]  # Add time dimension: [B, C, 1, H, W]
    L_layers = [L0]

    # First level of Sobel filtering
    G1 = F.conv2d(input=x_hf_agg, weight=sobel_kernel_x, padding='same', groups=dim)
    L1 = torch.sub(G0, G1)[:, :, None, ...]
    L_layers += [L1]

    # Second level of Sobel filtering
    G2 = F.conv2d(input=G1, weight=sobel_kernel_y, padding='same', groups=dim)
    L2 = torch.sub(G1, G2)[:, :, None, ...]
    L_layers += [L2]

    # Concatenate all levels
    lvl_cat = torch.cat(L_layers, dim=2)  # [B, C, L_num, H, W]

    # 3D convolution on the time dimension
    boundary_feats = boundary_lvl_agg(lvl_cat)  # [B, C, 1, H, W]

    # Adjust dimensions and apply linear transformation to get the final output
    boundary_att = boundary_feats[:, :, 0, ...].permute(0, 2, 3, 1)  # [B, H, W, C]
    boundary_att = linear_upsample(boundary_att).permute(0, 3, 1, 2)  # [B, 3, H, W]

    return boundary_att

# Wavelet变换模块
def get_wav(in_channels, pool=True):
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

# Wave class for wavelet transform
class Wave(nn.Module):
    def __init__(self, base_dim=768):
        super().__init__()
        self.dim = base_dim
        self.LL, self.LH, self.HL, self.HH = get_wav(self.dim)

# 处理图像函数
def process_image(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)  # 将数据移动到设备上
    B, L, C = data.shape
    data = data.permute(0, 2, 1)
    # 再将第二个维度拆分为 7 * 7
    data = data.view(B, C, 7, 7)
    # Perform the wavelet transform
    wave = Wave().to(device)  # 将模型移动到设备上
    LL_output = wave.LL(data)
    LH_output = wave.LH(data)
    HL_output = wave.HL(data)
    HH_output = wave.HH(data)

    feat_H = torch.cat([LH_output, HL_output, HH_output], dim=1)
    feat_L = LL_output

    # Add normalization and activation functions
    feat_H = F.relu(nn.BatchNorm2d(feat_H.size(1)).to(device)(feat_H))
    feat_L = F.relu(nn.BatchNorm2d(feat_L.size(1)).to(device)(feat_L))

    # Process the image using MF module
    z1 = feat_L
    z2 = boundary_attention(feat_H)
    z = F.interpolate(z1 + z2, size=(7, 7), mode='bicubic')

    # Add the result to the original image to get the enhanced image
    enhanced_image = z + data
    enhanced_image = enhanced_image.view(B, C, L).permute(0, 2, 1)
    enhanced_image = enhanced_image.view(B, L, -1)
    return enhanced_image

