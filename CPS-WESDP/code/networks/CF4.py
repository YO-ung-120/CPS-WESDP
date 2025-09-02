import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Wavelet transform function (unchanged)
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

# Wave class for wavelet transform (unchanged)
class Wave(nn.Module):
    def __init__(self, base_dim=3):
        super().__init__()
        self.dim = base_dim
        self.LL, self.LH, self.HL, self.HH = get_wav(self.dim)

# SE Block class (unchanged)
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# MF class for multi-scale feature aggregation (updated)
class MF(nn.Module):
    def __init__(self, channels):
        super(MF, self).__init__()
        self.mask_map = nn.Conv2d(channels, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.se = SE_Block(16, 16)
        self.final_conv = nn.Conv2d(16, 768, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        target_h, target_w = h, w
        x_mask = torch.mul(self.mask_map(x).repeat(1, 768, 1, 1), x)
        out = self.bottleneck(x_mask + x)
        out = self.se(out)
        out = self.final_conv(out)
        return out

# Load image function with data augmentation
def load_image(image_path, target_size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0)

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

# Boundary attention (updated)
def boundary_attention(x_hf_agg, dim=2304, kernel_shape=3, L_num=3):
    device = x_hf_agg.device  # 获取输入张量所在的设备

    # Generate Sobel kernels for edge detection
    sobel_kernel_x, sobel_kernel_y = get_kernel_sobel(kernel_size=kernel_shape, in_channels=dim)
    sobel_kernel_x = torch.from_numpy(sobel_kernel_x).float().to(device)
    sobel_kernel_y = torch.from_numpy(sobel_kernel_y).float().to(device)

    boundary_lvl_agg = nn.Sequential(
        nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(L_num, 1, 1), bias=False, groups=dim),
        nn.BatchNorm3d(dim),
        nn.ReLU(inplace=True)
    ).to(device)
    linear_upsample = nn.Linear(dim, 768).to(device)  # Output 768 channels

    G0 = x_hf_agg
    L0 = G0[:, :, None, ...]  # Add time dimension
    L_layers = [L0]

    G1 = F.conv2d(input=x_hf_agg, weight=sobel_kernel_x, padding='same', groups=dim)
    L1 = torch.sub(G0, G1)[:, :, None, ...]
    L_layers += [L1]

    G2 = F.conv2d(input=G1, weight=sobel_kernel_y, padding='same', groups=dim)
    L2 = torch.sub(G1, G2)[:, :, None, ...]
    L_layers += [L2]

    lvl_cat = torch.cat(L_layers, dim=2)
    boundary_feats = boundary_lvl_agg(lvl_cat)
    boundary_att = boundary_feats[:, :, 0, ...].permute(0, 2, 3, 1)
    boundary_att = linear_upsample(boundary_att).permute(0, 3, 1, 2)

    return boundary_att

# Updated process_image function with additional improvements
def process_image(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)  # 将输入数据移动到设备上

    B, L, C = data.shape
    data = data.permute(0, 2, 1)
    # 再将第二个维度拆分为 7 * 7
    data = data.view(B, C, 7, 7)

    wave = Wave(768).to(device)
    LL_output = wave.LL(data)
    LH_output = wave.LH(data)
    HL_output = wave.HL(data)
    HH_output = wave.HH(data)

    feat_H = torch.cat([LH_output, HL_output, HH_output], dim=1)
    feat_L = LL_output

    # Apply normalization and activation to high-frequency and low-frequency features
    feat_H = F.relu(nn.BatchNorm2d(feat_H.size(1)).to(device)(feat_H))
    feat_L = F.relu(nn.BatchNorm2d(feat_L.size(1)).to(device)(feat_L))

    # Apply multi-scale feature aggregation (MF) for better feature fusion
    mf = MF(768).to(device)
    z2 = mf(feat_L)

    # Apply boundary attention for edge refinement
    z1 = boundary_attention(feat_H)

    # Weighted feature fusion
    alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True).to(device))
    z = F.interpolate(alpha * z1 + (1 - alpha) * z2, size=(7, 7), mode='bicubic')

    # Add the result to the original image to enhance details
    enhanced_image = z + data
    enhanced_image = enhanced_image.view(B, C, L).permute(0, 2, 1)
    enhanced_image = enhanced_image.view(B, L, -1)
    return enhanced_image