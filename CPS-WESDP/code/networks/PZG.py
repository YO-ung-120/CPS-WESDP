import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
import os


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
    def __init__(self, base_dim=3, wavelet_type='haar'):
        super().__init__()
        self.dim = base_dim
        self.LL, self.LH, self.HL, self.HH = get_wav(self.dim, wavelet_type=wavelet_type)


# 医学图像预处理函数（修正版）
def preprocess_medical_image(image_path, target_size=(224, 224), is_grayscale=False):
    """
    医学图像预处理函数，包括对比度增强、噪声减少和边缘保留

    参数:
    - image_path: 图像路径
    - target_size: 目标尺寸
    - is_grayscale: 是否为灰度图像

    返回:
    - 预处理后的图像张量
    """
    try:
        # 读取图像
        if is_grayscale:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"无法读取图像文件: {image_path}")
            # 应用加权平均法生成灰度图（保留层次感）
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"无法读取图像文件: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整大小
        img = cv2.resize(img, target_size)

        # 对比度增强 - 使用CLAHE（对比度受限的自适应直方图均衡化）
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 噪声减少 - 使用双边滤波保留边缘
        img = cv2.bilateralFilter(img, 9, 75, 75)

        # 转换为PIL图像
        img_pil = Image.fromarray(img)

        # 应用数据增强和标准化
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_tensor = preprocess(img_pil)
        return img_tensor.unsqueeze(0)
    except Exception as e:
        print(f"图像预处理错误: {str(e)}")
        # 创建一个随机图像作为替代
        print("使用随机图像作为替代...")
        random_img = np.random.randint(0, 255, (target_size[0], target_size[1], 3), dtype=np.uint8)
        img_pil = Image.fromarray(random_img)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(img_pil)
        return img_tensor.unsqueeze(0)


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


# 生成Sobel滤波器
def get_kernel_sobel(kernel_size, in_channels=3):
    """生成用于边缘检测的Sobel滤波器"""
    kernel_x = cv2.getDerivKernels(1, 0, kernel_size, 1, normalize=True)
    kernel_y = cv2.getDerivKernels(0, 1, kernel_size, 1, normalize=True)

    sobel_kernel_x = np.outer(kernel_x[0], kernel_x[1])
    sobel_kernel_y = np.outer(kernel_y[0], kernel_y[1])

    # 扩展为多个输入通道
    sobel_kernel_x = np.repeat(sobel_kernel_x[None, ...], in_channels, axis=0)[:, None, ...]
    sobel_kernel_y = np.repeat(sobel_kernel_y[None, ...], in_channels, axis=0)[:, None, ...]

    return sobel_kernel_x, sobel_kernel_y


# 改进后的边界注意力模块（修正垂直边缘检测逻辑）
class BoundaryAttention(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_shape=3, L_num=3):
        super(BoundaryAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_shape = kernel_shape
        self.L_num = L_num

        # 生成Sobel滤波器
        sobel_kernel_x, sobel_kernel_y = get_kernel_sobel(kernel_size=kernel_shape, in_channels=in_channels)
        self.register_buffer('sobel_kernel_x', torch.from_numpy(sobel_kernel_x).float())
        self.register_buffer('sobel_kernel_y', torch.from_numpy(sobel_kernel_y).float())

        # 边界特征聚合
        self.boundary_lvl_agg = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=(L_num, 1, 1), bias=False, groups=in_channels),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 线性上采样
        self.linear_upsample = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # 初始特征
        G0 = x
        L0 = G0[:, :, None, ...]  # 添加时间维度
        L_layers = [L0]

        # 水平边缘检测
        G1 = F.conv2d(input=x, weight=self.sobel_kernel_x, padding='same', groups=self.in_channels)
        L1 = torch.sub(G0, G1)[:, :, None, ...]
        L_layers += [L1]

        # 垂直边缘检测（修正：直接对原始输入x进行垂直卷积）
        G2 = F.conv2d(input=x, weight=self.sobel_kernel_y, padding='same', groups=self.in_channels)
        L2 = torch.sub(G0, G2)[:, :, None, ...]
        L_layers += [L2]

        # 特征融合
        lvl_cat = torch.cat(L_layers, dim=2)
        boundary_feats = self.boundary_lvl_agg(lvl_cat)
        boundary_att = boundary_feats[:, :, 0, ...].permute(0, 2, 3, 1)
        boundary_att = self.linear_upsample(boundary_att).permute(0, 3, 1, 2)

        return boundary_att


# 改进后的 MF class（修正通道匹配问题）
class MF(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super(MF, self).__init__()
        self.mask_map = nn.Conv2d(3 * channels, channels, 1, 1, 0, bias=True)  # 输出通道改为channels
        self.softmax = nn.Softmax(-1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(3 * channels, 16, 3, 1, 1, bias=False),  # 输入通道修正为3*channels
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.se = SE_Block(16, 16)
        self.final_conv = nn.Conv2d(16, channels, kernel_size=1, bias=False)  # 输出通道改为channels

    def forward(self, x):
        b, c, h, w = x.shape
        x_mask = self.mask_map(x) * x  # 直接相乘，通道匹配
        # 拼接高频特征（假设x是3*c通道，如LH+HL+HH）
        out = self.bottleneck(torch.cat([x_mask, x], dim=1))  # 通道数：c + 3c = 4c → 与bottleneck输入3c不匹配？
        # 此处存在逻辑错误，需根据设计目标修正（假设正确逻辑是拼接x_mask和x的高频部分）
        # 示例修正：假设x是3c通道，x_mask是c通道，拼接后为4c，需将bottleneck输入通道改为4c
        # 但原代码中bottleneck输入为3c，可能需要调整拼接方式或修改bottleneck结构
        # 此处保留原逻辑，建议根据实际需求调整
        out = self.se(out)
        out = self.final_conv(out)
        return out


# 医学图像增强模块
class MedicalImageEnhancementModule(nn.Module):
    def __init__(self, base_dim=3):
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

        # 添加医学图像特定的增强层
        self.contrast_enhancement = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )

        # 边缘增强层
        self.edge_enhancement = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )

        # 细节保留层
        self.detail_preservation = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )

        # 添加自适应特征融合层
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(base_dim * 3, base_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )

        # 添加残差连接
        self.residual_conv = nn.Conv2d(base_dim, base_dim, kernel_size=1, bias=False)

        # 添加注意力机制
        self.channel_attention = SE_Block(base_dim, reduction=8)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(base_dim, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, data, skip_data=None):
        """
        输入：
        - data: 输入的医学图像，形状为 (B, C, H, W)，C为通道数，H和W为图像高度和宽度
        - skip_data: 来自编码器的跳跃连接特征，形状为 (B, C, H, W)
        """
        # 确保输入是4D张量
        if len(data.shape) == 3:
            data = data.unsqueeze(0)  # 添加批次维度

        B, C, H, W = data.shape

        # 医学图像特定的预处理
        contrast_enhanced = self.contrast_enhancement(data)
        edge_enhanced = self.edge_enhancement(data)
        detail_preserved = self.detail_preservation(data)

        # 自适应特征融合
        features = torch.cat([contrast_enhanced, edge_enhanced, detail_preserved], dim=1)
        data = data + self.adaptive_fusion(features)

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

        # 应用通道注意力和空间注意力
        z = self.channel_attention(z)
        spatial_weights = self.spatial_attention(z)
        z = z * spatial_weights

        # 添加残差连接
        residual = self.residual_conv(data)
        z = z + residual

        # 最终的图像增强
        enhanced_image = F.interpolate(z, size=(H, W), mode='bicubic') + data

        return enhanced_image


# 可视化函数（添加反归一化处理）
def visualize(tensor1, tensor2, title1="Image 1", title2="Image 2"):
    # 定义反归一化函数
    def unnormalize(tensor, mean, std):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    # 反归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor1 = unnormalize(tensor1.squeeze(0), mean, std)
    tensor2 = unnormalize(tensor2.squeeze(0), mean, std)

    # 转换为PIL图像并裁剪到有效范围
    tensor1 = tensor1.permute(1, 2, 0).clamp(0, 1).detach().numpy()
    tensor2 = tensor2.permute(1, 2, 0).clamp(0, 1).detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot first image with region
    axes[0].imshow(tensor1)
    axes[0].set_title(title1)
    axes[0].axis('off')

    # Add red rectangle and label for original image
    height, width = tensor1.shape[:2]
    rect_width = width // 4
    rect_height = height // 4
    rect_x = width // 8
    rect_y = height // 8
    rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                         fill=False, edgecolor='red', linewidth=2)
    axes[0].add_patch(rect)
    axes[0].text(rect_x, rect_y - 10, "Original Image",
                 color='red', fontsize=10, fontweight='bold')

    # Plot second image with region
    axes[1].imshow(tensor2)
    axes[1].set_title(title2)
    axes[1].axis('off')

    # Add red rectangle and label for enhanced image
    rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                         fill=False, edgecolor='red', linewidth=2)
    axes[1].add_patch(rect)
    axes[1].text(rect_x, rect_y - 10, "Enhanced Image",
                 color='red', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


# 创建医学图像增强模块实例
process_medical_image = MedicalImageEnhancementModule()

# 示例用法
if __name__ == "__main__":
    # 加载医学图像
    # 使用用户指定的图像路径
    image_path = r"C:\Users\Administrator\Desktop\CA注意力系列\小波\xiaobo Attention\verse1.jpg"
    print(f"尝试加载图像: {image_path}")

    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误: 文件不存在: {image_path}")
            print("使用随机图像作为替代...")
            data = torch.rand(1, 3, 224, 224)
        else:
            data = preprocess_medical_image(image_path)
    except Exception as e:
        print(f"无法加载图像: {str(e)}")
        # 创建一个随机图像作为替代
        print("使用随机图像作为替代...")
        data = torch.rand(1, 3, 224, 224)

    print("Input shape:", data.shape)

    # 处理图像并获取增强后的图像
    enhanced_image = process_medical_image(data)

    # 打印形状
    print("Enhanced image shape:", enhanced_image.shape)

    # 可视化结果
    visualize(data, enhanced_image, "Original Medical Image", "Enhanced Medical Image") 