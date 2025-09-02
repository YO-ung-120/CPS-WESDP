import numpy as np
import torch

import monai.metrics as metric
from scipy.ndimage import zoom
import utils.distributed_utils as utils
import torch.nn.functional as F

''' old
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if gt.sum() == 0:
            hd95 = 0  # 或者给出某个默认值，或者跳过该样本
        else:
            hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0
'''
def calculate_metric_percase(pred, gt):
    # 二值化预测和标签
    pred = torch.where(pred > 0, torch.tensor(1, dtype=pred.dtype, device=pred.device), pred)
    gt = torch.where(gt > 0, torch.tensor(1, dtype=gt.dtype, device=gt.device), gt)

    # 如果预测区域为空，则返回全0
    if pred.sum() > 0:
        # 计算 Dice 系数
        dice = metric.DiceMetric(include_background=False, reduction='mean')(pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0)).item()

        # 计算 IOU
        intersection = torch.sum(pred * gt)  # 交集
        union = torch.sum(pred) + torch.sum(gt) - intersection  # 并集
        iou = (intersection / union) if union > 0 else 0  # IOU计算

        # 计算 HD95
        if gt.sum() == 0:
            hd95 = 0  # 如果标签中没有目标，则HD95设为0
        else:
            hd95 = metric.HausdorffDistanceMetric(percentile=95, include_background=False, reduction='mean')(pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0)).item()

        return iou, dice, hd95
    else:
        return 0, 0, 0  # 如果预测没有目标，返回0,0,0
'''
def calculate_metric_percase(pred, gt):
    # 二值化预测和标签
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    # 如果预测区域为空，则返回全0
    if pred.sum() > 0:
        # 计算 Dice 系数
        dice = metric.binary.dc(pred, gt)

        # 计算 IOU
        intersection = np.sum(pred * gt)  # 交集
        union = np.sum(pred) + np.sum(gt) - intersection  # 并集
        iou = intersection / union if union > 0 else 0  # IOU计算

        # 计算 HD95
        if gt.sum() == 0:
            hd95 = 0  # 如果标签中没有目标，则HD95设为0
        else:
            hd95 = metric.binary.hd95(pred, gt)

        return iou, dice, hd95
    else:
        return 0, 0, 0  # 如果预测没有目标，返回0,0,0,0
'''
'''old

def test_single_volume1(image, label, pt, p_label, net, classes):
    image, label, pt, p_label = image.cuda().detach().float(), label.cuda().detach().float(), pt.cuda().detach().float(), p_label
    point_coords = pt
    coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
    labels_torch = torch.as_tensor(p_label, dtype=torch.int)
    if (len(p_label.shape) == 1):  # only one point prompt
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    pt = (coords_torch, labels_torch)

    net.eval()
    with torch.no_grad():

        imge = net.image_encoder(image)
        se, de = net.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None,
        )
        pred, _ = net.mask_decoder(
            image_embeddings=imge,
            image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=(True),
        )
        outputs1 = F.interpolate(pred, size=(224, 224))
        out = torch.argmax(torch.softmax(outputs1, dim=1), dim=1)
        prediction = out.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
'''
def test_single_volume1(image, label, pt, point_labels, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    
    # Print shapes for debugging
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Prediction shape: {prediction.shape}")
    
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        # Convert single channel to three channels by repeating
        slice = np.stack([slice] * 3, axis=0)  # Shape: (3, H, W)
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()  # Shape: (1, 3, H, W)
        
        # Process point prompts
        point_coords = pt[ind].unsqueeze(0).cuda()  # Shape: (1, N, 2)
        point_labels = point_labels[ind].unsqueeze(0).cuda()  # Shape: (1, N)
        
        # Ensure point_coords has correct shape (B, N, 2)
        if len(point_coords.shape) == 2:
            point_coords = point_coords.unsqueeze(0)
        if len(point_labels.shape) == 1:
            point_labels = point_labels.unsqueeze(0)
            
        # Scale point coordinates to match the input size
        point_coords = point_coords * torch.tensor([patch_size[1]/x, patch_size[0]/y], device=point_coords.device)
        
        # Get image embeddings
        with torch.no_grad():
            image_embeddings = net.image_encoder(input)
            if isinstance(image_embeddings, tuple):
                image_embeddings = image_embeddings[0]
            
            # Get prompt embeddings
            sparse_embeddings, dense_embeddings = net.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            
            # Ensure dense_embeddings has correct shape
            if dense_embeddings is not None:
                dense_embeddings = dense_embeddings.repeat(image_embeddings.shape[0] // dense_embeddings.shape[0], 1, 1, 1)
            
            # Get mask predictions
            pred, _ = net.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            
            # Take only first two channels
            pred = pred[:, :2, :, :]
            
            # Resize to original size
            pred = F.interpolate(pred, size=(x, y), mode='bilinear', align_corners=False)
            pred = torch.argmax(pred, dim=1).squeeze().cpu().detach().numpy()
            
            # Ensure prediction has correct shape
            if len(pred.shape) == 2:  # If pred is (H, W)
                prediction[ind] = pred
            else:
                print(f"Warning: Unexpected prediction shape: {pred.shape}")
                # Try to reshape if possible
                if pred.size == x * y:
                    pred = pred.reshape(x, y)
                    prediction[ind] = pred
                else:
                    raise ValueError(f"Cannot reshape prediction of shape {pred.shape} to match target shape ({x}, {y})")

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_volume2(image, label, net, classes):
    image, label = image.cuda().detach().float(), label.cuda().detach().float()

    net.eval()
    with torch.no_grad():
        x1 = net(image)
        out = torch.argmax(torch.softmax(x1, dim=1), dim=1)

    metric_list = []
    for i in range(1, classes):
        metric_i = calculate_metric_percase(
            out == i, label == i)
        # 确保所有元素都不是 GPU 张量，如果有需要转换
        metric_i = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for val in metric_i]
        metric_list.append(metric_i)
    return metric_list
'''
def test_single_volume2(image, label, net, classes):
    image, label = image.cuda().detach().float(), label.cuda().detach().float()

    net.eval()
    with torch.no_grad():
        x1 = net(image)
        out = torch.argmax(torch.softmax(x1, dim=1), dim=1)
        prediction = out.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
'''
