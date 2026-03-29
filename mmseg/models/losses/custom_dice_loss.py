import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class CustomDiceLoss(nn.Module):
    """
    为文档图像篡改检测量身定制的 Dice Loss
    """
    def __init__(self, use_sigmoid=False, loss_weight=1.0, loss_name='loss_dice'):
        super(CustomDiceLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        # pred 形状: [Batch, Num_Classes, H, W] (未经过 softmax 的 logits)
        # target 形状: [Batch, H, W] (真实的类别索引，0 是背景，1 是篡改)

        # 1. 转换为概率值 [0~1]
        pred = torch.softmax(pred, dim=1)

        # 2. 将 target 转为 One-hot 编码，并对齐维度
        num_classes = pred.shape[1]
        valid_mask = (target != ignore_index).float()
        safe_target = target.clone()
        safe_target[target == ignore_index] = 0
        target_one_hot = torch.nn.functional.one_hot(
            safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 3. 核心：只提取篡改类别（索引 1）进行计算，彻底无视背景（索引 0）
        pred_tamper = pred[:, 1, :, :]
        target_tamper = target_one_hot[:, 1, :, :]
        if weight is not None:
            valid_mask = valid_mask * weight.float()

        smooth = 1e-5  # 平滑项，防止分母为 0

        # 计算交集与并集
        intersect = torch.sum(
            pred_tamper * target_tamper * valid_mask, dim=(1, 2))
        union = torch.sum(pred_tamper * valid_mask, dim=(1, 2)) + torch.sum(
            target_tamper * valid_mask, dim=(1, 2))

        # 计算 Dice 分数
        dice_score = (2. * intersect + smooth) / (union + smooth)

        # Loss 转换为需要最小化的标量
        loss = 1. - dice_score

        return loss.mean() * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name
