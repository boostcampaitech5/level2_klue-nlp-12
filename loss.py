from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha   # 각 클래스에 대한 가중치
        self.gamma = gamma   # "focus" 매개변수로 어려운 예시에 더 많은 주의를 기울이는 역할
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor):
        ce_loss = nn.CrossEntropyLoss(reduction=self.reduction)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, weight=None, reduction: str = 'mean'):
        super(LovaszSoftmaxLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def lovasz_grad(self, true_sorted):
        p = len(true_sorted)
        gts = true_sorted.sum()
        intersection = gts - true_sorted.cumsum(0)
        union = gts + (1 - true_sorted).cumsum(0)
        jaccard = 1 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax(self, log_probs, labels):
        C = log_probs.shape[1]
        losses = []
        for c in range(C):
            fg = (labels == c).float()  # foreground for class c
            if fg.sum() == 0:
                continue
            errors = (fg - log_probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self.lovasz_grad(fg_sorted)))
        return torch.stack(losses)

    def forward(self, inputs: Tensor, targets: Tensor):
        log_probs = F.log_softmax(inputs, dim=1)
        lovasz_loss = self.lovasz_softmax(log_probs, targets)

        if self.reduction == 'mean':
            return torch.mean(lovasz_loss)
        elif self.reduction == 'sum':
            return torch.sum(lovasz_loss)
        else:
            return lovasz_loss


class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean'):
        super(MulticlassDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor):
        # Softmax over the inputs
        inputs = torch.softmax(inputs, dim=1)


        # One-hot encode targets
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])

        # Move targets_one_hot to device of inputs
        targets_one_hot = targets_one_hot.to(inputs.device)

        # Calculate Dice Loss for each class
        dice_loss = 0
        for i in range(inputs.shape[1]):
            intersection = 2 * (inputs[:, i] * targets_one_hot[:, i]).sum()
            union = inputs[:, i].sum() + targets_one_hot[:, i].sum()
            dice_loss += (1 - (intersection + self.smooth) / (union + self.smooth))

        # Average the dice loss for all classes
        dice_loss /= inputs.shape[1]

        if self.reduction == 'mean':
            return dice_loss
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss
