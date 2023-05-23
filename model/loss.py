from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """
    Dense detection을 위한 RetinaNet에서 제안된 loss: https://arxiv.org/abs/1708.02002.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean') -> None:
        """
        Args:
            alpha (float): 개구간 (0, 1) 내의 실수값을 가지는 가중치 factor.
                    양성 및 음성 예제간의 균형을 맞추는 역할. 기본값: 0.25.
            gamma (float): 쉬운 예제와 어려운 예제 간의 균형을 맞추는 역할을 하는 modulating factor의 지수.
                    기본값: 2.0
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: 평균 출력 반환.
                    ``'sum'``: 합계 출력 반환. 기본값: none.
        Returns:
            Loss Tensor
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha   # 각 클래스에 대한 가중치
        self.gamma = gamma   # "focus" 매개변수로 어려운 예시에 더 많은 주의를 기울이는 역할
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs (Tensor): (bsz, 30) 사이즈의 Float Tensor.
                    각 예제에 대한 예측.
            targets (Tensor): (30,) 사이즈의 true class 정보. 0부터 29까지의 정수가 담긴 Long Tensor.
                    음성 클래스: 0, 양성 클래스: 1.
        """
        p = torch.sigmoid(inputs)
        targets = F.one_hot(targets, num_classes=30).float()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError(
                f'Invalid Value for arg "reduction": {self.reduction} \n Supported reduction modes: "none", "mean", "sum"'
            )
        return loss

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha   # 각 클래스에 대한 가중치
        self.gamma = gamma   # "focus" 매개변수로 어려운 예시에 더 많은 주의를 기울이는 역할
        self.reduction = reduction

        # alpha가 None인 경우, 모든 클래스에 동일한 가중치 적용
        # alpha가 텐서인 경우, alpha의 각 요소는 해당 클래스의 가중치로 설정
        self.ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, weight=None, reduction: str = 'mean') -> None:
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

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
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

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
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
