import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class FocalLoss(nn.Module):
    """
    Focal Loss to mitigate class imbalance and focus gradient updates
    on subtle early-stage ulcer margins.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DFUClassifier(nn.Module):
    """
    EfficientNet-V2-S backbone with a regularized clinical classification head.
    """

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.3):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = efficientnet_v2_s(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)