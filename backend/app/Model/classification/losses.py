"""
    This module calculates DICE Loss and Combo Loss from logits and targets.
    
"""
import torch


# Helper class for focal loss, wraps around sigmoid_focal_loss so that there is no need to manually change
# the criterion parameter to not use nn.module
class focal_loss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        pt = pt.clamp(min=1e-6, max=1.0)
        alphat = self.alpha * targets + (1-self.alpha) * (1-targets)

        fl = -alphat * ((1 - pt) ** self.gamma) * torch.log(pt)
        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        return fl
    