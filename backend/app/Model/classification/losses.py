"""
    This module calculates DICE Loss and Combo Loss from logits and targets.
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]  (raw outputs from model)
        targets: [B, H, W]    (ionteger class labels)
        """
        
        
        
        """
        
        
        TO DO
        
        
        """
        probability = torch.softmax(logits, dim=1)
        pred = torch.argmax(probability, dim=1)
        intersection = (pred * targets).sum(dim=(1, 2))
        total = (pred + targets ).sum(dim=(1, 2))
        dice_score = (2. * intersection + self.smooth) / (total + self.smooth)
        dice_loss = 1 - dice_score.mean()
        
        
        
        return dice_loss

class ComboLoss(nn.Module): 
    def __init__(self, weight:torch.tensor, alpha=0.3):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.bce =  torch.nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, logits, targets): 
        dice = self.dice(logits, targets)
        bce = self.bce(logits, targets.long())

        return self.alpha * bce + (1 - self.alpha) * dice
    
    