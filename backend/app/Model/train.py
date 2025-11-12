"""
This module provides utility functions for training, validating, and testing
a PyTorch segmentation model. It includes per-epoch training, validation with
metric computation, and final model evaluation.

Functions:
    train_one_epoch: Train the model for a single epoch.
    validate: Evaluate model performance on the validation set.
    test: Evaluate model performance on the test set and save the prediction along with the input for comparison.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from metrics import calculate_metrics, accuracy_score
from typing import Tuple
from torchvision.utils import save_image
import os

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch on the provided training set.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader providing training batches.
        criterion (nn.Module): Loss function used for optimization.
        optimizer (optim.Optimizer): Optimizer for model parameter updates.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Average training loss and Accuracy for the epoch.
    """

    
    
    """


    TO DO


    """
    model.train()
    total_train_loss = 0.0
    all_preds, all_masks = [], []

    for imgs, masks, _ in dataloader:
        # Move input data and labels to the target device
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        total_train_loss += loss.item() * imgs.size(0)

        probability = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(probability, dim=1)

        all_preds.extend(pred_classes.cpu().numpy().flatten())
        all_masks.extend(masks.cpu().numpy().flatten())

    avg_loss = total_train_loss / len(dataloader.dataset)
    acc = accuracy_score(all_masks, all_preds)    
    
    return avg_loss, acc

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    """
    Validate the model on the provided validation set and compute detailed metrics.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing validation data.
        criterion (nn.Module): Loss function used for evaluation.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, float, float, float, float, float]:
            Average validation loss, Accuracy, DICE, IoU, FPR and FNR.
    """

    
    
    """


    TO DO


    """
    model.eval()
    total_val_loss = 0.0
    all_preds, all_masks = [], []
    for imgs, masks, _ in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)
        total_val_loss += loss.item() * imgs.size(0)

        probability = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(probability, dim=1)

        all_preds.extend(pred_classes.cpu().numpy().flatten())
        all_masks.extend(masks.cpu().numpy().flatten())

    avg_loss = total_val_loss / len(dataloader.dataset)
    acc, dice, iou, fpr, fnr = calculate_metrics(all_masks, all_preds)
    
    return avg_loss, acc, dice, iou, fpr, fnr

@torch.no_grad()
def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_image_dir: str
) -> Tuple[float, float, float, float, float]:
    """
    Test the model on the provided testing set and compute detailed metrics.
    Save the models' prediction with the input and ground truth.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing validation data.
        device (torch.device): Device to perform computations on.
        save_image_dir (str): directory to save the image, mask and predicted mask
                              outputs/model version/image_files/FOLD_n/image_id/[image/mask/pred]
    Returns:
        Tuple[float, float, float, float, float]:
            Accuracy, DICE, IoU, FPR and FNR.
    """

    
    
    """


    TO DO


    """
    model.eval()
    all_preds, all_masks = [], []

    for imgs, masks, image_ids in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        probability = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(probability, dim=1)

        all_preds.extend(pred_classes.cpu().numpy().flatten())
        all_masks.extend(masks.cpu().numpy().flatten())

        for i, imgage_id in enumerate(image_ids):
            fold_dir = os.path.join(save_image_dir, imgage_id)
            os.makedirs(fold_dir, exist_ok=True)

            save_image(imgs[i], os.path.join(fold_dir, "original.png"))
            save_image(masks[i].float(), os.path.join(fold_dir, "mask.png"))
            save_image(pred_classes[i].float(), os.path.join(fold_dir, "pred.png"))

    acc, dice, iou, fpr, fnr = calculate_metrics(all_masks, all_preds)
    
    return acc, dice, iou, fpr, fnr