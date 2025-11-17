"""
    This module calculates relevant metrics from ground truth and predicted labels.
"""
from typing import Sequence, Tuple
import torch
import torch.nn as nn
import numpy as np

def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[float, float, float, float, float]:
    """
    This function computes the following metrics based on ground truth labels and predicted labels:
      - Accuracy
      - DICE
      - IoU
      - FPR
      - FNR

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A tuple containing:
        (accuracy, dice, iou, fpr, fnr)
    """

    
    
    """


    TO DO


    """
    NUM_STABILITY_CONST = 1e-5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    dice = 2 * tp / (2 * tp + fp + fn + NUM_STABILITY_CONST)
    iou = tp / (tp + fp + fn + NUM_STABILITY_CONST)
    fpr = fp / (fp + tn + NUM_STABILITY_CONST)
    fnr = fn / (fn + tp + NUM_STABILITY_CONST)
    
    return acc, dice, iou, fpr, fnr

def confusion_matrix(    
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[int, int, int, int]:
    """
    This function computes the following metrics based on ground truth labels and predicted labels:
      - TN
      - FP
      - FN
      - TP

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A tuple containing:
        (tn, fp, fn, tp)
    """

    
    """


    TO DO


    """  
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    return tn, fp, fn, tp



def accuracy_score(    
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """
    This function computes the accuracy score based on ground truth labels and predicted labels.

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A float value of the accuracy score
    """

    
    """


    TO DO


    """    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return acc