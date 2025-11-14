"""
This module performs K-fold cross-validation training, validation, and testing
of an image segmentation model using PyTorch. It manages data loading,
training loops, model evaluation, and logging of per-fold metrics.

The training pipeline includes:
    - Reproducible seeding
    - K-fold dataset splitting
    - Model training and validation per epoch
    - Metrics logging and saving for each fold
    - Aggregation of fold-level test results
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from losses import ComboLoss
from dataset import CSVDataset
from model import UNet, ResNetUNet
from train import train_one_epoch, validate, test
from config import config_args
from typing import Any


def run_training(args: Any) -> None:
    """
    Execute K-fold cross-validation training and evaluation.

    Args:
        args: Object containing configuration parameters such as dataset directories and image settings.        
    Returns:
        None. Results and logs are written to disk.
    """
    # Restrict visible GPUs and set the target device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    csv_file_path = os.path.join(args.dataset_dir, args.csv_file)
    df = pd.read_csv(csv_file_path)

    """
        
        
    TO DO
        
        
    """
    model_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(model_dir, exist_ok=True)
    kf = KFold(args.num_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    for fold, (train_val_index, test_index) in enumerate(kf.split(df)):
        # create output folders
        fold_dir = os.path.join(args.output_dir, args.version, f"fold_{fold + 1}")
        os.makedirs(os.path.join(fold_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "images"), exist_ok=True)
        
        # define indices for training and validation
        train_df, val_df = train_test_split(df.iloc[train_val_index], test_size=0.2, random_state=args.seed)  
        # split dataset
        test_df = df.iloc[test_index]   

        train_ds = CSVDataset(args, train_df)
        val_ds = CSVDataset(args, val_df)
        test_ds = CSVDataset(args, test_df)

        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=1, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=1, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=1, pin_memory=True)

        model = UNet(num_classes=args.num_classes).to(DEVICE)
        """
        model = ResNetUNet(args).to(DEVICE)
        """
        loss_weight = torch.tensor([0.2, 0.8]).to(DEVICE)
        criterion = ComboLoss(weight=loss_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        log_records = []
        for epoch in range(args.epochs):
            train_avg_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
            val_avg_loss, val_acc, val_dice, val_iou, val_fpr, val_fnr= validate(model, val_dl, criterion, DEVICE)

            log_records.append({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": train_avg_loss,
                "train_acc": train_acc,
                "val_loss": val_avg_loss,
                "val_acc": val_acc,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_fpr": val_fpr,
                "val_fnr": val_fnr
            })

            print(
                f"Epoch [{epoch + 1:02d}/{args.epochs}] "
                f"Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val Dice: {val_dice:.4f}"
            )

        # Save per-fold logs
        log_df = pd.DataFrame(log_records)
        
        # ./outputs/UNet/fold_x/logs/fold_x_log.csv
        log_path = os.path.join(args.output_dir, args.version, f"fold_{fold + 1}","logs", f"fold_{fold + 1}_log.csv")
        log_df.to_csv(log_path, index=False)

        test_path = os.path.join(args.output_dir, args.version, f"fold_{fold + 1}", "images")

        test_acc, test_dice, test_iou, test_fpr, test_fnr = test(model, test_dl, DEVICE, test_path)

        fold_results.append({
            "fold": fold + 1,
            "test_acc": test_acc,
            "test_dice": test_dice,
            "test_iou": test_iou,
            "test_fpr": test_fpr,
            "test_fnr": test_fnr
        })

        del model
        torch.cuda.empty_cache()
        
    # Save aggregated test results
    results_path = os.path.join(args.output_dir, args.version, "testing_results.csv")
    pd.DataFrame(fold_results).to_csv(results_path, index=False)
            

if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)
