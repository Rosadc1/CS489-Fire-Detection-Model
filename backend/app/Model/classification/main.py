import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from dataset import ClassifierDataset, SubsetDataset
from model import CNNModel
from train import train_one_epoch, validate, test
from config import config_args
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import csv

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
    print(f"Model Version: {args.version}")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # make output directory
    os.makedirs(os.path.join(args.output_dir, args.version, "logs"), exist_ok=True)

    # KFold split
    classification_ds = ClassifierDataset(args)
    image_paths = classification_ds.image_paths
    labels = classification_ds.labels

    skf = StratifiedKFold(args.num_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    folds_pred, folds_labels = [], []

    for fold, (train_val_index, test_index) in enumerate(skf.split(image_paths, labels)):
        # define indices for training and validation
        train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(image_paths[train_val_index], labels[train_val_index], test_size=0.2, random_state=args.seed)

        # split dataset
        test_img_paths, test_ds_labels = image_paths[test_index], labels[test_index]

        # Create datasets for train, val, and test
        train_ds = SubsetDataset(train_img_paths, train_labels, classification_ds.img_transform)
        val_ds   = SubsetDataset(val_img_paths,   val_labels,   classification_ds.img_transform)
        test_ds  = SubsetDataset(test_img_paths,  test_ds_labels,  classification_ds.img_transform)

        # Create dataloaders
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=1)
        val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=1)
        test_dl  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=1)

        # Initialize model, loss, and optimizer
        model = CNNModel(args.num_classes).to(DEVICE)
        
        #loss
        # weights = torch.tensor([1.0, 3.0])
        # criterion = torch.nn.CrossEntropyLoss(weight=weights.to(DEVICE))
        criterion = torch.nn.CrossEntropyLoss()


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        log_records = []

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_prec, val_rec, val_spec, val_f1, val_roc, val_pr, val_tn, val_fp, val_fn, val_tp,  = validate(
                model, val_dl, criterion, DEVICE
            )

            log_records.append({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_spec": val_spec,
                "val_f1": val_f1,
                "val_roc_auc": val_roc,
                "val_pr_auc": val_pr,
                "val_tn": val_tn,
                "val_fp": val_fp,
                "val_fn": val_fn,
                "val_tp": val_tp,
            })

            print(
                f"Epoch [{epoch + 1:02d}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
            )

        # Save per-fold logs
        log_df = pd.DataFrame(log_records)
        log_path = os.path.join(args.output_dir, args.version, "logs", f"fold_{fold + 1}_log.csv")
        log_df.to_csv(log_path, index=False)

        # Final test evaluation
        test_acc, test_prec, test_rec, test_spec, test_f1, test_roc, test_pr, test_tn, test_fp, test_fn, test_tp, test_probs, test_labels = test(model, test_dl, DEVICE)
        fold_results.append({
            "fold": fold + 1,
            "test_acc": test_acc,
            "test_prec": test_prec,
            "test_rec": test_rec,
            "test_spec":test_spec,
            "test_f1": test_f1,
            "test_roc_auc": test_roc,
            "test_pr_auc": test_pr,
            "test_tn": test_tn,
            "test_fp": test_fp,
            "test_fn": test_fn,
            "test_tp": test_tp,
        })

        folds_pred.append({f"fold_{fold + 1}":list(test_probs)})
        folds_labels.append({f"fold_{fold + 1}":list(test_labels)})

        # Save Model and free memory and delete model to prevent leakage over folds
        save_dir = f"./{args.version}_saved_models"
        os.makedirs(save_dir, exist_ok=True)

        # File name
        file_name = f"_fold_{fold+1}.pth"
        file_path = os.path.join(save_dir, file_name)

        # Save model state_dict
        torch.save(model.state_dict(), file_path)
        
        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save aggregated test results
    results_path = os.path.join(args.output_dir, args.version, "testing_results.csv")
    pd.DataFrame(fold_results).to_csv(results_path, index=False)
    plot_cv_roc_pr(folds_pred, folds_labels)
    save_fold_results(folds_pred, folds_labels)

def plot_cv_roc_pr(folds_pred, folds_labels, save_folder="curve graph"):
    """
    Plots ROC and PR curves for k-fold cross-validation results and saves them as images.

    Parameters:
        folds_pred : list of dicts
            Example: [{ "fold_1": [0.1, 0.9, ...] }, { "fold_2": [...] }, ...]
        folds_labels : list of dicts
            Example: [{ "fold_1": [0, 1, ...] }, { "fold_2": [...] }, ...]
        save_folder : str
            Folder where the plots will be saved. Default is "curve graph".
    """
    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # --- ROC Curve ---
    plt.figure(figsize=(8, 6))
    for i, (pred_dict, label_dict) in enumerate(zip(folds_pred, folds_labels)):
        fold_name = list(pred_dict.keys())[0]
        y_score = pred_dict[fold_name]
        y_true = label_dict[fold_name]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{fold_name} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Fold')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save ROC figure
    roc_path = os.path.join(save_folder, "roc_curves_per_fold.png")
    plt.savefig(roc_path, bbox_inches='tight')
    print(f"ROC curve saved to {roc_path}")
    plt.close()  # Close figure to free memory

    # --- PR Curve ---
    plt.figure(figsize=(8, 6))
    for i, (pred_dict, label_dict) in enumerate(zip(folds_pred, folds_labels)):
        fold_name = list(pred_dict.keys())[0]
        y_score = pred_dict[fold_name]
        y_true = label_dict[fold_name]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)

        plt.plot(recall, precision, lw=2, label=f'{fold_name} (AUC={pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves per Fold')
    plt.legend(loc='lower left')
    plt.grid(True)

    # Save PR figure
    pr_path = os.path.join(save_folder, "pr_curves_per_fold.png")
    plt.savefig(pr_path, bbox_inches='tight')
    print(f"PR curve saved to {pr_path}")
    plt.close()

def save_fold_results(folds_pred, folds_labels, filename="fold_results.csv"):
    """
    Saves fold-wise predictions and labels to a CSV.

    CSV format: fold, true_label, pred_prob
    """
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "true_label", "pred_prob"])

        for fold_idx, (pred_dict, label_dict) in enumerate(zip(folds_pred, folds_labels)):
            fold_name = list(pred_dict.keys())[0]
            y_score = pred_dict[fold_name]
            y_true = label_dict[fold_name]

            for t, p in zip(y_true, y_score):
                writer.writerow([fold_name, t, p])

    print(f"Fold predictions and labels saved to {filename}")


if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)
