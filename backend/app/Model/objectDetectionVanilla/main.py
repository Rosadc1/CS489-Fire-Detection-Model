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
import torch
import numpy as np
from pathlib import Path
from config import config_args
from collections import Counter
from ultralytics import YOLO
from sklearn.model_selection import KFold
from typing import Any
import datetime
import pandas as pd
import yaml

def run_training(args: Any) -> None:
    # Restrict visible GPUs and set the target device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create output directory 
    os.makedirs(os.path.join(args.output_dir, args.version, "logs"), exist_ok=True)

    # Prepare labels dataframe
    dataset_dir = Path(args.dataset_dir)
    labels = sorted((dataset_dir / "labels").rglob("*.txt"))

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'r') as y:
        classes = yaml.safe_load(y)['names']
    class_index = list(range(len(classes)))
    index = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=class_index, index=index)

    # Populate labels dataframe to get correct number of instances per class
    for label in labels:
        lbl_counter = Counter()
        with open(label) as lf:
            lines = lf.readlines()

        for line in lines:
            lbl_counter[int(line.split(" ", 1)[0])] += 1
        row = [lbl_counter.get(cls, 0) for cls in class_index]
        labels_df.loc[label.stem] = row

    # K-Fold Cross Validation found from: https://docs.ultralytics.com/guides/kfold-cross-validation/#k-fold-dataset-split
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    folds = [f"split_{n}" for n in range(1, args.num_folds + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=class_index)

    for i, (train_index, val_index) in enumerate(kf.split(labels_df)):
        folds_df.loc[labels_df.iloc[train_index].index, f"split_{i + 1}"] = "train"
        folds_df.loc[labels_df.iloc[val_index].index, f"split_{i + 1}"] = "val"

        train_totals = labels_df.iloc[train_index].sum()
        val_totals = labels_df.iloc[val_index].sum()

        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{i + 1}"] = ratio
    
    # create list of all images
    images = [] 
    images.extend(sorted((dataset_dir / "images").rglob("*.jpg")))

    # create folder for day of cross fold validation
    save_path = Path(dataset_dir / f"{datetime.date.today().isoformat()}_{args.num_folds}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []
    
    for split in folds_df.columns:
        # create fold_i directory within day folder for cross fold validation
        kfold_split_dir = save_path / split
        kfold_split_dir.mkdir(parents=True, exist_ok=True)

        (kfold_split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (kfold_split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (kfold_split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (kfold_split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        dataset_yaml = kfold_split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": kfold_split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )
    import shutil
    from tqdm import tqdm

    for image, label in tqdm(zip(images, labels), total=len(images), desc="Copying files"):
        if "checkpoint" in image.stem:
            continue
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"
            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)
    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

    current_path = Path(__file__).parent
    weights_path = current_path / args.weights

    folds_results = []

    DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
    for k, dataset_yaml in enumerate(ds_yamls):
        print(f"\n--- Starting Fold {k + 1}/{args.num_folds} ---\n")
        model = YOLO(weights_path, task="detect")
        # Train the model
        fold_save_dir = Path(args.output_dir) / args.version / f"fold_{k + 1}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)

        folds_train_result = model.train(
            data=dataset_yaml.as_posix(),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.image_size,
            lr0=args.lr,
            patience=args.patience,
            seed=args.seed,
            device=DEVICE_STR,
            save=True,
            project=fold_save_dir.parent.as_posix(),
            name=fold_save_dir.name,
            exist_ok=True,
            verbose=False
        )


        # Validate the model
        fold_val_result = model.val(
            data=dataset_yaml.as_posix(),
            batch=args.batch,
            imgsz=args.image_size,
            seed=args.seed,
            device=DEVICE_STR,
            verbose=False
        )
        # Save fold results
        fold_metrics = {}
        print("folds train result")
        print(folds_train_result)
        print("Fold val result")
        print(fold_val_result)
        
        # Prefix train metrics
        for key, value in folds_train_result.results_dict().items():
            fold_metrics[f"train_{key}"] = value

        # Prefix validation metrics
        for key, value in fold_val_result.results_dict().items():
            fold_metrics[f"val_{key}"] = value
        
        fold_metrics['fold'] = k + 1
        folds_results.append(fold_metrics)


    # Save all folds results
    folds_results_df = pd.DataFrame(folds_results)
    folds_results_df.to_csv(Path(args.output_dir) / args.version / "kfold_folds_results.csv", index=False)

if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)
