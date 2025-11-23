import os
import torch
import numpy as np
from pathlib import Path
from config import config_args
from ultralytics import YOLO
from typing import Any
import pandas as pd

def run_training(args: Any) -> None:
    # Restrict visible GPUs and set the target device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
    
    save_dir = Path(args.output_dir) / args.version
    save_dir.mkdir(parents=True, exist_ok=True)
    # Train the model 
    current_path = Path(__file__).parent
    weights_path = current_path / args.weights
    model = YOLO(weights_path)
    
    train_result = model.train(
        data='Dataset/data.yaml',
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.image_size,
        lr0=args.lr,
        patience=args.patience,
        seed=args.seed,
        device=DEVICE_STR,
        save=True,
        project=save_dir.as_posix(),
        name="train_results",
        exist_ok=True,
        verbose=False
    )


    # Validate the model
    val_result = model.val(
        data='Dataset/data.yaml',
        batch=args.batch,
        imgsz=args.image_size,
        seed=args.seed,
        device=DEVICE_STR,
        verbose=False
    )
    # Save results
    metrics = {}
        
    # Prefix train metrics
    for key, value in train_result.results_dict.items():
        metrics[f"train_{key}"] = value

    # Prefix validation metrics
    for key, value in val_result.results_dict.items():
        metrics[f"val_{key}"] = value


    # Save all folds results
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(Path(args.output_dir) / args.version / "results.csv", index=False)

if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)
