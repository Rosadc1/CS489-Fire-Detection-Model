"""
    This module creates a custom PyTorch Dataset for loading prepared and pre-processed image data and corresponding mask from a CSV DataFrame.

    Each row in the DataFrame contains image_id which is the file name for the image and mask files.
    
"""
import os
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CSVDataset(Dataset):
    """
    Attributes:
        df (pd.DataFrame): DataFrame containing image file names.
        args: Object containing configuration parameters such as dataset directories and image settings.
        img_transform (transforms.Compose): Composed transformation pipeline applied to each input image.
        mask_transform (transforms.Compose): Composed transformation pipeline applied to each mask.
    """

    def __init__(self, args, df: pd.DataFrame) -> None:
        """
        Initialize the dataset with arguments and data.
        Initialize the image and mask transformation pipeline for pre-processing.

        Args:
            args: Configuration ArgumentParser.
            df (pd.DataFrame): DataFrame containing 'image_id' column.
        """
        
        """
        
        
        TO DO
        
        
        """
        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.args = args
        self.img_transform: transforms.Compose = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        """ Data augmentation ResUNet5
        self.img_transform: transforms.Compose = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=(args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        """
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieve a single image_id from the dataset and use it to open input image and its mask.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Image tensor of shape (C, H, W) after transformations.
                - Mask tensor of shape (H, W) after transformation, typically 0 or 1 for binary classification.
                - Image ID of type string used to save the prediction output as the same name.
        """
        """
        
        
        TO DO
        
        
        """
        row = self.df.iloc[i]
        image_path = os.path.join(
            self.args.dataset_dir,
            self.args.image_dir,
            f"{row['image_id']}.jpg"
        )

        image = Image.open(image_path).convert("RGB")
        image = self.img_transform(image)

        return image, row["image_id"]