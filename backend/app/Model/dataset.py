"""
    This module creates a custom PyTorch Dataset for loading prepared and pre-processed image data.
    
"""
import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ClassifierDataset(Dataset):
    """
    Attributes:
        args: Configuration ArgumentParser.
        img_transform: Transformation pipeline for input images.
        classification_map: Mapping of class names to integer labels.
        image_paths: List of file paths to the images.
        labels: List of integer labels corresponding to each image.
    """

    def __init__(self, args) -> None:
        """
        Initialize the dataset with arguments and data.
        Initialize the image transformation pipeline for pre-processing.

        Args:
            args: Configuration ArgumentParser.
        """
        self.args = args
        self.img_transform: transforms.Compose = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        self.classification_map = {
            "Non_fire": 0,
            "Fire": 1
        }

        self.image_paths = []
        self.labels = []

        for name, label in self.classification_map.items():
            class_dir = os.path.join(self.args.dataset_dir, name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)
        
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieve the image and its corresponding label at index i.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, str]: Transformed image tensor and its label.
        """
        image_path = self.image_paths[i]
        label = self.labels[i]

        image = Image.open(image_path).convert("RGB")
        image = self.img_transform(image)

        return image, label
