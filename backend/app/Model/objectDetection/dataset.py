"""
    This module creates a custom PyTorch Dataset for loading prepared and pre-processed image data.
    
"""
import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class objectDetectionDataset(Dataset):
    """
    Attributes:
        args: Configuration ArgumentParser.
        img_transform: Transformation pipeline for input images.
        annotation_transform: Transformation pipeline for annotations.
        data: List of tuples containing image paths and their corresponding annotations.
    """

    def __init__(self, args, data) -> None:
        """
        Initialize the dataset with arguments and data.
        Initialize the image transformation pipeline for pre-processing.

        Args:
            args: Configuration ArgumentParser.
            data: List of tuples containing image paths and their corresponding annotations.
        """
        self.args = args
        self.data = data
        self.img_transform: transforms.Compose = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the image and its corresponding annotations at index i.

        Args:
            i (int): Index of the sample to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image tensor and its annotations.
        """
        image_path, annotations = self.data[i]

        image = Image.open(image_path).convert("RGB")
        image = self.img_transform(image)

        # Convert annotations to tensor
        annotations_tensor = torch.tensor(annotations, dtype=torch.float32)

        return image, annotations_tensor