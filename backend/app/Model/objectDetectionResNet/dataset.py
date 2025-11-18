import os
from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
class objectDetectionDataset(Dataset):
    """
    Attributes:
        args: Configuration ArgumentParser.
        img_transform: Transformation pipeline for input images.
        samples: List of tuples containing image paths and their corresponding annotations."""

    def __init__(self, args) -> None:
        """
        Initialize the dataset with arguments and data.
        Initialize the image transformation pipeline for pre-processing.

        Args:
            args: Configuration ArgumentParser.
        """
        self.args = args
        self.img_transform: transforms.Compose = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
        ])
        
        image_dir = os.path.join(args.dataset_dir, args.image_dir)
        annotation_dir = os.path.join(args.dataset_dir, args.annotation_dir)
        self.samples = []

        for img_name in os.listdir(image_dir):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            
            img_path = os.path.join(image_dir, img_name)
            
            # annotation is assumed to have same name but .txt extension
            ann_name = os.path.splitext(img_name)[0] + ".txt"
            ann_path = os.path.join(annotation_dir, ann_name)

            if os.path.exists(ann_path):
                self.samples.append((img_path, ann_path))
            else:
                print(f"Warning: No annotation found for {img_name}")
                
        self.samples = np.array(self.samples, dtype=object)


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the image and its corresponding annotations at index i.

        Args:
            i (int): Index of the sample to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image tensor and its annotations.
        """
        image_path, annotation_path = self.samples[i]

        image = Image.open(image_path).convert("RGB")
        image = self.img_transform(image)

        # Convert annotations to tensor
        annotations = []
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                ann_class, x_center, y_center, width, height = parts
                annotations.append([
                    float(ann_class),
                    float(x_center),
                    float(y_center),
                    float(width),
                    float(height),
                ])

        annotations_tensor = torch.tensor(annotations, dtype=torch.float32)

        return image, annotations_tensor
class SubsetDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the subset dataset.
        """
        return len(self.samples)

    def __getitem__(self, i):
        image_path, annotation_path = self.samples[i]

        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        annotations = []
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split()

                ann_class, x_center, y_center, width, height = parts
                annotations.append([
                    float(ann_class),
                    float(x_center),
                    float(y_center),
                    float(width),
                    float(height),
                ])

        annotations_tensor = torch.tensor(annotations, dtype=torch.float32)
        return img, annotations_tensor