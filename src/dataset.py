import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DFUDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Foot Ulcer (DFU) classification.
    Expects a DataFrame containing 'image_id' and 'label'.
    """

    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = str(row["image_id"])

        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fname = f"{fname}.jpg"

        img_path = os.path.join(self.img_dir, fname)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"DFU image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dfu_transforms(img_size: int = 256):
    """
    Photometric and geometric augmentations calibrated for clinical skin lesions.
    Restricts hue shift to preserve erythema (redness) and ischemia cues.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform