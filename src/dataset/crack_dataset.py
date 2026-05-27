import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class CrackDataset(Dataset):
    """
    PyTorch Dataset representing bridge defect imagery
    and annotated segmentation binary masks.
    """
    def __init__(self, img_paths: list, mask_paths: list, transform: A.Compose = None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple:
        # Load imagery and convert to RGB channels
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load annotation mask in grayscale
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Retrieve binary values from grayscale mask
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, axis=-1).astype(np.float32)

        # Match image and mask transformations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask


# Augmentation using albumentations recipe
def get_train_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_test_transform() -> A.Compose:
    """
    Deterministic normalization transform for validation/test crack detection.
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
