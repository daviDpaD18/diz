from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ── Label config ──────────────────────────────────────────────────────────────

LABEL_COLS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

# Uncertain (-1) mapped to 1.0 (positive) — CheXpert paper recommendation
U_ONES = {'Atelectasis', 'Edema', 'Pleural Effusion', 'Consolidation'}
# Uncertain (-1) mapped to 0.0 (negative)
U_ZEROS = {'Cardiomegaly'}
# All other labels: uncertain also mapped to 0.0


def remap_uncertain(df, label_cols=None):
    """Return a copy of df with -1 uncertain labels replaced by 0 or 1
    according to the CheXpert paper policy. NaN (absent label) → 0."""
    label_cols = label_cols or LABEL_COLS
    df = df.copy()
    for col in label_cols:
        if col not in df.columns:
            continue
        if col in U_ONES:
            df[col] = df[col].replace(-1.0, 1.0)
        else:
            df[col] = df[col].replace(-1.0, 0.0)
        df[col] = df[col].fillna(0.0).astype(float)
    return df


def age_group(age):
    try:
        a = float(age)
    except (TypeError, ValueError):
        return 'unknown'
    if a < 40:
        return '<40'
    if a <= 60:
        return '40-60'
    return '>60'


# ── Transforms ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Dataset class ─────────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    """
    Loads CheXpert images and returns everything needed for training,
    evaluation, and post-hoc bias analysis.

    Expects df to have:
      - 'Path'               relative path from image_root, e.g. train/patient.../view.jpg
      - one column per label in label_cols, already remapped to binary floats
      - 'Sex', 'Age', 'AP/PA' (optional — returned as empty string if absent)

    Returns per item:
      img_tensor   float32 [3, 224, 224]
      labels       float32 [14]
      patient_id   str      extracted from path, e.g. 'patient00001'
      sex          str      'Male' | 'Female' | ''
      age_grp      str      '<40' | '40-60' | '>60' | 'unknown'
      orig_size    tuple    (W, H) of the original image before any transforms
    """

    def __init__(self, df, image_root, transform, label_cols=None):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.label_cols = label_cols or LABEL_COLS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(self.image_root / row['Path'])
        orig_size = img.size  # (W, H) before any resize

        img_tensor = self.transform(img)

        labels = torch.tensor(
            row[self.label_cols].values.astype(float),
            dtype=torch.float32,
        )

        patient_id = row['Path'].split('/')[1]          # 'patientXXXXX'
        sex        = str(row.get('Sex', ''))
        age_grp    = age_group(row.get('Age', None))

        return img_tensor, labels, patient_id, sex, age_grp, orig_size
