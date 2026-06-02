"""
Environment-aware path resolution.

MacBook:  images + splits + weights live inside the project directory.
Colab:    code is cloned from GitHub into /content/dizertatie_project;
          images, splits, weights, and checkpoints live on Google Drive
          at MyDrive/dizertatie/.

Usage on Colab (before importing this module):
    from google.colab import drive
    drive.mount('/content/drive')
    # then just import — paths resolve automatically
"""

from pathlib import Path


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


if _is_colab():
    PROJECT_ROOT = Path('/content/dizertatie_project')
    DRIVE_ROOT   = Path('/content/drive/MyDrive/dizertatie')
    IMAGE_ROOT   = DRIVE_ROOT
    SPLITS_DIR   = DRIVE_ROOT / 'splits'
    WEIGHTS_DIR  = DRIVE_ROOT / 'weights'
    CKPT_DIR     = DRIVE_ROOT / 'checkpoints'
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    IMAGE_ROOT   = PROJECT_ROOT
    SPLITS_DIR   = PROJECT_ROOT / 'splits'
    WEIGHTS_DIR  = PROJECT_ROOT / 'weights'
    CKPT_DIR     = PROJECT_ROOT / 'checkpoints'
