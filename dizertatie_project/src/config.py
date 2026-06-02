"""
Environment-aware path resolution.

MacBook:  everything lives inside the project directory.
Colab:    code is cloned from GitHub into /content/dizertatie_project.
          Data setup (run once per session):
            1. copy train.zip from Drive to /content/ and unzip  → /content/train/
            2. copy valid/    from Drive to /content/valid/
          After setup IMAGE_ROOT = /content/  (local SSD, full A100 throughput).
          If setup cell was skipped, falls back to Drive FUSE mount (slow).
          splits/, weights/, checkpoints/ always live on Drive (persistent).
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
    _local       = Path('/content')
    # Use local SSD if data has been copied there; fall back to Drive FUSE
    IMAGE_ROOT   = _local if (_local / 'train').exists() else DRIVE_ROOT
    SPLITS_DIR   = DRIVE_ROOT / 'splits'
    WEIGHTS_DIR  = DRIVE_ROOT / 'weights'
    CKPT_DIR     = DRIVE_ROOT / 'checkpoints'
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    IMAGE_ROOT   = PROJECT_ROOT
    SPLITS_DIR   = PROJECT_ROOT / 'splits'
    WEIGHTS_DIR  = PROJECT_ROOT / 'weights'
    CKPT_DIR     = PROJECT_ROOT / 'checkpoints'
