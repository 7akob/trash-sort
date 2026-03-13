"""
Trains YOLOv11n on the trash-sort dataset.

NOTE: Training is done on Google Colab via colab_train.ipynb.
This file is kept as a reference for hyperparameters and run config.
"""

import os

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_YAML   = Path.home() / "code" / "TrashDataset" / "split" / "data.yaml"
PROJECT_DIR = Path(__file__).parent / "runs"
RUN_NAME    = "trash_v1"
BASE_MODEL  = "yolo11n.pt"

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS   = 50
BATCH    = 16
IMGSZ    = 640
LR0      = 0.01
PATIENCE = 15   # early-stopping patience (epochs without improvement)
DEVICE   = 0    # GPU device index (run via Colab, not locally)

# ── Augmentation & regularization (important for small datasets) ───────────────
FLIPUD          = 0.2   # vertical flip
MIXUP           = 0.2   # blend two images + labels
COPY_PASTE      = 0.1   # paste objects from other images
DEGREES         = 15.0  # random rotation ±15°
SHEAR           = 5.0   # shear distortion
DROPOUT         = 0.2   # randomly disable 20% of neurons
LABEL_SMOOTHING = 0.1   # soften labels (1.0 → 0.9) to reduce overconfidence


def main() -> None:
    from ultralytics import YOLO
    import wandb

    wandb.init(project="trash-sort", name=RUN_NAME, config={
        "model":    BASE_MODEL,
        "epochs":   EPOCHS,
        "batch":    BATCH,
        "imgsz":    IMGSZ,
        "lr0":      LR0,
        "patience": PATIENCE,
        "device":   DEVICE,
    })

    model = YOLO(BASE_MODEL)

    model.train(
        data            = str(DATA_YAML),
        epochs          = EPOCHS,
        batch           = BATCH,
        imgsz           = IMGSZ,
        lr0             = LR0,
        patience        = PATIENCE,
        device          = DEVICE,
        project         = str(PROJECT_DIR),
        name            = RUN_NAME,
        exist_ok        = True,
        # augmentation
        flipud          = FLIPUD,
        mixup           = MIXUP,
        copy_paste      = COPY_PASTE,
        degrees         = DEGREES,
        shear           = SHEAR,
        # regularization
        dropout         = DROPOUT,
        label_smoothing = LABEL_SMOOTHING,
    )

    wandb.finish()
    print(f"\nWeights saved to: {PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
