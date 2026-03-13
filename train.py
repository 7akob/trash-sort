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
EPOCHS   = 100
BATCH    = 16
IMGSZ    = 640
LR0      = 0.01
PATIENCE = 20   # early-stopping patience (epochs without improvement)
DEVICE   = 0    # GPU device index (run via Colab, not locally)


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
        data      = str(DATA_YAML),
        epochs    = EPOCHS,
        batch     = BATCH,
        imgsz     = IMGSZ,
        lr0       = LR0,
        patience  = PATIENCE,
        device    = DEVICE,
        project   = str(PROJECT_DIR),
        name      = RUN_NAME,
        exist_ok  = True,
    )

    wandb.finish()
    print(f"\nWeights saved to: {PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
