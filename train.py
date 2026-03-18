"""
Tränar YOLOv11n på trash-sort datasetet.

OBS: Träningen kördes på Google Colab via colab_train.ipynb.
Den här filen sparas som referens för hyperparametrar och körningskonfiguration.
"""

import os

from pathlib import Path

# ── Sökvägar ───────────────────────────────────────────────────────────────────
DATA_YAML   = Path.home() / "code" / "TrashDataset" / "split" / "data.yaml"
PROJECT_DIR = Path(__file__).parent / "runs"
RUN_NAME    = "trash_v1"
BASE_MODEL  = "yolo11n.pt"  # minsta YOLO11-modellen (nano)

# ── Hyperparametrar ────────────────────────────────────────────────────────────
EPOCHS   = 50
BATCH    = 16
IMGSZ    = 640
LR0      = 0.01
PATIENCE = 15   # tidig stopp om ingen förbättring på 15 epoker
DEVICE   = 0    # GPU-index (körs via Colab, inte lokalt)

# ── Augmentering & regularisering (viktigt för små dataset) ───────────────────
FLIPUD          = 0.2   # vertikal spegling
MIXUP           = 0.2   # blandar två bilder och deras labels
COPY_PASTE      = 0.1   # klistrar in objekt från andra bilder
DEGREES         = 15.0  # slumpmässig rotation ±15°
SHEAR           = 5.0   # skjuvning av bilden
DROPOUT         = 0.2   # stänger av 20% av neuroner slumpmässigt
LABEL_SMOOTHING = 0.1   # mjukar upp labels för att minska överanpassning


def main() -> None:
    from ultralytics import YOLO
    import wandb

    # Logga träningen till Weights & Biases
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
        # augmentering
        flipud          = FLIPUD,
        mixup           = MIXUP,
        copy_paste      = COPY_PASTE,
        degrees         = DEGREES,
        shear           = SHEAR,
        # regularisering
        dropout         = DROPOUT,
        label_smoothing = LABEL_SMOOTHING,
    )

    wandb.finish()
    print(f"\nVikter sparade till: {PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
