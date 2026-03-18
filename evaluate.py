"""
Utvärderar den tränade modellen mot test-spliten.
Kör: python evaluate.py
"""

from pathlib import Path

# ── Sökvägar ───────────────────────────────────────────────────────────────────
MODEL_PATH    = Path(__file__).parent / "runs" / "trash_v1" / "weights" / "best.pt"
DATA_YAML     = Path(r"C:\Users\sandb\Documents\TrashDataset\split\data.yaml")
OUTPUT_DIR    = Path(__file__).parent / "runs" / "trash_v1" / "eval"

# ── Konfiguration ──────────────────────────────────────────────────────────────
DEVICE = "cpu"
IMGSZ  = 640
BATCH  = 16


def main() -> None:
    from ultralytics import YOLO
    import matplotlib.pyplot as plt

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run train.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    # Kör validering mot test-spliten
    metrics = model.val(
        data    = str(DATA_YAML),
        split   = "test",
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = DEVICE,
        project = str(OUTPUT_DIR),
        name    = "results",
        exist_ok= True,
    )

    class_names = model.names  # {0: 'carton', 1: 'tin', 2: 'can'}
    nc = len(class_names)

    # ── Resultat per klass ─────────────────────────────────────────────────────
    print("\n── Evaluation results ─────────────────────────────────────")
    print(f"{'Class':<12} {'mAP@50':>8} {'mAP@50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("─" * 52)

    map50     = metrics.box.ap50   # mAP@50 per klass
    map50_95  = metrics.box.ap     # mAP@50-95 per klass
    precision = metrics.box.p      # precision per klass
    recall    = metrics.box.r      # recall per klass

    for i in range(nc):
        print(f"{class_names[i]:<12} {map50[i]:>8.3f} {map50_95[i]:>10.3f} "
              f"{precision[i]:>10.3f} {recall[i]:>8.3f}")

    print("─" * 52)
    print(f"{'all':<12} {metrics.box.map50:>8.3f} {metrics.box.map:>10.3f} "
          f"{metrics.box.mp:>10.3f} {metrics.box.mr:>8.3f}")

    # ── Konfusionsmatris ───────────────────────────────────────────────────────
    cm = metrics.confusion_matrix.matrix  # numpy-array (nc+1, nc+1)
    labels = [class_names[i] for i in range(nc)] + ["background"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix (test split)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Skriv antal i varje cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()

    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    print(f"\nKonfusionsmatris sparad till: {cm_path}")


if __name__ == "__main__":
    main()
