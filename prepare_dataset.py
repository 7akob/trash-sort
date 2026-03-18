"""
Delar upp TrashDataset i train/val/test (80/10/10) och skapar data.yaml.
Bara bild-label-par inkluderas — bilder utan label-fil hoppas över.
"""

import random
import shutil
from pathlib import Path

# ── Sökvägar ───────────────────────────────────────────────────────────────────
DATASET_DIR = Path(r"C:\Users\sandb\Documents\TrashDataset")
IMAGES_DIR  = DATASET_DIR / "images"
LABELS_DIR  = DATASET_DIR / "labels"
SPLIT_DIR   = DATASET_DIR / "split"

# ── Konfiguration ──────────────────────────────────────────────────────────────
RANDOM_SEED  = 42   # fast seed för reproducerbar uppdelning
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
# TEST_RATIO  = 0.10  (resten)

# Klassnamn måste matcha index-ordningen i classes.txt:
#   0 = Dryckeskartong → carton
#   1 = Konservburk    → tin
#   2 = Pantburk       → can
CLASS_NAMES = ["carton", "tin", "can"]


def collect_pairs() -> list[Path]:
    """Returnerar bara de bilder som har en matchande label-fil."""
    images = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    paired = [img for img in images if (LABELS_DIR / img.with_suffix(".txt").name).exists()]
    skipped = len(list(IMAGES_DIR.glob("*"))) - len(paired)
    print(f"Found {len(paired)} paired samples ({skipped} images skipped — no label).")
    return paired


def split(pairs: list[Path]) -> tuple[list, list, list]:
    """Blandar och delar upp bilderna i train/val/test."""
    rng = random.Random(RANDOM_SEED)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def copy_split(pairs: list[Path], subset: str) -> None:
    """Kopierar bilder och labels till rätt undermapp (train/val/test)."""
    img_dst = SPLIT_DIR / subset / "images"
    lbl_dst = SPLIT_DIR / subset / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for img_path in pairs:
        shutil.copy2(img_path, img_dst / img_path.name)
        lbl_src = LABELS_DIR / img_path.with_suffix(".txt").name
        shutil.copy2(lbl_src, lbl_dst / lbl_src.name)


def write_yaml() -> None:
    """Skapar data.yaml som YOLO använder för att hitta datasetet."""
    yaml_path = SPLIT_DIR / "data.yaml"
    lines = [
        f"path: {SPLIT_DIR}",
        "train: train/images",
        "val:   val/images",
        "test:  test/images",
        "",
        f"nc: {len(CLASS_NAMES)}",
        f"names: {CLASS_NAMES}",
        "",
    ]
    yaml_path.write_text("\n".join(lines))
    print(f"Wrote {yaml_path}")


def main() -> None:
    # Rensa gammal split om den finns
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
        print(f"Removed existing {SPLIT_DIR}")

    pairs = collect_pairs()
    train, val, test = split(pairs)

    for subset, bucket in [("train", train), ("val", val), ("test", test)]:
        copy_split(bucket, subset)
        print(f"  {subset:5s}: {len(bucket)} samples")

    write_yaml()
    print("Done.")


if __name__ == "__main__":
    main()
