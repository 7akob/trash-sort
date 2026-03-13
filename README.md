# Trash Sort — YOLOv11n Waste Sorting

Realtime waste-sorting system using [Ultralytics YOLOv11n](https://docs.ultralytics.com/).
Detects and groups **cans**, **cartons**, and **tins** from a camera or video feed.

Training is done on **Google Colab** (`colab_train.ipynb`). Evaluation and the live app run locally.

---

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install ultralytics wandb opencv-python matplotlib
```

---

## Workflow

### 1. Prepare dataset (local)
Splits the raw dataset 80/10/10 (train/val/test) and generates `data.yaml`.

```bash
python prepare_dataset.py
```

Output: `~/code/TrashDataset/split/`

---

### 2. Train model (Google Colab)
Open `colab_train.ipynb` in [Google Colab](https://colab.research.google.com) and run the cells top to bottom.

The notebook will:
- Mount your Google Drive
- Copy the dataset from `MyDrive/TrashDataset` to local Colab storage
- Split the dataset and generate `data.yaml`
- Train YOLOv11n for 100 epochs on a GPU
- Save weights to `MyDrive/trash-sort-runs/trash_v1/weights/`
- Log metrics to [Weights & Biases](https://wandb.ai)

**If the session disconnects:** re-run Cell 1 (mount Drive) then Cell 5 (train) — it resumes automatically from the last checkpoint.

After training, download `best.pt` from `MyDrive/trash-sort-runs/trash_v1/weights/` and place it at:
```
runs/trash_v1/weights/best.pt
```

---

### 3. Evaluate model (local)
Runs the trained model against the test split.

```bash
python evaluate.py
```

Outputs:
- mAP@50, mAP@50-95, precision, recall per class (carton, tin, can)
- `runs/trash_v1/eval/confusion_matrix.png`

---

### 4. Run the sorting app (local)
Realtime detection and sorting visualization.

```bash
# Webcam
python waste_sorting_app.py --source 0

# Video file
python waste_sorting_app.py --source path/to/video.mp4

# Image file
python waste_sorting_app.py --source path/to/image.jpg

# Custom model weights
python waste_sorting_app.py --source 0 --model path/to/best.pt
```

Press **Q** to quit.

---

## Sorting logic

Objects detected in each frame are grouped by proximity (centroids within `PROXIMITY_PX` pixels of each other).

| Box color | Meaning |
|-----------|---------|
| **Green** | Group of objects — all the same class (e.g. two cans) |
| **Red**   | Group of objects — mixed classes (e.g. a can and a carton) |
| **Yellow**| Single lone object |

- Green groups show the **plural** class name (e.g. "cans")
- Red groups show each object's **singular** class name
- A connecting line is drawn between grouped objects

The `PROXIMITY_PX` constant in `waste_sorting_app.py` controls how close objects must be to be grouped.

---

## Classes

| ID | Swedish name     | English name |
|----|-----------------|--------------|
| 0  | Dryckeskartong  | carton       |
| 1  | Konservburk     | tin          |
| 2  | Pantburk        | can          |
