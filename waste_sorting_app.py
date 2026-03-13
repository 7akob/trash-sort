"""
Realtime waste-sorting app using OpenCV and YOLO.

Usage:
    python waste_sorting_app.py --source 0            # webcam
    python waste_sorting_app.py --source video.mp4
    python waste_sorting_app.py --source image.jpg
    python waste_sorting_app.py --model path/to/best.pt
"""

import argparse
import math
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU (local GPU/ROCm unsupported)

import cv2
import numpy as np

# ── Paths & defaults ───────────────────────────────────────────────────────────
DEFAULT_MODEL  = Path(__file__).parent / "runs" / "trash_v1" / "weights" / "best.pt"
DEFAULT_SOURCE = 0  # webcam index

# ── Visual config ──────────────────────────────────────────────────────────────
PROXIMITY_PX   = 150   # max centroid distance to be considered "grouped"
CONF_THRESHOLD = 0.35

COLOR_SAME   = (0, 200, 0)    # green  — group, same class
COLOR_MIXED  = (0, 0, 220)    # red    — group, mixed classes
COLOR_ALONE  = (0, 200, 220)  # yellow — lone object

CLASS_PLURAL = {
    "carton": "cartons",
    "tin":    "tins",
    "can":    "cans",
}

LEGEND_ITEMS = [
    (COLOR_SAME,  "Same class (group)"),
    (COLOR_MIXED, "Mixed classes (group)"),
    (COLOR_ALONE, "Lone object"),
]


# ── Geometry helpers ───────────────────────────────────────────────────────────

def centroid(box):
    """Return (cx, cy) from xyxy box."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def group_detections(detections):
    """
    Simple greedy grouping: merge any two detections whose centroids are
    within PROXIMITY_PX into the same group.
    Returns list of groups, each group is a list of detection indices.
    """
    n = len(detections)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    centroids = [centroid(d["box"]) for d in detections]
    for i in range(n):
        for j in range(i + 1, n):
            if distance(centroids[i], centroids[j]) <= PROXIMITY_PX:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_legend(frame):
    x, y0 = 10, 10
    for color, label in LEGEND_ITEMS:
        cv2.rectangle(frame, (x, y0), (x + 18, y0 + 18), color, -1)
        cv2.putText(frame, label, (x + 24, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 26


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_group(frame, group_indices, detections):
    classes = [detections[i]["class_name"] for i in group_indices]
    all_same = len(set(classes)) == 1
    lone = len(group_indices) == 1

    if lone:
        color = COLOR_ALONE
    elif all_same:
        color = COLOR_SAME
    else:
        color = COLOR_MIXED

    for i in group_indices:
        det = detections[i]
        x1, y1, x2, y2 = det["box"]
        cx, cy = centroid(det["box"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)

        if lone:
            label = det["class_name"]
        elif all_same:
            label = CLASS_PLURAL.get(det["class_name"], det["class_name"] + "s")
        else:
            label = det["class_name"]

        conf_str = f"{det['conf']:.2f}"
        text = f"{label} {conf_str}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # draw lines between grouped objects
    if not lone:
        pts = [centroid(detections[i]["box"]) for i in group_indices]
        for k in range(len(pts) - 1):
            cv2.line(frame, pts[k], pts[k + 1], color, 1)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(source, model_path):
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # Determine if source is an image file
    source_path = Path(str(source))
    is_image = source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {source_path}")
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        frame = process_frame(frame, results, fps=0)
        cv2.imshow("Waste Sorting", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video / camera
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        frame = process_frame(frame, results, fps)

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        cv2.imshow("Waste Sorting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_frame(frame, results, fps):
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            detections.append({"box": (x1, y1, x2, y2), "conf": conf,
                                "class_name": class_name})

    if detections:
        groups = group_detections(detections)
        for group in groups:
            draw_group(frame, group, detections)

    draw_legend(frame)
    if fps > 0:
        draw_fps(frame, fps)
    return frame


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Waste sorting realtime app")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Video source: 0 for webcam, or path to video/image")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Path to YOLO weights (.pt)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(source=args.source, model_path=args.model)
