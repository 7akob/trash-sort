"""
Realtidsapplikation för avfallssortering med OpenCV och YOLO.

Användning:
    python waste_sorting_app.py --source 0            # webbkamera
    python waste_sorting_app.py --source video.mp4
    python waste_sorting_app.py --source image.jpg
    python waste_sorting_app.py --model path/to/best.pt
    python waste_sorting_app.py --display-width 900   # anpassa fönsterstorlek
"""

import argparse
import math
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kör på CPU (ingen GPU lokalt)

import cv2
import numpy as np

# ── Sökvägar & standardvärden ──────────────────────────────────────────────────
DEFAULT_MODEL  = Path(__file__).parent / "runs" / "trash_v1" / "weights" / "best.pt"
DEFAULT_SOURCE = 0  # webbkamera index

# ── Visuell konfiguration ──────────────────────────────────────────────────────
PROXIMITY_PX   = 150   # max avstånd i pixlar för att räknas som samma grupp
CONF_THRESHOLD = 0.35  # minsta konfidens för att visa en detektion

# Färger per status (BGR-format)
COLOR_SAME   = (0, 200, 0)    # grön  — sorterat, samma klass
COLOR_MIXED  = (0, 0, 220)    # röd   — osorterat, blandat
COLOR_ALONE  = (0, 200, 220)  # gul   — ensamt objekt

# Svenska visningsnamn per klass
CLASS_DISPLAY = {
    "carton": "Dryckeskartong",
    "tin":    "Konservburk",
    "can":    "Pantburk",
}

# Plural-etiketter för sorterade grupper
CLASS_PLURAL = {
    "carton": "DRYCKESKARTONGER",
    "tin":    "KONSERVBURKAR",
    "can":    "PANTBURKAR",
}

# Förklaring som visas i hörnet av bilden
LEGEND_ITEMS = [
    (COLOR_SAME,  "SORTERAT — samma klass"),
    (COLOR_MIXED, "OSORTERAT — blandat"),
    (COLOR_ALONE, "ENSAMT objekt"),
]


# ── Geometrihjälpfunktioner ────────────────────────────────────────────────────

def centroid(box):
    """Beräknar mittpunkten för en bounding box (xyxy-format)."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def distance(a, b):
    """Beräknar euklidiskt avstånd mellan två punkter."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def group_detections(detections):
    """
    Grupperar detektioner med union-find.
    Två objekt hamnar i samma grupp om deras mittpunkter
    är inom PROXIMITY_PX pixlar från varandra.
    """
    n = len(detections)
    parent = list(range(n))

    def find(i):
        # Hitta gruppens rot med path compression
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    centroids = [centroid(d["box"]) for d in detections]

    # Jämför alla par och slå ihop de som är tillräckligt nära
    for i in range(n):
        for j in range(i + 1, n):
            if distance(centroids[i], centroids[j]) <= PROXIMITY_PX:
                union(i, j)

    # Samla ihop grupper baserat på rot
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


# ── Ritfunktioner ──────────────────────────────────────────────────────────────

def draw_legend(frame):
    """Ritar förklaringsrutan i övre vänstra hörnet."""
    x, y0 = 10, 10
    for color, label in LEGEND_ITEMS:
        cv2.rectangle(frame, (x, y0), (x + 18, y0 + 18), color, -1)
        cv2.putText(frame, label, (x + 24, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 26


def draw_fps(frame, fps):
    """Visar FPS i nedre vänstra hörnet."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _put_label(frame, text, x1, y1, color):
    """Ritar en textetikett med fylld bakgrund ovanför en bounding box."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_group(frame, group_indices, detections):
    """
    Ritar boxar och etiketter för en grupp objekt.
    - Grön + plural-etikett om alla är av samma klass
    - Röd + individuella etiketter om klasserna är blandade
    - Gul om objektet är ensamt
    """
    classes = [detections[i]["class_name"] for i in group_indices]
    all_same = len(set(classes)) == 1
    lone = len(group_indices) == 1

    # Välj färg baserat på grupptyp
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

        # Enskild etikett visas bara för ensamma eller blandade objekt
        if all_same and not lone:
            pass
        else:
            display_name = CLASS_DISPLAY.get(det["class_name"], det["class_name"])
            conf_str = f"{det['conf']:.2f}"
            _put_label(frame, f"{display_name} {conf_str}", x1, y1, color)

    # En gemensam plural-etikett för sorterade grupper
    if all_same and not lone:
        all_x1 = min(detections[i]["box"][0] for i in group_indices)
        all_y1 = min(detections[i]["box"][1] for i in group_indices)
        plural = CLASS_PLURAL.get(classes[0], classes[0].upper() + "S")
        _put_label(frame, plural, all_x1, all_y1, color)

    # Linje mellan objekten i gruppen
    if not lone:
        pts = [centroid(detections[i]["box"]) for i in group_indices]
        for k in range(len(pts) - 1):
            cv2.line(frame, pts[k], pts[k + 1], color, 1)


# ── Huvudloop ──────────────────────────────────────────────────────────────────

def resize_for_display(frame, width):
    """Skalar om bilden proportionellt till angiven bredd."""
    if width is None:
        return frame
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


def run(source, model_path, display_width=None):
    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # Kontrollera om källan är en bildfil
    source_path = Path(str(source))
    is_image = source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {source_path}")
        # Skala om innan detektering så att text och boxar ser rätt ut
        frame = resize_for_display(frame, display_width)
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        frame = process_frame(frame, results, fps=0)
        cv2.imshow("Waste Sorting", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video eller webbkamera
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skala om varje bildruta innan detektering
        frame = resize_for_display(frame, display_width)
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        frame = process_frame(frame, results, fps)

        # Beräkna FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        cv2.imshow("Waste Sorting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_frame(frame, results, fps):
    """Hämtar detektioner, grupperar dem och ritar allt på bildrutan."""
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


# ── Startpunkt ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Waste sorting realtime app")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Videokälla: 0 för webbkamera, eller sökväg till video/bild")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Sökväg till YOLO-vikter (.pt)")
    parser.add_argument("--display-width", type=int, default=None,
                        help="Skala visningsfönster till denna bredd i pixlar (t.ex. 900)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(source=args.source, model_path=args.model, display_width=args.display_width)
