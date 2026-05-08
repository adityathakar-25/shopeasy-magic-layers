"""
stage2_detect.py — Object detection using YOLOv8.

  Fixes applied (v3):
  ✅ Default model upgraded from yolov8n → yolov8s (better accuracy, fewer misclassifications)
  ✅ Default confidence lowered from 0.35 → 0.25 (catches more products in clean studio shots)
  ✅ Emoji badge replaced with ASCII "[P]" — OpenCV cannot render Unicode characters
  ✅ GPU auto-detection with FP16 half-precision on CUDA
  ✅ Inference timing logged for every run
  ✅ Configurable imgsz (default 640 — YOLO's native resolution)
  ✅ Adaptive box thickness scaled to image resolution
  ✅ Object index shown in label for easy cross-referencing with Stage 3
  ✅ Warm-up function — call at app launch to eliminate first-click lag
  ✅ GPU memory cleaned up after inference
  ✅ Corrupt .pt file auto-detected and re-downloaded (validated once per session)
"""

import os
import time
import zipfile
import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from utils import get_color, draw_text_with_bg, cv2_to_pil, pil_to_cv2

# ── Device ──────────────────────────────────────────────────────────────────
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_USE_HALF = (_DEVICE == "cuda")   # FP16 only on GPU

# ── Model registry ──────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "YOLOv8n (fastest, ~6 MB)":   "yolov8n.pt",
    "YOLOv8s (balanced, ~22 MB)": "yolov8s.pt",
    "YOLOv8m (accurate, ~52 MB)": "yolov8m.pt",
}
DEFAULT_MODEL = "YOLOv8s (balanced, ~22 MB)"

# Cache: model_name → loaded YOLO instance
_model_cache: dict = {}

# ── E-commerce class sets ────────────────────────────────────────────────────
# COCO classes relevant to product/item photography on Shopeasy.ai
ECOMMERCE_PRODUCT_CLASSES: set = {
    # Fashion & Accessories
    "handbag", "backpack", "suitcase", "tie", "umbrella",
    # Electronics
    "cell phone", "laptop", "keyboard", "mouse", "remote",
    "tv", "clock", "microwave", "oven", "toaster", "refrigerator",
    # Home & Kitchen
    "chair", "couch", "bed", "dining table", "potted plant",
    "vase", "bottle", "cup", "bowl", "wine glass",
    "fork", "knife", "spoon",
    # Food (for food product shots)
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    # Sports & Outdoor
    "sports ball", "skateboard", "surfboard", "tennis racket",
    "bicycle",
    # Tools & Stationery
    "scissors", "book",
}

# Classes that are almost always background noise in product photography
BACKGROUND_NOISE_CLASSES: set = {
    "person", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
}

# Track which weight files have already been validated this session
# so _is_valid_pt_file() is never called twice for the same path
_validated_files: set = set()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _is_valid_pt_file(path: str) -> bool:
    """
    PyTorch .pt files are ZIP archives internally.
    Returns True only if the file exists AND passes ZIP integrity check.
    A partial/corrupt download fails this.
    Result is cached per session via _validated_files.
    """
    if path in _validated_files:
        return True                      # already confirmed good this session
    if not os.path.exists(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()           # None = all OK; else = first bad entry
            if bad is None:
                _validated_files.add(path)
                return True
            return False
    except Exception:
        return False


def _adaptive_thickness(img_w: int, img_h: int) -> int:
    """
    Return box border thickness proportional to image area.
    Avoids hairline boxes on large images and chunky boxes on small ones.
    """
    base = max(img_w, img_h)
    if base >= 1000:
        return 3
    elif base >= 600:
        return 2
    return 1


# ── Model loading ────────────────────────────────────────────────────────────

def get_model(model_name: str = DEFAULT_MODEL) -> YOLO:
    """
    Return a cached YOLO model, loading/downloading it if needed.
    Corrupt weight files are detected and deleted so YOLO re-downloads them.
    """
    if model_name not in _model_cache:
        weight_file = MODEL_OPTIONS.get(model_name)
        if weight_file is None:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(MODEL_OPTIONS.keys())}"
            )

        # ── Integrity check (runs once per file per session) ───────────────
        if os.path.exists(weight_file):
            if _is_valid_pt_file(weight_file):
                print(f"[Stage 2] ✅ Valid cached weights: {weight_file}")
            else:
                print(
                    f"[Stage 2] ⚠️  Corrupt model file: {weight_file}\n"
                    f"           Deleting and re-downloading..."
                )
                os.remove(weight_file)
        else:
            print(f"[Stage 2] Weight file not found locally: {weight_file}")

        # ── Load / auto-download ───────────────────────────────────────────
        t0 = time.perf_counter()
        print(f"[Stage 2] Loading '{model_name}' on {_DEVICE.upper()}...")
        try:
            _model_cache[model_name] = YOLO(weight_file)
            elapsed = time.perf_counter() - t0
            print(f"[Stage 2] ✅ Model ready in {elapsed:.1f}s | device={_DEVICE.upper()} | half={_USE_HALF}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model '{weight_file}'.\n"
                f"Reason: {e}\n"
                "Ensure 'ultralytics' is installed and you have internet "
                "access for the first download."
            ) from e

    return _model_cache[model_name]


def warmup(model_name: str = DEFAULT_MODEL) -> None:
    """
    Pre-load the model now so the first user click has no loading lag.
    Call this at app startup.
    """
    print(f"[Stage 2] Warming up '{model_name}'...")
    model = get_model(model_name)
    # Run one silent inference on a tiny dummy image
    dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
    model(dummy, verbose=False, imgsz=64)
    print(f"[Stage 2] Warm-up complete.")


# ── Main inference ────────────────────────────────────────────────────────────

def detect_objects(
    image: Image.Image,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    model_name: str = DEFAULT_MODEL,
    imgsz: int = 640,
    ecommerce_mode: bool = False,
):
    """
    Run YOLOv8 detection on a PIL Image.

    Args:
        image                : PIL Image (RGB)
        confidence_threshold : Keep detections above this score (0.0–1.0)
        iou_threshold        : NMS overlap threshold — lower = fewer overlapping boxes
        model_name           : One of the keys in MODEL_OPTIONS
        imgsz                : Internal inference resolution (default 640 — YOLO native)
        ecommerce_mode       : When True, suppresses background-noise classes
                               (people, vehicles, animals) and focuses on product
                               classes relevant to Shopeasy.ai e-commerce.

    Returns:
        detections : list of dicts —
            {
              "label":      str   — COCO class name, e.g. "handbag"
              "confidence": float — 0.87
              "box":        [x1, y1, x2, y2] — pixel coords in the original image
              "color_bgr":  (B, G, R) tuple  — unique per-object colour
              "index":      int              — 0-based detection index
              "is_product": bool             — True if class is in ECOMMERCE_PRODUCT_CLASSES
            }
        annotated_image : PIL Image with coloured boxes and labels drawn
    """
    model = get_model(model_name)

    t0 = time.perf_counter()
    results = model(
        image,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        half=_USE_HALF,      # FP16 on GPU — ~40% faster, same quality
        device=_DEVICE,
        verbose=False,
    )[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    # ── Build annotated image ─────────────────────────────────────────────
    img_cv2 = pil_to_cv2(image).copy()
    thickness = _adaptive_thickness(*image.size)
    raw_detections = []
    for i, box in enumerate(results.boxes):
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        label    = model.names[class_id]
        is_product = label in ECOMMERCE_PRODUCT_CLASSES

        # ── E-commerce filter ─────────────────────────────────────────────
        if ecommerce_mode and label in BACKGROUND_NOISE_CLASSES:
            print(f"[Stage 2] 🛍️  E-commerce mode: skipping '{label}' (background noise)")
            continue

        raw_detections.append({
            "label":      label,
            "confidence": round(conf, 3),
            "box":        [x1, y1, x2, y2],
            "color_bgr":  get_color(i),
            "index":      i,
            "is_product": is_product,
        })

    # Re-index after filtering so indices are always 0-based and contiguous
    detections = []
    img_cv2    = pil_to_cv2(image).copy()

    for new_idx, det in enumerate(raw_detections):
        det["index"] = new_idx
        color = det["color_bgr"]
        x1, y1, x2, y2 = det["box"]
        conf  = det["confidence"]
        label = det["label"]
        detections.append(det)

        # Box — gold border for product classes, normal for others
        border_color = (0, 215, 255) if det["is_product"] else color   # gold BGR
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), border_color, thickness=thickness)

        # Label — ASCII only, OpenCV cannot render Unicode/emoji
        # "[P]" badge for product classes instead of emoji
        badge = "[P]" if det["is_product"] else ""
        text  = f"#{new_idx} {label} {conf:.0%} {badge}".strip()
        draw_text_with_bg(
            img_cv2, text, (x1, y1),
            font_scale=0.50 if image.size[0] < 800 else 0.60,
            bg_bgr=border_color,
            color_bgr=(0, 0, 0) if det["is_product"] else (255, 255, 255),
        )

    annotated_image = cv2_to_pil(img_cv2)

    # ── GPU cleanup ────────────────────────────────────────────────────────
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    n_product = sum(1 for d in detections if d["is_product"])
    print(
        f"[Stage 2] {len(detections)} object(s) detected in {infer_ms:.0f} ms "
        f"[conf≥{confidence_threshold:.2f} | iou≤{iou_threshold:.2f} | "
        f"imgsz={imgsz} | ecommerce={ecommerce_mode} | device={_DEVICE.upper()}] "
        f"→ {[d['label'] for d in detections]} "
        f"({n_product} product class{'es' if n_product!=1 else ''})"
    )

    return detections, annotated_image


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from stage1_preprocess import load_and_preprocess

    if len(sys.argv) < 2:
        print("Usage: python stage2_detect.py <image_path> [conf] [iou] [imgsz]")
        print("Example: python stage2_detect.py photo.jpg 0.35 0.45 640")
        sys.exit(1)

    img, _ = load_and_preprocess(sys.argv[1])
    conf   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    iou    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
    isz    = int(sys.argv[4])   if len(sys.argv) > 4 else 640

    detections, annotated = detect_objects(
        img,
        confidence_threshold=conf,
        iou_threshold=iou,
        imgsz=isz,
    )

    print("\n── Detections ──────────────────────────")
    for d in detections:
        print(
            f"  [#{d['index']}] {d['label']:15s} "
            f"conf={d['confidence']:.2f}  box={d['box']}"
        )

    out_path = "output_stage2.jpg"
    annotated.save(out_path)
    print(f"\nSaved → {out_path}")