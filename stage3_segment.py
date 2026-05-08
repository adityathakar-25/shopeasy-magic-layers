"""
stage3_segment.py — Pixel-precise segmentation using SAM (Segment Anything Model).

Improvements over v1:
  ✅ Image encoding cache — set_image() skipped if same image re-submitted (saves 2–3s)
  ✅ Batched box inference — predict_torch() runs ALL objects in one forward pass
     instead of N separate predict() calls  (was the #1 bottleneck)
  ✅ Numpy-only overlay pipeline — reduced PIL↔cv2 conversions from N×4 → 2 total
  ✅ GPU auto-detection + torch.cuda.empty_cache() after inference
  ✅ SAM download progress bar (urllib reporthook)
  ✅ Inference timing logged for every run
  ✅ Graceful fallback: if batched path fails, reverts to per-object predict()
"""

import hashlib
import time
import os
import numpy as np
import cv2
import torch
import urllib.request
from PIL import Image

from utils import get_color, bgr_to_rgba


# ── Constants ────────────────────────────────────────────────────────────────
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_EXPECTED_SIZE_MB = 375

# ── Module-level singletons ──────────────────────────────────────────────────
_sam_predictor = None
_last_image_hash: str = ""   # MD5 of the last image passed to set_image()


# ── Download helper ──────────────────────────────────────────────────────────

def _make_reporthook():
    """Returns a urllib reporthook that prints a simple download progress bar."""
    start = [time.time()]

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb_done = downloaded / 1_048_576
            mb_total = total_size / 1_048_576
            elapsed = time.time() - start[0]
            speed = mb_done / elapsed if elapsed > 0 else 0
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"{mb_done:.1f}/{mb_total:.0f} MB  "
                f"{speed:.1f} MB/s",
                end="", flush=True,
            )
            if downloaded >= total_size:
                print()   # newline after 100%

    return reporthook


def _download_sam_weights() -> None:
    """Download SAM ViT-B weights with a progress bar. Skips if already present."""
    if os.path.exists(SAM_CHECKPOINT):
        size_mb = os.path.getsize(SAM_CHECKPOINT) / 1_048_576
        if size_mb > SAM_EXPECTED_SIZE_MB * 0.95:   # within 5% of expected size
            return   # looks complete
        print(
            f"[Stage 3] ⚠️  Incomplete download detected "
            f"({size_mb:.0f} MB < {SAM_EXPECTED_SIZE_MB} MB). Re-downloading..."
        )
        os.remove(SAM_CHECKPOINT)

    print(f"[Stage 3] Downloading SAM ViT-B weights (~{SAM_EXPECTED_SIZE_MB} MB) → {SAM_CHECKPOINT}")
    print("          This happens only once. Please wait...")
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT, reporthook=_make_reporthook())
    print("[Stage 3] ✅ Download complete.")


# ── Model loading ────────────────────────────────────────────────────────────

def get_sam_predictor():
    """
    Lazy-load and cache the SAM predictor.
    Downloads weights on first call, loads on GPU if available.
    """
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor

    _download_sam_weights()

    # Late import — module loads fine even if segment_anything isn't installed yet
    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 3] Loading SAM ViT-B on {device.upper()}...")

    t0 = time.perf_counter()
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    _sam_predictor = SamPredictor(sam)

    print(f"[Stage 3] ✅ SAM ready in {time.perf_counter() - t0:.1f}s on {device.upper()}.")
    return _sam_predictor


def warmup() -> None:
    """Pre-load SAM at app startup so first user click has no loading lag."""
    print("[Stage 3] Warming up SAM...")
    get_sam_predictor()
    print("[Stage 3] Warm-up complete.")


# ── Image hash helper ────────────────────────────────────────────────────────

def _compute_image_hash(img_np: np.ndarray) -> str:
    """Fast MD5 hash of a numpy image array used for encoding cache."""
    return hashlib.md5(img_np.tobytes()).hexdigest()


# ── Overlay builder (pure numpy — no per-object PIL/cv2 round-trips) ─────────

def _build_overlay(
    img_np: np.ndarray,
    best_masks: list,
    detections: list,
    alpha: int = 55,
    contour_thickness: int = 2,
) -> np.ndarray:
    """
    Composite all object highlights onto the original image in numpy/cv2.

    Steps:
      1. Paint each object's colour onto an RGBA canvas at mask pixels   (numpy)
      2. Alpha-blend the canvas onto the original image                  (numpy)
      3. Draw contour outlines on the blended result                     (cv2)
      4. Return a uint8 RGB numpy array                                  (no PIL yet)

    Old approach: N objects × 4 PIL↔cv2 round-trips = e.g. 20 conversions for 5 objects
    New approach: 2 cv2 color-space calls total, regardless of N
    """
    H, W = img_np.shape[:2]

    # ── Step 1: Paint mask colours on RGBA canvas ─────────────────────────
    overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for det, mask in zip(detections, best_masks):
        b_bgr, g_bgr, r_bgr = det["color_bgr"]   # color_bgr is (B,G,R)
        # canvas is RGBA → store as R, G, B, A
        overlay_rgba[mask] = [r_bgr, g_bgr, b_bgr, alpha]

    # ── Step 2: Alpha blend in numpy ─────────────────────────────────────
    # blended[px] = overlay_color * a + original * (1 - a)
    a = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    blended = (
        overlay_rgba[:, :, :3].astype(np.float32) * a
        + img_np.astype(np.float32) * (1.0 - a)
    ).astype(np.uint8)

    # ── Step 3: Draw contours (cv2 expects BGR) ───────────────────────────
    result_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    for det, mask in zip(detections, best_masks):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result_bgr, contours, -1, det["color_bgr"], thickness=contour_thickness)

    # ── Step 4: Back to RGB numpy ─────────────────────────────────────────
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# ── Main segmentation ─────────────────────────────────────────────────────────

def segment_objects(image: Image.Image, detections: list):
    """
    Generate pixel-level masks for each detected object using SAM.

    Key optimisations:
      • Image encoding is cached — calling with the same image twice skips
        the expensive set_image() ViT encode step (~2–3 s on CPU).
      • All bounding boxes are passed in a single predict_torch() call,
        replacing the old per-object predict() loop.
      • Overlay compositing is fully numpy-based (no PIL round-trips per object).

    Args:
        image      : Preprocessed PIL Image (RGB) from Stage 1
        detections : List of detection dicts from Stage 2

    Returns:
        masks_dict       : {"label_index": np.ndarray bool (H×W)} — one mask per object
        highlighted_image: PIL Image with semi-transparent colour overlays + contours
    """
    if not detections:
        print("[Stage 3] No detections — skipping segmentation.")
        return {}, image.copy()

    predictor = get_sam_predictor()
    img_np    = np.array(image)      # HxWx3 uint8 RGB

    # ── Image encoding cache ──────────────────────────────────────────────
    global _last_image_hash
    current_hash = _compute_image_hash(img_np)

    t_encode = 0.0
    if current_hash != _last_image_hash:
        t0 = time.perf_counter()
        predictor.set_image(img_np)           # expensive ViT backbone encode
        t_encode = time.perf_counter() - t0
        _last_image_hash = current_hash
        print(f"[Stage 3] Image encoded in {t_encode*1000:.0f} ms")
    else:
        print("[Stage 3] ⚡ Same image — skipping encode (cache hit)")

    # ── Batched box inference ─────────────────────────────────────────────
    # predict_torch() accepts ALL boxes at once → single forward pass
    # ── Batched box inference ─────────────────────────────────────────────
    # Expand boxes by 8% to capture straps/handles YOLO's tight box clips
    H_img, W_img = img_np.shape[:2]
    raw_boxes = np.array([d["box"] for d in detections], dtype=np.float32)
    expanded  = raw_boxes.copy()
    widths    = raw_boxes[:, 2] - raw_boxes[:, 0]
    heights   = raw_boxes[:, 3] - raw_boxes[:, 1]
    pad_x     = widths  * 0.08
    pad_y     = heights * 0.08
    expanded[:, 0] = np.clip(raw_boxes[:, 0] - pad_x, 0, W_img)
    expanded[:, 1] = np.clip(raw_boxes[:, 1] - pad_y, 0, H_img)
    expanded[:, 2] = np.clip(raw_boxes[:, 2] + pad_x, 0, W_img)
    expanded[:, 3] = np.clip(raw_boxes[:, 3] + pad_y, 0, H_img)
    boxes_np  = expanded

    best_masks: list = []
    t0 = time.perf_counter()

    try:
        boxes_tensor = torch.as_tensor(
            boxes_np, dtype=torch.float, device=predictor.device
        )
        # Apply SAM's own coordinate transform (handles padding/scaling)
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_tensor, img_np.shape[:2]
        )
        with torch.no_grad():
            # masks_batch : (N, 3, H, W) — 3 candidates per object
            # scores_batch: (N, 3)
            masks_batch, scores_batch, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=True,
            )

        # Pick highest-scoring mask for each object
        for i in range(len(detections)):
            best_idx = int(scores_batch[i].argmax())
            best_masks.append(masks_batch[i, best_idx].cpu().numpy())   # (H,W) bool

        t_infer = time.perf_counter() - t0
        print(
            f"[Stage 3] ✅ Batched inference: {len(detections)} object(s) "
            f"in {t_infer*1000:.0f} ms  (encode: {t_encode*1000:.0f} ms)"
        )

    except Exception as exc:
        # ── Fallback: per-object predict() ───────────────────────────────
        print(f"[Stage 3] ⚠️  Batched path failed ({exc}), falling back to per-object predict()...")
        best_masks = []
        for det in detections:
            box_np = np.array(det["box"], dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=box_np,
                multimask_output=True,
            )
            best_masks.append(masks[int(np.argmax(scores))])

    # ── Build masks_dict ──────────────────────────────────────────────────
    masks_dict = {}
    for det, mask in zip(detections, best_masks):
        key = f"{det['label']}_{det['index']}"
        masks_dict[key] = mask
        print(
            f"[Stage 3]   '{key}'  "
            f"mask_pixels={int(mask.sum())}  "
            f"coverage={mask.mean()*100:.1f}%"
        )

    # ── Build highlighted image (numpy pipeline) ──────────────────────────
    result_rgb = _build_overlay(img_np, best_masks, detections)
    highlighted = Image.fromarray(result_rgb)

    # ── GPU cleanup ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return masks_dict, highlighted


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from stage1_preprocess import load_and_preprocess
    from stage2_detect     import detect_objects

    if len(sys.argv) < 2:
        print("Usage: python stage3_segment.py <image_path>")
        sys.exit(1)

    img, _         = load_and_preprocess(sys.argv[1])
    detections, _  = detect_objects(img)
    masks, highlighted = segment_objects(img, detections)

    print(f"\nMask keys: {list(masks.keys())}")

    out_path = "output_stage3.jpg"
    highlighted.save(out_path)
    print(f"Saved → {out_path}")

    # Demonstrate cache: running a second time should skip encoding
    print("\n── Cache demo: running again on same image ──")
    masks2, _ = segment_objects(img, detections)
    print("Done — encoding should have been skipped above.")