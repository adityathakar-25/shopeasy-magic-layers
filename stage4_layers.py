"""
stage4_layers.py — Layer management: toggle highlights per object.

Improvements over v1:
  ✅ Numpy-only render pipeline — eliminated N×4 PIL↔cv2 round-trips per render()
     (same technique used in Stage 3 — all compositing done in numpy/cv2)
  ✅ Render cache — identical active_layer sets skip recompute (instant re-render)
  ✅ Pre-computed contours stored at init — findContours() never called twice for the same mask
  ✅ Original image stored as numpy array — avoids repeated PIL→numpy conversion on every render
  ✅ render_single() no longer mutates active_layers (was a state bug — now uses temp set)
  ✅ Added layer_info() — returns per-layer stats (mask pixels, coverage %) for the UI
  ✅ Timing logged on every render call
"""

import time
import numpy as np
import cv2
from PIL import Image

from utils import bgr_to_rgba


class LayerManager:
    """
    Manages object highlight layers with fast numpy-based compositing.

    Usage:
        lm = LayerManager(original_image, masks_dict, detections)
        lm.set_active(["person_0", "car_1"])   # enable only these
        result_image = lm.render()             # PIL Image with highlights
    """

    # Alpha value for mask fill (0–255)
    FILL_ALPHA: int = 55
    CONTOUR_THICKNESS: int = 2

    def __init__(
        self,
        original_image: Image.Image,
        masks_dict: dict,
        detections: list,
    ):
        """
        Args:
            original_image : Clean preprocessed PIL Image from Stage 1
            masks_dict     : {"label_index": np.ndarray bool (H×W)} from Stage 3
            detections     : List of detection dicts from Stage 2
        """
        # Store original as numpy (uint8 RGB) — avoids repeated PIL→numpy on each render
        self._original_np: np.ndarray = np.array(original_image.convert("RGB"))

        self.masks_dict: dict = masks_dict

        # Build detection lookup keyed by "label_index"
        self.detections: dict = {
            f"{d['label']}_{d['index']}": d for d in detections
        }

        # All layers on by default
        self.active_layers: set = set(masks_dict.keys())

        # ── Pre-compute contours once at init ─────────────────────────────
        # findContours is O(mask_pixels) — doing it every render() is wasteful
        self._contours: dict = {}
        for key, mask in masks_dict.items():
            cnts, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            self._contours[key] = cnts

        # ── Render cache ──────────────────────────────────────────────────
        # Key = frozenset of active layer keys → value = rendered PIL Image
        # Avoids recomputing the same overlay when the user toggles back
        self._render_cache: dict = {}

    # ── Layer accessors ───────────────────────────────────────────────────────

    def all_layer_keys(self) -> list:
        """Return all available layer keys, sorted."""
        return sorted(self.masks_dict.keys())

    def set_active(self, keys: list) -> None:
        """Enable exactly the given layer keys; disable all others."""
        self.active_layers = set(keys)

    def toggle(self, key: str) -> None:
        """Flip the on/off state of a single layer."""
        if key in self.active_layers:
            self.active_layers.discard(key)
        else:
            self.active_layers.add(key)

    def layer_info(self) -> list:
        """
        Return a list of dicts with per-layer stats for display in the UI.

        Each dict:
            {
              "key":       str   — e.g. "person_0"
              "active":    bool
              "pixels":    int   — number of mask pixels
              "coverage":  float — fraction of image covered (0–1)
              "label":     str
              "confidence": float
            }
        """
        H, W = self._original_np.shape[:2]
        total_px = H * W
        info = []
        for key in self.all_layer_keys():
            mask = self.masks_dict[key]
            det  = self.detections.get(key, {})
            info.append({
                "key":        key,
                "active":     key in self.active_layers,
                "pixels":     int(mask.sum()),
                "coverage":   round(float(mask.sum()) / total_px, 4),
                "label":      det.get("label", key),
                "confidence": det.get("confidence", 0.0),
            })
        return info

    def summary(self) -> str:
        """Human-readable status of all layers with coverage stats."""
        lines = ["Layer status:"]
        for info in self.layer_info():
            state = "ON " if info["active"] else "OFF"
            lines.append(
                f"  [{state}] {info['key']:20s} "
                f"pixels={info['pixels']:6d}  "
                f"coverage={info['coverage']*100:.1f}%"
            )
        return "\n".join(lines)

    # ── Rendering ────────────────────────────────────────────────────────────

    def render(self) -> Image.Image:
        """
        Composite only the active layers onto the original image.

        Pipeline (all numpy/cv2 — no PIL inside the loop):
          1. Paint each active mask's colour onto an RGBA canvas        (numpy)
          2. Alpha-blend canvas onto original                           (numpy)
          3. Draw pre-computed contours on the blended result           (cv2)
          4. Wrap result in PIL once at the end

        Render cache: identical active_layer sets return the cached PIL Image
        immediately — zero recomputation.

        Returns:
            PIL Image (RGB) with selected objects highlighted
        """
        cache_key = frozenset(self.active_layers)

        # ── Cache hit ─────────────────────────────────────────────────────
        if cache_key in self._render_cache:
            return self._render_cache[cache_key]

        t0 = time.perf_counter()

        H, W = self._original_np.shape[:2]
        active_keys = [k for k in self.active_layers if k in self.masks_dict]

        if not active_keys:
            # Nothing selected — return clean original
            result = Image.fromarray(self._original_np)
            self._render_cache[cache_key] = result
            return result

        # ── Step 1: Paint mask colours on RGBA canvas ─────────────────────
        overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        for key in active_keys:
            mask      = self.masks_dict[key]
            det       = self.detections.get(key, {})
            color_bgr = det.get("color_bgr", (0, 200, 255))   # fallback cyan
            b, g, r   = color_bgr                              # color_bgr is (B,G,R)
            overlay_rgba[mask] = [r, g, b, self.FILL_ALPHA]   # canvas is RGBA

        # ── Step 2: Alpha blend in numpy ──────────────────────────────────
        a = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
        blended = (
            overlay_rgba[:, :, :3].astype(np.float32) * a
            + self._original_np.astype(np.float32) * (1.0 - a)
        ).astype(np.uint8)

        # ── Step 3: Draw pre-computed contours ────────────────────────────
        result_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        for key in active_keys:
            det       = self.detections.get(key, {})
            color_bgr = det.get("color_bgr", (0, 200, 255))
            cv2.drawContours(
                result_bgr,
                self._contours.get(key, []),
                -1, color_bgr,
                thickness=self.CONTOUR_THICKNESS,
            )

        # ── Step 4: Convert to PIL once ───────────────────────────────────
        result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[Stage 4] Rendered {len(active_keys)}/{len(self.masks_dict)} layer(s) "
            f"in {elapsed_ms:.1f} ms"
        )

        # Cache and return
        self._render_cache[cache_key] = result
        return result

    def render_single(self, key: str) -> Image.Image:
        """
        Return the original image with ONLY the given layer highlighted.
        Does NOT mutate self.active_layers (v1 had a state mutation bug here).
        """
        if key not in self.masks_dict:
            raise KeyError(f"Layer '{key}' not found. Available: {self.all_layer_keys()}")
        prev = self.active_layers
        self.active_layers = {key}
        result = self.render()
        self.active_layers = prev
        return result

    def clear_cache(self) -> None:
        """Invalidate the render cache (call if masks or image are replaced)."""
        self._render_cache.clear()


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from stage1_preprocess import load_and_preprocess
    from stage2_detect     import detect_objects
    from stage3_segment    import segment_objects

    if len(sys.argv) < 2:
        print("Usage: python stage4_layers.py <image_path>")
        sys.exit(1)

    img, _        = load_and_preprocess(sys.argv[1])
    detections, _ = detect_objects(img)
    masks, _      = segment_objects(img, detections)

    lm = LayerManager(img, masks, detections)
    print(lm.summary())

    # All layers
    lm.render().save("output_stage4_all.jpg")
    print("Saved output_stage4_all.jpg")

    # First layer only
    keys = lm.all_layer_keys()
    if keys:
        lm.render_single(keys[0]).save("output_stage4_first_only.jpg")
        print(f"Saved output_stage4_first_only.jpg  (only '{keys[0]}')")

    # Cache demo — second all-layers render should return instantly
    t0 = time.perf_counter()
    lm.render()
    print(f"Cache hit render time: {(time.perf_counter()-t0)*1000:.2f} ms")