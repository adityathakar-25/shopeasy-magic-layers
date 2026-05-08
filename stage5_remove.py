"""
stage5_remove.py — Object removal with IOPaint LaMa (HTTP server) + Gaussian fill fallback.

Architecture:
  Primary  : IOPaint LaMa HTTP server  — model loads ONCE at startup, ~2s per removal
             Auto-starts server if not running. Works on any background.
  Fallback : Gaussian fill             — instant, CPU only, for flat studio backgrounds

IOPaint server lifecycle (managed automatically):
  - On first removal call, stage5 checks if IOPaint is running at localhost:8081
  - If not running, it starts it as a background process and waits for ready
  - Server stays running for the entire app session
  - On app exit, server is shut down cleanly

Setup:
  pip install iopaint
  Then just run: python app.py  (server starts automatically)
"""

import os
import io
import time
import base64
import subprocess
import tempfile
import threading
import shutil
import signal
import atexit
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from utils import pil_to_cv2, cv2_to_pil

# ── IOPaint server config ─────────────────────────────────────────────────────
IOPAINT_HOST    = "127.0.0.1"
IOPAINT_PORT    = 8081
IOPAINT_URL     = f"http://{IOPAINT_HOST}:{IOPAINT_PORT}"
IOPAINT_TIMEOUT = 120   # seconds to wait for server startup

# Module-level server process handle
_iopaint_proc: subprocess.Popen = None
_server_ready:  bool = False
_server_lock    = threading.Lock()


# ── Replicate LaMa model ID ───────────────────────────────────────────────────
# This is the official LaMa big model on Replicate
REPLICATE_MODEL = "andreasjansson/lama-cleaner-lqhq:c50e0bce-2a36-4de5-80b7-7f7a3f71b963"
# Fallback model if above is deprecated
REPLICATE_MODEL_FALLBACK = "stability-ai/stable-diffusion-inpainting"


# ── Mask helpers ──────────────────────────────────────────────────────────────

def _validate_mask(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    """Validate shape/dtype. Returns uint8 mask (0 or 1)."""
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Mask must be numpy array, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D (H×W), got {mask.shape}")
    img_w, img_h = image.size
    if mask.shape != (img_h, img_w):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image size {(img_h, img_w)}"
        )
    return (mask > 0).astype(np.uint8)


def _smart_dilate(mask_u8: np.ndarray, base_dilation: int = 8) -> np.ndarray:
    """
    Expand mask outward — radius scales with object size.
    Also closes interior holes (zipper hardware, buckles).
    Returns uint8 mask (0 or 255).
    """
    rows = np.where(np.any(mask_u8, axis=1))[0]
    cols = np.where(np.any(mask_u8, axis=0))[0]
    if rows.size and cols.size:
        obj_h    = int(rows[-1] - rows[0] + 1)
        obj_w    = int(cols[-1] - cols[0] + 1)
        scale    = min(max(obj_h, obj_w) / 200, 3.0)
        dilation = max(base_dilation, int(base_dilation * scale))
    else:
        dilation = base_dilation

    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
    )
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)

    # Close interior holes (zipper hardware, gaps between handles)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closed  = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_k)

    return (closed > 0).astype(np.uint8) * 255


def _add_shadow_region(mask_u8: np.ndarray) -> np.ndarray:
    """
    Extend mask downward to cover the drop shadow below the object.
    Shadow extension = 18% of object height.
    """
    rows = np.where(np.any(mask_u8, axis=1))[0]
    cols = np.where(np.any(mask_u8, axis=0))[0]
    if not (rows.size and cols.size):
        return mask_u8

    H_img        = mask_u8.shape[0]
    obj_h        = int(rows[-1] - rows[0] + 1)
    shadow_px    = int(obj_h * 0.18)
    result       = mask_u8.copy()

    y_start = min(int(rows[-1]), H_img - 1)
    y_end   = min(int(rows[-1]) + shadow_px, H_img - 1)
    x_start = int(cols[0])
    x_end   = int(cols[-1])

    result[y_start:y_end, x_start:x_end] = 255
    return result


# ── PIL ↔ bytes helpers ───────────────────────────────────────────────────────

def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ── Background fill (CPU fallback) ────────────────────────────────────────────

def _sample_boundary_color(img_np: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Sample median background color from a 60px ring outside the mask."""
    kernel   = np.ones((60, 60), np.uint8)
    dilated  = cv2.dilate(mask_u8, kernel, iterations=1)
    boundary = ((dilated - mask_u8) > 0)
    pixels   = img_np[boundary]
    if len(pixels) == 0:
        return np.array([240, 240, 240], dtype=np.uint8)
    return np.median(pixels, axis=0).astype(np.uint8)


def _gaussian_fill(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    CPU fallback: two-pass Gaussian fill for flat/gradient backgrounds.
    Quality: good for studio shots, poor for complex scenes.
    """
    img_np  = np.array(image.convert("RGB"), dtype=np.float32)
    mask_b  = (mask.astype(np.uint8) > 0)
    mask_u8 = mask_b.astype(np.uint8)

    bg_color = _sample_boundary_color(img_np.astype(np.uint8), mask_u8)

    # Pre-fill masked region with background color
    filled          = img_np.copy()
    filled[mask_b]  = bg_color.astype(np.float32)

    # Two-pass Gaussian blur
    H, W = img_np.shape[:2]
    k1   = min(int(max(H, W) * 0.4) | 1, 501)
    k2   = min(int(max(H, W) * 0.15) | 1, 201)

    pass1          = cv2.GaussianBlur(filled, (k1, k1), k1 / 3.0)
    seeded         = filled.copy()
    seeded[mask_b] = pass1[mask_b]

    pass2          = cv2.GaussianBlur(seeded, (k2, k2), k2 / 3.0)
    result         = img_np.astype(np.uint8).copy()
    result[mask_b] = pass2[mask_b].clip(0, 255).astype(np.uint8)

    print(f"[Stage 5] Gaussian fill complete  bg=RGB{tuple(bg_color)}")
    return Image.fromarray(result)


# ── IOPaint server management ─────────────────────────────────────────────────

def _is_server_running() -> bool:
    """Check if IOPaint server is already running by hitting its health endpoint."""
    try:
        import urllib.request
        urllib.request.urlopen(f"{IOPAINT_URL}/api/v1/model", timeout=3)
        return True
    except Exception:
        return False


def _start_iopaint_server() -> bool:
    """
    Start IOPaint as a background process and wait until it is ready.
    Returns True if server is ready, False if startup failed.

    Key design decisions:
    - Uses Popen (not run) so it doesn't block — runs in background
    - Polls /api/v1/model every second until ready or timeout
    - Registers atexit handler so server shuts down cleanly when app exits
    - Thread-safe via _server_lock — safe to call from multiple threads
    """
    global _iopaint_proc, _server_ready

    with _server_lock:
        # Already confirmed running this session
        if _server_ready:
            return True

        # Someone else started it externally — that's fine, use it
        if _is_server_running():
            print("[Stage 5] IOPaint server already running — using it.")
            _server_ready = True
            return True

        # Find iopaint executable
        iopaint_exe = shutil.which("iopaint")
        if not iopaint_exe:
            print("[Stage 5] ⚠️  'iopaint' not found in PATH. Run: pip install iopaint")
            return False

        print("[Stage 5] Starting IOPaint LaMa server (loads model once)...")
        print("          First startup takes ~15s to load LaMa weights.")
        print("          All subsequent removals will take ~2s.")

        try:
            # Start server as background process
            # stdout/stderr to DEVNULL keeps our terminal clean
            _iopaint_proc = subprocess.Popen(
                [
                    iopaint_exe, "start",
                    "--model", "lama",
                    "--device", "cpu",
                    "--host", IOPAINT_HOST,
                    "--port", str(IOPAINT_PORT),
                    "--no-half",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[Stage 5] ⚠️  Failed to start IOPaint: {e}")
            return False

        # Register shutdown hook — cleans up server when Python exits
        atexit.register(_stop_iopaint_server)

        # Poll until ready
        deadline = time.time() + IOPAINT_TIMEOUT
        while time.time() < deadline:
            if _iopaint_proc.poll() is not None:
                print("[Stage 5] ⚠️  IOPaint process exited unexpectedly.")
                return False
            if _is_server_running():
                print(f"[Stage 5] ✅ IOPaint server ready at {IOPAINT_URL}")
                _server_ready = True
                return True
            time.sleep(1)

        print(f"[Stage 5] ⚠️  IOPaint server did not start within {IOPAINT_TIMEOUT}s.")
        _iopaint_proc.terminate()
        return False


def _stop_iopaint_server():
    """Gracefully stop the IOPaint server. Called automatically on app exit."""
    global _iopaint_proc, _server_ready
    if _iopaint_proc and _iopaint_proc.poll() is None:
        print("[Stage 5] Shutting down IOPaint server...")
        _iopaint_proc.terminate()
        try:
            _iopaint_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _iopaint_proc.kill()
        _server_ready = False
        print("[Stage 5] IOPaint server stopped.")


def _remove_with_iopaint_server(image: Image.Image, mask_255: np.ndarray) -> Image.Image:
    """
    Send image + mask to IOPaint HTTP server using base64 JSON.
    Multipart binary causes UnicodeDecodeError in IOPaint 1.6.0 (known bug).
    Base64 JSON is the correct workaround.
    """
    import urllib.request
    import json

    # Encode image and mask as base64 PNG strings
    img_buf  = io.BytesIO()
    mask_buf = io.BytesIO()
    image.convert("RGB").save(img_buf, format="PNG")
    Image.fromarray(mask_255).convert("L").save(mask_buf, format="PNG")

    img_b64  = base64.b64encode(img_buf.getvalue()).decode("utf-8")
    mask_b64 = base64.b64encode(mask_buf.getvalue()).decode("utf-8")

    payload = json.dumps({
        "image": img_b64,
        "mask":  mask_b64,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{IOPAINT_URL}/api/v1/inpaint",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    print("[Stage 5] Sending to IOPaint LaMa server...")
    with urllib.request.urlopen(req, timeout=120) as resp:
        result_bytes = resp.read()

    elapsed = time.perf_counter() - t0
    print(f"[Stage 5] ✅ IOPaint done in {elapsed:.1f}s")
    return Image.open(io.BytesIO(result_bytes)).convert("RGB")


def ensure_iopaint_ready() -> bool:
    """
    Public function — call this at app startup to pre-load the LaMa model.
    Returns True if IOPaint is ready, False if unavailable.
    Calling this at startup means the first user removal takes ~2s not ~17s.
    """
    return _start_iopaint_server()


def _remove_with_iopaint_local(image: Image.Image, mask_255: np.ndarray) -> Image.Image:
    """
    Main IOPaint removal function.
    Starts server automatically if not running, then calls HTTP API.
    """
    if not _start_iopaint_server():
        raise RuntimeError(
            "IOPaint server could not be started.\n"
            "Ensure iopaint is installed: pip install iopaint"
        )
    return _remove_with_iopaint_server(image, mask_255)


def _remove_with_replicate(image: Image.Image, mask_255: np.ndarray) -> Image.Image:
    """
    Send image + mask to Replicate LaMa API. Returns inpainted PIL Image.

    Args:
        image    : Original PIL Image RGB
        mask_255 : uint8 mask, 255 = region to remove
    """
    try:
        import replicate
    except ImportError:
        raise RuntimeError(
            "replicate package not installed.\n"
            "Run: pip install replicate"
        )

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN not set.\n"
            "Run in PowerShell: $env:REPLICATE_API_TOKEN = 'your_token'\n"
            "Or permanently:    setx REPLICATE_API_TOKEN 'your_token'"
        )

    # ── Prepare image bytes ───────────────────────────────────────────────
    img_bytes  = _pil_to_bytes(image.convert("RGB"))
    mask_pil   = Image.fromarray(mask_255).convert("L")
    mask_bytes = _pil_to_bytes(mask_pil)

    # Replicate accepts file-like objects directly
    img_file  = io.BytesIO(img_bytes)
    mask_file = io.BytesIO(mask_bytes)

    print("[Stage 5] Sending to Replicate LaMa API...")
    t0 = time.perf_counter()

    try:
        import urllib.request

        output = replicate.run(
            "zylim0702/remove-object:0e3a841c913f597c1e4c321560aa69e2bc1f15c65f8c366caafc379240efd8ba",
            input={
                "image": img_file,
                "mask":  mask_file,
            }
        )

        # output is a URL string or file-like — handle both
        if hasattr(output, "read"):
            result_bytes = output.read()
        else:
            with urllib.request.urlopen(str(output)) as resp:
                result_bytes = resp.read()

        elapsed = time.perf_counter() - t0
        print(f"[Stage 5] ✅ Replicate LaMa done in {elapsed:.1f}s")
        return _bytes_to_pil(result_bytes)

    except Exception as e:
        raise RuntimeError(f"Replicate API call failed: {e}") from e


# ── Background type detection ─────────────────────────────────────────────────

def detect_background_type(image: Image.Image, mask: np.ndarray) -> str:
    """
    Classify background as 'uniform', 'gradient', or 'complex'.
    Used to decide whether cloud API is needed or Gaussian fill is sufficient.
    """
    img_np  = np.array(image.convert("RGB"), dtype=np.float32)
    mask_u8 = (mask > 0).astype(np.uint8)

    # Sample a ring of pixels outside the mask
    kernel   = np.ones((30, 30), np.uint8)
    dilated  = cv2.dilate(mask_u8, kernel, iterations=1)
    ring     = ((dilated - mask_u8) > 0)
    pixels   = img_np[ring]

    if len(pixels) < 10:
        return "uniform"

    mean_std = float(pixels.std(axis=0).mean())
    print(f"[Stage 5] Background analysis: boundary_std={mean_std:.1f}")

    if mean_std < 12:
        return "uniform"
    elif mean_std < 60:
        return "gradient"
    return "complex"


# ── Main entry point ──────────────────────────────────────────────────────────

def remove_object(
    image: Image.Image,
    mask: np.ndarray,
    feather_px: int = 0,
    use_poisson: bool = False,
    method: str = "auto",
) -> Image.Image:
    """
    Remove the masked object from the image.

    Args:
        image  : Original PIL Image (RGB)
        mask   : Boolean/uint8 np.ndarray (H×W) — True = region to remove
        feather_px : kept for backward compatibility with app.py
        use_poisson : kept for backward compatibility with app.py
        method : "auto"     — local LaMa for non-uniform bg, Gaussian for simple
                 "local"    — Force local IOPaint LaMa
                 "lama"     — same as "local"
                 "replicate" — Force Replicate LaMa (needs API key)
                 "fill"      — Force Gaussian fill (instant, CPU, studio shots only)
                 "opencv" / "ns" — map to Gaussian fill for UI compatibility

    Returns:
        PIL Image with object removed
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    mask_u8      = _validate_mask(mask, image)
    dilated      = _smart_dilate(mask_u8, base_dilation=8)
    with_shadow  = _add_shadow_region(dilated)
    final_mask   = with_shadow   # uint8, values 0 or 255

    method = method.strip().lower()

    # ── Force fill ────────────────────────────────────────────────────────
    if method in ("fill", "opencv", "ns"):
        print("[Stage 5] Method: Gaussian fill (forced)")
        return _gaussian_fill(image, final_mask)

    # ── Force local LaMa ──────────────────────────────────────────────────
    if method in ("local", "lama"):
        print("[Stage 5] Method: Local IOPaint LaMa (forced)")
        try:
            return _remove_with_iopaint_local(image, final_mask)
        except Exception as e:
            print(f"[Stage 5] ⚠️  Local IOPaint failed: {e}\n[Stage 5] Falling back to Gaussian fill.")
            return _gaussian_fill(image, final_mask)

    # ── Force Replicate ───────────────────────────────────────────────────
    if method == "replicate":
        print("[Stage 5] Method: Replicate LaMa (forced)")
        try:
            return _remove_with_replicate(image, final_mask)
        except Exception as e:
            print(f"[Stage 5] ⚠️  Replicate failed: {e}\n[Stage 5] Falling back to local IOPaint LaMa.")
            try:
                return _remove_with_iopaint_local(image, final_mask)
            except Exception:
                return _gaussian_fill(image, final_mask)

    # ── Auto: choose based on background type ─────────────────────────────
    bg_type = detect_background_type(image, mask)
    print(f"[Stage 5] Auto mode → background='{bg_type}'")

    if bg_type == "uniform":
        print("[Stage 5] → Gaussian fill (uniform background)")
        return _gaussian_fill(image, final_mask)

    print("[Stage 5] → Local IOPaint LaMa (non-uniform background)")
    try:
        return _remove_with_iopaint_local(image, final_mask)
    except Exception as e:
        print(f"[Stage 5] ⚠️  Local IOPaint failed: {e}")
        token = bool(os.environ.get("REPLICATE_API_TOKEN", "").strip())
        if token:
            try:
                print("[Stage 5] → Replicate LaMa fallback")
                return _remove_with_replicate(image, final_mask)
            except Exception as e2:
                print(f"[Stage 5] ⚠️  Replicate failed: {e2}")
        print("[Stage 5] → Gaussian fill fallback")
        return _gaussian_fill(image, final_mask)

def remove_multiple(
    image: Image.Image,
    masks: dict,
    keys: list,
    method: str = "auto",
) -> Image.Image:
    """
    Remove multiple objects in one pass — union all masks then inpaint once.
    Avoids artifacts from sequential inpainting passes.

    Args:
        image  : Original PIL Image
        masks  : masks_dict from Stage 3
        keys   : list of layer keys to remove e.g. ["handbag_0", "person_1"]
        method : same as remove_object()
    """
    if not keys:
        return image

    H, W      = np.array(image).shape[:2]
    union     = np.zeros((H, W), dtype=np.uint8)
    for key in keys:
        if key in masks:
            union = np.clip(union + (masks[key] > 0).astype(np.uint8), 0, 1)

    print(f"[Stage 5] Removing {len(keys)} object(s) in one pass: {keys}")
    return remove_object(image, union, method=method)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from stage1_preprocess import load_and_preprocess
    from stage2_detect     import detect_objects
    from stage3_segment    import segment_objects

    if len(sys.argv) < 2:
        print("Usage: python stage5_remove.py <image> [object_key] [method]")
        print("  method: auto | local | lama | replicate | fill | opencv | ns")
        print("  example: python stage5_remove.py img-3.png handbag_0 auto")
        sys.exit(1)

    path   = sys.argv[1]
    key    = sys.argv[2] if len(sys.argv) > 2 else None
    method = sys.argv[3] if len(sys.argv) > 3 else "auto"

    img, _        = load_and_preprocess(path)
    detections, _ = detect_objects(img)
    masks, _      = segment_objects(img, detections)

    print(f"\nAvailable objects: {list(masks.keys())}")

    if key is None:
        key = list(masks.keys())[0]
        print(f"No key specified — using first object: '{key}'")

    if key not in masks:
        print(f"Key '{key}' not found. Available: {list(masks.keys())}")
        sys.exit(1)

    result   = remove_object(img, masks[key], method=method)
    out_path = f"output_stage5_{key}_{method}.jpg"
    result.save(out_path)
    print(f"\n✅ Saved → {out_path}")