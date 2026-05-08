"""
stage1_preprocess.py — Image upload and preprocessing.

What this does:
  1. Accepts a file path or PIL Image as input
  2. Validates the input and raises clear errors on bad files
  3. Converts to RGB (handles RGBA, grayscale, palette images)
  4. Resizes so longest edge <= max_size px (keeps aspect ratio)
  5. Returns a clean PIL Image + metadata dict ready for all downstream stages

Why:
  - YOLOv8 and SAM both expect RGB, not BGR or RGBA
  - Capping resolution keeps inference fast without losing detail
  - Metadata lets the UI show original vs processed dimensions
"""

from PIL import Image, UnidentifiedImageError
import numpy as np
from utils import resize_to_max


def load_and_preprocess(image_input, max_size: int = 1024):
    """
    Load an image from a file path or PIL Image object.

    Args:
        image_input : str (file path) | PIL.Image.Image | np.ndarray
        max_size    : Resize so longest edge <= this value (pixels)

    Returns:
        img      : PIL.Image.Image  — RGB, longest side <= max_size
        metadata : dict             — original_size, final_size, mode, source

    Raises:
        ValueError  — wrong input type or unreadable file
        FileNotFoundError — path does not exist
    """
    original_size = None
    source = "unknown"

    # ── Step 1: Load from the supplied input ──────────────────────────────
    if isinstance(image_input, str):
        source = image_input
        try:
            img = Image.open(image_input)
            img.verify()                  # catches truncated / corrupt files
            img = Image.open(image_input) # reopen after verify (verify() closes)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_input!r}")
        except UnidentifiedImageError:
            raise ValueError(f"Cannot identify image file (corrupt or unsupported format): {image_input!r}")
        except Exception as e:
            raise ValueError(f"Failed to open image '{image_input}': {e}")

    elif isinstance(image_input, np.ndarray):
        # Gradio sometimes delivers numpy arrays instead of PIL Images
        source = "numpy_array"
        try:
            img = Image.fromarray(image_input.astype(np.uint8))
        except Exception as e:
            raise ValueError(f"Could not convert numpy array to PIL Image: {e}")

    elif isinstance(image_input, Image.Image):
        source = "pil_image"
        img = image_input

    else:
        raise ValueError(
            f"Expected file path (str), PIL Image, or numpy array — got {type(image_input).__name__}"
        )

    original_size = img.size   # (width, height) before any conversion

    # ── Step 2: Convert any mode to plain RGB ─────────────────────────────
    # Handles: RGBA (transparency), P (palette/GIF), L (grayscale), CMYK, etc.
    original_mode = img.mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    # ── Step 3: Resize so longest side is at most max_size px ─────────────
    img = resize_to_max(img, max_size=max_size)

    metadata = {
        "source":        source,
        "original_size": original_size,
        "original_mode": original_mode,
        "final_size":    img.size,
        "max_size_cap":  max_size,
        "was_resized":   img.size != original_size,
    }

    print(
        f"[Stage 1] Loaded | original: {original_size} {original_mode} "
        f"→ final: {img.size} RGB | source: {source}"
    )
    return img, metadata


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stage1_preprocess.py <image_path> [max_size]")
        print("Example: python stage1_preprocess.py photo.jpg 1024")
        sys.exit(1)

    path = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

    try:
        result, meta = load_and_preprocess(path, max_size=size)
    except (ValueError, FileNotFoundError) as err:
        print(f"ERROR: {err}")
        sys.exit(1)

    print("\n── Metadata ─────────────────────────────")
    for k, v in meta.items():
        print(f"  {k:>15}: {v}")

    out_path = "output_stage1.jpg"
    result.save(out_path)
    print(f"\nSaved preprocessed image → {out_path}")