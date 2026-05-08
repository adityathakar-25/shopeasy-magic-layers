"""
utils.py — Shared helper functions used across all stages.
"""

import numpy as np
from PIL import Image
import cv2
import random

# Fixed color palette for up to 20 objects — BGR for OpenCV, RGBA for PIL
COLORS_BGR = [
    (255, 56,  56 ), (255, 157, 151), (255, 112, 31 ), (255, 178, 29 ),
    (207, 210, 49 ), (72,  249, 10 ), (146, 204, 23 ), (61,  219, 134),
    (26,  147, 52 ), (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,  69,  147), (100, 115, 255), (0,   24,  236), (132, 56,  255),
    (82,  0,   133), (203, 56,  255), (255, 149, 200), (255, 55,  199),
]

def get_color(index):
    """Return (B, G, R) color for a given object index."""
    return COLORS_BGR[index % len(COLORS_BGR)]

def bgr_to_rgba(bgr_color, alpha=160):
    """Convert (B,G,R) to (R,G,B,A) for PIL overlay."""
    b, g, r = bgr_color
    return (r, g, b, alpha)

def pil_to_cv2(pil_img):
    """Convert PIL Image (RGB) to OpenCV array (BGR)."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_img):
    """Convert OpenCV array (BGR) to PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def resize_to_max(image: Image.Image, max_size=1024) -> Image.Image:
    """Resize image so longest side <= max_size, keeping aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Image.Resampling.LANCZOS is the correct API in Pillow >= 10
    # Fallback keeps compatibility with older Pillow
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((new_w, new_h), resample)

def draw_text_with_bg(img_cv2, text, pos, font_scale=0.55, color_bgr=(255,255,255), bg_bgr=(0,0,0)):
    """Draw text with a solid background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Background rectangle
    cv2.rectangle(img_cv2, (x, y - th - baseline - 4), (x + tw + 4, y + baseline), bg_bgr, -1)
    # Text
    cv2.putText(img_cv2, text, (x + 2, y - 2), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
    return img_cv2