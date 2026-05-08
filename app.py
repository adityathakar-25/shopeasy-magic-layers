"""
app.py — Main Gradio UI for the Shopeasy image processing pipeline.

Tabs:
  Tab 1 — Detect & Segment : Upload → tune settings → YOLO + SAM → objects highlighted
  Tab 2 — Layer Viewer     : Toggle individual object layers on/off
  Tab 3 — Remove Object    : Select one object → inpaint it out → download result

Run:
    python app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
from PIL import Image
import numpy as np

from stage1_preprocess import load_and_preprocess
from stage2_detect      import detect_objects, MODEL_OPTIONS, DEFAULT_MODEL, warmup as warmup_yolo
from stage3_segment     import segment_objects, warmup as warmup_sam
from stage4_layers      import LayerManager
from stage5_remove      import remove_object, detect_background_type, ensure_iopaint_ready

# ── Global state ─────────────────────────────────────────────────────────────
STATE: dict = {
    "original":   None,   # PIL Image (preprocessed)
    "detections": [],     # list of detection dicts from Stage 2
    "masks":      {},     # {key: np.ndarray} from Stage 3
    "layer_mgr":  None,   # LayerManager instance from Stage 4
    "meta":       {},     # metadata from Stage 1
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_image_info(meta: dict) -> str:
    """Build a Markdown string summarising the loaded image."""
    if not meta:
        return ""
    orig  = meta.get("original_size", "?")
    final = meta.get("final_size", "?")
    mode  = meta.get("original_mode", "?")
    cap   = meta.get("max_size_cap", "?")
    was_resized = meta.get("was_resized", False)

    resize_note = (
        f"→ resized to **{final[0]}×{final[1]}** (cap: {cap} px)"
        if was_resized else "*(no resize needed)*"
    )
    return (
        f"📐 **Original:** {orig[0]}×{orig[1]} px &nbsp;|&nbsp; "
        f"**Mode:** `{mode}` &nbsp;|&nbsp; {resize_note}"
    )


def _build_status_md(detections: list) -> str:
    """Format detection results as a Markdown table."""
    if not detections:
        return (
            "### ⚠️ No objects detected\n"
            "Try **lowering the Confidence threshold** or switching to a larger model."
        )
    rows = ["### ✅ Detection Results\n",
            "| # | Object | Confidence | Bounding Box |",
            "|---|--------|-----------|--------------|"]
    for d in detections:
        bar  = "█" * int(d["confidence"] * 10) + "░" * (10 - int(d["confidence"] * 10))
        box  = d["box"]
        rows.append(
            f"| {d['index']} | **{d['label']}** | `{d['confidence']:.0%}` {bar} "
            f"| `[{box[0]}, {box[1]}, {box[2]}, {box[3]}]` |"
        )
    return "\n".join(rows)


# ── Tab 1 callbacks ──────────────────────────────────────────────────────────

def on_image_upload(uploaded_image):
    """Show image metadata immediately on upload — before running detection."""
    if uploaded_image is None:
        return ""
    try:
        img = Image.fromarray(uploaded_image) if isinstance(uploaded_image, np.ndarray) else uploaded_image
        w, h = img.size
        return f"📐 **Uploaded:** {w}×{h} px &nbsp;|&nbsp; **Mode:** `{img.mode}`"
    except Exception:
        return "📐 Image received."


def run_pipeline(uploaded_image, conf_thresh, iou_thresh, model_name, max_size, ecommerce_mode):
    """
    Full pipeline: preprocess → detect → segment → init layers.
    """
    _empty_layers = gr.CheckboxGroup(choices=[], value=[])
    _empty_dd     = gr.Dropdown(choices=[], value=None)

    if uploaded_image is None:
        return None, None, "", "⚠️ Please upload an image first.", _empty_layers, _empty_dd

    max_size = int(max_size)

    # ── Stage 1: Preprocess ──────────────────────────────────────────────────
    try:
        img, meta = load_and_preprocess(uploaded_image, max_size=max_size)
    except (ValueError, FileNotFoundError) as e:
        return None, None, "", f"### ❌ Stage 1 — Preprocess Error\n```\n{e}\n```", _empty_layers, _empty_dd

    STATE["original"] = img
    STATE["meta"]     = meta
    img_info_md = _format_image_info(meta)

    # ── Stage 2: Detect ──────────────────────────────────────────────────────
    try:
        detections, annotated = detect_objects(
            img,
            confidence_threshold=float(conf_thresh),
            iou_threshold=float(iou_thresh),
            model_name=model_name,
            ecommerce_mode=bool(ecommerce_mode),
        )
    except Exception as e:
        return None, None, img_info_md, f"### ❌ Stage 2 — Detection Error\n```\n{e}\n```", _empty_layers, _empty_dd

    STATE["detections"] = detections

    if not detections:
        return annotated, img, img_info_md, _build_status_md([]), _empty_layers, _empty_dd

    # ── Stage 3: Segment ─────────────────────────────────────────────────────
    try:
        masks, highlighted = segment_objects(img, detections)
    except Exception as e:
        return annotated, None, img_info_md, f"### ❌ Stage 3 — Segmentation Error\n```\n{e}\n```", _empty_layers, _empty_dd

    STATE["masks"] = masks

    # ── Stage 4: Layer manager ───────────────────────────────────────────────
    try:
        STATE["layer_mgr"] = LayerManager(img, masks, detections)
    except Exception as e:
        return annotated, highlighted, img_info_md, f"### ❌ Stage 4 — Layer Manager Error\n```\n{e}\n```", _empty_layers, _empty_dd

    # ── Auto-populate Tab 2 & Tab 3 selectors ───────────────────────────────
    layer_keys = STATE["layer_mgr"].all_layer_keys()
    mask_keys  = list(STATE["masks"].keys())

    return (
        annotated,
        highlighted,
        img_info_md,
        _build_status_md(detections),
        gr.CheckboxGroup(choices=layer_keys, value=layer_keys),
        gr.Dropdown(choices=mask_keys, value=mask_keys[0] if mask_keys else None),
    )


# ── Tab 2 callbacks ──────────────────────────────────────────────────────────

def render_selected_layers(selected_keys):
    lm = STATE["layer_mgr"]
    if lm is None:
        return None
    lm.set_active(selected_keys)
    return lm.render()


def refresh_layers():
    if STATE["layer_mgr"] is None:
        return gr.CheckboxGroup(choices=[], value=[])
    keys = STATE["layer_mgr"].all_layer_keys()
    return gr.CheckboxGroup(choices=keys, value=keys)


# ── Tab 3 callbacks ──────────────────────────────────────────────────────────

def remove_selected_object(object_key, method, feather_px, use_poisson):
    """
    Remove one selected object using Stage 5 inpainting.
    feather_px and use_poisson are kept for UI compatibility,
    even though Local LaMa handles blending internally.
    """

    if not STATE["masks"]:
        return None, None, "⚠️ **Run detection first** (Tab 1)."

    if not object_key or object_key not in STATE["masks"]:
        return None, None, f"⚠️ Object `{object_key}` not found."

    image = STATE["original"]
    mask  = STATE["masks"][object_key]

    # Detect background type for status display
    bg_type = detect_background_type(image, mask)

    # ── Method mapping ─────────────────────────────────────────────
    method_lower = method.strip().lower()

    # Map old UI names → new Stage 5 names
    if method_lower == "opencv":
        method_lower = "fill"

    elif method_lower == "ns":
        method_lower = "fill"

    elif method_lower == "lama":
        method_lower = "local"

    elif method_lower == "auto":
        method_lower = "auto"

    # ── Run removal ────────────────────────────────────────────────
    try:
        result = remove_object(
            image,
            mask,
            feather_px=int(feather_px),   # backward compatible
            method=method_lower,
        )

    except Exception as e:
        return None, None, f"❌ Removal failed:\n\n```{e}```"

    # ── Status message ─────────────────────────────────────────────
    bg_note = {
        "uniform": "solid background",
        "gradient": "gradient background",
        "complex": "complex scene"
    }.get(bg_type, "")

    status = (
        f"✅ Removed `{object_key}` using **{method}** "
        f"| background: **{bg_type}** ({bg_note}) "
        f"| feather: {feather_px}px "
        f"| poisson: {use_poisson}"
    )

    return result, result, status

def refresh_object_dropdown():
    keys = list(STATE["masks"].keys())
    return gr.Dropdown(choices=keys, value=keys[0] if keys else None)


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* ── Global font & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #0f1117 !important;
}


/* ── Tab bar ── */
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 22px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg,#6d28d9,#4f46e5) !important;
    color: #fff !important;
    border-color: transparent !important;
}

/* ── Section cards ── */
.settings-card {
    background: #1a1d27 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* ── Primary run button ── */
#run-btn {
    background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    min-height: 52px !important;
    letter-spacing: 0.3px;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.4) !important;
}
#run-btn:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
}

/* ── Remove button ── */
#remove-btn {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    min-height: 48px !important;
    box-shadow: 0 4px 14px rgba(220,38,38,0.35) !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
#remove-btn:hover {
    opacity: 0.91 !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary buttons ── */
#refresh-layers-btn, #refresh-obj-btn, #render-btn {
    background: #1e2235 !important;
    border: 1px solid #3b4263 !important;
    color: #a5b4fc !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    min-height: 40px !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
}
#refresh-layers-btn:hover, #refresh-obj-btn:hover, #render-btn:hover {
    background: #252b44 !important;
    border-color: #6366f1 !important;
}

/* ── Image outputs — equal height ── */
.image-pair img {
    max-height: 440px !important;
    object-fit: contain !important;
    border-radius: 10px !important;
}

/* ── Sliders accent ── */
input[type='range'] {
    accent-color: #6366f1 !important;
}

/* ── Textbox / Markdown boxes ── */
.status-box {
    background: #12151f !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    font-size: 0.88rem !important;
}

/* ── Info badge ── */
#img-info {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    padding: 6px 0 2px 0 !important;
}

/* ── Dropdown / CheckboxGroup ── */
.svelte-1gfkn6j, select {
    background: #1a1d27 !important;
    border-color: #2a2d3e !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Section heading ── */
.section-title {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #6366f1 !important;
    margin-bottom: 10px !important;
    border-bottom: 1px solid #2a2d3e !important;
    padding-bottom: 6px !important;
}
"""


# ── Build Gradio UI ───────────────────────────────────────────────────────────

with gr.Blocks(
    title="Image Object Removal",
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:



    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Detect & Segment
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Tab("Detect & Segment"):

        with gr.Row(equal_height=False):

            # ── Left: Upload ─────────────────────────────────────────────────
            with gr.Column(scale=3, min_width=320):

                inp_image = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=380,
                    show_label=False,
                )
                img_info_md = gr.Markdown("", elem_id="img-info")

            # ── Right: Settings ──────────────────────────────────────────────
            with gr.Column(scale=2, min_width=280):


                conf_slider = gr.Slider(
                    minimum=0.10, maximum=0.95, value=0.35, step=0.05,
                    label="Confidence Threshold",
                    info="Min score to accept a detection. Lower = more objects (but more false positives).",
                )
                iou_slider = gr.Slider(
                    minimum=0.10, maximum=0.90, value=0.45, step=0.05,
                    label="IOU / NMS Threshold",
                    info="Duplicate-box suppression. Lower = fewer overlapping boxes.",
                )
                model_selector = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value=DEFAULT_MODEL,
                    label="YOLO Model",
                    info="Larger = more accurate, slower. Weights auto-download on first use.",
                )
                max_size_slider = gr.Slider(
                    minimum=512, maximum=1280, value=1024, step=128,
                    label="Max Image Size (px)",
                    info="Longest edge is capped here before inference. Higher = more detail, slower.",
                )
                ecommerce_toggle = gr.Checkbox(
                    value=False,
                    label="E-commerce Mode",
                    info=(
                        "Suppresses background noise (people, vehicles, animals). "
                        "Highlights product-class objects with a gold border."
                    ),
                )

                btn_run = gr.Button(
                    "Run Detection + Segmentation",
                    variant="primary",
                    elem_id="run-btn",
                )

        # ── Output row ────────────────────────────────────────────────────────

        with gr.Row(equal_height=True, elem_classes=["image-pair"]):
            out_detected    = gr.Image(label="YOLO — Detected Objects", height=420)
            out_highlighted = gr.Image(label="SAM — Segmented Masks",   height=420)

        out_status_md = gr.Markdown("", elem_classes=["status-box"])

        # ── Wire image-upload info ─────────────────────────────────────────
        inp_image.change(
            fn=on_image_upload,
            inputs=[inp_image],
            outputs=[img_info_md],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Layer Viewer
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Tab("Layer Viewer"):



        with gr.Row():
            with gr.Column(scale=1):

                btn_refresh_layers = gr.Button(
                    "Refresh Layers",
                    elem_id="refresh-layers-btn",
                )
                layer_checks = gr.CheckboxGroup(
                    choices=[],
                    label="",
                    show_label=False,
                )
                btn_render = gr.Button(
                    "Render",
                    elem_id="render-btn",
                )

            with gr.Column(scale=3):

                out_layer = gr.Image(label="", show_label=False, height=480)

        btn_refresh_layers.click(fn=refresh_layers, outputs=[layer_checks])
        btn_render.click(
            fn=render_selected_layers,
            inputs=[layer_checks],
            outputs=[out_layer],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Remove Object
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Tab("Remove Object"):



        with gr.Row():
            with gr.Column(scale=1):


                with gr.Row():
                    object_dropdown = gr.Dropdown(
                        choices=[],
                        label="Object to Remove",
                        scale=3,
                    )
                    btn_refresh_obj = gr.Button(
                        "🔄",
                        elem_id="refresh-obj-btn",
                        scale=1,
                        min_width=48,
                    )


                remove_method = gr.Radio(
                    choices=["Auto", "OpenCV", "NS", "LaMa"],
                    value="Auto",
                    label="Method",
                    show_label=True,
                    info=(
                        "Auto: detects background type → best method. "
                        "OpenCV: fast Telea. NS: Navier-Stokes (smooth). "
                        "LaMa: deep learning best quality."
                    ),
                )
                feather_slider = gr.Slider(
                    minimum=0, maximum=12, value=4, step=1,
                    label="Mask Feathering (px)",
                    info="Softens mask edges for seamless blending. 0 = hard edge.",
                )
                poisson_toggle = gr.Checkbox(
                    value=True,
                    label="Poisson Blend Post-processing",
                    info="Fixes color/brightness seams at mask boundary after inpainting.",
                )

                btn_remove = gr.Button(
                    "Remove Object",
                    variant="stop",
                    elem_id="remove-btn",
                )

                out_remove_status = gr.Markdown("", elem_classes=["status-box"])

            with gr.Column(scale=3):

                out_removed = gr.Image(label="", show_label=False, height=480)

        btn_refresh_obj.click(fn=refresh_object_dropdown, outputs=[object_dropdown])
        btn_remove.click(
            fn=remove_selected_object,
            inputs=[object_dropdown, remove_method, feather_slider, poisson_toggle],
            outputs=[out_removed, out_removed, out_remove_status],
        )

    # ── Wire Tab 1 run button (outputs span all tabs) ─────────────────────────
    btn_run.click(
        fn=run_pipeline,
        inputs=[inp_image, conf_slider, iou_slider, model_selector, max_size_slider, ecommerce_toggle],
        outputs=[
            out_detected,
            out_highlighted,
            img_info_md,
            out_status_md,
            layer_checks,
            object_dropdown,
        ],
    )


if __name__ == "__main__":
    print("\n── Shopeasy Image Processor — startup ──────────────────────")
    print("Pre-loading models so first user click has no lag...")

    # Warm up YOLO (fast — just loads weights into memory)
    warmup_yolo()

    # Warm up SAM (downloads weights on first run if needed)
    warmup_sam()

    # Start IOPaint LaMa server in background
    # This loads the LaMa model ONCE — all removals this session take ~2s
    print("\nStarting IOPaint LaMa server in background...")
    iopaint_ok = ensure_iopaint_ready()
    if iopaint_ok:
        print("✅ IOPaint ready — LaMa removal will take ~2s per object")
    else:
        print("⚠️  IOPaint unavailable — will use Gaussian fill for removal")
        print("   Install iopaint: pip install iopaint")

    print("────────────────────────────────────────────────────────────\n")

    demo.launch(
        share=False,
        server_port=7860,
        server_name="0.0.0.0",
        show_error=True,
    )