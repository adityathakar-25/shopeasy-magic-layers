# Shopeasy AI Image Processor

AI-powered image processing pipeline for [spark.shopeasy.ai](https://spark.shopeasy.ai) — detect objects, highlight layers, and remove any object from product images. Built as the backend for the Magic Layers feature.

---

## Features

- **Object Detection** — YOLOv8s detects products with class labels and confidence scores
- **Pixel-Precise Segmentation** — SAM (Segment Anything Model) generates exact pixel masks
- **Interactive Layers** — Each object is a separate toggleable layer with colour highlight
- **Object Removal** — IOPaint LaMa deep inpainting removes objects cleanly from any background
- **Smart Routing** — Simple backgrounds use instant Gaussian fill; complex scenes use LaMa

---

## Project Structure

```
shopeasy-image-processor/
├── app.py                  ← Gradio UI (runs the full pipeline)
├── stage1_preprocess.py    ← Image loading, RGB conversion, resize
├── stage2_detect.py        ← YOLOv8s object detection
├── stage3_segment.py       ← SAM pixel-level segmentation
├── stage4_layers.py        ← LayerManager — toggle and render layers
├── stage5_remove.py        ← Object removal via IOPaint LaMa + Gaussian fill
├── utils.py                ← Shared helpers (colors, PIL/cv2 conversions)
├── requirements.txt        ← All dependencies
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/shopeasy-image-processor.git
cd shopeasy-image-processor
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

SAM requires a separate install from GitHub:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 4. Model weights

Model weights are downloaded automatically on first run:

| Model | Size | Downloaded to |
|---|---|---|
| YOLOv8s | ~22 MB | `yolov8s.pt` |
| SAM ViT-B | ~375 MB | `sam_vit_b_01ec64.pth` |
| LaMa (via IOPaint) | ~200 MB | `~/.cache/torch/hub/checkpoints/` |

---

## Running the App

```bash
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

On first launch:
- YOLOv8s and SAM are pre-loaded into memory
- IOPaint LaMa server starts automatically in the background (~15s first time)
- All subsequent object removals take ~2s

---

## Usage

### Tab 1 — Detect & Segment
1. Upload any product image
2. Click **Run Pipeline**
3. See detected objects with bounding boxes (YOLO) and pixel masks (SAM)

### Tab 2 — Layer Viewer
1. Click **Load Layers** after running Tab 1
2. Check/uncheck objects to toggle their highlights
3. Click **Render** to update the view

### Tab 3 — Remove Object
1. Click **Load Objects** after running Tab 1
2. Select the object to remove from the dropdown
3. Choose method: **Auto** (recommended), **LaMa**, or **Fill**
4. Click **Remove Object**
5. Download the result

---

## How It Works

```
Image Upload
    │
    ▼
Stage 1 — Preprocess
    Convert to RGB, resize to 1024px max
    │
    ▼
Stage 2 — Detect (YOLOv8s)
    Bounding boxes + class labels + confidence scores
    │
    ▼
Stage 3 — Segment (SAM ViT-B)
    Pixel-precise masks using YOLO boxes as prompts
    Batched inference — all objects in one forward pass
    │
    ▼
Stage 4 — Layer Manager
    Each object = one layer
    Toggle highlights, render cache for instant re-render
    │
    ▼
Stage 5 — Remove (IOPaint LaMa)
    Auto-detects background type:
      Uniform/gradient → Gaussian fill (instant)
      Complex scene    → LaMa inpainting (~2s)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Object detection | YOLOv8s (Ultralytics) |
| Segmentation | SAM ViT-B (Meta) |
| Inpainting | LaMa via IOPaint |
| Image processing | OpenCV, PIL, NumPy |
| UI | Gradio |
| Deep learning | PyTorch |

---

## Requirements

- Python 3.10+
- 8GB RAM minimum (16GB recommended)
- CPU inference supported (no GPU required)
- GPU optional — speeds up SAM encoding and LaMa inference significantly

---

## Environment Variables (optional)

```bash
# Optional: Replicate API for cloud GPU inpainting
REPLICATE_API_TOKEN=your_token_here
```

---

## Known Limitations

- SAM encoding takes ~8s on CPU per image (cached after first run on same image)
- LaMa inpainting takes ~2s on CPU after server warmup
- YOLOv8s is limited to 80 COCO classes — custom product categories not supported yet

---

## Roadmap

- [ ] Multi-object removal in one pass
- [ ] FastAPI backend for spark.shopeasy.ai integration
- [ ] React frontend with Canva-style layer panel
- [ ] Cloud deployment on Google Cloud Run
- [ ] Layer grouping feature
- [ ] Click-on-canvas object selection

---

## Project Context

Built as an intern project at [Shopeasy.ai](https://shopeasy.ai) for the Magic Layers feature of spark.shopeasy.ai. The goal is to give e-commerce sellers the ability to remove unwanted objects, isolate products, and clean up product photography directly in the browser — no Photoshop needed.

---

## License

MIT
