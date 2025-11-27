## Deep Video Forensics (Liveness + Deepfake Detection)

Flask web app that runs a liveness-style facial landmark analysis alongside two deepfake pipelines:
- HyperGuard hyperspectral detector (MSTPP reconstruction + HSViT classifier) with optional Grad-CAM visualization.
- RGB ensemble (Xception, ViT-B/16, SwinV2-B) for cross-checking deepfake predictions.


### Project layout
- `app.py` — Flask routes, frame sampling, landmark analysis, deepfake calls, Grad-CAM.
- `templates/`, `static/` — UI, JS player/analysis dashboard, styling.
- `models/` — Haar cascades for face/eyes/nose/mouth detection.
- `DeepfakeDetector/` — HyperGuard pipeline + weights.
- `SADeepfakeDetector/` — RGB ensemble + weights.
- `Test Videos/` — small sample clips you can upload to try the app.

### Prerequisites
- Python 3.10+ recommended.
- `pip` plus a working C++ build chain for `opencv-python`.
- `wget` (or swap `wget` for `curl -L -O` in the commands below).
- GPU optional: CUDA or Apple MPS is auto-detected; CPU works but will be slower.

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Download model weights (required)
Place the checkpoints exactly in the model folders below.

```bash
# RGB ensemble (Xception / ViT-B16 / SwinV2-B)
cd SADeepfakeDetector/models
wget https://huggingface.co/jwstes/SA-Coursework2-Models/resolve/main/best_xception_deepfake.pth
wget https://huggingface.co/jwstes/SA-Coursework2-Models/resolve/main/best_vit_b16_deepfake.pth
wget https://huggingface.co/jwstes/SA-Coursework2-Models/resolve/main/best_swinv2_b_deepfake.pth

# HyperGuard (hyperspectral reconstruction + classifier)
cd ../../DeepfakeDetector/models
wget https://huggingface.co/jwstes/Hyperguard-DF/resolve/main/HSDataSet-MSTPP.pt
wget https://huggingface.co/jwstes/Hyperguard-DF/resolve/main/hyperguard.pt
cd ../../
```

### Run the app
```bash
# from the repo root
export FLASK_APP=app.py
flask run --debug  # or: python app.py
```