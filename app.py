import os
import sys
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import (
    Flask,
    abort,
    render_template,
    request,
    redirect,
    url_for,
    Response,
    jsonify,
)
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_DIR = os.path.join(BASE_DIR, "DeepfakeDetector")
if DETECTOR_DIR not in sys.path:
    sys.path.insert(0, DETECTOR_DIR)

SA_DETECTOR_DIR = os.path.join(BASE_DIR, "SADeepfakeDetector")
if SA_DETECTOR_DIR not in sys.path:
    sys.path.insert(0, SA_DETECTOR_DIR)

from DeepfakeDetector.HyperGuard import detectDF
from DeepfakeDetector.hsi2dfclass import load_model_and_cfg, DEFAULT_CHECKPOINT_DIR
from SADeepfakeDetector.ensembleInfer import detectDF as detectDF_ensemble


UPLOAD_FOLDER = "static/uploads"
FACE_SIZE = 256
# Higher values include more of the head/scene (1.0 = tight face crop).
FACE_ZOOM_SCALE = 1.30
CASCADE_DIR = os.path.join(BASE_DIR, "models")
ANALYSIS_SAMPLE_COUNT = 6
ANALYSIS_OUTPUT_SUBDIR = "analysis"
WAVELENGTHS = [400 + 10 * i for i in range(31)]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, UPLOAD_FOLDER)
ANALYSIS_OUTPUT_DIR = os.path.join(app.root_path, "static", ANALYSIS_OUTPUT_SUBDIR)
ANALYSIS_ABS_ROOT = os.path.abspath(ANALYSIS_OUTPUT_DIR)

# Ensure upload directory exists at startup.
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

HYPER_MODEL = None
HYPER_CFG = None
HYPER_DEVICE = None


def _load_cascade(filename: str, label: str) -> cv2.CascadeClassifier:
    path = os.path.join(CASCADE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} cascade not found at {path}")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade for {label}.")
    return cascade


def _analysis_abs_path(rel_path: str) -> str:
    if not rel_path:
        raise ValueError("Missing analysis path.")
    candidate = os.path.normpath(os.path.join(ANALYSIS_ABS_ROOT, rel_path))
    if not candidate.startswith(ANALYSIS_ABS_ROOT):
        raise ValueError("Invalid analysis path.")
    return candidate


def _get_hyper_model():
    global HYPER_MODEL, HYPER_CFG, HYPER_DEVICE
    if HYPER_DEVICE is None:
        if torch.backends.mps.is_available():
            HYPER_DEVICE = torch.device("mps")
        elif torch.cuda.is_available():
            HYPER_DEVICE = torch.device("cuda")
        else:
            HYPER_DEVICE = torch.device("cpu")
    if HYPER_MODEL is None or HYPER_CFG is None:
        HYPER_MODEL, HYPER_CFG = load_model_and_cfg(DEFAULT_CHECKPOINT_DIR, HYPER_DEVICE)
    return HYPER_MODEL, HYPER_CFG, HYPER_DEVICE


FACE_DETECTOR = _load_cascade("haarcascade_frontalface_default.xml", "face")
EYE_DETECTOR = _load_cascade("haarcascade_eye.xml", "eyes")
NOSE_DETECTOR = _load_cascade("haarcascade_mcs_nose.xml", "nose")
MOUTH_DETECTOR = _load_cascade("haarcascade_mcs_mouth.xml", "mouth")

# Calculates geometric distances and ratios between facial landmarks
def _calculate_relationships(features: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not features:
        return {"eye_dist": None, "eye_mouth_ratio": None}

    eyes = features.get("eyes", [])
    mouth = features.get("mouth")

    if len(eyes) < 2:
        return {"eye_dist": None, "eye_mouth_ratio": None}

    # The eyes are sorted by x-coordinate in _serialize_landmark_frame
    left_eye = eyes[0].get("center")
    right_eye = eyes[1].get("center")

    if not left_eye or not right_eye:
        return {"eye_dist": None, "eye_mouth_ratio": None}

    # 1. Calculate Eye-to-Eye Distance
    eye_dist = math.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)

    eye_mouth_ratio = None
    if mouth and mouth.get("center"):
        mouth_center = mouth["center"]
        
        eye_midpoint_y = (left_eye[1] + right_eye[1]) / 2.0
        
        # 2. Calculate Vertical Eye-to-Mouth Distance
        eye_mouth_dist = abs(mouth_center[1] - eye_midpoint_y)

        # 3. Calculate the Inter-Ocular Ratio
        if eye_dist > 0:
            eye_mouth_ratio = eye_mouth_dist / eye_dist

    return {
        "eye_dist": eye_dist,
        "eye_mouth_ratio": eye_mouth_ratio
    }


@app.route("/")
def upload_form():
    """Display the landing page with the file upload form."""
    return render_template("upload.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Handle the file upload and redirect to the player page."""
    if request.method == "GET":
        return render_template("upload.html")

    uploaded_file = request.files.get("video")
    if uploaded_file is None or uploaded_file.filename == "":
        error = "Please choose a video file to upload."
        return render_template("upload.html", error=error), 400

    filename = secure_filename(uploaded_file.filename)
    if not filename:
        error = "Invalid filename provided."
        return render_template("upload.html", error=error), 400

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    final_name = f"{timestamp}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], final_name)
    uploaded_file.save(save_path)

    return redirect(url_for("player", video=final_name))


@app.route("/player")
def player():
    """Render the player page that hosts the HTML video element and face preview."""
    video_filename = request.args.get("video")
    if not video_filename:
        return redirect(url_for("upload_form"))

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_filename)
    if not os.path.exists(video_path):
        return redirect(url_for("upload_form"))

    video_url = url_for("static", filename=f"uploads/{video_filename}")
    return render_template("player.html", video_filename=video_filename, video_url=video_url)


def extract_face_crop(frame: np.ndarray) -> np.ndarray:
    """Detect a face in the given frame and return an annotated 256x256 crop."""
    annotated, _, _ = _process_face_frame(frame)
    return annotated


def _process_face_frame(
    frame: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """Core processing pipeline that returns annotated crop, metadata, and clean face."""
    placeholder = _placeholder_face()
    if frame is None:
        return placeholder, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return placeholder, None, None

    # Choose the largest detected face (by area).
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    x = max(x, 0)
    y = max(y, 0)
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = _expanded_face_bounds(x, y, w, h, frame_w, frame_h)
    nose_center = _find_nose_center(gray, (x1, y1, x2, y2))
    if nose_center is not None:
        x1, y1, x2, y2 = _crop_centered_on_point(
            frame_w, frame_h, nose_center, x2 - x1, y2 - y1
        )
    face_img = frame[y1:y2, x1:x2]
    if face_img.size == 0:
        return placeholder, None, None

    resized = cv2.resize(face_img, (FACE_SIZE, FACE_SIZE))
    clean_face = resized.copy()
    features = _detect_facial_features(clean_face)
    annotated = _overlay_facial_features(clean_face.copy(), features)
    return annotated, features, clean_face


def _expanded_face_bounds(x, y, w, h, frame_w, frame_h):
    """Expand the bounding box according to FACE_ZOOM_SCALE while staying in frame bounds."""
    scale = max(FACE_ZOOM_SCALE, 1.0)
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    new_w = w * scale
    new_h = h * scale

    x1 = max(int(center_x - new_w / 2.0), 0)
    y1 = max(int(center_y - new_h / 2.0), 0)
    x2 = min(int(center_x + new_w / 2.0), frame_w)
    y2 = min(int(center_y + new_h / 2.0), frame_h)
    return x1, y1, x2, y2


def _find_nose_center(
    gray_frame: np.ndarray, bounds: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int]]:
    """Detect the primary nose within the given bounds and return absolute coordinates."""
    x1, y1, x2, y2 = bounds
    roi = gray_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    noses = NOSE_DETECTOR.detectMultiScale(
        roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )
    if len(noses) == 0:
        return None
    nx, ny, nw, nh = max(noses, key=lambda rect: rect[2] * rect[3])
    return (x1 + nx + nw // 2, y1 + ny + nh // 2)


def _crop_centered_on_point(
    frame_w: int, frame_h: int, center: Tuple[int, int], crop_w: int, crop_h: int
) -> Tuple[int, int, int, int]:
    """Return new crop bounds of size (crop_w, crop_h) centered on the provided point."""
    cx, cy = center
    x1 = int(round(cx - crop_w / 2))
    y1 = int(round(cy - crop_h / 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_w:
        shift = x2 - frame_w
        x1 -= shift
        x2 = frame_w
    if y2 > frame_h:
        shift = y2 - frame_h
        y1 -= shift
        y2 = frame_h

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame_w)
    y2 = min(y2, frame_h)
    return x1, y1, x2, y2


def _save_hsi_band_images(
    hsi_cube: Optional[np.ndarray],
    dest_dir: str,
    video_id: str,
    base_name: str,
) -> List[Dict[str, Any]]:
    """Persist HSI bands as grayscale images for visualization."""
    if hsi_cube is None or hsi_cube.size == 0:
        return []

    bands = int(hsi_cube.shape[0])
    band_dir_name = f"{base_name}_bands"
    band_dir = os.path.join(dest_dir, band_dir_name)
    os.makedirs(band_dir, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    for band_index in range(bands):
        band_img = hsi_cube[band_index]
        norm_img = cv2.normalize(band_img, None, 0, 255, cv2.NORM_MINMAX)
        norm_img = norm_img.astype(np.uint8)
        filename = f"{base_name}_band_{band_index:02d}.png"
        path = os.path.join(band_dir, filename)
        cv2.imwrite(path, norm_img)

        wavelength = WAVELENGTHS[band_index] if band_index < len(WAVELENGTHS) else None
        image_url = url_for(
            "static",
            filename=f"{ANALYSIS_OUTPUT_SUBDIR}/{video_id}/{band_dir_name}/{filename}",
            _external=False,
        )
        entries.append(
            {
                "band_index": band_index,
                "wavelength": wavelength,
                "image_url": image_url,
            }
        )

    return entries


def _generate_gradcam_image(
    hsi_rel_path: str,
    face_rel_path: str,
    class_id: int,
) -> str:
    if class_id not in (0, 1):
        raise ValueError("Unsupported class id.")

    model, cfg, device = _get_hyper_model()
    hsi_path = _analysis_abs_path(hsi_rel_path)
    face_path = _analysis_abs_path(face_rel_path)

    hsi_cube = np.load(hsi_path, allow_pickle=False)
    if hsi_cube.ndim != 3:
        raise ValueError("Invalid HSI cube.")

    tensor = torch.from_numpy(hsi_cube).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    target_size = cfg.get("img_size", cfg.get("img_size_for_vit", FACE_SIZE))
    if tensor.shape[-1] != target_size or tensor.shape[-2] != target_size:
        tensor = F.interpolate(
            tensor,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

    logits = model(tensor).view(-1)[0]
    model.zero_grad(set_to_none=True)
    if tensor.grad is not None:
        tensor.grad.zero_()

    target = logits if class_id == 1 else -logits
    target.backward()

    grads = tensor.grad.detach().cpu().numpy()[0]
    activations = tensor.detach().cpu().numpy()[0]
    heatmap = np.maximum(grads * activations, 0).sum(axis=0)
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (FACE_SIZE, FACE_SIZE))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    face_img = cv2.imread(face_path, cv2.IMREAD_COLOR)
    if face_img is None:
        face_img = np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
    else:
        face_img = cv2.resize(face_img, (FACE_SIZE, FACE_SIZE))

    overlay = cv2.addWeighted(face_img, 0.5, heatmap_color, 0.5, 0)

    base_dir = os.path.dirname(face_rel_path)
    gradcam_rel_dir = os.path.join(base_dir, "gradcam")
    gradcam_abs_dir = _analysis_abs_path(gradcam_rel_dir)
    os.makedirs(gradcam_abs_dir, exist_ok=True)

    gradcam_filename = f"{os.path.splitext(os.path.basename(face_rel_path))[0]}_class{class_id}.png"
    gradcam_abs_path = os.path.join(gradcam_abs_dir, gradcam_filename)
    gradcam_rel_path = os.path.join(gradcam_rel_dir, gradcam_filename).replace(os.sep, "/")
    if not os.path.exists(gradcam_abs_path):
        cv2.imwrite(gradcam_abs_path, overlay)

    return url_for(
        "static",
        filename=f"{ANALYSIS_OUTPUT_SUBDIR}/{gradcam_rel_path}",
    )


def perform_landmark_analysis(video_path: str) -> Dict[str, Any]:
    """Iterate through the video and capture facial landmark coordinates over time."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {"frames": [], "fps": 0, "frames_analyzed": 0}

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_index = 0
    frames: List[Dict[str, Any]] = []

    while True:
        success, frame = capture.read()
        if not success:
            break
        timestamp = frame_index / fps if fps > 0 else frame_index
        _, features, _ = _process_face_frame(frame)
        
        # 1. Get the original record
        record = _serialize_landmark_frame(timestamp, features)
        
        # 2. Calculate new relationship metrics
        relationship_metrics = _calculate_relationships(features)
        
        # 3. Add the new metrics to the record for this frame
        record.update(relationship_metrics)
        frames.append(record)
        frame_index += 1

    capture.release()
    return {
        "frames": frames,
        "fps": fps,
        "frames_analyzed": len(frames),
        "duration": frame_index / fps if fps > 0 else frame_index,
        "df_samples": sample_frames_for_detection(video_path, ANALYSIS_SAMPLE_COUNT),
    }


def sample_frames_for_detection(video_path: str, sample_count: int) -> List[Dict[str, Any]]:
    """Randomly sample frames for downstream classification."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return []

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0 or fps <= 0:
        capture.release()
        return []

    sample_count = max(1, min(sample_count, total_frames))
    frame_indices = random.sample(range(total_frames), sample_count)
    results: List[Dict[str, Any]] = []

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    dest_dir = os.path.join(ANALYSIS_OUTPUT_DIR, video_id)
    os.makedirs(dest_dir, exist_ok=True)

    for idx in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = capture.read()
        if not success or frame is None:
            continue

        _, _, face_crop = _process_face_frame(frame)
        if face_crop is None:
            continue

        timestamp = idx / fps
        filename = f"sample_{idx}.jpg"
        file_path = os.path.join(dest_dir, filename)
        cv2.imwrite(file_path, face_crop)
        image_rel_path = os.path.join(video_id, filename).replace(os.sep, "/")

        detectors_bundle: Dict[str, Any] = {}
        hsi_cube = None
        try:
            hyperspectral_raw, hsi_cube = detectDF(file_path, return_hsi=True)
            detectors_bundle["hyperspectral"] = _normalize_detection_result(hyperspectral_raw)
        except TypeError:
            try:
                hyperspectral_raw = detectDF(file_path)
                detectors_bundle["hyperspectral"] = _normalize_detection_result(hyperspectral_raw)
            except Exception as exc:  # pragma: no cover - defensive
                detectors_bundle["hyperspectral"] = {"error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            detectors_bundle["hyperspectral"] = {"error": str(exc)}

        try:
            ensemble_raw = detectDF_ensemble(file_path)
        except Exception as exc:  # pragma: no cover - defensive
            ensemble_raw = {"error": str(exc)}

        detectors_bundle.update(_normalize_ensemble_results(ensemble_raw))

        gradcam_rel_source = None
        bands_meta: List[Dict[str, Any]] = []
        if hsi_cube is not None:
            hsi_dir = os.path.join(dest_dir, "hsi")
            os.makedirs(hsi_dir, exist_ok=True)
            hsi_filename = f"{os.path.splitext(filename)[0]}.npy"
            hsi_abs_path = os.path.join(hsi_dir, hsi_filename)
            np.save(hsi_abs_path, hsi_cube)
            gradcam_rel_source = os.path.join(video_id, "hsi", hsi_filename).replace(os.sep, "/")
            bands_meta = _save_hsi_band_images(
                hsi_cube,
                dest_dir,
                video_id,
                os.path.splitext(filename)[0],
            )

        image_url = url_for(
            "static",
            filename=f"{ANALYSIS_OUTPUT_SUBDIR}/{video_id}/{filename}",
            _external=False,
        )
        results.append(
            {
                "id": f"{video_id}_{idx}",
                "time": round(timestamp, 2),
                "frame_index": idx,
                "image_url": image_url,
                "detectors": detectors_bundle,
                "bands": bands_meta,
                "gradcam_source": gradcam_rel_source,
                "image_file": image_rel_path,
            }
        )

    capture.release()
    return results


def _normalize_detection_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize hyperspectral detector output."""
    label_map = {"real": 0, "fake": 1}
    raw_label = result.get("pred_class")
    if not raw_label:
        raw_label = result.get("pred_label")
    if isinstance(raw_label, (int, float)):
        raw_label = "fake" if int(raw_label) == 1 else "real"
    label_str = str(raw_label).lower()
    if label_str not in label_map:
        label_str = "unknown"
    numeric_label = label_map.get(label_str, -1)
    prob_real = result.get("prob_real")
    prob_fake = result.get("prob_fake")
    return {
        "label": label_str,
        "numeric_label": numeric_label,
        "prob_real": float(prob_real) if prob_real is not None else None,
        "prob_fake": float(prob_fake) if prob_fake is not None else None,
    }


def _normalize_ensemble_results(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ensemble detector outputs for each model."""
    detector_keys = {
        "xception": "xception",
        "vit_b16": "vit_b16",
        "swin_v2_b": "swin_v2_b",
    }
    normalized: Dict[str, Any] = {}

    if not isinstance(result, dict):
        for key in detector_keys.values():
            normalized[key] = {"error": "No ensemble results"}
        return normalized

    if "error" in result:
        for key in detector_keys.values():
            normalized[key] = {"error": str(result["error"])}
        return normalized

    for field, key in detector_keys.items():
        normalized[key] = _format_general_model_result(result.get(field))

    return normalized


def _format_general_model_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Format a single model's detection result."""
    if not result:
        return {"error": "No data"}
    if "error" in result:
        return {"error": str(result["error"])}

    label = str(result.get("label", "unknown")).lower()
    confidence = result.get("confidence")
    probs = result.get("probs") or []
    prob_real = probs[0] if len(probs) > 0 else None
    prob_fake = probs[1] if len(probs) > 1 else None

    return {
        "label": label,
        "confidence": float(confidence) if confidence is not None else None,
        "prob_real": float(prob_real) if prob_real is not None else None,
        "prob_fake": float(prob_fake) if prob_fake is not None else None,
    }


def _serialize_landmark_frame(
    timestamp: float, features: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Convert landmark metadata into JSON-serializable primitives."""
    record: Dict[str, Any] = {"time": round(timestamp, 3)}
    if not features:
        record["eyes"] = {"left": None, "right": None}
        record["nose"] = None
        record["mouth"] = None
        return record

    eye_entries = features.get("eyes", [])
    # Sort by x-coordinate to get left and right consistently
    eye_entries.sort(key=lambda entry: entry["center"][0])
    
    record["eyes"] = {
        "left": _eye_dict(eye_entries[0]) if len(eye_entries) > 0 else None,
        "right": _eye_dict(eye_entries[1]) if len(eye_entries) > 1 else None,
    }
    nose_feature = features.get("nose")
    record["nose"] = _point_dict(nose_feature["center"] if nose_feature else None)
    mouth_feature = features.get("mouth")
    record["mouth"] = _point_dict(mouth_feature["center"] if mouth_feature else None)
    return record


def _point_dict(point: Optional[Tuple[int, int]]) -> Optional[Dict[str, float]]:
    if point is None:
        return None
    return {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}


def _eye_dict(entry: Dict[str, Any]) -> Dict[str, float]:
    cx, cy = entry["center"]
    _, _, w, h = entry["bbox"]
    diameter = (w + h) / 2.0
    return {
        "x": round(float(cx), 2),
        "y": round(float(cy), 2),
        "diameter": round(float(diameter), 2),
    }


def _detect_facial_features(face_img: np.ndarray) -> Dict[str, Any]:
    """Return landmark metadata for the given aligned face crop."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    features: Dict[str, Any] = {"eyes": []}

    # Eyes
    eyes_roi = gray[: FACE_SIZE // 2, :]
    eyes = EYE_DETECTOR.detectMultiScale(
        eyes_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20),
    )
    for (ex, ey, ew, eh) in sorted(eyes, key=lambda rect: rect[2] * rect[3], reverse=True)[:2]:
        features["eyes"].append(
            {
                "center": (ex + ew // 2, ey + eh // 2),
                "bbox": (ex, ey, ew, eh),
            }
        )

    # Nose
    nose_start = FACE_SIZE // 3
    nose_roi = gray[nose_start : (FACE_SIZE * 2) // 3, :]
    noses = NOSE_DETECTOR.detectMultiScale(
        nose_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )
    if len(noses) > 0:
        nx, ny, nw, nh = max(noses, key=lambda rect: rect[2] * rect[3])
        features["nose"] = {
            "center": (nx + nw // 2, nose_start + ny + nh // 2),
            "bbox": (nx, nose_start + ny, nw, nh),
        }
    else:
        features["nose"] = None

    # Mouth
    mouth_start = FACE_SIZE // 2
    mouth_roi = gray[mouth_start:, :]
    mouths = MOUTH_DETECTOR.detectMultiScale(
        mouth_roi,
        scaleFactor=1.15,
        minNeighbors=15,
        minSize=(30, 20),
    )
    if len(mouths) > 0:
        mx, my, mw, mh = max(mouths, key=lambda rect: rect[2] * rect[3])
        features["mouth"] = {
            "center": (mx + mw // 2, mouth_start + my + mh // 2),
            "bbox": (mx, mouth_start + my, mw, mh),
        }
    else:
        features["mouth"] = None

    return features


def _overlay_facial_features(
    face_img: np.ndarray, features: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """Annotate eyes, nose, and mouth detections on the cropped face."""
    annotated = face_img.copy()
    if features is None:
        features = _detect_facial_features(face_img)

    for eye in features.get("eyes", []):
        cx, cy = eye["center"]
        _, _, ew, eh = eye["bbox"]
        cv2.circle(annotated, (cx, cy), max(ew, eh) // 3, (0, 255, 0), 2)

    nose = features.get("nose")
    if nose:
        cv2.circle(
            annotated,
            tuple(map(int, nose["center"])),
            max(nose["bbox"][2], nose["bbox"][3]) // 3,
            (0, 165, 255),
            2,
        )

    mouth = features.get("mouth")
    if mouth:
        x, y, w, h = mouth["bbox"]
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        cv2.rectangle(annotated, pt1, pt2, (255, 0, 0), 2)

    # Draw crosshair at the center to illustrate stabilization target.
    center = FACE_SIZE // 2
    cv2.line(annotated, (center, 0), (center, FACE_SIZE), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(annotated, (0, center), (FACE_SIZE, center), (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def _placeholder_face() -> np.ndarray:
    """Create a simple placeholder image that indicates no face was found."""
    placeholder = np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        "NO FACE",
        (20, FACE_SIZE // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return placeholder


def read_frame_at_time(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """Grab a frame from the video at the requested timestamp."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return None

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        capture.release()
        return None

    frame_index = max(int(math.floor(timestamp * fps)), 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    capture.release()
    if not success:
        return None
    return frame


@app.route("/frame_crop")
def frame_crop():
    """Return a 256x256 face crop image for the requested time in the video."""
    video_filename = request.args.get("video")
    if not video_filename:
        abort(400, description="Missing video parameter.")

    t = request.args.get("t", "0")
    try:
        timestamp = max(float(t), 0.0)
    except ValueError:
        abort(400, description="Invalid timestamp.")

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_filename)
    if not os.path.exists(video_path):
        abort(404, description="Video not found.")

    frame = read_frame_at_time(video_path, timestamp)
    crop = extract_face_crop(frame)
    success, buffer = cv2.imencode(".jpg", crop)
    if not success:
        abort(500, description="Could not encode crop image.")

    response = Response(buffer.tobytes(), mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.route("/analyze", methods=["POST"])
def analyze_video():
    """Perform landmark analysis over the entire uploaded video."""
    payload = request.get_json(silent=True) or {}
    video_filename = payload.get("video")
    if not video_filename:
        abort(400, description="Missing video parameter.")

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_filename)
    if not os.path.exists(video_path):
        abort(404, description="Video not found.")

    analysis = perform_landmark_analysis(video_path)
    return jsonify(analysis)


@app.route("/gradcam", methods=["POST"])
def gradcam():
    """Generate and return a Grad-CAM overlay for a saved sample."""
    payload = request.get_json(silent=True) or {}
    class_id = int(payload.get("class_id", 1))
    hsi_source = payload.get("hsi_source")
    face_source = payload.get("face_source")

    if hsi_source is None or face_source is None:
        abort(400, description="Missing Grad-CAM source paths.")

    try:
        image_url = _generate_gradcam_image(hsi_source, face_source, class_id)
    except ValueError as exc:
        abort(400, description=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        abort(500, description=str(exc))

    return jsonify({"image_url": image_url, "class_id": class_id})


if __name__ == "__main__":
    app.run(debug=True)
