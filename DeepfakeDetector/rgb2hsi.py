from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import cv2
import torch

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, ".."))
MODELS_DIR = os.path.join(MODULE_DIR, "models")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import MSTPP


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_rgb_as_model_input(image_path, resize=None):
    """
    Load a single RGB image and convert it to a normalized tensor
    suitable for MST++: [1, 3, H, W], float32, per-image min-max normalized.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Handle possible channel formats
    if img.ndim == 2:
        # Grayscale -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # BGRA -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resize is not None:
        # OpenCV expects (width, height)
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0  # ensure [0,1]

    # Per-image min-max normalization (same logic as your batch code)
    vmin = float(img.min())
    vmax = float(img.max())
    if vmax > vmin:
        img_norm = (img - vmin) / (vmax - vmin)
    else:
        img_norm = np.zeros_like(img, dtype=np.float32)

    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor.to(device)


def gethHsiFP32(hsi_cube):
    """
    Save an HSI cube to a .npz file as float32 (compressed).
    Shape is [C, H, W].
    """
    hsi_fp32 = np.asarray(hsi_cube, dtype=np.float32)

    return hsi_fp32


def load_mstpp_model(checkpoint_path):
    """
    Build MST++ model and load weights.
    """
    model = MSTPP.MST_Plus_Plus(in_channels=3, out_channels=31, n_feat=31, stage=3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def rgb_to_hsi_npz(
    image_path,
    checkpoint_path=os.path.join(MODELS_DIR, "HSDataSet-MSTPP.pt"),
    resize=(256, 256),
):
    """
    Convert a single RGB image to a hyperspectral cube and save as .npz.
    """
    model = load_mstpp_model(checkpoint_path)
    input_tensor = load_rgb_as_model_input(image_path, resize=resize)

    with torch.no_grad():
        hsi_batch = model(input_tensor).float()

    # [1, C, H, W] -> [C, H, W]
    hsi_cube = hsi_batch.squeeze(0).cpu().numpy()
    hsi_fp32 = gethHsiFP32(hsi_cube)

    return hsi_fp32


def getHSIfromRGB(imagePath):
    input_image_path = imagePath
    resize = (256, 256)
    checkpoint_path = os.path.join(MODELS_DIR, "HSDataSet-MSTPP.pt")

    hsi_fp32 = rgb_to_hsi_npz(
        image_path=input_image_path,
        checkpoint_path=checkpoint_path,
        resize=resize,
    )

    return hsi_fp32
