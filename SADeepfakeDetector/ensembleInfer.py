#!/usr/bin/env python3
import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(MODULE_DIR, "models")

# =====================
# Config
# =====================
IMG_SIZE = 256  # MUST be 256x256 for all three models

XCEPTION_WEIGHTS = os.path.join(MODELS_DIR, "best_xception_deepfake.pth")
VIT_WEIGHTS = os.path.join(MODELS_DIR, "best_vit_b16_deepfake.pth")
SWIN_WEIGHTS = os.path.join(MODELS_DIR, "best_swinv2_b_deepfake.pth")

LABEL_MAP = {
    0: "real",
    1: "fake",
}

# =====================
# Device selection (CUDA / Apple Silicon / CPU)
# =====================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple Silicon
else:
    DEVICE = torch.device("cpu")

print(f"[INFO] Using device: {DEVICE}")

# =====================
# Common image transform (256x256)
# =====================
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# =====================
# Model builders (must match training architectures)
# =====================
def build_xception(num_classes: int = 2) -> nn.Module:
    model = timm.create_model(
        "xception",  # timm maps this to legacy_xception
        pretrained=False,
        num_classes=num_classes,
    )
    return model


def build_vit_b16(num_classes: int = 2, img_size: int = IMG_SIZE) -> nn.Module:
    """
    ViT-B/16 with img_size=256, matching the training script.
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size,
    )

    # Ensure patch_embed is consistent with 256x256
    if hasattr(model, "patch_embed"):
        model.patch_embed.img_size = (img_size, img_size)
        patch_h, patch_w = model.patch_embed.patch_size
        grid_h = img_size // patch_h
        grid_w = img_size // patch_w
        model.patch_embed.grid_size = (grid_h, grid_w)

    return model


def build_swinv2_b(num_classes: int = 2, img_size: int = IMG_SIZE) -> nn.Module:
    """
    Swin V2 Base @ 256x256, matching the training script.
    """
    model = timm.create_model(
        "swinv2_base_window8_256",
        pretrained=False,
        num_classes=num_classes,
    )

    # Ensure internal patch_embed is consistent with 256x256
    if hasattr(model, "patch_embed"):
        model.patch_embed.img_size = (img_size, img_size)
        patch_h, patch_w = model.patch_embed.patch_size
        grid_h = img_size // patch_h
        grid_w = img_size // patch_w
        model.patch_embed.grid_size = (grid_h, grid_w)

    return model


# =====================
# Weight loading helpers
# =====================
def load_model_with_weights(
    builder_fn,
    weights_path: str,
    model_name: str,
) -> nn.Module:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"[ERROR] Weights file for {model_name} not found: {weights_path}"
        )

    model = builder_fn()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Loaded {model_name} weights from {weights_path}")
    return model


# =====================
# Global model instances (loaded once when module is imported)
# =====================
try:
    XCEPTION_MODEL = load_model_with_weights(
        build_xception, XCEPTION_WEIGHTS, "Xception"
    )
except FileNotFoundError as e:
    print(e)
    XCEPTION_MODEL = None

try:
    VIT_MODEL = load_model_with_weights(
        build_vit_b16, VIT_WEIGHTS, "ViT-B/16"
    )
except FileNotFoundError as e:
    print(e)
    VIT_MODEL = None

try:
    SWIN_MODEL = load_model_with_weights(
        build_swinv2_b, SWIN_WEIGHTS, "SwinV2-B"
    )
except FileNotFoundError as e:
    print(e)
    SWIN_MODEL = None


# =====================
# Preprocessing & prediction helpers
# =====================
def _preprocess_image(image_path: str) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    tensor = IMAGE_TRANSFORM(img).unsqueeze(0)  # shape: [1, C, H, W]
    return tensor.to(DEVICE)


@torch.no_grad()
def _predict_single_model(
    model: nn.Module,
    image_tensor: torch.Tensor,
) -> Dict[str, Any]:
    """
    Returns: {
        "class": int,
        "label": str,
        "confidence": float,
        "probs": [p0, p1]
    }
    """
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])

    return {
        "class": pred_class,
        "label": LABEL_MAP.get(pred_class, str(pred_class)),
        "confidence": confidence,
        "probs": probs.tolist(),
    }


# =====================
# Public API: detectDF(imagePath)
# =====================
def detectDF(image_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Run deepfake detection using all available models.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    Dict[str, Dict[str, Any]]:
        {
          "xception": { "class": 0/1, "label": "real"/"fake", "confidence": float, "probs": [p0, p1] },
          "vit_b16":  { ... },
          "swin_v2_b":{ ... }
        }
    """
    image_tensor = _preprocess_image(image_path)

    results: Dict[str, Dict[str, Any]] = {}

    if XCEPTION_MODEL is not None:
        results["xception"] = _predict_single_model(XCEPTION_MODEL, image_tensor)
    else:
        results["xception"] = {"error": "Xception model not loaded"}

    if VIT_MODEL is not None:
        results["vit_b16"] = _predict_single_model(VIT_MODEL, image_tensor)
    else:
        results["vit_b16"] = {"error": "ViT-B/16 model not loaded"}

    if SWIN_MODEL is not None:
        results["swin_v2_b"] = _predict_single_model(SWIN_MODEL, image_tensor)
    else:
        results["swin_v2_b"] = {"error": "SwinV2-B model not loaded"}

    return results



# if __name__ == "__main__":
#     test_path = "sample_35.jpg"
#     if os.path.exists(test_path):
#         out = detectDF(test_path)
#         print("Detection results:")
#         for model_name, info in out.items():
#             print(f"\n[{model_name}]")
#             print(info)
#     else:
#         print("No test image found; set 'test_path' to a real file to test.")
