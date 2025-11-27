import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(MODULE_DIR, "models")

# =====================================================================================
# Model components (copied to match customArch.py)
# =====================================================================================

class SpectralStatsMLP(nn.Module):
    """
    Wide MLP that maps per-band stats -> FiLM (gamma, beta) for ViT tokens.
    """
    def __init__(self, in_dim: int, embed_dim: int, hidden: int = 8192, depth: int = 2,
                 train_last_only: bool = True, dropout: float = 0.0):
        super().__init__()
        assert depth >= 1
        self.fc_in = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(max(0, depth - 1))]
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # produce gamma and beta for FiLM, shape [B, 2*embed_dim]
        self.fc_out = nn.Linear(hidden, 2 * embed_dim)

        # Freeze layers if requested
        if train_last_only:
            for p in self.fc_in.parameters():
                p.requires_grad = False
            for layer in self.hiddens:
                for p in layer.parameters():
                    p.requires_grad = False
            # Keep only final projection trainable
            for p in self.fc_out.parameters():
                p.requires_grad = True

    def forward(self, stats):  # stats: [B, in_dim]
        x = self.fc_in(stats)
        x = self.act(x)
        x = self.drop(x)
        for layer in self.hiddens:
            x = layer(x)
            x = self.act(x)
            x = self.drop(x)
        x = self.fc_out(x)  # [B, 2*embed_dim]
        gamma, beta = x.chunk(2, dim=-1)
        return gamma, beta


class HSViTL_Adapter(nn.Module):
    """
    - ViT-L backbone (31-channel input) with most blocks frozen
    - Spectral gating MLP conditions tokens using 31-band statistics
    - Train last N transformer blocks + head
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        vit_name = cfg["vit_name"]
        in_chans = cfg["in_chans"]
        img_size = cfg["img_size_for_vit"]

        # Create ViT-L; timm adapts patch_embed for in_chans != 3 when possible
        self.backbone = timm.create_model(
            vit_name,
            pretrained=True,
            in_chans=in_chans,
            img_size=img_size,
            num_classes=0,  # headless
        )
        embed_dim = self.backbone.num_features
        self.embed_dim = embed_dim

        # Spectral stats dimension
        spec_dim = in_chans
        if cfg["spec_stats_use_mean_std"]:
            spec_dim = in_chans * 2  # per-band mean and std

        self.spec_gate = SpectralStatsMLP(
            in_dim=spec_dim,
            embed_dim=embed_dim,
            hidden=cfg["spec_hidden"],
            depth=cfg["spec_depth"],
            train_last_only=cfg["spec_train_last_only"],
            dropout=cfg["dropout"],
        )

        # Classification head (binary) with small MLP
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(embed_dim // 2, 1),  # BCEWithLogits
        )

        # Freeze entire backbone by default
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last N transformer blocks
        n_blocks = len(self.backbone.blocks)
        n_trainable = min(cfg["num_trainable_blocks"], n_blocks)
        self.n_trainable_blocks = n_trainable
        if n_trainable > 0 and not cfg["train_head_only"]:
            for blk in self.backbone.blocks[-n_trainable:]:
                for p in blk.parameters():
                    p.requires_grad = True

        # Freeze patch_embed
        if hasattr(self.backbone, "patch_embed"):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False

        # Freeze pos_embed (it's a Parameter, not a Module)
        if hasattr(self.backbone, "pos_embed") and isinstance(
            self.backbone.pos_embed, torch.nn.Parameter
        ):
            self.backbone.pos_embed.requires_grad = False

        # Freeze cls_token if present
        if hasattr(self.backbone, "cls_token") and isinstance(
            self.backbone.cls_token, torch.nn.Parameter
        ):
            self.backbone.cls_token.requires_grad = False

        # Freeze final norm
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = False

        # Keep head trainable
        for p in self.head.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _forward_frozen_prefix(self, x):
        # x: [B, C=31, H, W], no grad
        b = x.shape[0]
        x = self.backbone.patch_embed(x)  # [B, N, D]
        # cls token
        if hasattr(self.backbone, "cls_token"):
            cls_token = self.backbone.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_token, x), dim=1)  # [B, 1+N, D]
        # absolute pos embed (if present)
        if hasattr(self.backbone, "pos_embed"):
            x = x + self.backbone.pos_embed
        # dropout after pos
        if hasattr(self.backbone, "pos_drop"):
            x = self.backbone.pos_drop(x)

        # Run the frozen prefix of blocks
        if self.n_trainable_blocks == 0:
            for blk in self.backbone.blocks:
                x = blk(x)
        else:
            for blk in self.backbone.blocks[:-self.n_trainable_blocks]:
                x = blk(x)
        return x  # [B, 1+N, D]

    def _apply_gating(self, tokens, gamma, beta):
        # tokens: [B, 1+N, D]; gamma/beta: [B, D]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return tokens * (1 + gamma) + beta

    def _compute_stats(self, x):
        # x: [B, 31, H, W] -> per-band mean (+ std)
        B, C, H, W = x.shape
        bands_mean = x.mean(dim=(2, 3))  # [B, C]
        if self.cfg["spec_stats_use_mean_std"]:
            bands_std = (
                x.var(dim=(2, 3), unbiased=False).sqrt().clamp_min(1e-8)
            )  # [B, C]
            stats = torch.cat([bands_mean, bands_std], dim=1)  # [B, 2C]
        else:
            stats = bands_mean
        return stats

    def forward(self, x):
        # x: [B, 31, H, W], float32 in [0,1]
        stats = self._compute_stats(x)  # [B, spec_dim]
        gamma, beta = self.spec_gate(stats)  # [B, D] each

        # Frozen prefix
        x_frozen = self._forward_frozen_prefix(x)  # [B, 1+N, D]

        # Apply gating before trainable tail
        x_mod = self._apply_gating(x_frozen, gamma, beta)  # FiLM

        # Trainable tail (last blocks)
        if self.n_trainable_blocks > 0 and not self.cfg["train_head_only"]:
            for blk in self.backbone.blocks[-self.n_trainable_blocks:]:
                x_mod = blk(x_mod)

        # Final norm (frozen)
        x_mod = self.backbone.norm(x_mod)
        cls = x_mod[:, 0]  # [B, D]
        logits = self.head(cls).squeeze(-1)  # [B]
        return logits


# =====================================================================================
# Helper: load & preprocess a single HSI .npz (match training behaviour)
# =====================================================================================

def _minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalize to [0,1] like in HsiNpzDataset."""
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        return (x - mn) / (mx - mn)
    return torch.zeros_like(x)


def load_single_hsi(hsi_fp32: np.ndarray, img_size: int) -> torch.Tensor:
    """
    Take an in-memory HSI cube [C, H, W] (float32 or convertible)
    and normalize + resize it like the training dataset.
    Returns a torch.Tensor [C, img_size, img_size].
    """
    if hsi_fp32.ndim != 3:
        raise ValueError(f"Expected 3D array [C,H,W], got {hsi_fp32.shape}")

    # Ensure float32 and torch tensor
    x = torch.from_numpy(hsi_fp32.astype(np.float32))  # [C,H,W]

    # Normalize like training dataset
    x = _minmax(x)

    # Resize if needed
    if x.shape[1] != img_size or x.shape[2] != img_size:
        x = F.interpolate(
            x.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return x  # [C, img_size, img_size]


# =====================================================================================
# Load checkpoint, run prediction on a single input, print class & confidences
# =====================================================================================

def load_model_and_cfg(checkpoint_dir: str, device: torch.device):
    """
    Load best.pt checkpoint and build HSViTL_Adapter with stored cfg.
    """
    ckpt_path = os.path.join(checkpoint_dir, "hyperguard.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["cfg"]
    model = HSViTL_Adapter(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg


def pred(hsi_fp32):
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Directory used in training script
    checkpoint_dir = DEFAULT_CHECKPOINT_DIR

    # Load model & cfg
    model, cfg = load_model_and_cfg(checkpoint_dir, device)

    # Load and preprocess single HSI
    img_size = cfg.get("img_size", cfg.get("img_size_for_vit", 256))
    x = load_single_hsi(hsi_fp32, img_size=img_size)  # [C,H,W]
    x = x.unsqueeze(0).to(device)  # [1,C,H,W]

    with torch.no_grad():
        logits = model(x)  # [1]
        prob_fake = torch.sigmoid(logits)[0].item()

    prob_real = 1.0 - prob_fake

    # Label mapping from training: 0 = real, 1 = fake
    class_names = {0: "real", 1: "fake"}
    pred_label = 1 if prob_fake >= 0.5 else 0
    pred_class = class_names[pred_label]

    return {
        "pred_class": pred_class,
        "pred_label": pred_label,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
    }
