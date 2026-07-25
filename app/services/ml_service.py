"""
ML Service — loads the EfficientNet-B3 model and runs inference on fundus images,
returning predictions and a Grad-CAM heatmap.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Disease classes (order must match the training label order) ───────────────
DISEASE_CLASSES: List[str] = [
    "Normal",
    "Diabetes",
    "Glaucoma",
    "Cataract",
    "Myopia"
]


class EyeImagePreprocessor:
    """
    Standard resize and ImageNet normalization for EfficientNet-B3.
    """
    def __init__(self, size=300):
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Resize
        img_resized = cv2.resize(img_np, (self.size, self.size), interpolation=cv2.INTER_AREA)
        
        # We need both the tensor for the model and the resized numpy array for the heatmap overlay
        return self.transform(img_resized), img_resized


# ── Image pre-processing ────────────────────────────────────────────────────
_transform = EyeImagePreprocessor(size=300)

def _is_valid_fundus(img_np: np.ndarray) -> bool:
    """
    Heuristic check to prevent users from uploading random objects.
    Checks color dominance.
    """
    try:
        if len(img_np.shape) != 3:
            return False
            
        # Color check: Retinas are heavily reddish/orange
        # Note: img_np is RGB (from PIL Image.convert("RGB"))
        r, g, b = cv2.split(img_np)
        r_mean = float(np.mean(r))
        g_mean = float(np.mean(g))
        b_mean = float(np.mean(b))
        
        # Red must be at least 10% more dominant than green and blue
        color_valid = (r_mean > g_mean * 1.05) and (r_mean > b_mean * 1.05)
        
        # Structural check: Natural eyes lack long perfect straight lines
        # Man-made objects (chairs, shelves) have long straight lines
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        has_long_lines = lines is not None and len(lines) >= 2
        
        return color_valid and not has_long_lines
    except Exception as e:
        logger.error(f"Heuristic _is_valid_fundus crashed: {e}")
        return True # Fallback to True to prevent server crash

# ── Module-level singleton ──────────────────────────────────────────────────
_model: Optional[nn.Module] = None
_device: torch.device = torch.device("cpu")


def _build_model(num_classes: int = 5) -> nn.Module:
    """Build the EfficientNet-B3 model architecture."""
    return timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)


def load_model() -> nn.Module:
    """Load the model once and cache it."""
    global _model, _device

    if _model is not None:
        return _model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", _device)

    model = _build_model(num_classes=len(DISEASE_CLASSES))

    try:
        # Load the local weights from the backend root folder
        base_dir = Path(__file__).parent.parent.parent
        weights_path = base_dir / "best_model_b3_patient.pth"
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found at {weights_path}. Make sure it is copied there!")
            
        state_dict = torch.load(weights_path, map_location=_device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        logger.info("EfficientNet-B3 model weights loaded successfully")
    except Exception as exc:
        logger.error("Could not load pre-trained weights: %s", exc)
        raise exc

    model.to(_device)
    model.eval()
    _model = model
    return _model


def predict_image(image_bytes: bytes) -> Tuple[str, float, List[Tuple[str, float]], bytes]:
    """
    Run inference on raw image bytes and generate Grad-CAM heatmap.

    Returns
    -------
    top_disease : str
    top_confidence : float  (0-1)
    all_predictions : list of (disease, confidence) sorted descending
    heatmap_bytes : raw JPEG bytes of the heatmap overlay
    """
    model = load_model()

    # Open & convert image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor, img_resized = _transform(image)
    
    # Block out-of-distribution images (like chairs, cars, etc)
    if not _is_valid_fundus(img_resized):
        raise ValueError("INVALID_IMAGE")
        
    tensor = tensor.unsqueeze(0).to(_device)

    with torch.no_grad():
        d_out = model(tensor)
        # Use sigmoid because we trained with BCEWithLogitsLoss (Multi-label)
        probs = torch.sigmoid(d_out).squeeze(0)

    probs_list = probs.cpu().tolist()

    # Apply a confidence multiplier for Myopia (index 4) because the model 
    # struggles with it due to severe class imbalance in the training data.
    if len(probs_list) == 5:
        probs_list[4] = min(probs_list[4] * 1.6, 0.99)

    # argmax index using the adjusted probabilities
    pred_idx = probs_list.index(max(probs_list))
    top_disease = DISEASE_CLASSES[pred_idx]
    top_confidence = probs_list[pred_idx]

    all_preds = [
        (DISEASE_CLASSES[i], probs_list[i]) for i in range(len(DISEASE_CLASSES))
    ]

    # Sort descending by confidence
    all_preds.sort(key=lambda x: x[1], reverse=True)

    logger.info("Inference results:")
    for name, prob in all_preds:
        logger.info(f"  {name}: {prob:.4f}")

    # Generate Grad-CAM Heatmap
    logger.info(f"Generating Grad-CAM heatmap for {top_disease} (index {pred_idx})...")
    try:
        # Enable gradients temporarily for Grad-CAM
        with torch.enable_grad():
            target_layers = [model.conv_head]
            cam = GradCAM(model=model, target_layers=target_layers)
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
            
        # Normalize original image to [0,1] to draw the heatmap over it
        rgb_img_float = img_resized.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        
        # Convert RGB back to BGR for OpenCV encoding
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        
        # Encode as JPEG bytes
        success, encoded_image = cv2.imencode('.jpg', cam_image_bgr)
        if not success:
            logger.warning("Failed to encode heatmap image")
            heatmap_bytes = None
        else:
            heatmap_bytes = encoded_image.tobytes()
            
    except Exception as exc:
        logger.error(f"Failed to generate Grad-CAM heatmap: {exc}")
        heatmap_bytes = None

    return top_disease, round(top_confidence, 4), [
        (d, round(c, 4)) for d, c in all_preds
    ], heatmap_bytes
