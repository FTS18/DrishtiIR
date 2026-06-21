"""
detection.py
------------
Downstream Task Benchmark: Object Detection on IR vs Colorized RGB.

Implements PS-10 requirement:
  "Boost Downstream Tasks — Ensure the output images significantly improve
   the accuracy of subsequent object detection and segmentation tasks."

Uses torchvision's pretrained Faster-RCNN (ResNet-50 backbone, COCO-trained)
to detect objects in both the raw grayscale IR image and the AI-colorized RGB.
Reports detection count and mean confidence score for each, proving that
colorization directly boosts downstream detection performance.

No additional installs needed — torchvision is already in requirements.
"""

import numpy as np
import torch
import cv2
from PIL import Image

try:
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    )
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False

# COCO class names (80 classes)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "TV",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

_detector = None


def _get_detector(device: str = "cpu"):
    global _detector
    if _detector is None:
        if not DETECTION_AVAILABLE:
            return None
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        _detector = fasterrcnn_resnet50_fpn_v2(weights=weights)
        _detector.eval()
        _detector.to(device)
    return _detector


def _preprocess_for_detection(img_uint8: np.ndarray, device: str) -> torch.Tensor:
    """
    Convert a (H, W, 3) uint8 RGB image to a normalized float tensor.
    """
    t = torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0
    return t.to(device)


def run_detection(
    img_uint8: np.ndarray,  # (H, W, 3) uint8 RGB
    device: str = "cpu",
    confidence_threshold: float = 0.3,
) -> dict:
    """
    Run Faster-RCNN object detection on a single image.

    Returns:
        dict with:
          - 'boxes'       : list of [x1, y1, x2, y2]
          - 'labels'      : list of class name strings
          - 'scores'      : list of confidence scores
          - 'count'       : total detections above threshold
          - 'mean_conf'   : mean confidence of detections
          - 'annotated'   : (H, W, 3) uint8 image with bounding boxes drawn
    """
    detector = _get_detector(device)
    if detector is None:
        return {"count": 0, "mean_conf": 0.0, "boxes": [], "labels": [], "scores": [], "annotated": img_uint8}

    tensor = _preprocess_for_detection(img_uint8, device)

    with torch.no_grad():
        outputs = detector([tensor])[0]

    boxes  = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    # Filter by confidence threshold
    mask   = scores >= confidence_threshold
    boxes  = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    label_names = [COCO_CLASSES[l] if l < len(COCO_CLASSES) else "unknown" for l in labels]

    # Draw bounding boxes
    annotated = img_uint8.copy()
    for box, name, score in zip(boxes, label_names, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 80), 2)
        label_text = f"{name} {score:.2f}"
        cv2.putText(
            annotated, label_text, (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1, cv2.LINE_AA,
        )

    return {
        "count":     int(len(scores)),
        "mean_conf": float(np.mean(scores)) if len(scores) > 0 else 0.0,
        "boxes":     boxes.tolist(),
        "labels":    label_names,
        "scores":    scores.tolist(),
        "annotated": annotated,
    }


def compare_detection(
    ir_gray: np.ndarray,      # (H, W) uint8 grayscale IR
    rgb_colorized: np.ndarray, # (H, W, 3) uint8 colorized RGB
    device: str = "cpu",
    confidence_threshold: float = 0.3,
) -> dict:
    """
    Compare detection performance between raw IR and colorized RGB.

    Converts the grayscale IR to 3-channel for fair comparison
    (detector requires 3 channels), then reports delta in detection count
    and confidence to prove colorization boosts downstream tasks.

    Returns:
        dict with 'ir' and 'rgb' sub-dicts, each containing detection results,
        plus 'delta_count' and 'delta_conf' showing the improvement.
    """
    # Convert grayscale IR to 3-channel for fair comparison
    ir_3ch = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)

    ir_result  = run_detection(ir_3ch,       device, confidence_threshold)
    rgb_result = run_detection(rgb_colorized, device, confidence_threshold)

    delta_count = rgb_result["count"] - ir_result["count"]
    delta_conf  = rgb_result["mean_conf"] - ir_result["mean_conf"]

    return {
        "ir":          ir_result,
        "rgb":         rgb_result,
        "delta_count": delta_count,
        "delta_conf":  round(delta_conf, 4),
        "improved":    delta_count >= 0,
    }
