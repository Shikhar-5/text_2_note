"""CPU inference adapter for the published Handwriting Transformers model."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .style_crops import WordCrop


MODEL_DIR = Path(__file__).resolve().parent / "hwt_model"


class HandwritingModelError(RuntimeError):
    """A handwriting profile cannot be synthesized with the local model."""


def model_directory() -> Path:
    return Path(os.environ.get("TEXT2NOTE_HWT_DIR", str(MODEL_DIR))).resolve()


@lru_cache(maxsize=1)
def _load_model():
    root = model_directory()
    checkpoint = root / "files" / "iam_model.pth"
    if not checkpoint.is_file() or not (root / "models" / "model.py").is_file():
        raise HandwritingModelError(
            "Personal handwriting model is not installed. Run "
            "python scripts/setup_handwriting_model.py first."
        )

    try:
        import torch
        import torchvision.models as torchvision_models
    except ImportError as error:
        raise HandwritingModelError(
            "PyTorch and torchvision are required for personal handwriting."
        ) from error

    sys.path.insert(0, str(root))
    from models.OCR_network import strLabelConverter
    from models.model import Generator
    from params import ALPHABET, resolution

    # Generator() asks torchvision to fetch unrelated ImageNet weights. The
    # published HWT checkpoint replaces every generator weight immediately.
    original_resnet18 = torchvision_models.resnet18
    torchvision_models.resnet18 = lambda *args, **kwargs: original_resnet18(weights=None)
    try:
        generator = Generator().cpu()
    finally:
        torchvision_models.resnet18 = original_resnet18

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    generator.load_state_dict(state)
    generator.eval()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    return generator, strLabelConverter(ALPHABET), ALPHABET, resolution


def _style_tensor(crops: list[WordCrop]):
    import torch

    if len(crops) < 5:
        raise HandwritingModelError(
            "Not enough clear handwriting was found. Upload a page with at "
            "least five separated handwritten words."
        )

    selected = crops[:15]
    if len(selected) < 15:
        selected = (selected * (15 // len(selected) + 1))[:15]
    images = []
    for item in selected:
        image = item.image
        ys, xs = np.where(image < 245)
        if len(xs):
            image = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        width = max(1, min(192, round(image.shape[1] * 32 / image.shape[0])))
        resized = cv2.resize(image, (width, 32), interpolation=cv2.INTER_AREA)
        padded = np.full((32, 192), 255, dtype=np.uint8)
        padded[:, :width] = resized
        images.append((padded.astype(np.float32) / 127.5) - 1)
    return torch.from_numpy(np.stack(images, axis=0)).unsqueeze(0)


def generate_word_masks(crops: list[WordCrop], words: list[str]) -> list[np.ndarray]:
    """Render each requested word as a grayscale ink alpha mask.

    Unsupported characters fail explicitly so the output never silently
    changes the requested text.
    """
    if not words:
        return []
    generator, converter, alphabet, resolution = _load_model()
    unsupported = sorted({char for word in words for char in word if char not in alphabet})
    if unsupported:
        shown = " ".join(repr(char) for char in unsupported)
        raise HandwritingModelError(f"Personal handwriting model cannot write: {shown}")

    import torch

    style = _style_tensor(crops)
    results = []
    for start in range(0, len(words), 12):
        batch = words[start : start + 12]
        encoded, lengths = converter.encode([word.encode("utf-8") for word in batch])
        with torch.inference_mode():
            generated = generator.Eval(style, encoded.unsqueeze(0))
        if len(generated) != len(batch):
            raise HandwritingModelError("Model returned the wrong number of handwritten words.")
        for length, tensor in zip(lengths.tolist(), generated):
            pixels = tensor[0, 0, :, : length * resolution].clamp(-1, 1)
            gray = ((pixels.cpu().numpy() + 1) * 127.5).astype(np.uint8)
            alpha = np.clip((255 - gray.astype(np.float32) - 20) * 1.85, 0, 255).astype(
                np.uint8
            )
            contrast = float(np.percentile(alpha, 95))
            if 0 < contrast < 220:
                alpha = np.clip(alpha.astype(np.float32) * min(1.6, 220 / contrast), 0, 255).astype(np.uint8)
            if alpha.max() < 30:
                raise HandwritingModelError("Model returned an empty handwritten word.")
            results.append(alpha)
    if len(results) != len(words):
        raise HandwritingModelError("Model did not return every requested word.")
    return results
