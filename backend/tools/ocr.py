"""
OCR for scanned / image documents — REQ-050 (stretch).

Real OCR via pytesseract + Pillow, used as a fallback when a PDF page has no
extractable born-digital text (i.e. it is a scanned image or a damage photo).

Graceful degradation: if Pillow, pytesseract, or the underlying tesseract
engine is not installed, every function returns an empty result instead of
raising — so the born-digital pipeline is never affected. Nothing here is on
the deterministic decision path; OCR only supplies text that is then scored
with the same confidence/bbox logic as any other extraction.
"""

from __future__ import annotations

import io
from pathlib import Path


def ocr_available() -> bool:
    """True only if Pillow, pytesseract, AND the tesseract engine are usable."""
    try:
        import PIL
        import pytesseract

        _ = PIL.__version__  # confirm Pillow is installed
        pytesseract.get_tesseract_version()  # confirm the engine is installed
        return True
    except Exception:
        return False


def _ocr_pil_image(image) -> tuple[str, float]:
    """Run tesseract on a PIL image → (text, mean_confidence 0..1). Never raises."""
    try:
        import pytesseract

        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        words = [w for w in data.get("text", []) if str(w).strip()]
        confs = [
            int(c)
            for c in data.get("conf", [])
            if str(c).lstrip("-").isdigit() and int(c) >= 0
        ]
        text = " ".join(words)
        confidence = round((sum(confs) / len(confs)) / 100.0, 4) if confs else 0.0
        return text, confidence
    except Exception:
        return "", 0.0


def ocr_image(image_path: str | Path) -> tuple[str, float]:
    """OCR an image file → (text, confidence). Returns ("", 0.0) on any failure."""
    try:
        from PIL import Image

        with Image.open(str(image_path)) as img:
            return _ocr_pil_image(img)
    except Exception:
        return "", 0.0


def ocr_image_bytes(data: bytes) -> tuple[str, float]:
    """OCR raw image bytes → (text, confidence). Returns ("", 0.0) on failure."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return _ocr_pil_image(img)
    except Exception:
        return "", 0.0
