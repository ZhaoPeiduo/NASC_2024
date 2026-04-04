"""Extracted OCR logic — wraps EasyOCR for Japanese text extraction."""
import base64
import io
from functools import lru_cache

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _get_reader():
    import easyocr
    return easyocr.Reader(["ja", "en"])


def extract_text(
    image_b64: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    num_options: int,
) -> dict:
    """
    Decode base64 image, crop to bounding box, run OCR, split into
    question and options. Returns {"question": str, "options": [str, ...]}.
    """
    image_bytes = base64.b64decode(image_b64.split(",")[-1])
    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)

    x1_, x2_ = min(x1, x2), max(x1, x2)
    y1_, y2_ = min(y1, y2), max(y1, y2)
    crop = img_array[y1_:y2_, x1_:x2_]

    reader = _get_reader()
    results = reader.readtext(crop)
    texts = [r[1] for r in sorted(results, key=lambda r: r[0][0][1])]

    if len(texts) <= num_options:
        return {"question": " ".join(texts), "options": []}

    question = " ".join(texts[:-num_options])
    options = texts[-num_options:]
    return {"question": question, "options": options}
