"""Image compression and preprocessing for Vision API (saves tokens)."""
import base64
import io
import logging
from pathlib import Path

from app.config import MAX_IMAGE_PX, IMAGE_QUALITY

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def compress_image(image_bytes: bytes, max_px: int = MAX_IMAGE_PX, quality: int = IMAGE_QUALITY) -> bytes:
    """Resize and compress image to save Vision API tokens."""
    if not HAS_PIL:
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert RGBA to RGB
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        # Resize if larger than max_px
        w, h = img.size
        if max(w, h) > max_px:
            ratio = max_px / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        # Compress
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Image compression failed: {e}, using original")
        return image_bytes


def load_and_compress_b64(file_path: str) -> tuple[str, str]:
    """Load image file, compress, return (base64_string, mime_type)."""
    ext = Path(file_path).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/jpeg")

    with open(file_path, "rb") as f:
        raw = f.read()

    compressed = compress_image(raw)
    return base64.b64encode(compressed).decode(), mime


def should_compress_image(file_path: str) -> bool:
    """Check if image is large enough to need compression."""
    mb = Path(file_path).stat().st_size / (1024 * 1024)
    return mb > 2
