"""Decode bounded in-memory images. No remote fetches or server file paths."""
import base64
import binascii
import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError


def decode_images(inputs):
    images = []
    for source in inputs:
        try:
            data = base64.b64decode(source.data_url.split(",", 1)[1], validate=True)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("Each image must be at most 4 MiB")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as raw:
                    if raw.format not in {"PNG", "JPEG", "WEBP"}:
                        raise ValueError("Unsupported image format")
                    if raw.width * raw.height > 16_000_000:
                        raise ValueError("Image exceeds 16 megapixels")
                    # Correct phone-camera orientation and bound vision work.
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    images.append(image.copy())
        except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise ValueError("Invalid or oversized image") from exc
    return images
