"""Wire encoding for the Porthole bus.

Implements the payload tagging rules of `docs/bus-protocol.md` §3. Nothing in
this module knows about sockets, clients, or topics — it turns Python values
into bytes and back, and that is all.
"""

from __future__ import annotations

import time
from typing import Any

import msgpack

# Wire dtypes. Anything not listed here is rejected rather than silently
# widened, because a UI that receives an unexpected dtype has no sane fallback.
ALLOWED_DTYPES = frozenset({
    "float32", "float64",
    "int8", "uint8", "int16", "uint16",
    "int32", "uint32", "int64", "uint64",
})

ALLOWED_MIMES = frozenset({"image/jpeg", "image/png"})


def array_payload(array: Any) -> dict:
    """Tag a numpy array for the wire as ``{"$nd": {...}}``.

    Forces C-contiguous little-endian layout, because the browser decoder does
    not handle Fortran order or byte swapping — see protocol §3.1.

    Prefer float32 for geometry: a 12k-point cloud is 144 KB as float32 versus
    288 KB as float64, at precision no display can resolve.
    """
    import numpy as np

    array = np.ascontiguousarray(array)
    if array.dtype.byteorder == ">":
        array = array.astype(array.dtype.newbyteorder("<"))

    name = str(array.dtype)
    if name not in ALLOWED_DTYPES:
        raise TypeError(
            f"dtype {name!r} cannot cross the bus. Allowed: "
            f"{', '.join(sorted(ALLOWED_DTYPES))}. Cast before publishing."
        )
    return {"$nd": {"dtype": name, "shape": list(array.shape), "data": array.tobytes()}}


def image_payload(data: bytes, mime: str = "image/jpeg", w: int = 0, h: int = 0) -> dict:
    """Tag already-encoded image bytes as ``{"$img": {...}}``."""
    if mime not in ALLOWED_MIMES:
        raise ValueError(f"mime {mime!r} not allowed; use one of {sorted(ALLOWED_MIMES)}")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(
            "image_payload() takes encoded image bytes, not a pixel array. "
            "Use jpeg_payload() to encode an array first."
        )
    return {"$img": {"mime": mime, "w": int(w), "h": int(h), "data": bytes(data)}}


def jpeg_payload(array: Any, quality: int = 80, bgr: bool = False) -> dict:
    """Encode an HxWx3 uint8 array to JPEG and tag it.

    Raw pixel buffers must never reach the socket: an 848x480 RGB frame is
    1.2 MB raw and ~40 KB at q80, which at 15 fps is 18 MB/s versus 600 KB/s.

    Set ``bgr=True`` if the array is already in OpenCV's channel order.
    """
    import numpy as np

    array = np.ascontiguousarray(array)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise TypeError(
            f"jpeg_payload() expects HxWx3 uint8, got {array.shape} {array.dtype}"
        )

    h, w = array.shape[:2]
    data = _encode_jpeg(array if bgr else array[:, :, ::-1], quality)
    return image_payload(data, "image/jpeg", w, h)


def _encode_jpeg(bgr_array: Any, quality: int) -> bytes:
    """Encode BGR uint8 to JPEG bytes, preferring cv2 and falling back to PIL."""
    try:
        import cv2
    except ImportError:
        pass
    else:
        ok, buf = cv2.imencode(".jpg", bgr_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

    try:
        import io

        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "JPEG encoding needs either opencv or pillow installed"
        ) from exc

    stream = io.BytesIO()
    Image.fromarray(bgr_array[:, :, ::-1]).save(stream, "JPEG", quality=quality)
    return stream.getvalue()


def encode_envelope(topic: str, seq: int, payload: Any, ts: float | None = None) -> bytes:
    """Build one msgpack envelope frame. See protocol §2."""
    return msgpack.packb(
        {
            "topic": topic,
            "seq": seq,
            "ts": time.time() if ts is None else float(ts),
            "payload": payload,
        },
        use_bin_type=True,
    )
