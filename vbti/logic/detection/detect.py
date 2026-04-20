"""Object detection for duck-in-cup task using Grounding DINO.

Replaces OWLv2 — significantly better detection rates especially for
the duck on top camera and during grasp/transport phases.

Includes area filtering to reject false positives where the model
detects the robot arm as the duck.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-base"
TEXT_PROMPT = "a blue rubber duck. a red cup."
LABEL_NAMES = ["duck", "cup"]

# Max detection box area as fraction of image — larger boxes are likely the robot arm.
# Gripper camera allows larger boxes since objects fill more of the frame.
DEFAULT_MAX_AREA = 0.08
GRIPPER_MAX_AREA = 0.35

DEFAULT_ONNX_PATH = Path.home() / ".cache" / "vbti" / "grounding_dino_base.onnx"

_trt_loaded = False

def _ensure_trt_libs():
    """Preload TensorRT shared libs into the process so ONNX Runtime's
    TensorRT EP can find them via dlopen. Only runs once."""
    global _trt_loaded
    if _trt_loaded:
        return
    _trt_loaded = True

    import sys, ctypes
    trt_dir = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "tensorrt_libs"
    if not trt_dir.exists():
        return
    # Load core libs in dependency order
    for lib_name in ["libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"]:
        lib_path = trt_dir / lib_name
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class DuckDetector:
    """Detects duck and cup in camera images using Grounding DINO."""

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.1,
        max_area: float = DEFAULT_MAX_AREA,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_area = max_area

        print(f"[DuckDetector] Loading {MODEL_ID} on {device}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        # Grounding DINO doesn't support fp16 — use fp32
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID
        ).to(device)
        self.model.eval()

        vram = torch.cuda.memory_allocated() / 1e6
        print(f"[DuckDetector] Ready — {vram:.0f} MB VRAM (threshold={confidence_threshold})")

    @torch.no_grad()
    def detect(self, image: np.ndarray, max_area: float | None = None) -> dict:
        """Detect duck and cup in a single image.

        Args:
            image: RGB numpy array (H, W, 3) uint8
            max_area: Override max area ratio for this call (e.g. for gripper cam)

        Returns:
            Dict with "duck" and "cup" keys, each containing:
                found, center, center_norm, bbox, confidence
        """
        if image is None:
            return self._empty_result()

        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)

        inputs = self.processor(
            images=pil_image, text=TEXT_PROMPT, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold,
            target_sizes=[(h, w)],
        )[0]

        return self._parse_detections(results, h, w, max_area or self.max_area)

    @torch.no_grad()
    def detect_batch(self, images: list[np.ndarray], max_area: float | None = None) -> list[dict]:
        """Batch detection — processes images one by one (G-DINO text encoder
        doesn't batch as cleanly as OWLv2, but inference is still fast).

        Args:
            images: List of RGB numpy arrays (H, W, 3) uint8
            max_area: Override max area ratio for all images in this batch

        Returns:
            List of detection dicts (same format as detect())
        """
        return [self.detect(img, max_area=max_area) for img in images]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_detections(self, results: dict, h: int, w: int, max_area: float) -> dict:
        """Pick the highest-confidence detection per class, with area filtering."""
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["text_labels"]

        img_area = h * w
        out = {}

        for name in LABEL_NAMES:
            # Find all detections for this object
            candidates = []
            for i, label in enumerate(labels):
                if name not in label:
                    continue
                box = boxes[i]
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                area_ratio = box_area / img_area

                # Skip boxes that are too large (likely robot arm)
                if area_ratio > max_area:
                    continue

                candidates.append((scores[i], box))

            if not candidates:
                out[name] = self._empty_obj()
                continue

            # Pick highest confidence
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_box = candidates[0]

            cx = (best_box[0] + best_box[2]) / 2
            cy = (best_box[1] + best_box[3]) / 2

            out[name] = {
                "found": True,
                "center": (float(cx), float(cy)),
                "center_norm": (float(cx / w), float(cy / h)),
                "bbox": tuple(float(v) for v in best_box),
                "confidence": float(best_score),
            }

        return out

    @staticmethod
    def _empty_obj() -> dict:
        return {
            "found": False,
            "center": (0.0, 0.0),
            "center_norm": (0.0, 0.0),
            "bbox": (0.0, 0.0, 0.0, 0.0),
            "confidence": 0.0,
        }

    def _empty_result(self) -> dict:
        return {name: self._empty_obj() for name in LABEL_NAMES}


class OnnxDuckDetector:
    """Detects duck and cup using ONNX-exported Grounding DINO.

    Same interface as DuckDetector but runs via ONNX Runtime with
    TensorRT/CUDA execution providers for faster inference.
    """

    def __init__(
        self,
        onnx_path: str | Path = DEFAULT_ONNX_PATH,
        device: str = "cuda",
        confidence_threshold: float = 0.1,
        max_area: float = DEFAULT_MAX_AREA,
    ):
        # Preload TensorRT shared libs so ORT TRT EP can find them
        _ensure_trt_libs()

        import onnxruntime as ort

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_area = max_area

        onnx_path = Path(onnx_path)
        meta_path = onnx_path.with_suffix(".json")

        print(f"[OnnxDuckDetector] Loading ONNX model: {onnx_path}")

        # Load pre-tokenized text from export metadata
        with open(meta_path) as f:
            text_meta = json.load(f)
        self._input_ids = np.array(text_meta["input_ids"], dtype=np.int64)
        self._attention_mask = np.array(text_meta["attention_mask"], dtype=np.int64)
        self._token_type_ids = np.array(text_meta["token_type_ids"], dtype=np.int64)

        # Expected processor output shape (from export)
        self._expected_pv_shape = tuple(text_meta.get("pixel_values_shape", []))
        self._expected_pm_shape = tuple(text_meta.get("pixel_mask_shape", []))

        # Load HF processor for image preprocessing and post-processing
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Set up ONNX Runtime session
        providers = []
        if device == "cuda":
            # Try TensorRT first, then CUDA — ORT falls back automatically
            available = ort.get_available_providers()
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.log_severity_level = 2  # suppress ScatterND warnings

        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts, providers=providers,
        )

        active_provider = self.session.get_providers()[0]
        print(f"[OnnxDuckDetector] Ready — provider={active_provider} "
              f"(threshold={confidence_threshold})")

    def detect(self, image: np.ndarray, max_area: float | None = None) -> dict:
        """Detect duck and cup in a single image.

        Args:
            image: RGB numpy array (H, W, 3) uint8
            max_area: Override max area ratio for this call

        Returns:
            Dict with "duck" and "cup" keys (same format as DuckDetector)
        """
        if image is None:
            return self._empty_result()

        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)

        # Use HF processor for image preprocessing only
        img_inputs = self.processor(
            images=pil_image, text=TEXT_PROMPT, return_tensors="np"
        )

        pixel_values = img_inputs["pixel_values"].astype(np.float32)
        pixel_mask = img_inputs["pixel_mask"].astype(np.int64)

        # Validate shape matches export (Swin backbone breaks on mismatched H/W)
        if self._expected_pv_shape and tuple(pixel_values.shape) != tuple(self._expected_pv_shape):
            raise RuntimeError(
                f"Processor output shape {pixel_values.shape} doesn't match ONNX export "
                f"shape {self._expected_pv_shape}. Re-export with matching input resolution."
            )

        # Run ORT inference with pre-tokenized text + processed image
        ort_inputs = {
            "input_ids": self._input_ids,
            "attention_mask": self._attention_mask,
            "token_type_ids": self._token_type_ids,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
        }

        logits, pred_boxes = self.session.run(None, ort_inputs)

        # Reconstruct output format for HF post-processing
        outputs = _OrtOutputProxy(
            torch.tensor(logits),
            torch.tensor(pred_boxes),
        )
        input_ids = torch.tensor(self._input_ids)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold,
            target_sizes=[(h, w)],
        )[0]

        return self._parse_detections(results, h, w, max_area or self.max_area)

    def detect_batch(self, images: list[np.ndarray], max_area: float | None = None) -> list[dict]:
        """Batch detection — processes images one by one."""
        return [self.detect(img, max_area=max_area) for img in images]

    def _parse_detections(self, results: dict, h: int, w: int, max_area: float) -> dict:
        """Pick the highest-confidence detection per class, with area filtering."""
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["text_labels"]

        img_area = h * w
        out = {}

        for name in LABEL_NAMES:
            candidates = []
            for i, label in enumerate(labels):
                if name not in label:
                    continue
                box = boxes[i]
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                area_ratio = box_area / img_area
                if area_ratio > max_area:
                    continue
                candidates.append((scores[i], box))

            if not candidates:
                out[name] = self._empty_obj()
                continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_box = candidates[0]

            cx = (best_box[0] + best_box[2]) / 2
            cy = (best_box[1] + best_box[3]) / 2

            out[name] = {
                "found": True,
                "center": (float(cx), float(cy)),
                "center_norm": (float(cx / w), float(cy / h)),
                "bbox": tuple(float(v) for v in best_box),
                "confidence": float(best_score),
            }

        return out

    @staticmethod
    def _empty_obj() -> dict:
        return {
            "found": False,
            "center": (0.0, 0.0),
            "center_norm": (0.0, 0.0),
            "bbox": (0.0, 0.0, 0.0, 0.0),
            "confidence": 0.0,
        }

    def _empty_result(self) -> dict:
        return {name: self._empty_obj() for name in LABEL_NAMES}


class _OrtOutputProxy:
    """Minimal proxy matching the model output interface expected by
    processor.post_process_grounded_object_detection()."""

    def __init__(self, logits: torch.Tensor, pred_boxes: torch.Tensor):
        self.logits = logits
        self.pred_boxes = pred_boxes


def create_detector(
    device: str = "cuda",
    confidence_threshold: float = 0.1,
    max_area: float = DEFAULT_MAX_AREA,
    use_onnx: bool = True,
) -> DuckDetector | OnnxDuckDetector:
    """Create detector -- uses ONNX if available, falls back to PyTorch.

    Args:
        device: Torch/ORT device.
        confidence_threshold: Minimum detection confidence.
        max_area: Max box area ratio (filters out robot arm).
        use_onnx: Prefer ONNX backend if model file exists.

    Returns:
        DuckDetector or OnnxDuckDetector instance.
    """
    if use_onnx and DEFAULT_ONNX_PATH.exists():
        print(f"[create_detector] Using ONNX backend: {DEFAULT_ONNX_PATH}")
        return OnnxDuckDetector(
            onnx_path=DEFAULT_ONNX_PATH,
            device=device,
            confidence_threshold=confidence_threshold,
            max_area=max_area,
        )
    else:
        if use_onnx:
            print(f"[create_detector] ONNX model not found at {DEFAULT_ONNX_PATH}, "
                  f"falling back to PyTorch")
        else:
            print(f"[create_detector] Using PyTorch backend")
        return DuckDetector(
            device=device,
            confidence_threshold=confidence_threshold,
            max_area=max_area,
        )


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from pathlib import Path

    test_dir = Path("/tmp/det_test_frames")
    if test_dir.exists():
        detector = DuckDetector()
        frames = sorted(test_dir.glob("*.png"))[:5]
        for p in frames:
            img = np.array(Image.open(p).convert("RGB"))
            cam = p.name.split("_")[0]
            ma = GRIPPER_MAX_AREA if cam == "gripper" else DEFAULT_MAX_AREA
            t0 = time.perf_counter()
            result = detector.detect(img, max_area=ma)
            dt = time.perf_counter() - t0
            print(f"\n{p.name} ({dt:.3f}s):")
            for obj_name in LABEL_NAMES:
                det = result[obj_name]
                if det["found"]:
                    print(f"  {obj_name}: conf={det['confidence']:.3f} center_norm={det['center_norm']}")
                else:
                    print(f"  {obj_name}: not found")
