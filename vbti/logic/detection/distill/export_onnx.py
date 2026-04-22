"""Export Grounding DINO to ONNX for fast inference.

Usage:
    python -m vbti.logic.detection.export_onnx [--output path/to/model.onnx]

Exports the IDEA-Research/grounding-dino-base model with a fixed text prompt
("a blue rubber duck. a red cup.") baked in. The exported model takes only
image inputs (pixel_values, pixel_mask) plus the pre-tokenized text tensors.

Default output: ~/.cache/vbti/grounding_dino_base.onnx
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from vbti.logic.detection.detect import MODEL_ID, TEXT_PROMPT, DEFAULT_ONNX_PATH

# Image size for the cameras used in this project (480x640 and 240x320 both
# produce 800x1066 after the HF processor's resize+pad). The ONNX model is
# exported at this fixed resolution — the OnnxDuckDetector forces images to
# this size via the processor's size config.
TRACE_INPUT_H, TRACE_INPUT_W = 480, 640


class _ExportWrapper(torch.nn.Module):
    """Wrapper that maps positional args to model kwargs for ONNX tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids,
                pixel_values, pixel_mask):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            return_dict=True,
        )
        return outputs.logits, outputs.pred_boxes


def export_onnx(output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading {MODEL_ID} (disable_custom_kernels=True)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, disable_custom_kernels=True
    )
    model.eval()

    # Tokenize fixed text prompt + get processor output shape
    print(f"[export] Text prompt: '{TEXT_PROMPT}'")
    dummy_image = np.zeros((TRACE_INPUT_H, TRACE_INPUT_W, 3), dtype=np.uint8)
    from PIL import Image as PILImage
    inputs = processor(
        images=PILImage.fromarray(dummy_image),
        text=TEXT_PROMPT,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    pixel_values = inputs["pixel_values"]
    pixel_mask = inputs["pixel_mask"]

    print(f"[export] input_ids shape: {input_ids.shape}")
    print(f"[export] pixel_values shape: {pixel_values.shape}")
    print(f"[export] pixel_mask shape: {pixel_mask.shape}")

    # Save tokenized text + expected shapes alongside the ONNX model
    proc_h, proc_w = pixel_values.shape[2], pixel_values.shape[3]
    text_meta = {
        "input_ids": input_ids.numpy().tolist(),
        "attention_mask": attention_mask.numpy().tolist(),
        "token_type_ids": token_type_ids.numpy().tolist(),
        "text_prompt": TEXT_PROMPT,
        "pixel_values_shape": [1, 3, proc_h, proc_w],
        "pixel_mask_shape": [1, proc_h, proc_w],
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(text_meta, f)
    print(f"[export] Saved text metadata: {meta_path}")

    # Wrap model for ONNX export (maps positional args to kwargs)
    wrapper = _ExportWrapper(model)
    wrapper.eval()

    # Export
    print(f"[export] Exporting to ONNX (opset 16)...")
    t0 = time.perf_counter()

    input_names = [
        "input_ids", "attention_mask", "token_type_ids",
        "pixel_values", "pixel_mask",
    ]
    output_names = ["logits", "pred_boxes"]

    # Only batch is dynamic — Swin backbone concat ops break with dynamic H/W.
    # The processor always produces the same output size for our camera resolutions.
    dynamic_axes = {
        "pixel_values": {0: "batch"},
        "pixel_mask": {0: "batch"},
        "logits": {0: "batch"},
        "pred_boxes": {0: "batch"},
    }

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
        do_constant_folding=True,
    )

    export_time = time.perf_counter() - t0

    # Validate
    print("[export] Validating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model, full_check=True)

    file_size_mb = output_path.stat().st_size / 1e6
    print(f"[export] Done!")
    print(f"[export] Output: {output_path}")
    print(f"[export] Size: {file_size_mb:.1f} MB")
    print(f"[export] Export time: {export_time:.1f}s")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Grounding DINO to ONNX")
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_ONNX_PATH),
        help=f"Output ONNX file path (default: {DEFAULT_ONNX_PATH})",
    )
    args = parser.parse_args()
    export_onnx(Path(args.output))


if __name__ == "__main__":
    main()
