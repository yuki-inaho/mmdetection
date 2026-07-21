#!/usr/bin/env python3
"""mmdet cu12 GPU smoke: import mmdet (runs the mmcv/mmengine version guards) and
run bbox_overlaps on CUDA tensors.

    uv run is avoided; call via: .venv/bin/python tools/cu12/smoke_mmdet.py --output-json ...
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import run_smoke, standard_argv  # noqa: E402


def runner() -> dict[str, Any]:
    import torch
    from mmdet.structures.bbox import bbox_overlaps

    a = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]], device="cuda")
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]], device="cuda")
    ious = bbox_overlaps(a, b)
    torch.cuda.synchronize()
    return {
        "op": "mmdet.structures.bbox.bbox_overlaps",
        "cuda_op_device": ious.device.type,
        "output_shape": list(ious.shape),
        "output_dtype": str(ious.dtype),
        "self_iou_top_left": float(ious[0, 0].item()),
    }


def main() -> int:
    args = standard_argv("mmdet cu12 GPU smoke")
    return run_smoke(module_name="mmdet", runner=runner, output_json=args.output_json)


if __name__ == "__main__":
    sys.exit(main())
