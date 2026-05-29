# Running this mmdetection fork on cu12 (torch 2.1.0 + cu121)

Part of the OpenMMLab cu12 migration (end goal: train
`tomato_pipe_rgbd_rtmdet_obb_training` on cu12). Validated stack:
**Python 3.10 / torch 2.1.0+cu121 / mmcv 2.1.0 / mmengine 0.10.7 / mmdet 3.3.0**.

## Quick start (uv)

```bash
just sync          # uv sync (cu12 deps incl. mmcv 2.1.0 cu121 wheel) + this source on path (.pth)
just smoke         # import mmdet (mmcv/mmengine guards) + bbox_overlaps on GPU -> cuda_op_device: cuda
just env-doctor    # GPU + versions
```

`pyproject.toml` is a **virtual** uv project. mmcv comes from the prebuilt cu121
wheel; `mmcv<2.2.0` / `mmengine<1.0.0` (mmdet's `__init__.py` guards) are satisfied
by mmcv 2.1.0 + mmengine 0.10.7. mmdet is pure Python (no native ext), so `just sync`
makes it importable via a `.pth` file rather than a setuptools build.
