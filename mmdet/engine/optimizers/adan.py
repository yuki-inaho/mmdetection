# Copyright (c) OpenMMLab. All rights reserved.
"""Adan optimizer registered for OpenMMLab configs.

Use as ``optimizer=dict(type='AdanOptimizer', ...)`` (or ``mmdet.AdanOptimizer``
from another default scope). The ``adan`` backend (sail-sg/Adan) is optional and
imported lazily so ``import mmdet`` never fails when it is absent; instantiating the
optimizer without it raises a clear error. ``fused=False`` (the default) uses the
pure-Python path and needs no CUDA fused kernel.
"""
from mmdet.registry import OPTIMIZERS

try:
    from adan import Adan as _Adan
except ImportError:
    _Adan = None

_Base = _Adan if _Adan is not None else object


@OPTIMIZERS.register_module()
class AdanOptimizer(_Base):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.98, 0.92, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=0.0,
        no_prox=False,
        foreach=True,
        fused=False,
        **kwargs,
    ):
        if _Adan is None:
            raise ImportError(
                "AdanOptimizer requires the 'adan' package (sail-sg/Adan). Install it, "
                "e.g. `FORCE_CUDA=0 pip install git+https://github.com/sail-sg/Adan.git "
                "--no-build-isolation` for the pure-Python build.")
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            no_prox=no_prox,
            foreach=foreach,
            fused=fused,
        )
        # mmengine's optimizer constructor expects these in ``defaults``.
        self.defaults.update(
            dict(
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                no_prox=no_prox,
                foreach=foreach,
                fused=fused,
            ))
