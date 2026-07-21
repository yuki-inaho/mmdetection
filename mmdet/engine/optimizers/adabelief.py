# Copyright (c) OpenMMLab. All rights reserved.
"""AdaBelief optimizer registered for OpenMMLab configs.

Use as ``optimizer=dict(type='AdaBeliefOptimizer', ...)`` (or
``mmdet.AdaBeliefOptimizer`` from another default scope). The ``adabelief_pytorch``
backend is optional: it is imported lazily so ``import mmdet`` never fails when the
package is absent; instantiating the optimizer without it raises a clear error.
"""
from mmdet.registry import OPTIMIZERS

try:
    from adabelief_pytorch import AdaBelief as _AdaBelief
except ImportError:
    _AdaBelief = None

# Subclass the backend when present; fall back to ``object`` so the class (and its
# registry entry) always exists even if the backend is not installed.
_Base = _AdaBelief if _AdaBelief is not None else object


@OPTIMIZERS.register_module()
class AdaBeliefOptimizer(_Base):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-16,
        weight_decay=1e-2,
        weight_decouple=True,
        fixed_decay=False,
        rectify=True,
        amsgrad=False,
        **kwargs,
    ):
        if _AdaBelief is None:
            raise ImportError(
                "AdaBeliefOptimizer requires the 'adabelief-pytorch' package. "
                "Install it with `pip install adabelief-pytorch`.")
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            fixed_decay=fixed_decay,
            rectify=rectify,
            amsgrad=amsgrad,
        )
        # mmengine's optimizer constructor expects these in ``defaults``.
        self.defaults.update(
            dict(betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad))
