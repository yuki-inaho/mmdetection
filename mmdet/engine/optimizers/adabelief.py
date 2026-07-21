from adabelief_pytorch import AdaBelief
from mmdet.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class AdaBeliefOptimizer(AdaBelief):
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

        # mmdetectionが要求するdefaults値を設定
        self.defaults.update(dict(betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad))
