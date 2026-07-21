from adan import Adan
from mmdet.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class AdanOptimizer(Adan):
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

        # mmdetectionが要求するdefaults値を設定
        self.defaults.update(
            dict(
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                no_prox=no_prox,
                foreach=foreach,
                fused=fused,
            )
        )
