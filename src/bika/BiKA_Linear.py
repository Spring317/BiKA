import math
import torch

from torch import nn

from .functional import _BiKALinearFn


class BiKA_Linear(nn.Module):
    __constants__ = (
        "in_features",
        "out_features",
        "per_connection_bias",
    )

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.in_features = in_features
        self.out_features = out_features
        self.per_connection_bias = True

        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                **factory_kwargs,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    (out_features, in_features),
                    **factory_kwargs,
                )
            )
        else:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features, in_features),
                    **factory_kwargs,
                ),
                persistent=False,
            )

        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if isinstance(self.bias, nn.Parameter):
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(-1) != self.in_features:
            raise ValueError(
                f"BiKA_Linear: expected x shape "
                f"(B, {self.in_features}), got {tuple(x.shape)}"
            )

        return _BiKALinearFn.apply(
            x,
            self.weight,
            self.bias,
        )


__all__ = [
    "BiKA_Linear",
]