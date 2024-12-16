from typing import Tuple, List

import torch
from torch import nn

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@MODELS.register_module()
class SegmentationHead(BaseDecodeHead):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 384,
        embedding_dims: List[int] = [128, 64, 32],
        **kwargs
    ):
        super().__init__(num_classes=num_classes, **kwargs)
        self.upsample = Upsample((4, 4))
        embedding_dims = [input_dim] + embedding_dims

        stack = []
        for i in range(len(embedding_dims) - 1):
            stack.extend(
                [
                    nn.Conv2d(
                        in_channels=embedding_dims[i],
                        out_channels=embedding_dims[i + 1],
                        kernel_size=1,
                    ),
                    nn.ReLU(),
                ]
            )

        stack.append(
            nn.Conv2d(
                in_channels=embedding_dims[-1], out_channels=num_classes, kernel_size=1
            )
        )

        self.mlp = nn.Sequential(*stack)

    def forward(self, fuse):
        x = self.upsample(fuse)
        return self.mlp(x)


class Upsample(nn.Module):
    """Main class for upsampling 2D input with a specified factor"""

    def __init__(
        self,
        scale_factor: Tuple[int, int] = (2, 2),
        mode: str = "bilinear",
        align_corners: bool = True,
    ):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.mode = mode
        self.align_corners = align_corners
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x
