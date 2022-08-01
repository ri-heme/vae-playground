__all__ = ["ImageDecoder"]

from typing import Tuple

import torch
from torch import nn

from vaeplayland.models.encoders.image_encoder import calculate_output_shape


class ImageDecoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], num_latent_units: int) -> None:
        """Parametrizes p(x|z). Architecture originally described in
        `Wu & Goodman (2018) <https://arxiv.org/abs/1802.05335>`_, see
        Figure 8.

        Parameters
        ----------
        input_shape : Tuple[int, int]
            Width and height of input image
        num_latent_units : int
            Size of latent space
        """
        super().__init__()
        output_shape = calculate_output_shape(input_shape, 4, 2, 1)
        output_shape = calculate_output_shape(output_shape, 4, 2, 1)
        output_shape = calculate_output_shape(output_shape, 4, 2, 1)
        output_shape = calculate_output_shape(output_shape, 4, 1, 0)
        self.network = nn.Sequential(
            nn.Linear(num_latent_units, 256 * int.__mul__(*output_shape)),
            nn.SiLU(),
            nn.Unflatten(-1, (256, *output_shape)),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.num_latent_units = num_latent_units

    def forward(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self.network(batch), self.log_scale
