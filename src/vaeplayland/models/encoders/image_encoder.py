__all__ = ["ImageEncoder"]

import torch
from torch import nn


def calculate_output_shape(
    input_shape: tuple[int, int], kernel_size: int, stride: int, padding: int
) -> tuple[int, int]:
    """Calculate output shape of convolutional layer.

    Args:
        input_shape: Width and height of input matrix
        kernel_size: Kernel size of convolution operation
        stride: Stride of convolution operation
        padding: Padding of convolution operation

    Returns:
        Width and height of matrix after a convolution operation
    """
    output_shape = []
    for d in range(2):
        output_shape.append(
            (input_shape[d] + (2 * padding - kernel_size)) // stride + 1,
        )
    return tuple(output_shape)


class ImageEncoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int], num_latent_units: int) -> None:
        """Parameterize q(z|x). Architecture originally described in
        `Wu & Goodman (2018) <https://arxiv.org/abs/1802.05335>`_, see
        Figure 8.

        Args:
            input_shape: Width and height of input image
            num_latent_units: Size of latent space
        """
        super().__init__()
        output_shape = calculate_output_shape(input_shape, 4, 2, 1)
        output_shape = calculate_output_shape(output_shape, 4, 2, 1)
        output_shape = calculate_output_shape(output_shape, 4, 2, 1)
        output_shape = calculate_output_shape(output_shape, 4, 1, 0)
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * int.__mul__(*output_shape), 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2 * num_latent_units),
        )
        self.num_latent_units = num_latent_units

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.network(batch)
        loc, log_var = x.chunk(2, -1)
        return loc, log_var
