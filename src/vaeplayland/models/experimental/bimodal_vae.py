__all__ = ["ExperimentalBimodalDecoder", "ExperimentalBimodalVAE"]

from typing import Sequence, Union

import torch
from torch import nn

from vaeplayland.models.decoders.simple_decoder import create_decoder_network
from vaeplayland.models.vae import BimodalVAE


class ExperimentalBimodalDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        compress_dims: Union[int, Sequence[int]],
        embedding_dim: int,
        split: int,
        activation_fun_name: str = "ReLU",
        dropout_rate: float = 0.5,
    ) -> None:
        """Parameterize p(x|z). Note that x is bimodal, having two distinct
        distributions. The output of this network is split into the parameters
        of each distribution (a Categorical and Normal distribution).

        Args:
            input_dim:
                Size of input layer. Input is a concatenated matrix of two
                data modalities.
            compress_dims:
                Size of each layer.
            embedding_dim:
                Size of latent space.
            split:
                Index of input at which the two data modalities can be split.
            activation_fun_name:
                Name of activation function torch module. Default is "ReLU".
            dropout_rate:
                Fraction of elements to zero between activations. Default is
                0.5.
        """
        output_dim = 4 * input_dim - 3 * split  # eq. 4 * n_norm + n_cat

        layers = create_decoder_network(
            output_dim, compress_dims, embedding_dim, activation_fun_name, dropout_rate
        )
        self.network = nn.Sequential(*layers)
        self.split = split

    def forward(self, batch: torch.Tensor) -> Sequence[torch.Tensor]:
        x_params = self.network(batch)
        x_logits, x_norm_params = torch.tensor_split(x_params, [self.split], dim=-1)
        x_norm_params = torch.chunk(x_norm_params, 4, dim=-1)
        for i in range(1, 4, 2):
            x_norm_params[i] = x_norm_params[i].exp()
        # 5 params: logits (categorical) and 2 sets of loc/scale (normal)
        return x_logits, *x_norm_params


class ExperimentalBimodalVAE(BimodalVAE):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__(encoder, decoder)
