__all__ = ["SimpleEncoder"]

from typing import Sequence, Union

import torch
from torch import nn


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        compress_dims: Union[int, Sequence[int]],
        embedding_dim: int,
        activation_fun_name: str = "ReLU",
        batch_norm: bool = False,
        dropout_rate: float = 0.5,
    ) -> None:
        """Parameterize q(z|x).

        Args:
            input_dim:
                Size of input layer.
            compress_dims:
                Size of each layer.
            embedding_dim:
                Size of latent space.
            activation_fun_name:
                Name of activation function torch module. Default is "ReLU".
            batch_norm:
                Apply batch normalization.
            dropout_rate:
                Fraction of elements to zero between activations. Default is
                0.5.
        """
        super().__init__()

        if not isinstance(compress_dims, Sequence):
            compress_dims = (compress_dims,)

        activation_fun = getattr(nn, activation_fun_name)
        assert issubclass(activation_fun, nn.Module)

        layers = []
        layer_dims = [input_dim, *compress_dims]
        out_features = None
        for in_features, out_features in zip(layer_dims, layer_dims[1:]):
            layers.append(nn.Linear(in_features, out_features, bias=True))
            layers.append(activation_fun())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
        assert isinstance(out_features, int)
        self.network = nn.Sequential(
            *layers, nn.Linear(out_features, embedding_dim * 2, bias=True)
        )

    def forward(self, batch: torch.Tensor) -> Sequence[torch.Tensor]:
        return torch.chunk(self.network(batch), chunks=2, dim=-1)
