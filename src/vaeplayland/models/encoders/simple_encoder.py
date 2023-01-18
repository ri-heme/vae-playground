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
        dropout_rate: float = 0.5,
    ) -> None:
        """Parametrizes q(z|x).

        Parameters
        ----------
        input_dim : Tuple[int, int]
            Size of input data
        compress_dims : int or Sequence[int]
            Size of each layer
        embedding_dim : int
            Size of output. Will be doubled to account for location and scale
            of a Gaussian distribution
        activation_fun_name: str
            Name of activation function class (from torch.nn)
        dropout_rate: float
            Fraction of units to zero between activations
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
        assert isinstance(out_features, int)
        self.network = nn.Sequential(
            *layers, nn.Linear(out_features, embedding_dim * 2, bias=True)
        )

    def forward(self, batch: torch.Tensor) -> Sequence[torch.Tensor]:
        return torch.chunk(self.network(batch), chunks=2, dim=-1)
