__all__ = ["SimpleDecoder", "SimpleBimodalDecoder"]

from typing import Sequence, Union

import torch
from torch import nn


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        compress_dims: Union[int, Sequence[int]],
        embedding_dim: int,
        activation_fun_name: str = "ReLU",
        dropout_rate: float = 0.5,
        num_output_params: int = 2,
    ) -> None:
        """Parametrizes p(x|z).

        Parameters
        ----------
        input_dim: Tuple[int, int]
            Size of input data
        compress_dims: int or Sequence[int]
            Size of each layer
        embedding_dim: int
            Size of output. Will be doubled to account for location and scale
            of a Gaussian distribution
        activation_fun_name: str
            Name of activation function class (from torch.nn)
        dropout_rate: float
            Fraction of units to zero between activations
        num_output_params: int
            Number of parameters of output distribution (e.g., set to 1 for a
            Categorical or Bernouilli distribution, 2 for a Gaussian, or 3 for
            a Student's t-distribution)
        """
        super().__init__()

        input_dim *= num_output_params
        self.num_output_params = num_output_params
        if num_output_params < 1:
            raise ValueError("Must output at least one parameter")

        if not isinstance(compress_dims, Sequence):
            compress_dims = (compress_dims,)

        activation_fun = getattr(nn, activation_fun_name)
        assert issubclass(activation_fun, nn.Module)

        layers = []
        layer_dims = [*compress_dims, embedding_dim][::-1]
        out_features = None
        for in_features, out_features in zip(layer_dims, layer_dims[1:]):
            layers.append(nn.Linear(in_features, out_features, bias=True))
            layers.append(activation_fun())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        assert isinstance(out_features, int)
        self.network = nn.Sequential(
            *layers, nn.Linear(out_features, input_dim, bias=True)
        )

    def forward(self, batch: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.network(batch)
        return torch.chunk(x, chunks=self.num_output_params, dim=-1)


class SimpleBimodalDecoder(SimpleDecoder):
    def __init__(
        self,
        input_dim: int,
        compress_dims: Sequence[int],
        embedding_dim: int,
        split: int,
        activation_fun_name: str = "ReLU",
        dropout_rate: float = 0.5,
        num_output_params: tuple[int, int] = (1, 2),
    ) -> None:
        total_output_params = (
            split * num_output_params[0] + (input_dim - split) * num_output_params[1]
        )
        print(total_output_params)
        super().__init__(
            1,
            compress_dims,
            embedding_dim,
            activation_fun_name,
            dropout_rate,
            total_output_params,
        )
        self.split = split
        self.num_output_params = num_output_params

    def forward(self, batch: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.network(batch)
        x_params = []
        for x_i, chunks in zip(
            torch.tensor_split(x, [self.split], dim=-1), self.num_output_params
        ):
            if chunks > 1:
                x_params.extend(torch.chunk(x_i, chunks=chunks, dim=-1))
            else:
                x_params.append(x_i)
        return tuple(x_params)
