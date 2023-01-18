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
        """Parametrize p(x|z). The output is the parameters (location and
        scale) of a Gaussian distribution.

        Args:
            input_dim:
                Size of input layer.
            compress_dims:
                Size of each layer.
            embedding_dim:
                Size of latent space.
            activation_fun_name:
                Name of activation function torch module. Default is "ReLU".
            dropout_rate:
                Fraction of elements to zero between activations. Default is
                0.5.
            num_output_params:
                Number of parameters of output distribution. For example,
                for a Bernoulli distribution, this number would be 1; 2 for a
                Gaussian; or 3 for a Student's t-distribution. Default is 2.
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
        """Parametrize p(x|z). Note that x is bimodal, having two distinct
        distributions. The output of this network is split into the parameters
        of each distribution.

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
            num_output_params:
                Number of parameters of each output distribution. For example,
                for a combined Bernoulli-Gaussian distribution, the default is
                (1, 2).
        """
        if len(num_output_params) != 2:
            raise ValueError("Specify # output parameters of two distributions.")
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
