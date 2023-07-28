__all__ = ["SimpleDecoder", "SimpleBimodalDecoder"]

from typing import List, Sequence, Tuple, Union

import torch
from torch import nn


def create_decoder_network(
    output_dim: int,
    compress_dims: Union[int, Sequence[int]],
    embedding_dim: int,
    activation_fun_name: str,
    batch_norm: bool,
    dropout_rate: float,
) -> List[nn.Module]:
    """Create a decoder network that decompresses embedding dimension into
    output dimension by layering linear functions, non-linear activations, and
    dropout."""
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
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
    assert isinstance(out_features, int)
    layers.append(nn.Linear(out_features, output_dim, bias=True))
    return layers


def get_num_args(distribution_name: str) -> int:
    """Return number of parameters that describe a distribution."""
    if distribution_name in ("Bernoulli", "Categorical", "Exponential"):
        return 1
    elif distribution_name in ("Normal", "LogNormal"):
        return 2
    elif distribution_name == "StudentT":
        return 3
    else:
        raise ValueError("Unsupported distribution")


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        compress_dims: Union[int, Sequence[int]],
        embedding_dim: int,
        activation_fun_name: str = "ReLU",
        batch_norm: bool = False,
        dropout_rate: float = 0.5,
        output_distribution: Union[str, int] = "Normal",
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
            batch_norm:
                Apply batch normalization.
            dropout_rate:
                Fraction of elements to zero between activations. Default is
                0.5.
            output_distribution:
                Name of output distribution (as named in torch's distributions
                module). For example, 'Normal' or 'Categorical'. Alternatively,
                the number of total output parameters
        """
        super().__init__()

        if isinstance(output_distribution, str):
            num_output_dist_args = get_num_args(output_distribution)
        else:
            num_output_dist_args = output_distribution

        input_dim *= num_output_dist_args
        self.total_output_args = num_output_dist_args

        layers = create_decoder_network(
            input_dim,
            compress_dims,
            embedding_dim,
            activation_fun_name,
            batch_norm,
            dropout_rate,
        )
        self.network = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.network(batch)
        return torch.chunk(x, chunks=self.total_output_args, dim=-1)


class SimpleBimodalDecoder(SimpleDecoder):
    def __init__(
        self,
        input_dim: int,
        compress_dims: Union[int, Sequence[int]],
        embedding_dim: int,
        split: int,
        activation_fun_name: str = "ReLU",
        batch_norm: bool = False,
        dropout_rate: float = 0.5,
        output_distributions: Tuple[str, str] = ("Categorical", "Normal"),
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
            batch_norm:
                Apply batch normalization.
            dropout_rate:
                Fraction of elements to zero between activations. Default is
                0.5.
            output_distributions:
                Distribution of each data modality. The default is 'Categorical'
                and 'Normal'.
        """
        if len(output_distributions) != 2:
            raise ValueError("Specify only two output distributions.")
        self.output_distributions = output_distributions
        total_output_args = (
            split * self.num_output_args[0]
            + (input_dim - split) * self.num_output_args[1]
        )
        super().__init__(
            1,
            compress_dims,
            embedding_dim,
            activation_fun_name,
            batch_norm,
            dropout_rate,
            total_output_args,
        )
        self.split = split

    @property
    def num_output_args(self) -> Sequence[int]:
        return [get_num_args(dist) for dist in self.output_distributions]

    def forward(self, batch: torch.Tensor) -> Sequence[torch.Tensor]:
        x = self.network(batch)
        x_params = []
        for x_i, chunks in zip(
            torch.tensor_split(x, [self.split], dim=-1), self.num_output_args
        ):
            if chunks > 1:
                x_params.extend(torch.chunk(x_i, chunks=chunks, dim=-1))
            else:
                x_params.append(x_i)
        return tuple(x_params)
