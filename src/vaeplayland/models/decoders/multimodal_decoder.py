__all__ = ["MultimodalDecoder"]

from typing import Literal, Sequence, TypedDict, Union, cast, overload

import numpy as np
import torch
from torch import nn

from vaeplayland.models.decoders.simple_decoder import create_decoder_network


class CategoricalDistributionArgs(TypedDict):
    logits: torch.Tensor


class TDistributionArgs(TypedDict):
    df: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor


DecoderOutput = tuple[list[CategoricalDistributionArgs], list[TDistributionArgs]]


class MultimodalDecoder(nn.Module):
    """Parameterize p(x|z). Note that x is multimodal, having multiple distinct
    distributions. The output of this network is split into the parameters of
    each distribution (a Categorical for each categorical dataset and a
    t-distribution for each continuous dataset).

    Args:
        disc_dims:
            Dimensions (excluding batch dimension) of each categorical dataset,
            each shape expected to be a two-element tuple (num. features times
            cardinality).
        cont_dims:
            Dimensions (excluding batch dimension) of each continuous dataset.
        compress_dims:
            Size of each layer.
        embedding_dim:
            Size of latent space.
        activation_fun_name:
            Name of activation function torch module. Default is "ReLU".
        batch_norm:
            Apply batch normalization.
        dropout_rate:
            Fraction of elements to zero between activations. Default is 0.5.
    """

    def __init__(
        self,
        disc_dims: Sequence[tuple[int, int]],
        cont_dims: Sequence[int],
        compress_dims: Union[int, Sequence[int]],
        embedding_dim: int,
        activation_fun_name: str = "ReLU",
        batch_norm: bool = False,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.disc_dims = list(disc_dims)
        self.disc_dims_1d = [int.__mul__(*shape) for shape in disc_dims]
        self.cont_dims = list(cont_dims)
        self.disc_sz = sum(self.disc_dims_1d)
        self.cont_sz = sum(cont_dims)

        output_dim = self.disc_sz + 3 * self.cont_sz

        layers = create_decoder_network(
            output_dim,
            compress_dims,
            embedding_dim,
            activation_fun_name,
            batch_norm,
            dropout_rate,
        )
        self.network = nn.Sequential(*layers)

    @overload
    def reshape_data(
        self, x: torch.Tensor, *, return_args: Literal[False]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        ...

    @overload
    def reshape_data(
        self, x: torch.Tensor, *, return_args: Literal[True]
    ) -> DecoderOutput:
        ...

    def reshape_data(
        self, x: torch.Tensor, *, return_args: bool = True
    ) -> Union[tuple[list[torch.Tensor], list[torch.Tensor]], DecoderOutput]:
        """Split a tensor into the original shapes of the discrete and
        continuous datasets.

        Args:
            x:
                Input tensor
            return_args:
                If True, return the args of a distribution instead of tensors

        Returns:
            A tuple containing two lists: one with the reshaped categorical
            subsets and one with the reshaped continuous subsets of input `x`
        """
        # Split into discrete/continuous subsets
        disc_subset, cont_subset = torch.tensor_split(x, [self.disc_sz], dim=-1)

        # Split & reshape categorical distributions
        split_indices = np.cumsum(self.disc_dims_1d)[:-1].tolist()
        disc_subsets = torch.tensor_split(disc_subset, split_indices, dim=-1)
        x_disc = []
        for subset, shape in zip(disc_subsets, self.disc_dims):
            subset = subset.reshape(-1, *shape)
            if return_args:
                x_disc.append({"logits": subset})
            else:
                x_disc.append(subset)

        # Split continuous distributions
        if not return_args:
            split_indices = np.cumsum(self.cont_dims)[:-1].tolist()
            cont_subsets = torch.tensor_split(cont_subset, split_indices, dim=-1)
            return x_disc, cont_subsets

        num_chunks = 3

        split_indices = np.cumsum(np.multiply(self.cont_dims, num_chunks))[:-1].tolist()
        cont_subsets = torch.tensor_split(cont_subset, split_indices, dim=-1)

        # Transform distribution args to meet support
        x_cont: list = []
        for subset in cont_subsets:
            chunks = torch.chunk(subset, num_chunks, dim=-1)
            args = {
                "df": chunks[0].mul(-0.5).exp().mul(27.5).add(2.5).pow(-1),
                "loc": chunks[1],
                "scale": chunks[2].mul(0.5).exp(),
            }
            x_cont.append(args)
        return x_disc, x_cont

    def forward(self, batch: torch.Tensor) -> DecoderOutput:
        x_params = self.network(batch)
        return self.reshape_data(x_params, return_args=True)
