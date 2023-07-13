__all__ = []

from typing import Literal, Sequence, TypedDict, Union, cast, overload

import numpy as np
import torch
import torch.distributions
from torch import nn
from torch.distributions import Categorical, Exponential, HalfNormal, Normal, StudentT
from torch.distributions.constraints import Constraint
from torch.distributions.kl import kl_divergence

from vaeplayland.models.decoders.simple_decoder import create_decoder_network
from vaeplayland.models.experimental.multimodal_vae_t2 import MultimodalEncoderV2
from vaeplayland.models.loss import ELBODict, compute_kl_div, compute_log_prob
from vaeplayland.models.vae import VAE


class CategoricalDistributionArgs(TypedDict):
    logits: torch.Tensor


class DistributionArgs(TypedDict):
    df: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor


class VAEOutput(TypedDict):
    x_disc: list[CategoricalDistributionArgs]
    x_cont: list[DistributionArgs]
    z_loc: torch.Tensor
    z_scale: torch.Tensor


class MultimodalELBODictV2(ELBODict):
    disc_rec_loss: torch.Tensor
    cont_rec_loss: torch.Tensor


DecoderOutput = tuple[list[CategoricalDistributionArgs], list[DistributionArgs]]


class MultimodalDecoderV4(nn.Module):
    """Parameterize p(x|z). Note that x is multimodal, having multiple distinct
    distributions. The output of this network is split into the parameters of
    each distribution (a Categorical for each categorical dataset and a
    t-distribution for each continuous dataset).

    Args:
        disc_dims:
            Dimensions (excluding batch dimension) of each categorical dataset,
            expected to be a two-element tuple (num. features times
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

    distribution_arg_constraints = cast(dict[str, Constraint], StudentT.arg_constraints)

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
        self.distribution = StudentT
        if self.distribution.has_enumerate_support:
            raise ValueError("Distribution cannot be discrete.")

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


class MultimodalVAEv4(VAE[MultimodalEncoderV2, MultimodalDecoderV4]):
    def __init__(
        self, encoder: MultimodalEncoderV2, decoder: MultimodalDecoderV4
    ) -> None:
        super().__init__(encoder, decoder)

    def forward(self, batch: tuple[torch.Tensor, ...]) -> VAEOutput:
        x, *_ = batch
        z_loc, z_scale = self.encode(x)
        z = self.reparameterize(z_loc, z_scale)
        x_disc, x_cont = self.decoder(z)
        return {
            "x_disc": x_disc,
            "x_cont": x_cont,
            "z_loc": z_loc,
            "z_scale": z_scale,
        }

    def compute_loss(
        self, batch: tuple[torch.Tensor, ...], kl_weight: float
    ) -> MultimodalELBODictV2:
        # Split incoming data into discrete/continuous subset
        x, *_ = batch
        x_disc, x_cont = self.decoder.reshape_data(x, return_args=False)

        out = cast(VAEOutput, self(batch))

        # Calculate discrete reconstruction loss
        disc_rec_loss = torch.tensor(0.0)
        for i, args in enumerate(out["x_disc"]):
            y = torch.argmax(x_disc[i], dim=-1)
            disc_rec_loss += compute_log_prob(dist=Categorical, x=y, **args).mean()

        # Calculate continuous reconstruction loss
        cont_rec_loss = torch.tensor(0.0)
        for i, args in enumerate(out["x_cont"]):
            cont_rec_loss += compute_log_prob(
                dist=StudentT,
                x=x_cont[i],
                df=args["df"],
                loc=args["loc"],
                scale=args["scale"],
            ).mean()

        # Calculate overall reconstruction and regularization loss
        rec_loss = disc_rec_loss + cont_rec_loss
        reg_loss = compute_kl_div(out["z_loc"], out["z_scale"]).mean()

        elbo = disc_rec_loss + cont_rec_loss - kl_weight * reg_loss
        return {
            "elbo": elbo,
            "reg_loss": reg_loss,
            "rec_loss": rec_loss,
            "disc_rec_loss": disc_rec_loss,
            "cont_rec_loss": cont_rec_loss,
            "kl_weight": kl_weight,
        }
