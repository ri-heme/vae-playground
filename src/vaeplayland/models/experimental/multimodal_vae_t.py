__all__ = []

from math import log
from typing import Literal, Sequence, TypedDict, Union, cast, overload

import numpy as np
import torch
import torch.distributions
from torch import nn
from torch.distributions import Exponential, Categorical, Normal, StudentT, Uniform
from torch.distributions.constraints import Constraint

from vaeplayland.models.decoders.simple_decoder import create_decoder_network
from vaeplayland.models.encoders.simple_encoder import SimpleEncoder
from vaeplayland.models.loss import (
    BimodalELBODict,
    compute_log_prob,
    compute_cross_entropy,
    compute_kl_div,
    compute_t_log_prob,
)
from vaeplayland.models.vae import VAE


class CategoricalDistributionArgs(TypedDict):
    logits: torch.Tensor


class TDistributionArgs(TypedDict):
    df: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor


class VAEOutput(TypedDict):
    x_disc: list[CategoricalDistributionArgs]
    x_cont: list[TDistributionArgs]
    z: torch.Tensor
    z_loc: torch.Tensor
    z_scale: torch.Tensor


DecoderOutput = tuple[list[CategoricalDistributionArgs], list[TDistributionArgs]]


class MultimodalEncoder(SimpleEncoder):
    """Parameterize q(z|x). Note that x is multimodal, having a discrete and a
    continuous part.

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
        dropout_rate: float = 0.5,
    ) -> None:
        disc_dims_1d = [int.__mul__(*shape) for shape in disc_dims]
        input_dim = sum(disc_dims_1d) + sum(cont_dims)
        super().__init__(
            input_dim, compress_dims, embedding_dim, activation_fun_name, dropout_rate
        )


class MultimodalDecoder(nn.Module):
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
            output_dim, compress_dims, embedding_dim, activation_fun_name, dropout_rate
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

        split_indices = np.cumsum(
            np.multiply(self.cont_dims, len(self.distribution_arg_constraints))
        )[:-1].tolist()
        cont_subsets = torch.tensor_split(cont_subset, split_indices, dim=-1)

        # Transform distribution args to meet support
        x_cont: list[TDistributionArgs] = []
        for subset in cont_subsets:
            chunks = torch.chunk(subset, len(self.distribution_arg_constraints), dim=-1)
            args: TDistributionArgs = {
                "df": torch.pow(10, chunks[0]),
                "loc": chunks[1],
                "scale": chunks[2].exp(),
            }
            x_cont.append(args)
        return x_disc, x_cont

    def forward(self, batch: torch.Tensor) -> DecoderOutput:
        x_params = self.network(batch)
        return self.reshape_data(x_params, return_args=True)


class MultimodalVAE(VAE[MultimodalEncoder, MultimodalDecoder]):
    df_prior = Exponential(1.0)
    scale_prior = Uniform(log(1e-3), log(1e3), validate_args=False)

    def __init__(self, encoder: MultimodalEncoder, decoder: MultimodalDecoder) -> None:
        super().__init__(encoder, decoder)

    def forward(self, batch: tuple[torch.Tensor, ...]) -> VAEOutput:
        x, *_ = batch
        z_loc, z_scale = self.encode(x)
        z = self.reparameterize(z_loc, z_scale)
        x_disc, x_cont = self.decoder(z)
        return {
            "x_disc": x_disc,
            "x_cont": x_cont,
            "z": z,
            "z_loc": z_loc,
            "z_scale": z_scale,
        }

    def compute_loss(
        self, batch: tuple[torch.Tensor, ...], kl_weight: float
    ) -> BimodalELBODict:
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
                df=args["df"] * 27.5 + 2.5,
                loc=args["loc"],
                scale=args["scale"],
            ).mean()
            cont_rec_loss += self.compute_prior_log_prob(x_cont[i], **args)

        # Calculate overall reconstruction and regularization loss
        rec_loss = disc_rec_loss + cont_rec_loss
        reg_loss = compute_kl_div(out["z"], out["z_loc"], out["z_scale"]).mean()

        elbo = disc_rec_loss + cont_rec_loss - kl_weight * reg_loss
        return {
            "elbo": elbo,
            "reg_loss": reg_loss,
            "rec_loss": rec_loss,
            "cat_rec_loss": disc_rec_loss,
            "con_rec_loss": cont_rec_loss,
            "kl_weight": kl_weight,
        }

    @classmethod
    def compute_prior_log_prob(
        cls, x: torch.Tensor, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability density of estimated args fitting broad
        priors:

        - loc ~ Normal(x_mean, 1e3 * x_std)
        - log (scale / x_std) ~ Uniform(log 1e-3, log 1e3)
        - df ~ Exponential(1)
        """
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0)

        # loc ~ Normal
        loc_prior = Normal(mean, 1e3 * std)
        log_prob = loc_prior.log_prob(loc)

        # scale ~ Uniform
        log_prob += cls.scale_prior.log_prob(scale.log() - std.log())

        # df ~ Exponential
        log_prob += cls.df_prior.log_prob(df)

        return log_prob.mean(dim=0).sum()
