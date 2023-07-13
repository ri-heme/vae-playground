__all__ = []

from math import log, pi, sqrt
from typing import TypedDict, cast

import torch
import torch.distributions
from torch.distributions import (
    Categorical,
    Exponential,
    HalfNormal,
    Normal,
    StudentT,
    Uniform,
)
from torch.distributions.kl import kl_divergence

from vaeplayland.models.experimental.multimodal_vae_t2 import (
    MultimodalDecoderV2,
    MultimodalEncoderV2,
)
from vaeplayland.models.loss import ELBODict, compute_kl_div, compute_log_prob
from vaeplayland.models.vae import VAE


class CategoricalDistributionArgs(TypedDict):
    logits: torch.Tensor


class DistributionArgs(TypedDict):
    df_rate: torch.Tensor
    loc_loc: torch.Tensor
    loc_scale: torch.Tensor
    scale_scale: torch.Tensor


class VAEOutput(TypedDict):
    x_disc: list[CategoricalDistributionArgs]
    x_cont: list[DistributionArgs]
    z_loc: torch.Tensor
    z_scale: torch.Tensor


class MultimodalELBODictV2(ELBODict):
    disc_rec_loss: torch.Tensor
    cont_rec_loss: torch.Tensor
    kl_loss: torch.Tensor
    kl_loc: torch.Tensor
    kl_scale: torch.Tensor
    kl_df: torch.Tensor
    kl_weight_decoder: float


DecoderOutput = tuple[list[CategoricalDistributionArgs], list[DistributionArgs]]


class MultimodalVAEv3(VAE[MultimodalEncoderV2, MultimodalDecoderV2]):
    df_prior = Exponential(1.0)
    scale_prior = Uniform(log(1e-3), log(1e3), validate_args=False)

    def __init__(
        self, encoder: MultimodalEncoderV2, decoder: MultimodalDecoderV2
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
        self, batch: tuple[torch.Tensor, ...], kl_weight: tuple[float, float]
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

        kl_loc = torch.tensor(0.0)
        kl_scale = torch.tensor(0.0)
        kl_df = torch.tensor(0.0)
        for i, args in enumerate(out["x_cont"]):
            cont_rec_loss += compute_log_prob(
                dist=StudentT,
                x=x_cont[i],
                df=args["df_rate"].mul(27.5).add(2.5).pow(-1),
                loc=args["loc_loc"],
                scale=args["scale_scale"].mul(sqrt(2 / pi)),
            ).mean()
            loc_loss, scale_loss, df_loss = self.compute_kl_terms(x_cont[i], **args)
            kl_loc += loc_loss.mean()
            kl_scale += scale_loss.mean()
            kl_df += df_loss.mean()

        cont_kl_loss = kl_loc + kl_scale + kl_df

        # Calculate overall reconstruction and regularization loss
        rec_loss = disc_rec_loss + cont_rec_loss
        reg_loss = compute_kl_div(out["z_loc"], out["z_scale"]).mean()

        kl_weight_encoder, kl_weight_decoder = kl_weight
        elbo = (
            disc_rec_loss
            + cont_rec_loss
            - kl_weight_encoder * reg_loss
            - kl_weight_decoder * cont_kl_loss
        )
        return {
            "elbo": elbo,
            "reg_loss": reg_loss,
            "rec_loss": rec_loss,
            "disc_rec_loss": disc_rec_loss,
            "cont_rec_loss": cont_rec_loss,
            "kl_loss": cont_kl_loss,
            "kl_loc": kl_loc,
            "kl_scale": kl_scale,
            "kl_df": kl_df,
            "kl_weight": kl_weight_encoder,
            "kl_weight_decoder": kl_weight_decoder,
        }

    @classmethod
    def compute_kl_terms(
        cls,
        x: torch.Tensor,
        df_rate: torch.Tensor,
        loc_loc: torch.Tensor,
        loc_scale: torch.Tensor,
        scale_scale: torch.Tensor,
        eps: float = 1e-3,
    ) -> tuple:
        """Compute KL of estimated args fitting broad priors:

        - loc ~ Normal(batch_mean, 1e3 * (batch_std + eps))
        - log (scale / (batch_std + eps)) ~ Uniform(log 1e-3, log 1e3)
        - df ~ Exponential(1)
        """
        batch_mean = x.detach().mean(dim=0)
        batch_std = x.detach().std(dim=0, unbiased=False) + eps

        # loc ~ Normal
        q_loc = Normal(loc_loc, loc_scale)
        p_loc = Normal(batch_mean, 10 * batch_std)
        kl_loc = kl_divergence(q_loc, p_loc).sum(dim=-1)

        # scale ~ Uniform
        q_loc = HalfNormal(scale_scale)
        p_loc = HalfNormal(10 * batch_std)
        kl_scale = kl_divergence(q_loc, p_loc).sum(dim=-1)

        # df ~ Exponential
        q_loc = Exponential(df_rate)
        p_loc = Exponential(1.0)
        kl_df = kl_divergence(q_loc, p_loc).sum(dim=-1)

        return kl_loc, kl_scale, kl_df
