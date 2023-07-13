__all__ = ["MultimodalVAE"]

from typing import TypedDict, cast

import torch
from torch.distributions import Categorical, StudentT

from vaeplayland.models.decoders.multimodal_decoder import (
    CategoricalDistributionArgs,
    MultimodalDecoder,
    TDistributionArgs,
)
from vaeplayland.models.encoders.multimodal_encoder import MultimodalEncoder
from vaeplayland.models.loss import ELBODict, compute_kl_div, compute_log_prob
from vaeplayland.models.vae import VAE


class MultimodalVAEOutput(TypedDict):
    x_disc: list[CategoricalDistributionArgs]
    x_cont: list[TDistributionArgs]
    z_loc: torch.Tensor
    z_scale: torch.Tensor


class MultimodalELBODict(ELBODict):
    disc_rec_loss: torch.Tensor
    cont_rec_loss: torch.Tensor


class MultimodalVAE(VAE[MultimodalEncoder, MultimodalDecoder]):
    def __init__(self, encoder: MultimodalEncoder, decoder: MultimodalDecoder) -> None:
        super().__init__(encoder, decoder)

    def forward(self, batch: tuple[torch.Tensor, ...]) -> MultimodalVAEOutput:
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
    ) -> MultimodalELBODict:
        # Split incoming data into discrete/continuous subset
        x, *_ = batch
        x_disc, x_cont = self.decoder.reshape_data(x, return_args=False)

        out = cast(MultimodalVAEOutput, self(batch))

        # Calculate discrete reconstruction loss
        disc_rec_loss = torch.tensor(0.0)
        for i, args in enumerate(out["x_disc"]):
            y = torch.argmax(x_disc[i], dim=-1)
            disc_rec_loss += compute_log_prob(Categorical, y, **args).mean()

        # Calculate continuous reconstruction loss
        cont_rec_loss = torch.tensor(0.0)
        for i, args in enumerate(out["x_cont"]):
            cont_rec_loss += compute_log_prob(StudentT, x_cont[i], **args).mean()

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
