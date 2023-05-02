__all__ = ["compute_elbo"]

from typing import Optional, TypedDict, Union

import torch
from torch import nn
from torch.distributions import Categorical, Normal, StudentT


class ELBODict(TypedDict):
    elbo: torch.Tensor
    reg_loss: torch.Tensor
    rec_loss: torch.Tensor
    kl_weight: Union[float, torch.Tensor]


class BimodalELBODict(ELBODict):
    cat_rec_loss: torch.Tensor
    con_rec_loss: torch.Tensor


def compute_kl_div(z: torch.Tensor, qz_loc: torch.Tensor, qz_scale: torch.Tensor):
    """Compute the KL divergence between posterior q(z|x) and prior p(z). The
    prior has a Normal(0, 1) distribution."""
    qz = Normal(qz_loc, qz_scale)
    pz = Normal(0.0, 1.0)
    kl_div: torch.Tensor = qz.log_prob(z) - pz.log_prob(z)
    return kl_div.sum(-1)


def compute_gaussian_log_prob(
    x: torch.Tensor, px_loc: torch.Tensor, px_scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the log of the probability density of the likelihood p(x|z)."""
    if px_scale is None:
        px_scale = torch.ones(1)
    px = Normal(px_loc, px_scale)
    log_px: torch.Tensor = px.log_prob(x)
    return log_px.sum(dim=[*range(1, log_px.dim())])


def compute_t_log_prob(
    x: torch.Tensor, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Compute the log of the probability density of the likelihood p(x|z)."""
    px = StudentT(df, loc, scale)  # type: ignore
    log_px: torch.Tensor = px.log_prob(x)
    return log_px


def compute_cross_entropy(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Compute the log of the probability density of the likelihood p(x|z)."""
    px = Categorical(logits=logits)
    log_px: torch.Tensor = px.log_prob(x)
    return log_px.sum(dim=[*range(1, log_px.dim())])


def compute_elbo(
    model: nn.Module, batch: tuple[torch.Tensor, ...], kl_weight: float = 1.0
) -> ELBODict:
    """Compute the evidence lower bound objective."""
    x, _ = batch
    px_loc, px_scale, z, qz_loc, qz_scale = model(batch)
    rec_loss = compute_gaussian_log_prob(x, px_loc, px_scale).mean()
    reg_loss = compute_kl_div(z, qz_loc, qz_scale).mean()
    elbo = rec_loss - kl_weight * reg_loss
    return {
        "elbo": elbo,
        "reg_loss": reg_loss,
        "rec_loss": rec_loss,
        "kl_weight": kl_weight,
    }


def compute_bimodal_elbo(
    model: nn.Module, batch: tuple[torch.Tensor, ...], kl_weight: float = 1.0
) -> BimodalELBODict:
    x, y = batch
    x_logits, px_loc, px_log_scale, z, qz_loc, qz_scale = model(batch)
    _, x_con = torch.tensor_split(x, [getattr(model, "split")], dim=1)
    cat_rec_loss = compute_cross_entropy(y, x_logits).mean()
    con_rec_loss = compute_gaussian_log_prob(x_con, px_loc, px_log_scale.exp()).mean()
    rec_loss = cat_rec_loss + con_rec_loss
    reg_loss = compute_kl_div(z, qz_loc, qz_scale).mean()
    elbo = cat_rec_loss + con_rec_loss - kl_weight * reg_loss
    return {
        "elbo": elbo,
        "reg_loss": reg_loss,
        "rec_loss": rec_loss,
        "cat_rec_loss": cat_rec_loss,
        "con_rec_loss": con_rec_loss,
        "kl_weight": kl_weight,
    }
