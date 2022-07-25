__all__ = ["compute_elbo"]

from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal


def compute_kl_div(z: torch.Tensor, qz_loc: torch.Tensor, qz_scale: torch.Tensor):
    """Computes the KL divergence between posterior q(z|x) and prior p(z). The
    prior has a Normal(0, 1) distribution."""
    qz = Normal(qz_loc, qz_scale)
    pz = Normal(torch.zeros_like(qz_loc), torch.ones_like(qz_scale))
    kl_div: torch.Tensor = qz.log_prob(z) - pz.log_prob(z)
    return kl_div.sum(-1)


def compute_gaussian_log_prob(
    x: torch.Tensor, px_loc: torch.Tensor, px_scale: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes the log of the probability density of the likelihood p(x|z)."""
    if px_scale is None:
        px_scale = torch.ones(1)
    px = Normal(px_loc, px_scale)
    log_px: torch.Tensor = px.log_prob(x)
    return log_px.sum(dim=[*range(1, log_px.dim())])


def compute_elbo(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Computes the evidence lower bound objective."""
    x, = batch
    px_loc, z, qz_loc, qz_scale = model(batch)
    reg_loss = compute_kl_div(z, qz_loc, qz_scale)
    rec_loss = compute_gaussian_log_prob(x, px_loc)
    elbo = (reg_loss - rec_loss).mean()
    return elbo
