__all__ = ["compute_elbo"]

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Normal


def compute_kl_div(z: torch.Tensor, qz_loc: torch.Tensor, qz_scale: torch.Tensor):
    """Computes the KL divergence between posterior q(z|x) and prior p(z). The
    prior has a Normal(0, 1) distribution."""
    qz = Normal(qz_loc, qz_scale)
    pz = Normal(0.0, 1.0)
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


def compute_elbo(
    model: nn.Module, batch: Tuple[torch.Tensor, ...], kl_weight: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Computes the evidence lower bound objective."""
    x, _ = batch
    px_loc, px_scale, z, qz_loc, qz_scale = model(batch)
    rec_loss = compute_gaussian_log_prob(x, px_loc, px_scale).mean()
    reg_loss = compute_kl_div(z, qz_loc, qz_scale).mean()
    elbo = rec_loss - kl_weight * reg_loss
    return dict(
        elbo=elbo,
        reg_loss=reg_loss,
        rec_loss=rec_loss,
        kl_weight=kl_weight,
    )
