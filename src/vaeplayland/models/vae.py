__all__ = ["VAE"]

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        scale = (0.5 * logvar).exp()
        return Normal(mu, scale).rsample()

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x, _ = batch
        z_mu, z_logvar = self.encoder(x)
        z = self.reparametrize(z_mu, z_logvar)
        x_mu, x_logsigma = self.decoder(z)
        return x_mu, x_logsigma, z, z_mu, z_logvar
