from typing import Tuple

import torch
from torch import nn
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.new_empty().normal_())
        return eps.mul(std).add_(mu)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        mu, logvar = self.encoder(batch)
        z = self.reparametrize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
