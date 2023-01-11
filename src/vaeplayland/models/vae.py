__all__ = ["VAE"]

import torch
from torch import nn
from torch.distributions import Normal


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def z_dim(self) -> int:
        z_dim = getattr(self.encoder, "num_latent_units")
        assert z_dim == getattr(self.decoder, "num_latent_units")
        return z_dim

    @staticmethod
    def reparameterize(loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return Normal(loc, scale).rsample()

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        x, _ = batch
        z_loc, z_scale = self.encode(x)
        z = self.reparameterize(z_loc, z_scale)
        x_loc, x_scale = self.decode(z)
        return x_loc, x_scale, z, z_loc, z_scale

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Parameterizes q(z|x), a Gaussian distribution.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        z_loc, z_scale : tuple[torch.Tensor, ...]
        """
        z_loc, z_log_var = self.encoder(x)
        z_scale = torch.exp(0.5 * z_log_var)
        return z_loc, z_scale

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Parameterizes p(x|z), a Gaussian distribution.

        Parameters
        ----------
        z : torch.Tensor

        Returns
        -------
        x_loc, x_scale : tuple[torch.Tensor, ...]
        """
        x_loc, x_log_scale = self.decoder(z)
        x_scale = torch.exp(x_log_scale)
        return x_loc, x_scale
