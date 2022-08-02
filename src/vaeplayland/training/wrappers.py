__all__ = ["TrainingLogic", "PyroTrainingLogic"]

import math
from typing import Any, Tuple

import torch
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage

import pyro
import pyro.distributions
import pyro.infer
import pyro.optim
from pyro.optim import PyroOptim

import vaeplayland
from vaeplayland.models.loss import compute_elbo


class TrainingLogic(pl.LightningModule):
    def __init__(
        self,
        vae: "vaeplayland.models.VAE",
        kl_weight: float = 1.0,
        annealing_epochs: int = 20,
        annealing_schedule: str = "linear",
    ) -> None:
        """Encapsulates the training loop logic.

        Parameters
        ----------
        vae : vaeplayland.models.VAE
        kl_weight : float, optional
        annealing_epochs : int, optional
        annealing_schedule : str, optional
        """
        super().__init__()
        self.vae = vae
        self.save_hyperparameters(ignore="vae")

    @property
    def annealing_factor(self) -> float:
        epoch = self.current_epoch
        if (
            self.trainer is not None
            and self.trainer.state.stage == RunningStage.TRAINING
            and epoch < self.hparams.annealing_epochs
        ):
            if self.hparams.annealing_schedule == "stairs":
                return epoch / self.hparams.annealing_epochs
            num_batches = len(self.trainer.train_dataloader)
            step = self.global_step
            slope = 1 / (self.hparams.annealing_epochs * num_batches)
            if self.hparams.annealing_schedule == "sigmoid":
                # actual slope is 10 times slope of linear function
                # shift is half of the annealing epochs
                # equation below is factorized to avoid repeating terms
                shift = 0.5
                return 1 / (1 + math.exp(10 * (shift - step * slope)))
            elif self.hparams.annealing_schedule == "linear":
                return max(1e-8, step * slope)  # Pyro won't accept zero
        return 1.0

    @property
    def kl_weight(self) -> float:
        return self.annealing_factor * self.hparams.kl_weight

    def forward(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self.vae(batch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Defines the logic in the training loop.

        Parameters
        ----------
        batch : torch.Tensor

        Returns
        -------
        negative_elbo : torch.Tensor
        """
        output = compute_elbo(self, batch, self.kl_weight)
        for key, value in output.items():
            self.log(f"{self.trainer.state.stage}_{key}", value)
        # objetive: minimize negative ELBO
        return -output["elbo"]

    def training_step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.step(batch)

    def validation_step(self, batch: Tuple[torch.Tensor, ...]) -> None:
        self.step(batch)


class PyroTrainingLogic(TrainingLogic):
    def __init__(
        self,
        vae: "vaeplayland.models.VAE",
        kl_weight: float = 1.0,
        annealing_epochs: int = 20,
        annealing_schedule: str = "linear",
    ) -> None:
        super().__init__(vae, kl_weight, annealing_epochs, annealing_schedule)
        self.optimizer = PyroOptim(optim.Adam, dict(lr=1e-4))
        self.svi = pyro.infer.SVI(
            self.model, self.guide, self.optimizer, pyro.infer.Trace_ELBO()
        )
        self.automatic_optimization = False  # handled by the SVI interface

    def model(self, x: torch.Tensor) -> None:
        """Defines the generative model for the data, i.e., p(x|z). In this
        model, the data is assumed to be generated from a latent space
        that has a normal distribution (our prior)."""
        batch_size = x.size(0)
        event_dims = x.ndim - 1  # number of dependent dimensions
        z_dim = self.vae.z_dim
        pyro.module("decoder", self.vae.decoder)
        with pyro.plate("data", batch_size):
            with pyro.poutine.scale(scale=self.kl_weight):
                z_loc = torch.zeros((batch_size, z_dim))
                z_scale = torch.ones((batch_size, z_dim))
                z_dist = pyro.distributions.Normal(z_loc, z_scale).to_event(1)
                z = pyro.sample("latent", z_dist)  # sample from prior
            x_loc, x_scale = self.vae.decode(z)
            x_dist = pyro.distributions.Normal(x_loc, x_scale).to_event(event_dims)
            pyro.sample("obs", x_dist, obs=x)  # sample from likelihood

    def guide(self, x: torch.Tensor) -> None:
        """Defines the inference model (in the probabilistic aception), which
        serves as an approximation to the posterior q(z|x)."""
        batch_size = x.size(0)
        pyro.module("encoder", self.vae.encoder)
        with pyro.plate("data", batch_size), pyro.poutine.scale(scale=self.kl_weight):
            z_loc, z_scale = self.vae.encode(x)
            z_dist = pyro.distributions.Normal(z_loc, z_scale).to_event(1)
            pyro.sample("latent", z_dist)  # sample from posterior

    def configure_optimizers(self) -> None:
        return

    def step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, _ = batch
        negative_elbo = self.svi.step(x) / x.size(0)
        self.log("train_elbo", -negative_elbo)
        self.log("train_kl_weight", self.kl_weight)
        # objetive: minimize negative ELBO
        return torch.Tensor([negative_elbo])

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        pass

    def backward(self, *args: Any, **kwargs: Any) -> None:
        pass
