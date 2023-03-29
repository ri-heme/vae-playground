__all__ = ["TrainingLogic", "PyroTrainingLogic"]

import math
from typing import Any, Callable, Literal, Sized, cast

import pyro
import pyro.distributions
import pyro.infer
import pyro.optim
import pytorch_lightning as pl
import torch
from pyro.optim import PyroOptim
from pytorch_lightning.trainer.states import RunningStage
from torch import nn, optim

from vaeplayland.models.vae import VAE

AnnealingFunction = Literal["linear", "sigmoid", "stairs"]


class TrainingLogic(pl.LightningModule):
    def __init__(
        self,
        vae: VAE,
        kl_weight: float = 1.0,
        annealing_epochs: int = 20,
        annealing_function: AnnealingFunction = "linear",
        lr: float = 1e-4,
    ) -> None:
        """Encapsulate the training loop logic.

        Args:
            vae:
                A variational autoencoder instance
            kl_weight:
                Weight applied to the regularizing term
            annealing_epochs:
                Number of epochs, after which the full KL weight is applied. KL
                weight starts at 0 and is increased every training step until
                it reaches its full value. Set to 0 to disable KL warm-up.
            annealing_function:
                Monotonic function used to warm-up the KL term. Can be a
                linear, sigmoid, or stairstep function.
            lr:
                Learning rate of the Adam optimizer
        """
        super().__init__()
        self.vae = vae
        self.save_hyperparameters(ignore="vae")

    @property
    def annealing_factor(self) -> float:
        epoch = self.current_epoch
        annealing_epochs: int = getattr(self.hparams, "annealing_epochs")
        annealing_function: str = getattr(self.hparams, "annealing_function")
        if (
            self.trainer is not None
            and self.trainer.state.stage == RunningStage.TRAINING
            and epoch < annealing_epochs
        ):
            if annealing_function == "stairs":
                return epoch / annealing_epochs
            num_batches = len(cast(Sized, self.trainer.train_dataloader))
            step = self.global_step
            slope = 1 / (annealing_epochs * num_batches)
            if annealing_function == "sigmoid":
                # actual slope is 10 times slope of linear function
                # shift is half of the annealing epochs
                # equation below is factorized to avoid repeating terms
                shift = 0.5
                return 1 / (1 + math.exp(10 * (shift - step * slope)))
            elif annealing_function == "linear":
                return max(1e-8, step * slope)  # Pyro won't accept zero
        return 1.0

    @property
    def kl_weight(self) -> float:
        kl_weight: float = getattr(self.hparams, "kl_weight")
        return self.annealing_factor * kl_weight

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return self.vae(batch)

    def configure_optimizers(self):
        lr = getattr(self.hparams, "lr")
        return optim.Adam(self.parameters(), lr=lr)

    def step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Define the logic in the training loop.

        Args:
            batch: Batch of input data

        Returns:
            Negative ELBO
        """
        output = self.vae.compute_loss(batch, self.kl_weight)
        if self.trainer is not None:
            for key, value in output.items():
                self.log(f"{self.trainer.state.stage}_{key}", value)
        # objetive: minimize negative ELBO
        return -output["elbo"]

    def training_step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.step(batch)

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self.step(batch)


class PyroTrainingLogic(TrainingLogic):
    def __init__(
        self,
        vae: VAE,
        kl_weight: float = 1.0,
        annealing_epochs: int = 20,
        annealing_function: AnnealingFunction = "linear",
        lr: float = 1e-4,
    ) -> None:
        """Encapsulate the training loop logic. This wrapper defines and uses
        a Pyro probabilistic model.

        Args:
            vae:
                A variational autoencoder instance
            kl_weight:
                Weight applied to the regularizing term
            annealing_epochs:
                Number of epochs, after which the full KL weight is applied. KL
                weight starts at 0 and is increased every training step until
                it reaches its full value. Set to 0 to disable KL warm-up.
            annealing_function:
                Monotonic function used to warm-up the KL term. Can be a
                linear, sigmoid, or stairstep schedule.
            lr:
                Learning rate of the Adam optimizer
        """
        super().__init__(vae, kl_weight, annealing_epochs, annealing_function, lr)
        self.optimizer = PyroOptim(optim.Adam, dict(lr=lr))
        self.svi = pyro.infer.SVI(
            self.model, self.guide, self.optimizer, pyro.infer.Trace_ELBO()
        )
        self.automatic_optimization = False  # handled by the SVI interface

    def model(self, x: torch.Tensor) -> None:
        """Define the generative model for the data, i.e., p(x|z). In this
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
        """Define the inference model (in the probabilistic aception), which
        serves as an approximation to the posterior q(z|x)."""
        batch_size = x.size(0)
        pyro.module("encoder", self.vae.encoder)
        with pyro.plate("data", batch_size), pyro.poutine.scale(scale=self.kl_weight):
            z_loc, z_scale = self.vae.encode(x)
            z_dist = pyro.distributions.Normal(z_loc, z_scale).to_event(1)
            pyro.sample("latent", z_dist)  # sample from posterior

    def configure_optimizers(self) -> None:
        return

    def step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, _ = batch
        negative_elbo = cast(float, self.svi.step(x)) / x.size(0)
        self.log("train_elbo", -negative_elbo)
        self.log("train_kl_weight", self.kl_weight)
        # objetive: minimize negative ELBO
        return torch.Tensor([negative_elbo])

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        pass

    def backward(self, *args: Any, **kwargs: Any) -> None:
        pass
