__all__ = ["TrainingLogic", "PyroTrainingLogic"]

import math
from typing import Any, Literal, Tuple, cast

import pyro
import pyro.distributions
import pyro.infer
import pyro.optim
import pytorch_lightning as pl
import torch
from pyro.optim import PyroOptim
from pytorch_lightning.trainer.states import RunningStage
from torch import nn, optim

from vaeplayland.models.loss import ELBODict
from vaeplayland.models.vae import VAE

AnnealingFunction = Literal["linear", "cosine", "sigmoid", "stairs"]
AnnealingSchedule = Literal["monotonic", "cyclical"]


class TrainingLogic(pl.LightningModule):
    def __init__(
        self,
        vae: VAE,
        kl_weight: float = 1.0,
        annealing_epochs: int = 20,
        annealing_function: AnnealingFunction = "linear",
        annealing_schedule: AnnealingSchedule = "monotonic",
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
            annealing_schedule:
                Whether the KL term warm-up will be cyclical or monotonic.
            lr:
                Learning rate of the Adam optimizer
        """
        super().__init__()
        self.vae = vae
        self.save_hyperparameters(ignore="vae")
        self.fixed_kl_weight = False

    def annealing_epochs(self) -> int:
        return getattr(self.hparams, "annealing_epochs")

    def annealing_function(self) -> AnnealingFunction:
        return getattr(self.hparams, "annealing_function")

    def annealing_schedule(self) -> AnnealingSchedule:
        return getattr(self.hparams, "annealing_schedule")

    def annealing_factor(self) -> float:
        if (
            self.trainer is not None
            and self.trainer.state.stage == RunningStage.TRAINING
        ):
            epoch = self.current_epoch
            if (
                self.annealing_schedule == "monotonic" and epoch < self.annealing_epochs()
            ) or (self.annealing_schedule == "cyclical"):
                if self.annealing_function == "stairs":
                    num_epochs_cyc = self.trainer.max_epochs / self.num_cycles()
                    # location in cycle: 0 (start) - 1 (end)
                    loc = (epoch % math.ceil(num_epochs_cyc)) / num_epochs_cyc
                    # first half of the cycle, KL weight is warmed up
                    # second half, it is fixed
                    if loc <= 0.5:
                        return loc * 2
                else:
                    max_steps = self.trainer.estimated_stepping_batches
                    num_steps_cyc = max_steps / self.num_cycles()
                    step = self.global_step
                    loc = (step % math.ceil(num_steps_cyc)) / num_steps_cyc
                    if loc < 0.5:
                        if self.annealing_function == "linear":
                            return loc * 2
                        elif self.annealing_function == "sigmoid":
                            # ensure it reaches 0.5 at 1/4 of the cycle
                            shift = 0.25
                            slope = self.annealing_epochs()
                            return 1 / (1 + math.exp(slope * (shift - loc)))
                        elif self.annealing_function == "cosine":
                            return math.cos((loc - 0.5) * math.pi)
        return 1.0

    def kl_weight(self) -> float:
        kl_weight: float = getattr(self.hparams, "kl_weight")
        return self.annealing_factor() * kl_weight

    def num_cycles(self) -> float:
        if self.trainer is None:
            return float("-inf")
        return self.trainer.max_epochs / (self.annealing_epochs() * 2)

    def forward(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self.vae(batch)

    def configure_optimizers(self):
        lr = getattr(self.hparams, "lr")
        return optim.Adam(self.parameters(), lr=lr)

    def step(self, batch: Tuple[torch.Tensor, ...]) -> ELBODict:
        """Define the logic in the training loop.

        Args:
            batch: Batch of input data

        Returns:
            Negative ELBO
        """
        output = self.vae.compute_loss(batch, self.kl_weight())
        if self.trainer is not None:
            for key, value in output.items():
                self.log(f"{self.trainer.state.stage}_{key}", cast(torch.Tensor, value))
        # objetive: minimize negative ELBO
        return output

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        output = self.step(batch)
        return -output["elbo"]

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self.step(batch)

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> ELBODict:
        return self.step(batch)


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
        super().__init__(
            vae, kl_weight, annealing_epochs, annealing_function, "monotonic", lr
        )
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

    def step(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, _ = batch
        negative_elbo = cast(float, self.svi.step(x)) / x.size(0)
        self.log("train_elbo", -negative_elbo)
        self.log("train_kl_weight", self.kl_weight())
        # objetive: minimize negative ELBO
        return torch.Tensor([negative_elbo])

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        pass

    def backward(self, *args: Any, **kwargs: Any) -> None:
        pass
