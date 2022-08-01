__all__ = ["TrainingLogic"]

import math
from typing import Dict, Tuple

import torch
from torch import nn, optim

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage

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
        epoch = self.trainer.current_epoch
        if (
            self.trainer is not None
            and self.trainer.state.stage == RunningStage.TRAINING
            and epoch < self.hparams.annealing_epochs
        ):
            if self.hparams.annealing_schedule == "stairs":
                return epoch / self.hparams.annealing_epochs
            num_batches = len(self.trainer.train_dataloader)
            step = self.trainer.global_step
            slope = 1 / (self.hparams.annealing_epochs * num_batches)
            if self.hparams.annealing_schedule == "sigmoid":
                # actual slope is 10 times slope of linear function
                # shift is half of the annealing epochs
                # equation below is factorized to avoid repeating terms
                shift = 0.5
                return 1 / (1 + math.exp(10 * (shift - step * slope)))
            elif self.hparams.annealing_schedule == "linear":
                return step * slope
        return 1.0

    @property
    def kl_weight(self) -> float:
        return self.hparams.kl_weight

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.vae(batch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def step(self, batch: torch.Tensor) -> torch.Tensor:
        output = compute_elbo(self.vae, batch, self.kl_weight, self.annealing_factor)
        for key, value in output.items():
            self.log(f"{self.trainer.state.stage}_{key}", value)
        # objetive: minimize negative ELBO
        return -output["elbo"]

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        return self.step(batch)

    def validation_step(self, batch: torch.Tensor) -> None:
        self.step(batch)
