__all__ = []

import math
from typing import Literal, Sized, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import RunningStage
from torch import optim

from vaeplayland.models.loss import ELBODict

from vaeplayland.models.experimental.multimodal_vae_t3 import MultimodalVAEv3

AnnealingFunction = Literal["linear", "sigmoid", "stairs"]


class TrainingLogicV2(pl.LightningModule):
    def __init__(
        self,
        vae: MultimodalVAEv3,
        kl_weight_encoder: float = 1.0,
        kl_weight_decoder: float = 1.0,
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
    def kl_weight(self) -> tuple[float, float]:
        kl_weight_encoder: float = getattr(self.hparams, "kl_weight_encoder")
        kl_weight_decoder: float = getattr(self.hparams, "kl_weight_decoder")
        return (
            self.annealing_factor * kl_weight_encoder,
            self.annealing_factor * kl_weight_decoder,
        )

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return self.vae(batch)

    def configure_optimizers(self):
        lr = getattr(self.hparams, "lr")
        return optim.Adam(self.parameters(), lr=lr)

    def step(self, batch: tuple[torch.Tensor, ...]) -> ELBODict:
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
        return output

    def training_step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        output = self.step(batch)
        return -output["elbo"]

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self.step(batch)

    def test_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> ELBODict:
        return self.step(batch)
