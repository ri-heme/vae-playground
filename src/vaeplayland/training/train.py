__all__ = ["train"]

from typing import Optional, Tuple, cast

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger, WandbLogger
from torch.utils.data import DataLoader

from vaeplayland.config.logging import log_config
from vaeplayland.models.loss import ELBODict


@hydra.main("../config", "main", version_base=None)
def train(config: DictConfig) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Train a model using the given configuration.

    Args:
        config: A dict config
    """
    train_dataloader: DataLoader = hydra.utils.instantiate(config.data.train_dataloader)
    if hasattr(config.data, "valid_dataloader"):
        valid_dataloader = hydra.utils.instantiate(config.data.valid_dataloader)
    else:
        valid_dataloader = None
    model: pl.LightningModule = hydra.utils.instantiate(config.model)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer)
    log_config(config, trainer, model)
    trainer.fit(model, train_dataloader, valid_dataloader)
    if hasattr(config.data, "test_dataloader"):
        test_dataloader = hydra.utils.instantiate(config.data.test_dataloader)
        output, *_ = trainer.test(model, test_dataloader)
        test_losses = cast(
            Tuple[torch.Tensor, torch.Tensor],
            (output["test_reg_loss"], output["test_rec_loss"]),
        )
    else:
        test_losses = None
    if trainer.logger is not None:
        if isinstance(trainer.logger, WandbLogger):
            import wandb

            wandb.finish()
        elif isinstance(trainer.logger, NeptuneLogger):
            trainer.logger.experiment.stop()
    return test_losses


if __name__ == "__main__":
    train()
