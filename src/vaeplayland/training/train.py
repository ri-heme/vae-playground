__all__ = ["train"]

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger, WandbLogger
from torch.utils.data import DataLoader

from vaeplayland.config.logging import log_config


@hydra.main("../config", "main", version_base=None)
def train(config: DictConfig) -> None:
    """Train a model using the given configuration.

    Args:
        config: A dict config
    """
    train_dataloader: DataLoader = hydra.utils.instantiate(config.data.train_dataloader)
    model: pl.LightningModule = hydra.utils.instantiate(config.model)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer)
    log_config(config, trainer, model)
    trainer.fit(model, train_dataloader)
    if trainer.logger is not None:
        if isinstance(trainer.logger, WandbLogger):
            import wandb

            wandb.finish()
        elif isinstance(trainer.logger, NeptuneLogger):
            trainer.logger.experiment.stop()


if __name__ == "__main__":
    train()
