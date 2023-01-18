__all__ = ["train"]

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
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
    wandb.finish()


if __name__ == "__main__":
    train()
