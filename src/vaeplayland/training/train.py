__all__ = ["train"]

import hydra
from omegaconf import DictConfig

from torch.utils.data import DataLoader

import pytorch_lightning as pl
import wandb

from vaeplayland.config.logging import log_config


@hydra.main("../config", "main")
def train(config: DictConfig) -> None:
    """Train a model using the given configuration.

    Parameters
    ----------
    config : DictConfig
    """
    train_dataloader: DataLoader = hydra.utils.instantiate(config.data.train_dataloader)
    model: pl.LightningModule = hydra.utils.instantiate(config.model)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer)
    log_config(config, trainer, model)
    trainer.fit(model, train_dataloader)
    trainer.save_checkpoint(trainer.log_dir / "best.ckpt")
    wandb.finish()
    
if __name__ == "__main__":
    train()
