__all__ = ["log_config"]

from argparse import Namespace
from typing import Any, Generator, MutableMapping, Optional, cast

import pytorch_lightning as pl
from omegaconf import DictConfig


def flatten_config(
    config: MutableMapping,
    ignore: Optional[List[str]] = None,
    prefixes: Optional[str] = None,
) -> Generator[Tuple[str, Any], None, None]:
    """Flatten a hierarchical config.

    Args:
        config: A dict config
        ignore: List of keys to ignore
        prefixes: Prefix to prepend to each key

    Yields:
        A tuple of flattened key and its corresponding value

    Example:
        >>> k, v = next(flatten_config({"a": {"b": "c"}})); print(f"{k}={v}")
        a/b
    """
    if prefixes is None:
        prefixes = ""
    for key, value in config.items():
        if key in ignore:
            continue
        if isinstance(value, MutableMapping):
            yield from flatten_config(value, ignore, f"{prefixes}{key}/")
        else:
            yield prefixes + key, value


def log_config(
    config: DictConfig, trainer: pl.Trainer, model: pl.LightningModule
) -> None:
    """Record Hydra config.

    Args:
        config: A dict config
        trainer: A PyTorch Lightning trainer instance
        model: A Pytorch Lightning module instance
    """
    assert trainer.logger is not None
    ignore = ["logger", "valid_dataloader", "test_dataloader"]
    hparams = dict(flatten_config(config, ignore=ignore))
    trainable_params, non_trainable_params = [], []
    for param in model.parameters():
        (trainable_params if param.requires_grad else non_trainable_params).append(
            param.numel()
        )
    hparams["model/params/total"] = sum(trainable_params + non_trainable_params)
    hparams["model/params/trainable"] = sum(trainable_params)
    hparams["model/params/non_trainable"] = sum(non_trainable_params)
    trainer.logger.log_hyperparams(cast(Namespace, hparams))
