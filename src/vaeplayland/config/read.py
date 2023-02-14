__all__ = ["read_config"]

import hydra

from vaeplayland import config

HYDRA_VERSION_BASE = "1.2"


def read_config(*args, **kwargs):
    overrides = [*args] + [f"{key}={value}" for key, value in kwargs.items()]
    with hydra.initialize_config_module(
        config.__name__, version_base=HYDRA_VERSION_BASE
    ):
        return hydra.compose("main", overrides=overrides)
