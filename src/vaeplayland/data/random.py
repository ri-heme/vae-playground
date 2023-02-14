__all__ = ["get_dataloader"]

from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


def make_data(
    num_samples: int,
    num_features: int,
    num_associations: int,
    seed: int,
) -> torch.Tensor:
    if num_associations > num_features:
        raise ValueError("Number of associations cannot exceed number of features.")

    rng = np.random.RandomState(seed)

    loc = rng.randint(5, 10, size=num_features)
    con_data = rng.normal(loc=loc, scale=1, size=(num_samples, num_features))

    ids = rng.permutation(num_features)[:num_associations]

    mean = con_data[:, ids].mean(axis=0)
    std = con_data[:, ids].std(axis=0)

    effect = mean - rng.normal(loc=mean, scale=2 * std)

    con_data[num_samples // 2 :, ids] += effect

    return torch.FloatTensor(con_data)


def get_dataloader(
    num_samples: int = 300,
    num_features: int = 15,
    num_associations: int = 5,
    seed: int = 123,
    split: Literal["train", "test"] = "train",
    test_frac: float = 0.2,
    **dataloader_kwargs
) -> DataLoader:
    con_data = make_data(num_samples, num_features, num_associations, seed)

    train_ids = [
        *range(int(num_samples // 2 * (1 - test_frac))),
        *range(num_samples // 2, int(num_samples // 2 * (2 - test_frac))),
    ]
    test_ids = np.setdiff1d(range(num_samples), train_ids)

    mean = con_data[train_ids, :].mean(dim=0)
    scale = con_data[train_ids, :].std(dim=0)

    con_data = (con_data - mean) / scale

    cat_data = torch.zeros((num_samples, 2))
    cat_data[: (num_samples // 2), 0] = 1.0
    cat_data[(num_samples // 2) :, 1] = 1.0
    labels = cat_data.argmax(dim=-1)

    data = torch.cat((cat_data, con_data), dim=1)

    dataset = TensorDataset(data, labels)
    if split == "train":
        return DataLoader(Subset(dataset, train_ids), **dataloader_kwargs)
    elif split == "test":
        return DataLoader(Subset(dataset, test_ids), **dataloader_kwargs)

    raise ValueError("Unsupported split")
