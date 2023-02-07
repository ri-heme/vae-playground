from typing import Any, Literal

import torch
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, Subset, TensorDataset


def get_dataloader(
    split: Literal["train", "test"] = "train",
    seed: int = 42,
    test_frac: float = 0.2,
    **dataloader_kwargs
) -> DataLoader:
    """Load Iris dataset and return a training and validation dataloaders.

    Labels are one-hot encoded, and continuous values are standardized. Both
    data modalities are concatenated.
    """
    # Load
    data = load_iris()
    features = torch.FloatTensor(getattr(data, "data"))
    labels = torch.FloatTensor(getattr(data, "target"))
    num_samples = labels.size(0)
    num_categories = len(getattr(data, "target_names"))

    # One-hot encode labels
    ohe_data = torch.zeros((num_samples, num_categories))
    ohe_data[torch.arange(num_samples), labels.long()] = 1

    # Split train/test
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randperm(num_samples, generator=generator)
    train_size = int(num_samples * (1 - test_frac))
    train_ids, test_ids = [
        torch.sort(t).values for t in torch.tensor_split(ids, [train_size])
    ]

    # Standardize
    mean = features[train_ids, :].mean(dim=0)
    scale = features[train_ids, :].std(dim=0)

    features = (features - mean) / scale
    cat_data = torch.cat((ohe_data, features), dim=1)

    dataset = TensorDataset(cat_data, labels)
    if split == "train":
        return DataLoader(Subset(dataset, train_ids), **dataloader_kwargs)
    elif split == "test":
        return DataLoader(Subset(dataset, test_ids), **dataloader_kwargs)

    raise ValueError("Unsupported split")
