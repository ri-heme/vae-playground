from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloader(source_dir: str, split: str = "train", **dataloader_args):
    # Reload data
    source_path = Path(source_dir)
    cat1 = torch.load(source_path / f"cat1_{split}.pt")
    con = torch.load(source_path / f"con_{split}.pt")
    # Get labels
    lab1 = torch.argmax(cat1, dim=1)
    # Prep dataloader
    data = torch.cat((cat1, con), dim=1)
    dataset = TensorDataset(data, lab1)
    return DataLoader(dataset, **dataloader_args)
