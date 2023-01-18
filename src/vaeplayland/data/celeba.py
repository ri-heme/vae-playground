__all__ = ["get_dataloader"]

from pathlib import Path
from typing import Literal, Optional

from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CelebA

IMAGE_SIZE = 64
NORMALIZE_PARAMS = ((0.5,) * 3,) * 2


def get_dataloader(
    root: Optional[Path] = None,
    split: Literal["train", "valid", "test", "all"] = "train",
    **dataloader_kwargs
) -> DataLoader:
    """Return a dataloader for the CelebA dataset. The images are resized,
    cropped, and normalized. Each image in the dataset has three channels and
    is 64 x 64 px. Additionally, each image is accompanied with 40 labels,
    corresponding to binary attributes such as mustache and 5 o'clock shadow.

    Args:
        root: Path to dataset's location
        split: Split of dataset to load

    Raises:
        FileNotFoundError:
            If the CelebA dataset has not been previously downloaded

    Returns:
        A dataloader constructed with the specified keyword arguments and the
        CelebA dataset split
    """
    if root is None:
        root = Path.home() / ".pytorch"
    elif not isinstance(root, Path):
        root = Path(root)
    if not root.joinpath("celeba").exists():
        raise FileNotFoundError("CelebA has not been downloaded.")
    mean, std = NORMALIZE_PARAMS
    transform = T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    dataset = CelebA(str(root), split, transform=transform, download=False)
    return DataLoader(dataset, **dataloader_kwargs)
