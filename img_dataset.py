
import torch
from torch.utils.data import Dataset

from typing import NamedTuple, List
from util import CircleParams

class ImageData(NamedTuple):
    img: torch.Tensor
    params: CircleParams

class ImageDataset(Dataset):
    def __init__(self, data: List[ImageData]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        image, params = self.data[idx]
        return image, params
