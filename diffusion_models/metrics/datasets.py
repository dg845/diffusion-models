"""Implements a PyTorch Dataset which outputs noise samples for diffusion model sampling."""

from typing import Sequence

import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    """A PyTorch Dataset created from a list of diffusion model samples, for
    use in calculating metrics such as the Inception score."""
    def __init__(self, samples):
        self.samples = samples
        self.length = len(self.samples)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.samples[idx]


class GaussianNoiseDataset(Dataset):
    """Implements a simple PyTorch Dataset whose __getitem__ method just
    samples random Gaussian noise of the appropriate size for a diffusion
    model."""
    def __init__(
        self,
        shape: Sequence[int]=(3, 32, 32),
        data_len: int=50000,
    ) -> None:
        self.shape = shape
        self.data_len = data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        noise = torch.randn(self.shape)
        # No label
        return noise

