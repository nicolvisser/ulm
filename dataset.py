import torch
import random
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    def __init__(
        self,
        path_to_ids: str,
        block_size,
        num_samples=10000,
    ):
        super().__init__()

        self.path_to_ids = path_to_ids
        self.block_size = block_size + 1  # +1 for the target
        self.num_samples = num_samples

        self.ids = torch.load(path_to_ids)
        self.sample_numbers = [i for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        _ = self.sample_numbers[idx]  # ensures the iterator ends after num_samples
        random_start = random.randint(0, len(self.ids) - self.block_size)
        sequence = self.ids[random_start : random_start + self.block_size]
        return sequence
