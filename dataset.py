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


if __name__ == "__main__":
    dataset = TokenizedDataset(
        path_to_ids="/home/nicolvisser/workspace/ulm/data/train.pt",
        block_size=10,
        num_samples=100,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    for batch in dataloader:
        print(batch)
        break
