import torch
from torch.utils.data import Dataset

class LyricsDataset:
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, index):
        # TODO:
        # Get the input/output token sequences
        # Calculate x by grabbing the sublist of tokens starting at the index up to the block_size
        # Calculate y by grabbing the sublist of tokens starting at index + 1 up to block_size + 1
        x = self.tokens[index:index + self.block_size]

        y = self.tokens[index + 1:index + self.block_size + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
        pass