"""Dataset utilities."""
from torch.utils.data.sampler import Sampler


class IndexSampler(Sampler):
    """Samples the provided indices in sequence.

    Note, to be used with torch.utils.data.DataLoader.

    Args:
        indices: list of indices to sample.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
