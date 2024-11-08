"""Dataset utilities."""

from typing import List, Tuple

import numpy as np
from numpy.random import RandomState
from torch.hub import download_url_to_file
from torch.utils.data.sampler import Sampler

import flyvis


def random_walk_of_blocks(
    n_blocks: int = 20,
    block_size: int = 4,
    top_lum: float = 0,
    bottom_lum: float = 0,
    dataset_size: List[int] = [3, 20, 64, 64],
    noise_mean: float = 0.5,
    noise_std: float = 0.1,
    step_size: int = 4,
    p_random: float = 0.6,
    p_center_attraction: float = 0.3,
    p_edge_attraction: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a sequence dataset with blocks doing random walks.

    Args:
        n_blocks: Number of blocks.
        block_size: Size of blocks.
        top_lum: Luminance of the top of the block.
        bottom_lum: Luminance of the bottom of the block.
        dataset_size: Size of the dataset. (n_sequences, n_frames, h, w)
        noise_mean: Mean of the background noise.
        noise_std: Standard deviation of the background noise.
        step_size: Number of pixels to move in each step.
        p_random: Probability of moving randomly.
        p_center_attraction: Probability of moving towards the center.
        p_edge_attraction: Probability of moving towards the edge.
        seed: Seed for the random number generator.

    Returns:
        Dataset of shape (n_sequences, n_frames, h, w)
    """
    np.random.seed(seed)
    sequences = np.random.normal(loc=noise_mean, scale=noise_std, size=dataset_size)
    h, w = sequences.shape[2:]
    assert h == w

    y_coordinates = np.arange(h)
    x_coordinates = np.arange(w)

    def step(coordinate: int) -> int:
        ps = np.array([p_random, p_center_attraction, p_edge_attraction])
        ps /= ps.max()

        q = np.random.rand()
        if q < p_center_attraction:
            return (coordinate + np.sign(h // 2 - coordinate) * step_size) % h
        elif q > 1 - p_edge_attraction:
            return (coordinate + np.sign(coordinate - h // 2) * step_size) % h
        else:
            return (coordinate + np.random.choice([-1, 1]) * step_size) % h

    def block_at_coords(
        y: int, x: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        mask_top = np.meshgrid(
            np.arange(y - block_size // 2, y) % h,
            np.arange(x - block_size // 2, x + block_size // 2) % w,
        )
        mask_bottom = np.meshgrid(
            np.arange(y, y + block_size // 2) % h,
            np.arange(x - block_size // 2, x + block_size // 2) % w,
        )
        return mask_bottom, mask_top

    def initial_block() -> (
        Tuple[
            int, int, Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ]
    ):
        initial_x = np.random.choice(x_coordinates)
        initial_y = np.random.choice(y_coordinates)
        return initial_x, initial_y, block_at_coords(initial_x, initial_y)

    for _b in range(n_blocks):
        for i in range(sequences.shape[0]):
            for t in range(sequences.shape[1]):
                if t == 0:
                    x, y, (mask_bottom, mask_top) = initial_block()
                else:
                    x = step(x)
                    y = step(y)
                    mask_bottom, mask_top = block_at_coords(x, y)
                sequences[i, t, mask_bottom[0], mask_bottom[1]] = bottom_lum
                sequences[i, t, mask_top[0], mask_top[1]] = top_lum

    return sequences / sequences.max()


def load_moving_mnist(delete_if_exists: bool = False) -> np.ndarray:
    """Return Moving MNIST dataset.

    Args:
        delete_if_exists: If True, delete the dataset if it exists.

    Returns:
        Dataset of shape (n_sequences, n_frames, h, w)==(10000, 20, 64, 64).

    Note:
        This dataset (0.78GB) will be downloaded if not present. The download
        is stored in flyvis.root_dir / "mnist_test_seq.npy".
    """
    moving_mnist_path = flyvis.root_dir / "mnist_test_seq.npy"
    moving_mnist_url = (
        "https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    )

    if not moving_mnist_path.exists() or delete_if_exists:
        download_url_to_file(moving_mnist_url, moving_mnist_path)
    try:
        sequences = np.load(moving_mnist_path)
        return np.transpose(sequences, (1, 0, 2, 3)) / 255.0
    except ValueError as e:
        # delete broken download and load again
        print(f"broken file: {e}, restarting download...")
        return load_moving_mnist(delete_if_exists=True)


class CrossValIndices:
    """Returns folds of indices for cross-validation.

    Args:
        n_samples: Total number of samples.
        folds: Total number of folds.
        shuffle: Shuffles the indices.
        seed: Seed for shuffling.

    Attributes:
        n_samples: Total number of samples.
        folds: Total number of folds.
        indices: Array of indices.
        random: RandomState object for shuffling.

    """

    def __init__(self, n_samples: int, folds: int, shuffle: bool = True, seed: int = 0):
        self.n_samples = n_samples
        self.folds = folds
        self.indices = np.arange(n_samples)

        if shuffle:
            self.random = RandomState(seed)
            self.random.shuffle(self.indices)

    def __call__(self, fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns train and test indices for a fold.

        Args:
            fold: The fold number.

        Returns:
            A tuple containing train and test indices.
        """
        fold_sizes = np.full(self.folds, self.n_samples // self.folds, dtype=int)
        fold_sizes[: self.n_samples % self.folds] += 1
        current = sum(fold_sizes[:fold])
        start, stop = current, current + fold_sizes[fold]
        test_index = self.indices[start:stop]
        test_mask = np.zeros_like(self.indices, dtype=bool)
        test_mask[test_index] = True
        return self.indices[np.logical_not(test_mask)], self.indices[test_mask]

    def iter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Iterate over all folds.

        Yields:
            A tuple containing train and test indices for each fold.
        """
        for fold in range(self.folds):
            yield self(fold)


def get_random_data_split(
    fold: int, n_samples: int, n_folds: int, shuffle: bool = True, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices to split the data.

    Args:
        fold: The fold number.
        n_samples: Total number of samples.
        n_folds: Total number of folds.
        shuffle: Whether to shuffle the indices.
        seed: Seed for shuffling.

    Returns:
        A tuple containing train and validation indices.
    """
    cv_split = CrossValIndices(
        n_samples=n_samples,
        folds=n_folds,
        shuffle=shuffle,
        seed=seed,
    )
    train_seq_index, val_seq_index = cv_split(fold)
    return train_seq_index, val_seq_index


class IndexSampler(Sampler):
    """Samples the provided indices in sequence.

    Note:
        To be used with torch.utils.data.DataLoader.

    Args:
        indices: List of indices to sample.

    Attributes:
        indices: List of indices to sample.
    """

    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self) -> int:
        return len(self.indices)
