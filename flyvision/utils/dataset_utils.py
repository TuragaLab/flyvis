"""Dataset utilities."""

import numpy as np
from torch.hub import download_url_to_file
from torch.utils.data.sampler import Sampler

import flyvision


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


def random_walk_of_blocks(
    n_blocks=20,
    block_size=4,
    top_lum=0,
    bottom_lum=0,
    dataset_size=[3, 20, 64, 64],
    noise_mean=0.5,
    noise_std=0.1,
    step_size=4,
    p_random=0.6,
    p_center_attraction=0.3,
    p_edge_attraction=0.1,
    seed=42,
):
    """Generates a sequence dataset with blocks doing random walks.

    Args:
        n_blocks (int): Number of blocks.
        block_size (int): Size of blocks.
        top_lum (float): Luminance of the top of the block.
        bottom_lum (float): Luminance of the bottom of the block.
        dataset_size (list): Size of the dataset. (n_sequences, n_frames, h, w)
        noise_mean (float): Mean of the background noise.
        noise_std (float): Standard deviation of the background noise.
        step_size (int): Number of pixels to move in each step.
        p_random (float): Probability of moving randomly.
        p_center_attraction (float): Probability of moving towards the center.
        p_edge_attraction (float): Probability of moving towards the edge.
        seed (int): Seed for the random number generator.

    Returns:
        array: Dataset of shape (n_sequences, n_frames, h, w)
    """
    np.random.seed(seed)
    sequences = np.random.normal(loc=noise_mean, scale=noise_std, size=dataset_size)
    h, w = sequences.shape[2:]
    assert h == w

    y_coordinates = np.arange(h)
    x_coordinates = np.arange(w)

    def step(coordinate):
        ps = np.array([p_random, p_center_attraction, p_edge_attraction])
        ps /= ps.max()

        q = np.random.rand()
        if q < p_center_attraction:
            return (coordinate + np.sign(h // 2 - coordinate) * step_size) % h
        elif q > 1 - p_edge_attraction:
            return (coordinate + np.sign(coordinate - h // 2) * step_size) % h
        else:
            return (coordinate + np.random.choice([-1, 1]) * step_size) % h
        return coordinate

    def block_at_coords(y, x):
        mask_top = np.meshgrid(
            np.arange(y - block_size // 2, y) % h,
            np.arange(x - block_size // 2, x + block_size // 2) % w,
        )
        mask_bottom = np.meshgrid(
            np.arange(y, y + block_size // 2) % h,
            np.arange(x - block_size // 2, x + block_size // 2) % w,
        )
        return mask_bottom, mask_top

    def initial_block():
        initial_x = np.random.choice(x_coordinates)
        initial_y = np.random.choice(y_coordinates)
        return initial_x, initial_y, block_at_coords(initial_x, initial_y)

    for b in range(n_blocks):
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


def load_moving_mnist(delete_if_exists=False):
    """Return Moving MNIST dataset.

    Args:
        delete_if_exists (bool): If True, delete the dataset if it exists.

    Returns:
        array: Dataset of shape (n_sequences, n_frames, h, w)==(10000, 20, 64, 64).

    Note: this dataset (0.78GB) will be downloaded if not present. The download
        is stored in flyvision.root_dor / "mnist_test_seq.npy".
    """
    moving_mnist_path = flyvision.root_dir / "mnist_test_seq.npy"
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
