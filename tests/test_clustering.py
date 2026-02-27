import numpy as np

from flyvis.analysis.clustering import umap_embedding


def test_umap_embedding_single_nonzero_variance_row():
    """Test that umap_embedding handles the edge case where only one row has
    nonzero variance (all others are constant). UMAP should not be fitted and
    the function should return NaN embedding with None reducer."""
    rng = np.random.default_rng(0)
    # One row with variance, four constant rows
    X = np.zeros((5, 10))
    X[2] = rng.random(10)

    embedding, mask, reducer = umap_embedding(X)

    assert reducer is None
    assert np.all(np.isnan(embedding))
    # Only the one non-constant row should be True in the mask
    expected_mask = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(mask, expected_mask)


def test_umap_embedding_all_zero_variance_rows():
    """Test that umap_embedding handles all-constant rows gracefully."""
    X = np.ones((5, 10))

    embedding, mask, reducer = umap_embedding(X)

    assert reducer is None
    assert np.all(np.isnan(embedding))
    assert not np.any(mask)


def test_umap_embedding_returns_none_reducer_when_insufficient_data():
    """Test that reducer is None when fewer than 2 rows have nonzero variance."""
    X = np.zeros((4, 8))
    # Only one non-constant row
    X[0] = np.arange(8, dtype=float)

    embedding, mask, reducer = umap_embedding(X)

    assert reducer is None
    assert embedding.shape == (4, 2)
    assert np.all(np.isnan(embedding))
