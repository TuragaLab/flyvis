import numpy as np
import pytest

from flyvis import results_dir
from flyvis.analysis import response_norms
from flyvis.network.ensemble import Ensemble

pytestmark = pytest.mark.require_download


@pytest.fixture(scope="module")
def ensemble() -> Ensemble:
    models = [results_dir / f"flow/0000/{i:03}" for i in range(4)]
    return Ensemble(
        models,
        best_checkpoint_fn_kwargs={
            "validation_subdir": "validation",
            "loss_file_name": "loss",
        },
    )


def test_precomputed_norms_are_shipped():
    norms = response_norms.read_response_norms(
        response_norms.PRECOMPUTED_FILE, "flow/0000"
    )
    assert norms is not None, f"{response_norms.PRECOMPUTED_FILE} misses flow/0000"
    assert len(norms.model_names) == 50
    assert len(norms.cell_types) == 65
    assert norms.norm.shape == (50, 65)
    assert norms.rectified_norm.shape == (50, 65)
    assert np.all(norms.norm > 0)
    # cell types that a model rectifies away entirely have a zero rectified norm
    assert np.all(norms.rectified_norm >= 0)
    assert np.isfinite(norms.norm).all()
    assert np.isfinite(norms.rectified_norm).all()


def test_responses_norm_is_loaded_not_computed(ensemble, monkeypatch):
    """The whole point: no simulation for the released ensemble."""

    def fail(*args, **kwargs):
        raise AssertionError("normalization constants were recomputed")

    monkeypatch.setattr(response_norms, "compute_response_norms", fail)

    norm = ensemble.responses_norm()
    rectified = ensemble.responses_norm(rectified=True)
    assert norm.shape == (len(ensemble), 1, 1, 65)
    assert rectified.shape == norm.shape
    # rectified responses are a subset of the unrectified ones
    assert np.all(rectified <= norm + 1e-6)


def test_responses_norm_follows_model_order(ensemble):
    norm = ensemble.responses_norm()
    reversed_ensemble = ensemble[::-1]
    assert reversed_ensemble.names == ensemble.names[::-1]
    np.testing.assert_allclose(reversed_ensemble.responses_norm(), norm[::-1])


def test_responses_norm_of_unsorted_ensemble_matches_by_name():
    """Rows must follow the ensemble's own order, whatever that order is."""
    stored = response_norms.read_response_norms(
        response_norms.PRECOMPUTED_FILE, "flow/0000"
    )
    by_name = dict(zip(stored.model_names, stored.rectified_norm))

    shuffled = [f"flow/0000/{i:03}" for i in (7, 0, 42, 13, 2)]
    unsorted_ensemble = Ensemble(
        [results_dir / name for name in shuffled],
        best_checkpoint_fn_kwargs={
            "validation_subdir": "validation",
            "loss_file_name": "loss",
        },
    )
    assert unsorted_ensemble.names == shuffled

    norm = unsorted_ensemble.responses_norm(rectified=True)
    for i, name in enumerate(unsorted_ensemble.names):
        np.testing.assert_array_equal(norm[i, 0, 0], by_name[name])


def test_store_and_read_roundtrip(ensemble, tmp_path):
    norms = response_norms.read_response_norms(
        response_norms.PRECOMPUTED_FILE, ensemble.name
    )
    path = tmp_path / "responses_norm.h5"
    response_norms.write_response_norms(path, norms)
    read = response_norms.read_response_norms(path, ensemble.name)
    assert read.model_names == norms.model_names
    assert read.cell_types == norms.cell_types
    assert read.checkpoints == norms.checkpoints
    assert read.checkpoint_hashes == norms.checkpoint_hashes
    np.testing.assert_array_equal(read.norm, norms.norm)
    np.testing.assert_array_equal(read.rectified_norm, norms.rectified_norm)


def test_unknown_ensemble_returns_none(tmp_path):
    missing = response_norms.read_response_norms(tmp_path / "missing.h5", "flow/0000")
    assert missing is None
    assert (
        response_norms.read_response_norms(response_norms.PRECOMPUTED_FILE, "flow/9999")
        is None
    )


def test_covers_rejects_other_checkpoints():
    norms = response_norms.read_response_norms(
        response_norms.PRECOMPUTED_FILE, "flow/0000"
    )
    hashes = norms.checkpoint_hashes
    assert norms.covers(norms.model_names, norms.checkpoints, hashes)
    assert not norms.covers(norms.model_names, ["chkpt_99999"] * len(norms))
    assert not norms.covers(["flow/0000/999"], [norms.checkpoints[0]], [hashes[0]])


def test_covers_is_order_independent():
    norms = response_norms.read_response_norms(
        response_norms.PRECOMPUTED_FILE, "flow/0000"
    )
    order = list(range(len(norms)))[::-1]
    assert norms.covers(
        [norms.model_names[i] for i in order],
        [norms.checkpoints[i] for i in order],
        [norms.checkpoint_hashes[i] for i in order],
    )
    # a name paired with another model's checkpoint hash must be rejected
    shifted = norms.checkpoint_hashes[1:] + norms.checkpoint_hashes[:1]
    assert not norms.covers(norms.model_names, norms.checkpoints, shifted)


def test_precomputed_norms_carry_checkpoint_hashes():
    norms = response_norms.read_response_norms(
        response_norms.PRECOMPUTED_FILE, "flow/0000"
    )
    assert len(norms.checkpoint_hashes) == len(norms)
    # the checkpoint file name is the same string for every model, the hash is not
    assert len(set(norms.checkpoints)) == 1
    assert len(set(norms.checkpoint_hashes)) == len(norms)


def test_retrained_checkpoint_is_rejected(ensemble, monkeypatch):
    """A checkpoint modified in place keeps its file name but not its hash."""
    monkeypatch.setattr(response_norms, "checkpoint_hash", lambda path: "0" * 64)
    assert response_norms.load_response_norms(ensemble) is None


def test_falls_back_to_names_without_hashes(ensemble, tmp_path):
    """Constants written before hashes were recorded still load."""
    norms = response_norms.read_response_norms(
        response_norms.PRECOMPUTED_FILE, ensemble.name
    )
    norms.checkpoint_hashes = []
    path = tmp_path / "legacy.h5"
    response_norms.write_response_norms(path, norms)
    legacy = response_norms.read_response_norms(path, ensemble.name)
    assert legacy.checkpoint_hashes == []

    checkpoints, hashes = response_norms.checkpoint_identity(ensemble)
    assert legacy.covers(list(ensemble.names), checkpoints, hashes)
    wrong = ["chkpt_99999"] * len(ensemble)
    assert not legacy.covers(list(ensemble.names), wrong, hashes)
