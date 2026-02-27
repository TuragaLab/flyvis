from types import SimpleNamespace

import numpy as np

from flyvis.network import ensemble_view as ensemble_view_module
from flyvis.network.ensemble_view import EnsembleView


class _FakeFRI:
    def __init__(self, values, cell_types):
        self.values = values
        self.cell_type = SimpleNamespace(values=np.array(cell_types))
        self.custom = SimpleNamespace(where=self._where)

    def _where(self, cell_type):
        filtered = sorted(cell_type)
        return _FakeFRI(self.values, filtered)


class _DummyView:
    def flash_responses(self):
        return "responses"

    def task_error(self):
        return SimpleNamespace(values=np.array([0.5, 0.1, 0.2]))


def test_flash_response_index_uses_filtered_cell_type_order(monkeypatch):
    requested = ["Mi1", "Tm3", "CT1(M10)"]
    filtered = ["CT1(M10)", "Mi1", "Tm3"]
    fake_fris = _FakeFRI(np.ones((3, 2, 1)), filtered)

    monkeypatch.setattr(
        ensemble_view_module, "flash_response_index", lambda *_args, **_kwargs: fake_fris
    )

    captured = {}

    def _fake_plot_fris(fris, cell_types, **kwargs):
        captured["fris"] = fris
        captured["cell_types"] = list(cell_types)
        captured["sorted_type_list"] = kwargs.get("sorted_type_list")
        return "fig", "ax"

    monkeypatch.setattr(ensemble_view_module, "plot_fris", _fake_plot_fris)

    fig, ax = EnsembleView.flash_response_index(_DummyView(), cell_types=requested)

    assert (fig, ax) == ("fig", "ax")
    assert captured["fris"].shape == (3, 2, 1)
    assert captured["cell_types"] == filtered
    assert captured["sorted_type_list"] == requested
