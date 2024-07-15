from typing import Iterable
from itertools import product
import logging

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datamate import Directory, Namespace, root

from flyvision.rendering import BoxEye
from flyvision.rendering.utils import resample
from flyvision.datasets.datasets import SequenceDataset
from flyvision import utils, root_dir
from flyvision.utils.hex_utils import HexLattice, Hexal

logging = logging.getLogger()


@root(root_dir)
class RenderedFlashes(Directory):
    """Render a directory with flashes for the Flashes dataset.

    Args:
        boxfilter: parameters for the BoxEye filter.
        dynamic_range: range of intensities. E.g. [0, 1] renders flashes
            with decrement 0.5->0 and increment 0.5->1.
        t_stim: duration of the stimulus.
        t_pre: duration of the grey stimulus.
        dt: timesteps.
        radius: radius of the stimulus.
        alternations: sequence of alternations between lower or upper intensity and
            baseline of the dynamic range.
    """

    def __init__(
        self,
        boxfilter: dict = dict(extent=15, kernel_size=13),
        dynamic_range: list = [0, 1],
        t_stim=1.0,
        t_pre=1.0,
        dt=1 / 200,
        radius: list = [-1, 6],
        alternations=(0, 1, 0),
    ):
        boxfilter = BoxEye(**boxfilter)
        n_ommatidia = len(boxfilter.receptor_centers)
        dynamic_range = np.array(dynamic_range)
        baseline = 2 * (dynamic_range.sum() / 2,)

        intensity = dynamic_range.copy()
        values = np.array(list(zip(baseline, intensity)))
        samples = dict(v=values, r=radius)
        values = list(product(*(v for v in samples.values())))
        sequence = []  # samples, #frames, width, height
        for (baseline, intnsty), rad in tqdm(values, desc="Flashes"):
            sequence.append(
                get_flash(
                    n_ommatidia,
                    intnsty,
                    baseline,
                    t_stim,
                    t_pre,
                    dt,
                    alternations,
                    rad,
                )
            )

        self.flashes = np.array(sequence)


def get_flash(
    n_ommatidia, intensity, baseline, t_stim, t_pre, dt, alternations, radius
):
    """Generate a sequence of flashes on a hexagonal lattice.

    Args:
        n_ommatidia (int): number of ommatidia.
        intensity (float): intensity of the flash.
        baseline (float): intensity of the baseline.
        t_stim (float): duration of the stimulus.
        t_pre (float): duration of the grey stimulus.
        dt (float): timesteps.
        alternations (list): sequence of alternations between lower or upper intensity and
            baseline of the dynamic range.
        radius (int): radius of the stimulus.
    """
    stimulus = torch.ones(n_ommatidia)[None] * baseline

    if radius != -1:
        ring = HexLattice.filled_circle(
            radius=radius, center=Hexal(0, 0, 0), as_lattice=True
        )
        coordinate_index = ring.where(1)
    else:
        coordinate_index = np.arange(n_ommatidia)

    stimulus[:, coordinate_index] = intensity

    on = resample(stimulus, t_stim, dt)
    off = resample(torch.ones(n_ommatidia)[None] * baseline, t_pre, dt)

    whole_stimulus = []
    for switch in alternations:
        if switch == 0:
            whole_stimulus.append(off)
        elif switch == 1:
            whole_stimulus.append(on)
    return torch.cat(whole_stimulus, dim=0).cpu().numpy()


class Flashes(SequenceDataset):
    """Flashes dataset.

    Args:
        boxfilter: parameters for the BoxEye filter.
        dynamic_range: range of intensities. E.g. [0, 1] renders flashes
            with decrement 0.5->0 and increment 0.5->1.
        t_stim: duration of the stimulus.
        t_pre: duration of the grey stimulus.
        dt: timesteps.
        radius: radius of the stimulus.
        alternations: sequence of alternations between lower or upper intensity and
            baseline of the dynamic range.
    """

    augment = False
    n_sequences = 0
    dt = None
    framerate = None
    t_post = 0.0

    def __init__(
        self,
        boxfilter=dict(extent=15, kernel_size=13),
        dynamic_range=[0, 1],
        t_stim=1.0,
        t_pre=1.0,
        dt=1 / 200,
        radius=[-1, 6],
        alternations=(0, 1, 0),
    ):
        self.flashes_dir = RenderedFlashes(
            boxfilter=boxfilter,
            dynamic_range=dynamic_range,
            t_stim=t_stim,
            t_pre=t_pre,
            dt=dt,
            radius=radius,
            alternations=alternations,
        )
        self.config = self.flashes_dir.config
        baseline = 2 * (sum(dynamic_range) / 2,)
        intensity = dynamic_range.copy()

        params = [
            (p[0][0], p[0][1], p[1])
            for p in list(product(zip(baseline, intensity), radius))
        ]
        self.baseline = baseline[0]
        self.arg_df = pd.DataFrame(params, columns=["baseline", "intensity", "radius"])

        self.dt = dt

    @property
    def t_pre(self):
        return self.config.t_pre

    @property
    def t_stim(self):
        return self.config.t_stim

    def __len__(self):
        return len(self.arg_df)

    def _key(self, intensity, radius):
        try:
            return self.arg_df.query(
                f"radius=={radius}" f" & intensity == {intensity}"
            ).index.values.item()
        except ValueError:
            raise ValueError(f"radius: {radius}, intensity: {intensity} invalid.")

    def _params(self, key):
        return self.arg_df.iloc[key].values

    def get(self, intensity, radius):
        """
        #TODO: docstring
        """
        key = self._key(intensity, radius)
        return self[key]

    def get_item(self, key):
        """
        #TODO: docstring
        """
        return torch.Tensor(self.flashes_dir.flashes[key])

    def stimulus(self, intensity, radius):
        """
        #TODO: docstring
        """
        key = self._key(intensity, radius)
        stim = self[key][:, 360].cpu().numpy()
        return stim

    def mask(self, intensity=None, radius=None):

        values = self.arg_df.values

        def iterparam(param, name, axis, and_condition):
            condition = np.zeros(len(values)).astype(bool)
            if isinstance(param, Iterable):
                for p in param:
                    _new = values.take(axis, axis=1) == p
                    assert any(_new), f"{name} {p} not in dataset."
                    condition = np.logical_or(condition, _new)
            else:
                _new = values.take(axis, axis=1) == param
                assert any(_new), f"{name} {param} not in dataset."
                condition = np.logical_or(condition, _new)
            return condition & and_condition

        condition = np.ones(len(values)).astype(bool)
        if intensity is not None:
            condition = iterparam(intensity, "intensity", 1, condition)
        if radius is not None:
            condition = iterparam(radius, "radius", 2, condition)
        return condition

    def __repr__(self):
        return "Flashes dataset: \n{}".format(repr(self.arg_df))
