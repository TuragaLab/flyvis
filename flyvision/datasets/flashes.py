from typing import Iterable
from itertools import product
import logging

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datamate import Directory, Namespace, root

from flyvision.rendering import BoxEye, HexEye
from flyvision.datasets.datasets import SequenceDataset
from flyvision import utils, root_dir
from flyvision.utils.hex_utils import HexLattice, Hexal, resample

logging = logging.getLogger()


@root(root_dir)
class RenderedFlashes(Directory):
    def __init__(
        self,
        boxfilter: dict = dict(extent=15, kernel_size=13),
        dynamic_range: list = [0, 1],
        t_stim = 1.0,
        t_pre = 1.0,
        dt = 1 / 200,
        radius: list = [-1, 6],
        alternations = (0, 1, 0),
    ):
        boxfilter = BoxEye(**boxfilter)
        n_ommatidia = len(boxfilter.receptor_centers)
        dynamic_range = np.array(dynamic_range)
        baseline = 2 * (dynamic_range.sum() / 2,)

        intensity = dynamic_range.copy()
        values = np.array(list(zip(baseline, intensity)))
        samples = dict(v=values, r=radius)
        values = list(itertools.product(*(v for v in samples.values())))
        sequence = []  # samples, #frames, width, height
        for (bsln, intnsty), rad in tqdm(values, desc="Flashes"):
            sequence.append(
                get_flash(
                    n_ommatidia,
                    intnsty,
                    bsln,
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
    stimulus = torch.ones(n_ommatidia)[None] * baseline

    if radius != -1:
        ring = HexLattice.filled_ring(
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
    """ """

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


# def get_flash(
#     receptors,
#     intensity=1,
#     baseline=0,
#     t_stim=1.0,
#     t_pre=1.0,
#     dt=1 / 200,
#     alternations=(0, 1, 0),
#     padding=(50, 50),
#     hex_sample=True,
#     radius=-1,
#     ftype="median",
# ):
#     """Computes probe stimuli in form of oriented bars.

#     Args:
#         receptors (HexBoxFilter.
#         intensity (float): contrast of the background. 1 is white, 0 is black.
#         baseline (float): contrast of the baseline. 1 is white, 0 is black.
#         t_stim (float): stimulus time in ms.
#         t_pre (float): grey stimuli time in ms.
#         dt (float): timesteps.
#         alternations (list): alternating sequence between baseline and stim,
#             where 0 stands for baseline and 1 stands for intensity.
#         padding (tuple): (p_w, p_h), increases size of the stimulus image
#             so that the receptor do not see the padding added by potential
#             rotation.

#     Returns:
#         (array): sequences come in shape (n_frames, n_hexals).
#     """
#     min_frame_size = (
#         receptors.min_frame_size.cpu().numpy()
#         if isinstance(receptors.min_frame_size, torch.Tensor)
#         else receptors.min_frame_size
#     )

#     # Raw stimuli dimensions.
#     height, width = min_frame_size + padding

#     def create_circular_mask(h, w, radius=None):

#         center = [int(w / 2), int(h / 2)]

#         Y, X = np.ogrid[:h, :w]
#         dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

#         mask = dist_from_center <= radius * receptors.kernel_size
#         return mask

#     mask = (
#         create_circular_mask(height, width, radius=radius)
#         if radius != -1
#         else slice(None)
#     )
#     # Init stimuli.
#     baseline = np.ones([height, width]) * baseline
#     stim = np.ones([height, width]) * baseline
#     stim[mask] = intensity
#     # breakpoint()
#     # Sample stimuli.
#     if hex_sample:
#         baseline = receptors.sample(baseline, ftype=ftype)
#         stim = receptors.sample(stim, ftype=ftype)

#     # Repeat over time.
#     baseline = baseline[None, ...].repeat(int(t_pre / dt), axis=0)
#     stim = stim[None, ...].repeat(int(t_stim / dt), axis=0)

#     # Stack.
#     sequence = np.concatenate(np.array((baseline, stim))[np.array(alternations)])
#     return sequence
