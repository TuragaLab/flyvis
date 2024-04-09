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
class MultiParameterFlashes(Directory):
    class Conf:
        boxfilter: dict
        dynamic_range: list = [0, 1]
        t_stim = (1.0,)
        t_pre = (1.0,)
        dt = (1 / 200,)
        radius: list = [-1, 6]
        alternations = ((0, 1, 0),)
        filter_type: str = "median"
        hex_sample: bool = True

    def build(self, config):
        boxfilter = BoxEye(config.boxfilter)
        n_ommatidia = len(boxfilter.receptor_centers)
        dynamic_range = np.array(config.dynamic_range)
        baseline = 2 * (dynamic_range.sum() / 2,)

        intensity = config.dynamic_range.copy()
        values = np.array(list(zip(baseline, intensity)))
        samples = dict(v=values, r=config.radius)
        values = list(product(*(v for v in samples.values())))
        sequence = []  # samples, #frames, width, height
        for (baseline, intensity), radius in tqdm(values, desc="Flashes"):
            sequence.append(
                get_flash(
                    n_ommatidia,
                    intensity,
                    baseline,
                    config.t_stim,
                    config.t_pre,
                    config.dt,
                    config.alternations,
                    radius,
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
        filter_type="median",
        hex_sample=True,
        subdir="flashes",
        tnn=None,
    ):
        self.flashes_dir = MultiParameterFlashes(
            boxfilter=boxfilter,
            dynamic_range=dynamic_range,
            t_stim=t_stim,
            t_pre=t_pre,
            dt=dt,
            radius=radius,
            filter_type=filter_type,
            hex_sample=hex_sample,
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

        self.subdir = subdir

        self.tnn = tnn
        if self.tnn:
            self.centralactivity = utils.CentralActivity(
                self.tnn[subdir].network_states.nodes.activity_central[:],
                self.tnn.connectome,
                keepref=True,
            )

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

    def response(self, intensity=None, radius=None, cell_type=None):
        """
        #TODO: docstring
        """
        assert self.tnn
        mask = self.mask(intensity, radius)
        if cell_type is not None:
            return self.centralactivity[cell_type][mask].squeeze()
        return self.centralactivity[:][mask].squeeze()

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

    def argmax_fri(self):
        """
        Returns the argmax fri for each cell type.
        #TODO: docstring, check deprecated
        """
        cell_types = self.tnn.connectome.unique_cell_types[:].astype(str)
        fri = np.zeros(len(cell_types))
        argmax = np.zeros([3, len(cell_types)])
        for (baseline, intensity), radius in self.values:
            _fri = self.flash_response_index(baseline, intensity, radius)[0]
            mask = np.abs(_fri) > np.abs(fri)
            argmax[:, mask] = np.array([baseline, intensity, radius])[:, None]
            fri[mask] = _fri[mask]
        return argmax, {
            cell_type: argmax[:, i] for i, cell_type in enumerate(cell_types)
        }

    def flash_response_index(
        self,
        radius,
        mode="convention?",
        subtract_baseline=False,
        nonlinearity=False,
        nonnegative=True,
    ):
        """
        Returns the fri for each cell type.
        """
        # logging.warning(
        #     f"{subtract_baseline}, {nonlinearity}, {nonnegative}, {mode}"
        # )
        t_stim = self.config.t_stim
        n_alternations = len(
            self.config.alternations
        )  # alternations baseline - flash - baseline
        dt = self.config.dt
        time = np.arange(0, t_stim * n_alternations, dt)

        # in (-inf, inf) starting in (-inf, inf)
        r_on = self.response(1, radius)
        r_off = self.response(0, radius)

        # relevant time window
        # start one time step before stimulus onset at which both potentials
        # should be at the same resting state.
        mask = (time >= t_stim - dt) & (time < 2 * t_stim)
        r_on = r_on[mask]
        r_off = r_off[mask]
        # subtract baseline -- this baseline subtraction cancels out
        # for on_peak - off_peak
        # and for
        # on_peak + off_peak = (V_peak_on - V_peak_on(0)) + (V_peak_off - V_peak_off(0))
        # = V_peak_on + V_peak_off - 2 * V_peak(0)
        # in (-inf, inf) starting in 0
        if subtract_baseline:
            r_on -= r_on[0]
            r_off -= r_off[0]

        # in (0, inf)
        if nonlinearity:
            r_on = np.maximum(r_on, 0)
            r_off = np.maximum(r_off, 0)

        # because conventionally, index computed on nonnegative spike rates
        # or calcium traces. lifting the traces to nonnegative magnitudes
        if nonnegative:
            minimum = np.minimum(r_on, r_off).min(axis=0)
            r_on += np.abs(minimum)
            r_off += np.abs(minimum)

        # normalize over sample and temporal max
        all_responses = self.response(None, radius)[:, mask]

        # in (-inf, inf) starting in 0
        if subtract_baseline:
            all_responses -= all_responses[:, 0][:, None]

        # in (0, inf)
        if nonlinearity:
            all_responses = np.maximum(all_responses, 0)

        # max over intensities and time, leaving cell types
        # in (0, inf)
        _max_abs_responses = np.abs(all_responses).max(axis=(0, 1))
        # r_on = r_on / (_max_abs_responses + 1e-16)
        # r_off = r_off / (_max_abs_responses + 1e-16)

        if mode == "transient":
            on_peak = r_on.max(axis=0)
            off_peak = r_off.max(axis=0)
            fri = on_peak - off_peak
            fri /= 2 * _max_abs_responses + 1e-16
        elif mode == "transient_on_and_off":
            on_peak = r_on.max(axis=0)
            off_peak = r_off.max(axis=0)
            fri = on_peak + off_peak
            fri /= 2 * _max_abs_responses + 1e-16
        elif mode == "sustained":
            on_avg = r_on.mean(axis=0)
            off_avg = r_off.mean(axis=0)
            fri = on_avg - off_avg
            fri /= 2 * _max_abs_responses + 1e-16
        elif mode == "sustained_on_and_off":
            on_avg = r_on.mean(axis=0)
            off_avg = r_off.mean(axis=0)
            fri = on_avg + off_avg
            fri /= 2 * _max_abs_responses + 1e-16
        elif mode == "convention?":
            # seen in presentation Connectomics 22 (ON - Off) / (ON + OFF)
            on_peak = r_on.max(axis=0)
            off_peak = r_off.max(axis=0)
            fri = on_peak - off_peak
            fri /= on_peak + off_peak + 1e-16
        elif mode == "sustained_convention":
            on_avg = r_on.mean(axis=0)
            off_avg = r_off.mean(axis=0)
            fri = on_avg - off_avg
            fri /= on_avg + off_avg + 1e-16
        elif mode == "convention2?":
            # seen in presentation Connectomics 22 (ON - Off) / (ON + OFF)
            on_peak = r_on.max(axis=0)
            off_peak = r_off.max(axis=0)
            fri = on_peak - off_peak
            fri /= np.abs(on_peak) + np.abs(off_peak) + 1e-16
        else:
            raise ValueError(
                f"{mode} must be 'sustained', 'transient', 'transient_on_and_off', or 'sustained_on_and_off'"
            )

        return fri, {
            cell_type: fri[i]
            for i, cell_type in enumerate(
                self.tnn.connectome.unique_cell_types[:].astype(str)
            )
        }


def get_flash(
    receptors,
    intensity=1,
    baseline=0,
    t_stim=1.0,
    t_pre=1.0,
    dt=1 / 200,
    alternations=(0, 1, 0),
    padding=(50, 50),
    hex_sample=True,
    radius=-1,
    ftype="median",
):
    """Computes probe stimuli in form of oriented bars.

    Args:
        receptors (HexBoxFilter.
        intensity (float): contrast of the background. 1 is white, 0 is black.
        baseline (float): contrast of the baseline. 1 is white, 0 is black.
        t_stim (float): stimulus time in ms.
        t_pre (float): grey stimuli time in ms.
        dt (float): timesteps.
        alternations (list): alternating sequence between baseline and stim,
            where 0 stands for baseline and 1 stands for intensity.
        padding (tuple): (p_w, p_h), increases size of the stimulus image
            so that the receptor do not see the padding added by potential
            rotation.

    Returns:
        (array): sequences come in shape (n_frames, n_hexals).
    """
    min_frame_size = (
        receptors.min_frame_size.cpu().numpy()
        if isinstance(receptors.min_frame_size, torch.Tensor)
        else receptors.min_frame_size
    )

    # Raw stimuli dimensions.
    height, width = min_frame_size + padding

    def create_circular_mask(h, w, radius=None):

        center = [int(w / 2), int(h / 2)]

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius * receptors.kernel_size
        return mask

    mask = (
        create_circular_mask(height, width, radius=radius)
        if radius != -1
        else slice(None)
    )
    # Init stimuli.
    baseline = np.ones([height, width]) * baseline
    stim = np.ones([height, width]) * baseline
    stim[mask] = intensity
    # breakpoint()
    # Sample stimuli.
    if hex_sample:
        baseline = receptors.sample(baseline, ftype=ftype)
        stim = receptors.sample(stim, ftype=ftype)

    # Repeat over time.
    baseline = baseline[None, ...].repeat(int(t_pre / dt), axis=0)
    stim = stim[None, ...].repeat(int(t_stim / dt), axis=0)

    # Stack.
    sequence = np.concatenate(np.array((baseline, stim))[np.array(alternations)])
    return sequence
