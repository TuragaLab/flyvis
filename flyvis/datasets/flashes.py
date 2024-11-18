"""Rendering of circular flash sequences on a hexagonal lattice."""

import logging
from itertools import product
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datamate import Directory, root
from tqdm import tqdm

from flyvis import renderings_dir
from flyvis.utils.hex_utils import Hexal, HexLattice

from .datasets import SequenceDataset
from .rendering import BoxEye
from .rendering.utils import resample

logger = logging.getLogger(__name__)

__all__ = ["RenderedFlashes", "Flashes", "render_flash"]


@root(renderings_dir)
class RenderedFlashes(Directory):
    """Render a directory with flashes for the Flashes dataset.

    Args:
        boxfilter: Parameters for the BoxEye filter.
        dynamic_range: Range of intensities. E.g. [0, 1] renders flashes
            with decrement 0.5->0 and increment 0.5->1.
        t_stim: Duration of the stimulus.
        t_pre: Duration of the grey stimulus.
        dt: Timesteps.
        radius: Radius of the stimulus.
        alternations: Sequence of alternations between lower or upper intensity and
            baseline of the dynamic range.

    Attributes:
        flashes (ArrayFile): Array containing rendered flash sequences.
    """

    def __init__(
        self,
        boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
        dynamic_range: List[float] = [0, 1],
        t_stim: float = 1.0,
        t_pre: float = 1.0,
        dt: float = 1 / 200,
        radius: List[int] = [-1, 6],
        alternations: Tuple[int, ...] = (0, 1, 0),
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
        for (baseline, intensity), rad in tqdm(values, desc="Flashes"):
            sequence.append(
                render_flash(
                    n_ommatidia,
                    intensity,
                    baseline,
                    t_stim,
                    t_pre,
                    dt,
                    alternations,
                    rad,
                )
            )

        self.flashes = np.array(sequence)


def render_flash(
    n_ommatidia: int,
    intensity: float,
    baseline: float,
    t_stim: float,
    t_pre: float,
    dt: float,
    alternations: Tuple[int, ...],
    radius: int,
) -> np.ndarray:
    """Generate a sequence of flashes on a hexagonal lattice.

    Args:
        n_ommatidia: Number of ommatidia.
        intensity: Intensity of the flash.
        baseline: Intensity of the baseline.
        t_stim: Duration of the stimulus.
        t_pre: Duration of the grey stimulus.
        dt: Timesteps.
        alternations: Sequence of alternations between lower or upper intensity
            and baseline of the dynamic range.
        radius: Radius of the stimulus.

    Returns:
        Generated flash sequence.
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
        boxfilter: Parameters for the BoxEye filter.
        dynamic_range: Range of intensities. E.g. [0, 1] renders flashes
            with decrement 0.5->0 and increment 0.5->1.
        t_stim: Duration of the stimulus.
        t_pre: Duration of the grey stimulus.
        dt: Timesteps.
        radius: Radius of the stimulus.
        alternations: Sequence of alternations between lower or upper intensity and
            baseline of the dynamic range.

    Attributes:
        dt: Timestep.
        t_post: Post-stimulus time.
        flashes_dir: Directory containing rendered flashes.
        config: Configuration object.
        baseline: Baseline intensity.
        arg_df: DataFrame containing flash parameters.

    Note:
        Zero alternation is the prestimulus and baseline. One alternation is the
        central stimulus. Has to start with zero alternation. `t_pre` is the
        duration of the prestimulus and `t_stim` is the duration of the stimulus.
    """

    dt: Union[float, None] = None
    t_post: float = 0.0

    def __init__(
        self,
        boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
        dynamic_range: List[float] = [0, 1],
        t_stim: float = 1.0,
        t_pre: float = 1.0,
        dt: float = 1 / 200,
        radius: List[int] = [-1, 6],
        alternations: Tuple[int, ...] = (0, 1, 0),
    ):
        assert alternations[0] == 0, "First alternation must be 0."
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
    def t_pre(self) -> float:
        """Duration of the prestimulus and zero alternation."""
        return self.config.t_pre

    @property
    def t_stim(self) -> float:
        """Duration of the one alternation."""
        return self.config.t_stim

    def get_item(self, key: int) -> torch.Tensor:
        """Index the dataset.

        Args:
            key: Index of the item to retrieve.

        Returns:
            Flash sequence at the given index.
        """
        return torch.Tensor(self.flashes_dir.flashes[key])

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"Flashes dataset. Parametrization: \n{self.arg_df}"
