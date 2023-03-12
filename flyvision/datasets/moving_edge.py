from itertools import product
from datamate import Namespace
import numpy as np
import pandas as pd
from flyvision.datasets.moving_bar import Movingbar


class Movingedge(Movingbar):
    """
    Args:
        post_pad_mode (str): "value" or "continue". The latter pads with the
            last frame - for moving edges.
        shuffled_offsets (bool): to shuffle the
            offsets for removing the spatio-temporal correlation of the stimulus,
            which could serve as a normalizing stimulus.
    """

    def __init__(
        self,
        offsets=(-10, 11),  # in 1 * radians(2.25) led size
        intensities=[0, 1],
        speeds=[2.4, 4.8, 9.7, 13, 19, 25],  # in 1 * radians(5.8) / s
        height=9,  # in 1 * radians(2.25) led size
        dt=1 / 200,
        tnn=None,
        subwrap="movingbar",
        device="cuda",
        post_pad_mode="value",
        t_pre=1.0,
        t_post=1.0,
        build_stim_on_init=True,  # can speed up things downstream if only responses are needed
        shuffle_offsets=False,  # shuffle offsets to remove spatio-temporal correlation -- can be used as stimulus to compute a baseline of motion selectivity
        seed=0,  # only for shuffle_offsets
        angles=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    ):
        widths = [80]
        bar_loc_horizontal = np.radians(0)
        kwargs = vars()
        kwargs.pop("self")
        kwargs.pop("__class__")
        super().__init__(**kwargs)
