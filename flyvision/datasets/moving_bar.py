from typing import Iterable, List
from numbers import Number
from itertools import product
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import logging


from datamate import Directory, Namespace, root

import flyvision
from flyvision.datasets.base import StimulusDataset
from flyvision.rendering.eye import HexEye
from flyvision.rendering.utils import resample, pad, shuffle
from flyvision import utils


logging = logging.getLogger()


@root(flyvision.root_dir)
class Offsets(Directory):
    class Config:
        angles: List = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        widths = [1, 2, 4]
        intensities = [0, 1]
        offsets: list
        led_width = np.radians(2.25)
        height = np.radians(2.25) * 9
        n_bars = 1
        bg_intensity = 0.5
        bar_loc_horizontal = np.radians(90)

    def __init__(self, config: Config):

        eye = HexEye(721, 25)

        params = list(product(config.angles, config.widths, config.intensities))

        sequences = {}
        _tqdm = tqdm(total=len(params), desc="building stimuli")
        for angle in config.angles:  # np.unique(np.array(config.angles) % 180):
            for width in config.widths:
                for intensity in config.intensities:
                    # breakpoint()
                    offset_bars = eye.offset_bars(
                        bar_width_rad=width * config.led_width,
                        bar_height_rad=config.height,
                        n_bars=config.n_bars,
                        offsets=np.array(config.offsets) * config.led_width,
                        bar_intensity=intensity,
                        bg_intensity=config.bg_intensity,
                        moving_angle=angle,
                        bar_loc_horizontal=config.bar_loc_horizontal,
                    )
                    sequences[(angle, width, intensity)] = offset_bars
                    _tqdm.update()
                    # sequences[(angle + 180, width, intensity)] =
                    # torch.flip(offset_bars, (0, )) _tqdm.update()
        _tqdm.close()
        self.offsets = torch.stack([sequences[p] for p in params]).cpu().numpy()


class Movingbar(StimulusDataset):
    """
    Args:
        post_pad_mode (str): "value" or "continue". The latter pads with the
            last frame - for moving edges.
        shuffled_offsets (bool): to shuffle the
            offsets for removing the spatio-temporal correlation of the stimulus,
            which could serve as a normalizing stimulus.
    """

    augment = False
    dt = None
    n_sequences = 0
    framerate = None

    def __init__(
        self,
        widths=[1, 2, 4],  # in 1 * radians(2.25) led size
        offsets=(-10, 11),  # in 1 * radians(2.25) led size
        intensities=[0, 1],
        speeds=[2.4, 4.8, 9.7, 13, 19, 25],  # in 1 * radians(5.8) / s
        height=9,  # in 1 * radians(2.25) led size
        dt=1 / 200,
        tnn=None,
        subwrap="movingbar",
        device="cuda",
        bar_loc_horizontal=np.radians(90),
        post_pad_mode="value",
        t_pre=1.0,
        t_post=1.0,
        build_stim_on_init=True,  # can speed up things downstream if only responses are needed
        shuffle_offsets=False,  # shuffle offsets to remove spatio-temporal correlation -- can be used as stimulus to compute a baseline of motion selectivity
        seed=0,  # only for shuffle_offsets
        angles=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
    ):

        super().__init__()
        # Eye parameter
        self.omm_width = np.radians(5.8)  # radians(5.8 degree)

        # Monitor parameter
        self.led_width = np.radians(2.25)  # Gruntman et al. 2018

        _locals = locals()
        self.config = Namespace(
            {
                arg: _locals[arg]
                for arg in [
                    "widths",
                    "offsets",
                    "intensities",
                    "speeds",
                    "height",
                    "bar_loc_horizontal",
                    "shuffle_offsets",
                    "post_pad_mode",
                    "t_pre",
                    "t_post",
                    "dt",
                    "angles",
                ]
            }
        )

        # Stim parameter
        self.angles = np.array(angles)
        self.widths = np.array(widths)  # half ommatidia
        self.offsets = np.arange(*offsets)  # half ommatidia
        self.intensities = np.array(intensities)
        self.speeds = np.array(speeds)
        self.bg_intensity = 0.5
        self.n_bars = 1
        self.bar_loc_horizontal = bar_loc_horizontal

        self.t_stim = (len(self.offsets) * self.led_width) / (
            self.speeds * self.omm_width
        )
        self.t_stim_max = np.max(self.t_stim)

        self._speed_to_t_stim = dict(zip(self.speeds, self.t_stim))

        self.height = self.led_width * height

        self.post_pad_mode = post_pad_mode
        self._t_pre = t_pre
        self._t_post = t_post

        params = [
            (*p[:-1], *p[-1])
            for p in list(
                product(
                    self.angles,
                    self.widths,
                    self.intensities,
                    zip(self.t_stim, self.speeds),
                )
            )
        ]
        self.arg_df = pd.DataFrame(
            params, columns=["angle", "width", "intensity", "t_stim", "speed"]
        )

        self.arg_group_df = self.arg_df.groupby(
            ["angle", "width", "intensity"], sort=False, as_index=False
        ).all()

        self.tnn = tnn
        self.subwrap = subwrap

        if tnn is not None:
            self._init_tnn()

        self.device = device
        self.shuffle_offsets = shuffle_offsets
        self.randomstate = None
        if self.shuffle_offsets:
            self.randomstate = np.random.RandomState(seed=seed)

        self._dt = dt

        self._built = False
        if build_stim_on_init:
            self._build()
            self._resample()
            self._built = True

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if self.tnn is None or self._dt == value:
            self._dt = value
            if self._built:
                self._resample()
            return
        logging.warning(
            f"Cannot override dt={value} because responses with dt={self._dt} are initialized."
            f" Keeping dt={self._dt}."
        )

    def _init_tnn(self, tnn=None):
        tnn = tnn or self.tnn
        with exp_path_context():
            if isinstance(tnn, (str, Path)):
                self.tnn, _ = init_network_wrap(tnn, None, None)
            self.central_activity = utils.CentralActivity(
                self.tnn[self.subwrap].network_states.nodes.activity_central,
                self.tnn.connectome,
                keepref=True,
            )
        stored_config = self.tnn[self.subwrap].config
        if stored_config.is_value_matching_superset(self.config):
            self.config = stored_config
        else:
            flyvision.utils.warn_once(
                logging,
                (
                    "ValueError: stored config (other) is not a value-matching superset of the initialized config (self).\n"
                    f"Diff is:\n{self.config.diff(stored_config)}"
                ),
            )

    def __len__(self):
        return len(self.arg_df)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n"
            + "Config:\n"
            + repr(self.config)
            + "Stimulus parameter:\n"
            + repr(self.arg_df)
        )

    @property
    def t_pre(self):
        return self._t_pre

    @property
    def t_post(self):
        return self._t_post

    def _build(self):
        self.wrap = Offsets(
            dict(
                angles=self.angles,
                widths=self.widths,
                intensities=self.intensities,
                offsets=self.offsets,
                led_width=self.led_width,
                height=self.height,
                n_bars=self.n_bars,
                bg_intensity=self.bg_intensity,
                bar_loc_horizontal=self.bar_loc_horizontal,
            )
        )

        self._offsets = torch.tensor(self.wrap.offsets[:], device=self.device)
        self._built = True

    def _resample(self):
        # resampling at runtime to allow for changing dt and to save GB of
        # storage.
        self.sequences = {}
        for t, speed in zip(self.t_stim, self.speeds):
            sequence = resample(
                self._offsets,
                t,
                self.dt,
                dim=1,
                device=self.device,
            )
            if self.shuffle_offsets:
                # breakpoint()
                sequence = shuffle(sequence, self.randomstate)
            sequence = pad(
                sequence,
                t + self.t_pre,
                self.dt,
                mode="start",
                fill=self.bg_intensity,
            )
            sequence = pad(
                sequence,
                t + self.t_pre + self.t_post,
                self.dt,
                mode="end",
                pad_mode=self.post_pad_mode,
                fill=self.bg_intensity,
            )
            # Because we fix the distance that the bar moves but vary speeds we
            # have different stimulation times. To make all sequences equal
            # length for storing them in a single tensor, we pad them with nans
            # based on the maximal stimulation time (slowest speed). The nans
            # can later be removed before processing the traces.
            sequence = pad(
                sequence,
                self.t_stim_max + self.t_pre + self.t_post,
                self.dt,
                mode="end",
                fill=np.nan,
            )
            self.sequences[speed] = sequence

    # def _resample(self): # TODO: could flatten sequences datastructure to be
    #     able to index the sequences # in the sample dimension with the same
    #     mask as the responses # this requires however, to change the ordering
    #     of the stimulus parameters # which also effects the computation of the
    #     dsi and the order of all stored responses is mismatched to this order,
    #     careful self.sequences = [] for t, speed in zip(self.t_stim,
    #     self.speeds): sequence = resample( self._offsets, t, self.dt, dim=1,
    #     device=self.device,
    #         )
    #         if self.shuffle_offsets:
    #             # breakpoint()
    #             sequence = shuffle(sequence, self.randomstate)
    #         sequence = pad(
    #             sequence,
    #             t + self.t_pre,
    #             self.dt,
    #             mode="start",
    #             fill=self.bg_intensity,
    #         )
    #         sequence = pad(
    #             sequence,
    #             t + self.t_pre + self.t_post,
    #             self.dt,
    #             mode="end",
    #             pad_mode=self.post_pad_mode,
    #             fill=self.bg_intensity,
    #         )
    #         # Because we fix the distance that the bar moves but vary speeds
    #         # we have different stimulation times. To make all sequences equal length
    #         # for storing them in a single tensor, we pad them with nans based on the
    #         # maximal stimulation time (slowest speed).
    #         # The nans can later be removed before processing the traces.
    #         sequence = pad(
    #             sequence,
    #             self.t_stim_max + self.t_pre + self.t_post,
    #             self.dt,
    #             mode="end",
    #             fill=np.nan,
    #         )
    #         self.sequences.append(sequence)
    #     self.sequences = torch.cat(self.sequences)

    def _key(self, angle, width, intensity, speed):
        try:
            return self.arg_df.query(
                f"angle=={angle}"
                f" & width=={width}"
                f" & intensity == {intensity}"
                f" & speed == {speed}"
            ).index.values.item()
        except ValueError:
            raise ValueError(
                f"angle: {angle}, width: {width}, intensity: {intensity}, speed: {speed} invalid."
            )

    def get_sequence_id_from_arguments(self, angle, width, intensity, speed):
        # generic of _key
        return self._get_sequence_id_from_arguments(locals())

    def _params(self, key):
        return self.arg_df.iloc[key].values

    def _group_key(self, angle, width, intensity):
        return self.arg_group_df.query(
            f"angle=={angle}" f" & width=={width}" f" & intensity == {intensity}"
        ).index.values.item()

    def _group_params(self, key):
        return self.arg_group_df.iloc[key].values

    def get(self, angle, width, intensity, speed):
        key = self._key(angle, width, intensity, speed)
        return self[key]

    def arg_df(self):
        return self.arg_df

    def get_item(self, key):
        # return self.sequences[key]
        angle, width, intensity, _, speed = self._params(key)
        return self.sequences[speed][self._group_key(angle, width, intensity)]

    def mask(self, angle=None, width=None, intensity=None, speed=None, t_stim=None):
        # 22x faster than df.query
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
        if angle is not None:
            condition = iterparam(angle, "angle", 0, condition)
        if width is not None:
            condition = iterparam(width, "width", 1, condition)
        if intensity is not None:
            condition = iterparam(intensity, "intensity", 2, condition)
        if t_stim is not None:
            condition = iterparam(t_stim, "t_stim", 3, condition)
        if speed is not None:
            condition = iterparam(speed, "speed", 4, condition)
        return condition

    def stimulus(
        self,
        angle=None,
        width=None,
        intensity=None,
        speed=None,
        pre_stim=True,
        post_stim=True,
    ):
        """
        #TODO: docstring
        """
        key = self._key(angle, width, intensity, speed)
        stim = self[key][:, 360].cpu().numpy()
        if not post_stim:
            stim = filter_post([stim], self.t_post, self.dt).squeeze()
        if not pre_stim:
            stim = filter_pre(stim[None], self.t_pre, self.dt).squeeze()
        return stim

    def stimulus_parameters(self, angle=None, width=None, intensity=None, speed=None):
        def _number_to_list(*args):
            returns = tuple()
            for arg in args:
                if isinstance(arg, Number):
                    returns += ([arg],)
                else:
                    returns += (arg,)
            return returns

        angle, width, speed, intensity = _number_to_list(angle, width, speed, intensity)
        angle = angle or self.angles
        width = width or self.widths
        intensity = intensity or self.intensities
        speed = speed or self.speeds
        return angle, width, intensity, speed

    def response(
        self,
        cell_type=None,
        angle=None,
        width=None,
        intensity=None,
        speed=None,
        pre_stim=False,
        post_stim=False,
        nonlinearity=False,
        subtract_baseline=False,
        reshape=False,
    ):
        def _load_response():
            assert self.tnn
            mask = self.mask(
                angle=angle,
                width=width,
                intensity=intensity,
                t_stim=None,
                speed=speed,
            )
            # breakpoint()
            if cell_type is not None:
                responses = self.central_activity[cell_type][mask]
            else:
                responses = self.central_activity[:][:][mask]

            # breakpoint()
            if not post_stim:
                responses = filter_post(responses, self.t_post, self.dt)
            if not pre_stim:
                responses = filter_pre(responses, self.t_pre, self.dt)

            return responses.squeeze()

        responses = _load_response()

        if reshape:

            angle, width, intensity, speed = self.stimulus_parameters(
                angle, width, intensity, speed
            )

            shape = responses.shape
            if len(shape) == 2:
                n_cell_types = 1
                n_samples, n_frames = shape
            else:
                n_samples, n_frames, n_cell_types = shape

            if nonlinearity:
                responses = np.maximum(responses, 0)

            if subtract_baseline:
                responses -= responses[:, 0][:, None]

            responses = responses.reshape(
                len(angle),
                len(width),
                len(intensity),
                len(speed),
                n_frames,
                n_cell_types,
            )
        return responses

    def peak_response_angular(
        self,
        cell_type=None,
        width=None,
        intensity=None,
        speed=None,
        nonlinearity=True,
        pre_stim=False,
        post_stim=False,
        normalize=False,
        subtract_baseline=False,
    ):
        """To return the peak responses in the complex plane
        r * exp(theta * 1j), with r the peak responses and theta the stimulus
        movement angle.
        """
        if isinstance(width, Number):
            width = [width]
        if isinstance(speed, Number):
            speed = [speed]
        if isinstance(intensity, Number):
            intensity = [intensity]
        _stim_args = (width, intensity, speed)
        width = width or self.widths
        intensity = intensity or self.intensities
        speed = speed or self.speeds

        responses = self.response(
            angle=None,
            width=width,
            intensity=intensity,
            speed=speed,
            cell_type=None,
            pre_stim=pre_stim,
            post_stim=post_stim,
        )
        n_samples, n_frames, n_cell_types = responses.shape

        if nonlinearity:
            responses = np.maximum(responses, 0)

        if subtract_baseline:
            responses -= responses[:, 0][:, None]

        responses = responses.reshape(
            len(self.angles),
            len(width),
            len(intensity),
            len(speed),
            n_frames,
            n_cell_types,
        )

        # for speeds > min(speeds) we pad nans after the post stimulus duration
        # simply setting them to zero is wrong because it creates jumps
        # responses[np.isnan(responses)] = 0
        # breakpoint()
        peak_resp = np.nanmax(responses, axis=4)
        peak_resp_angular = (
            np.exp(np.radians(self.angles) * 1j)[:, None, None, None, None] * peak_resp
        )

        # it is important to normalize over both contrasts for each individual
        # contrast because otherwise the on-off motion dsi differences are not
        # visible
        if normalize:
            if normalize == "self":
                _peak_resp_angular = peak_resp_angular
            elif any(_stim_args):
                # to compute the normalizer as peak response given all possible
                # stimuli, we set width, intensity, speed to None
                _peak_resp_angular, _ = self.peak_response_angular(
                    cell_type=cell_type,
                    width=None,
                    intensity=None,
                    speed=None,
                    nonlinearity=nonlinearity,
                    pre_stim=pre_stim,
                    post_stim=post_stim,
                    normalize=False,
                    subtract_baseline=subtract_baseline,
                )
            else:
                # if width, intensity, speed is None peak_resp_angular contains
                # peak responses to all stimuli and therefore the normalizer
                _peak_resp_angular = peak_resp_angular
            # first absolute to get the undirectional vector lengths and then
            # sum to get the normalization constants for the superpositioned
            # directional vectors
            peak_resp_angular /= (
                np.nansum(np.abs(_peak_resp_angular), axis=(0, 1, 2, 3)) + 1e-15
            )

        if cell_type is not None:
            return np.take(
                peak_resp_angular, self.central_activity.index[cell_type], -1
            ), (width, intensity, speed)

        return peak_resp_angular, (width, intensity, speed)

    def dsi(
        self,
        cell_type=None,
        round_angle=False,
        width=None,
        intensity=None,
        speed=None,
        nonlinearity=True,
        pre_stim=False,  # debugging flag
        post_stim=False,  # debugging flag
        subtract_baseline=False,
        normalize=True,
    ):
        # peak_resp_angular, (
        #     width,
        #     intensity,
        #     speed,
        # ) = self.peak_response_angular(
        #     cell_type=None,
        #     width=width,
        #     intensity=intensity,
        #     speed=speed,
        #     nonlinearity=nonlinearity,
        #     pre_stim=pre_stim,
        #     post_stim=post_stim,
        #     normalize=normalize,
        #     subtract_baseline=subtract_baseline,
        # )
        # peak_resp_angular_sum = np.nansum(peak_resp_angular, axis=(0, 1, 2, 3))
        # dsis_absolute = np.abs(peak_resp_angular_sum)
        # raise NotImplementedError
        # breakpoint()
        peak_resp_angular, (width, _, speed,) = self.peak_response_angular(
            cell_type=None,
            width=width,
            intensity=None,
            speed=speed,
            nonlinearity=nonlinearity,
            pre_stim=pre_stim,
            post_stim=post_stim,
            normalize=False,
            subtract_baseline=subtract_baseline,
        )
        # breakpoint()
        dsis = dsi_from_peak_angular_responses(
            peak_resp_angular,
            angle_axis=0,
            width_axis=1,
            intensity_axis=2,
            speed_axis=3,
            average=True,
        )
        intensity = [intensity] if not isinstance(intensity, Iterable) else intensity
        intensity_index = [
            np.where(_intensity == self.intensities)[0].item()
            for _intensity in intensity
        ]
        # breakpoint()
        dsis_absolute = dsis[intensity_index].squeeze()
        all_theta_pref = utils.round_angles(
            np.degrees(
                theta_pref_from_peak_angular_responses(
                    peak_resp_angular,
                    angle_axis=0,
                    width_axis=1,
                    intensity_axis=2,
                    speed_axis=3,
                    average=True,
                )
            ),
            self.angles,
            round_angle,
        )
        theta_pref = all_theta_pref[intensity_index].squeeze()

        if cell_type is None:
            nodes_list, index = utils.order_nodes_list(
                self.tnn.connectome.unique_cell_types[:].astype(str)
            )
            # breakpoint()
            return (
                nodes_list,
                np.take(dsis_absolute, index, axis=-1),
                np.take(theta_pref, index, axis=-1),
            )
        # breakpoint()
        cell_type_index = self.central_activity.index[cell_type]
        dsi_cell_type = np.take(dsis_absolute, cell_type_index, axis=-1)
        # breakpoint()
        peak_resp_angular = peak_resp_angular[:, :, intensity_index][
            :, :, :, :, cell_type_index
        ]
        # _peak_resp_angular = _peak_resp_angular[:, :, :, :, cell_type_index]

        # summing over n_widths, n_intentieis, n_speeds to get
        # n_angles peak responses in complex space
        # breakpoint()
        # z_cell_type =
        # theta_pref = utils.round_angles(
        #     np.degrees(np.angle(z_cell_type.sum())), self.angles, round_angle
        # )
        # breakpoint()
        dsis = dsi_from_peak_angular_responses(
            peak_resp_angular,
            angle_axis=0,
            width_axis=1,
            intensity_axis=2,
            speed_axis=3,
            average=False,
        )

        all_theta_pref = utils.round_angles(
            np.degrees(
                theta_pref_from_peak_angular_responses(
                    peak_resp_angular,
                    angle_axis=0,
                    width_axis=1,
                    intensity_axis=2,
                    speed_axis=3,
                    average=False,
                )
            ),
            self.angles,
            round_angle,
        )

        values = np.array(list(product(width, intensity, speed)))
        # peak_resp_angular_sum = np.nansum(peak_resp_angular, axis=0)
        # dsi_tensor = np.abs(peak_resp_angular_sum)
        # theta_pref_tensor = np.degrees(np.angle(peak_resp_angular_sum))

        values = np.array(list(product(width, intensity, speed)))
        dsi_table = pd.DataFrame(
            dict(
                dsi=dsis.flatten(),
                theta_pref=all_theta_pref.flatten(),
                width=values[:, 0],
                intensity=values[:, 1],
                speed=values[:, 2],
            )
        )

        argmax_index = dsi_table.dsi.argmax()
        argmax = dsi_table.iloc[argmax_index].values[1:]
        theta_pref = all_theta_pref.flatten()[argmax_index]
        r = np.abs(np.nansum(peak_resp_angular, axis=(1, 2, 3)))

        return dsi_cell_type, theta_pref, argmax, (self.angles, r), dsi_table

    def max_dsi_trace(
        self,
        cell_type,
        width=None,
        speed=None,
        intensity=None,
        round_angle=True,
        angle=None,
    ):
        _, pd, argmax, _, _ = self.dsi(
            cell_type,
            round_angle=round_angle,
            width=width,
            speed=speed,
            intensity=intensity,
        )
        _, width, intensity, speed = argmax
        if angle is None:
            angle = pd
        elif angle == "null":
            angle = int((pd + 180) % 360)

        response = self.response(
            angle=angle,
            width=width,
            speed=speed,
            intensity=intensity,
            cell_type=cell_type,
        ).squeeze()
        stimulus = self.stimulus(
            angle=angle, width=width, speed=speed, intensity=intensity
        ).squeeze()
        return stimulus, response, self.dt, (angle, width, speed, intensity)

    def max_dsi(
        self,
        cell_type,
        width=None,
        speed=None,
        intensity=None,
        round_angle=True,
    ):
        _, _, argmax, _, _ = self.dsi(
            cell_type,
            round_angle=round_angle,
            width=width,
            speed=speed,
            intensity=intensity,
        )
        _, width, intensity, speed = argmax
        return self.dsi(
            cell_type,
            round_angle=True,
            width=[width],
            speed=[speed],
            intensity=[intensity],
        )

    def traces(
        self,
        cell_type,
        angle,
        width,
        speed,
        intensity,
        pre_stim=True,
        post_stim=True,
        zero_at="center",
        xlim=(-0.5, 1.0),
    ):
        dt = self.dt
        stim = self.stimulus(
            angle=angle,
            width=width,
            speed=speed,
            intensity=intensity,
            pre_stim=pre_stim,
            post_stim=post_stim,
        )
        resp = self.response(
            cell_type=cell_type,
            pre_stim=pre_stim,
            post_stim=post_stim,
            angle=angle,
            width=width,
            speed=speed,
            intensity=intensity,
        )
        opposite_resp = self.response(
            cell_type=cell_type,
            pre_stim=pre_stim,
            post_stim=post_stim,
            angle=(angle + 180) % 360,
            width=width,
            intensity=intensity,
            speed=speed,
        )

        _nans = np.isnan(stim)
        stim = stim[~_nans]
        resp = resp[
            ~_nans
        ]  # TODO: the response can be non-nan for longer -> nan takes some timesteps to reach the neuron.
        resp = resp - np.mean(resp)
        opposite_resp = opposite_resp[~_nans]
        opposite_resp = opposite_resp - np.mean(opposite_resp)

        if pre_stim:
            t_pre = self.t_pre
        else:
            t_pre = 0
        time_to_center = (
            np.abs(self.config.offsets[0])
            * np.radians(2.25)
            / (speed * np.radians(5.8))
        )
        if zero_at == "center":
            time = np.linspace(
                -(t_pre + time_to_center),
                len(resp) * dt - (t_pre + time_to_center),
                len(resp),
            )
        elif zero_at == "onset":
            time = np.linspace(-t_pre, len(resp) * dt - t_pre, len(resp))

        if xlim:
            mask = (time >= xlim[0]) & (time <= xlim[1])
            time = time[mask]
            resp = resp[mask]
            stim = stim[mask]
            opposite_resp = opposite_resp[mask]

        return time, stim, resp, opposite_resp

    def preferred_direction(
        self, cell_type, nonlinearity=True, round_angle=True, intensity=None
    ):
        dsi_cell_type, theta_pref, argmax, (angles, r), dsi_table = self.dsi(
            cell_type,
            nonlinearity=nonlinearity,
            round_angle=round_angle,
            intensity=intensity,
        )
        return theta_pref

    def sample_shape(
        self,
        angle=None,
        width=None,
        intensity=None,
        speed=None,
    ):
        if isinstance(angle, Number):
            angle = [angle]
        if isinstance(width, Number):
            width = [width]
        if isinstance(speed, Number):
            speed = [speed]
        if isinstance(intensity, Number):
            intensity = [intensity]
        angle = angle or self.angles
        width = width or self.widths
        intensity = intensity or self.intensities
        speed = speed or self.speeds
        return (
            len(angle),
            len(width),
            len(intensity),
            len(speed),
        )

    def peak_response(
        self,
        cell_type=None,
        angle=None,
        width=None,
        intensity=None,
        speed=None,
        nonlinearity=True,
        pre_stim=False,
        post_stim=False,
        normalize=False,
        subtract_baseline=False,
    ):
        """To return the peak responses.

        TODO: peak_response_angular and peak_response contain redundant code and
        could be modularized differently.
        """
        if isinstance(angle, Number):
            angle = [angle]
        if isinstance(width, Number):
            width = [width]
        if isinstance(speed, Number):
            speed = [speed]
        if isinstance(intensity, Number):
            intensity = [intensity]
        _stim_args = (angle, width, intensity, speed)
        angle = angle or self.angles
        width = width or self.widths
        intensity = intensity or self.intensities
        speed = speed or self.speeds

        responses = self.response(
            angle=angle,
            width=width,
            intensity=intensity,
            speed=speed,
            cell_type=None,
            pre_stim=pre_stim,
            post_stim=post_stim,
        )
        # breakpoint()
        n_samples, n_frames, n_cell_types = responses.shape

        if nonlinearity:
            responses = np.maximum(responses, 0)

        if subtract_baseline:
            responses -= responses[:, 0][:, None]

        responses = responses.reshape(
            len(angle),
            len(width),
            len(intensity),
            len(speed),
            n_frames,
            n_cell_types,
        )

        responses[np.isnan(responses)] = 0

        # temporal speed response
        peak_resp = responses.max(axis=4)

        if normalize:
            if any(_stim_args):
                # to compute the normalizer as peak response given all possible
                # stimuli, we set angle, width, intensity, speed to None
                _peak_resp, _ = self.peak_response(
                    self,
                    cell_type=cell_type,
                    angle=angle,
                    width=None,
                    intensity=None,
                    speed=None,
                    nonlinearity=nonlinearity,
                    pre_stim=pre_stim,
                    post_stim=post_stim,
                    normalize=False,
                    subtract_baseline=subtract_baseline,
                )
            else:
                # if angle, width, intensity, speed is None peak_resp_angular
                # contains peak responses to all stimuli and therefore the
                # normalizer
                _peak_resp = peak_resp
            # first absolute to get the undirectional vector lengths and then
            # sum to get the normalization constants for the superpositioned
            # directional vectors
            peak_resp /= np.abs(_peak_resp).sum(axis=(0, 1, 2, 3)) + 1e-15

        if cell_type is not None:
            return np.take(peak_resp, self.central_activity.index[cell_type], -1), (
                width,
                intensity,
                speed,
            )

        return peak_resp, (angle, width, intensity, speed)

    def time_to_center(self, speed: float) -> float:
        """Returns the time in s at which the bar reaches the center.

        Note: time = distance / velocity, i.e.
            time = (n_leds * led_width) / (speed * omm_width)
            with speed in ommatidia / s.
        """
        return (
            np.abs(self.config.offsets[0]) * self.led_width / (speed * self.omm_width)
        )

    def get_time_with_origin_at_onset(self, speed: float):
        """Time with 0 at the onset of the stimulus."""
        return np.linspace(
            -self.t_pre,
            self.t_stim_max - self.t_pre + self.t_post,
            int(self.t_stim_max / self.dt)
            + int(self.t_post / self.dt)
            + int(self.t_pre / self.dt),
        )

    def get_time_with_origin_at_center(self, speed: float):
        """Time with 0 where the bar reaches the central column."""
        time_to_center = self.time_to_center(speed)
        n_steps = (
            int(self.t_stim_max / self.dt)
            + int(self.t_post / self.dt)
            + int(self.t_pre / self.dt)
        )
        return np.linspace(
            -(self.t_pre + time_to_center),
            n_steps * self.dt - (self.t_pre + time_to_center),
            n_steps,
        )

    def postsynaptic_inputs(
        self,
        network: "flyvision.networks.Network",
        dt,
        angle,
        width,
        speed,
        intensity,
    ):
        """Stimulus, postsynaptic inputs and responses for configific parameters.

        Returns:
            time (n_frames)
            stims (n_samples, n_frames, n_hexals)
            postsynaptic_inputs: w_{ij}f(V_j) (n_samples, n_frames, n_edges)
            responses: V_i (n_samples, n_frames, n_nodes)
        """
        indices = [self._key(angle, width, intensity, speed)]
        time = self.get_time_with_origin_at_center(speed)
        stims, postsynaptic_inputs, responses = network.current_response(
            self, dt, indices, t_pre=0, t_fade_in=1.0
        )
        return time, stims, postsynaptic_inputs, responses


def filter_post(resp, t_post, dt):
    """To remove the post stimulus time from responses.

    Because we fix the distance that the bar moves but vary speeds we
    have different stimulation times. To make all sequences equal
    length for storing them in a single tensor, we pad them with nans
    based on the maximal stimulation time (slowest speed).

    The post stimulus time is per speed in
    'first_nan - int(t_post / dt):first_nan'.

    Assuming resp can be partitioned along the temporal dimension as |pre stim|
    stimulus   |    post stimulus       |        nan padding        |
    |0:t_pre:|t_pre + t_stim:|t_pre + t_stim + t_post:|t_pre + t_stim_max +
    t_post|

    Args: resp of shape (n_samples, n_frames, ...)
    """
    _resp = []
    for r in resp:
        mask = np.ones(r.shape[0]).astype(bool)
        _nans = np.isnan(r)
        if len(_nans.shape) == 2:
            _nans = _nans[:, 0]
        if any(_nans):
            _nan_index = np.where(_nans)[0][0]
            # print(f'cutting out {_nan_index * dt - int(t_post)} to {_nan_index
            # * dt}')
            mask[_nan_index - int(t_post / dt) : _nan_index] = False
        else:
            mask[-int(t_post / dt) :] = False
        _resp.append(r[mask])
    return np.array(_resp)


def filter_pre(resp, t_pre, dt):
    """
    Args: resp of shape (n_samples, n_frames, ...)
    """
    return resp[:, int(t_pre / dt) :]


def dsi_from_peak_angular_responses(
    peak_responses_angular,
    angle_axis=1,
    width_axis=2,
    intensity_axis=3,
    speed_axis=4,
    average=True,
):
    vector_sum = np.nansum(peak_responses_angular, axis=angle_axis, keepdims=True)
    vector_length = np.abs(vector_sum)
    normalization = np.max(
        np.nansum(np.abs(peak_responses_angular), axis=angle_axis, keepdims=True),
        axis=intensity_axis,
        keepdims=True,
    )
    dsi = vector_length / (normalization + 1e-15)
    if average:
        dsi = np.mean(dsi, axis=(width_axis, speed_axis)).squeeze()
    return dsi


def theta_pref_from_peak_angular_responses(
    peak_responses_angular,
    angle_axis=1,
    width_axis=2,
    intensity_axis=3,
    speed_axis=4,
    average=True,
):
    vector_sum = np.nansum(peak_responses_angular, axis=angle_axis, keepdims=True)
    theta_pref = np.angle(vector_sum)
    if average:
        theta_pref = np.angle(
            np.sum(vector_sum, axis=(width_axis, speed_axis)).squeeze()
        )
    return theta_pref
