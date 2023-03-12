"""Animations of neural activations.
"""
from fractions import Fraction

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from flyvision import utils
from flyvision.plots import plots
from flyvision.plots import plt_utils
from flyvision.animations.animations import Animation, AnimationCollector
from flyvision.animations.hexscatter import HexScatter
from flyvision.animations.traces import Trace


class ActivationPlusTrace(AnimationCollector):
    """Hex-scatter animation of an activation plus the central hexal's trace.

    Args:
        activation (array): shape (#samples, #frames, #hexals)
        cranges (array): shape (#samples).
        vmin (float): color minimal value.
        vmax (flot): color maximal value.
        fig (Figure): existing Figure instance or None.
        figsize (list): size of initialied figure if no fig instance given.
            Defaults to [3, 4].
        ratio (float): height ratio of the trace animation under the
            hex_scatter animation. Defaults to 0.3.
        trace_color (str or array): optional color of the trace.
        dynamic_ax_lims (bool): whether the ax limits of the trace are animated.
            Defaults to True.
        contour (array): optional background contour for trace in x direction.
        title (str): optional title of the animation.
        ylabel (str): optional y-label of the trace.
        trace_mode (str): 'center' for central hexal or 'sum' for sum over hexals.
            Defaults to 'center'.
        ylims (List[tuple]): static y-limits for the trace for each sample.
        ax (Axis): existing Axis instance or None.
        batch_sample (int): batch sample to start from. Defaults to 0.
        cmap (colormap): colormap for the hex-scatter. Defaults to
            cm.get_cmap("binary_r") (greyscale).
        edgecolor (str): edgecolor for the hexals. Defaults to "k" displaying
            edges.
        update (bool): whether to update the canvas after an animation step.
            Must be False if this animation is composed with others.
            Defaults to False.
        label (str): label of the animation. Defaults to 'Sample: {}\nFrame:{}',
            which is formatted with the current sample and frame number per frame.
        labelxy (tuple): normalized x and y location of the label. Defaults to
            (0, 1), i.e. top-left corner.
        fontsize (float): fontsize. Defaults to 10.

    #TODO: this could simply be a general HexScatter animation plus trace.
    """

    def __init__(
        self,
        activation,
        dt=1,
        cranges=None,
        vmin=None,
        vmax=None,
        fig=None,
        figsize=[3, 4],
        ratio=0.3,
        trace_color=None,
        dynamic_ax_lims=True,
        contour=None,
        title="",
        ylabel="",
        trace_mode="center",
        ylims=None,
        ax=None,
        batch_sample=0,
        cmap=cm.get_cmap("binary_r"),
        update=False,
        edgecolor="k",
        label="Sample: {}\nFrame: {}",
        labelxy=(0.0, 1.0),
        fontsize=10,
    ):

        self.n_samples, self.frames = activation.shape[:-1]

        if fig is None and ax is None:
            self.fig, self.ax = plt_utils.init_plot(figsize=figsize)
        else:
            self.fig = fig
            self.ax = ax

        fr = Fraction(str(ratio))
        matrix = np.zeros(fr.denominator).reshape(-1, 1)
        matrix[-fr.numerator :] = 1
        self.axes = plt_utils.divide_axis_to_grid(self.ax, matrix=matrix, hspace=0)

        animations = []
        animations.append(
            HexScatter(
                activation,
                vmin=vmin,
                vmax=vmax,
                cranges=cranges,
                labelxy=labelxy,
                label=label,
                fontsize=fontsize,
                update=update,
                title=title,
                cmap=cmap,
                edgecolor=edgecolor,
                fig=self.fig,
                ax=self.axes[0],
            )
        )

        if trace_mode == "center":
            n_hexals = activation.shape[-1]
            center = n_hexals // 2
            trace = activation[:, :, center]
        elif trace_mode == "sum":
            trace = activation.sum(axis=-1)
        else:
            raise ValueError(f"{trace_mode} invalid")
        animations.append(
            Trace(
                trace,
                dt=dt,
                fig=self.fig,
                ax=self.axes[1],
                update=False,
                color=trace_color,
                batch_sample=batch_sample,
                dynamic_ax_lims=dynamic_ax_lims,
                ylims=ylims,
                contour=contour,
                ylabel=ylabel,
                label="",
                fontsize=fontsize,
            )
        )
        self.animations = animations
        self.update = False
        self.batch_sample = batch_sample
        super().__init__(None, self.fig)


class ActivationGrid(AnimationCollector):
    """Hex-scatter animation grid for displaying the input, source and target
        activations for a particular target neuron type.

    Args:
        input_sources_target_activities (dict):
             containing 'input': array
                activities
                        'target': array
                        <source_type>: array
                        'summed_activity': array
                coordinates
                        'u': array
                        'v': array
        target (str): target neuron type.
        time_constatns (dicts): trained time constant.
        batch_sample (int): batch sample to start from. Defaults to 0.
    """

    def __init__(
        self,
        input_sources_target_activities,
        target,
        time_constants={},
        batch_sample=0,
    ):
        ista = input_sources_target_activities
        self.n_samples, self.frames = ista["input"].shape[:2]
        self.fig, self.axes, _ = plt_utils.get_axis_grid(range(len(ista) - 3))
        animations = []
        animations.append(
            HexScatter(
                ista["input"],
                vmin=0,
                vmax=1,
                fig=self.fig,
                ax=self.axes[0],
                title="input",
            )
        )
        tau = time_constants.get(target, "")
        animations.append(
            HexScatter(
                ista["target"],
                cranges=np.abs(ista["target"]).max(axis=(1, 2)),
                cmap=cm.get_cmap("seismic"),
                label=f"τ={tau*1000:.2G}ms" if tau else "",
                fig=self.fig,
                ax=self.axes[1],
                title=f"target act.: {target}",
            )
        )
        animations.append(
            HexScatter(
                ista["summed_activity"],
                cranges=np.abs(ista["summed_activity"]).max(axis=(1, 2)),
                cmap=cm.get_cmap("seismic"),
                label="",
                fig=self.fig,
                ax=self.axes[2],
                title=f"sum source act.",
            )
        )
        source_types = set(ista.keys()).difference(
            {"input", "target", "crange", "u", "v", "summed_activity"}
        )
        for i, source_type in enumerate(
            set(input_sources_target_activities.keys()).difference(
                {"input", "target", "crange", "u", "v", "summed_activity"}
            )
        ):
            tau = time_constants.get(source_type, "")
            animations.append(
                HexScatter(
                    ista[source_type],
                    cranges=ista["crange"],
                    cmap=cm.get_cmap("seismic"),
                    label=f"τ={tau*1000:.2G}ms" if tau else "",
                    fig=self.fig,
                    ax=self.axes[3 + i],
                    title=f"source act.: {source_type}",
                )
            )
        self.animations = animations
        self.update = False
        self.batch_sample = batch_sample
        super().__init__(None, self.fig)


class ActivationGridPlusTraces(AnimationCollector):
    """Hex-scatter animation grid for displaying the input, source and target
        activations for a particular target neuron type.

    Args:
        input_sources_target_activities (dict):
             containing 'input': array
                activities
                        'target': array
                        <source_type>: array
                        'summed_activity': array
                coordinates
                        'u': array
                        'v': array
        target (str): target neuron type.
        time_constatns (dicts): trained time constant.
        figsize (list): size of initialied figure if no fig instance given.
            Defaults to [22, 13].
        path (Path): path object to save animation to.
        batch_sample (int): batch sample to start from. Defaults to 0.
        fontsize (float): fontsize. Defaults to 10.
    """

    def __init__(
        self,
        input_sources_target_activities,
        dt=1,
        figsize=[22, 13],
        batch_sample=0,
        fontsize=9,
    ):
        ista = input_sources_target_activities

        target = ista["target_type"]
        time_constants = ista["time_const"]
        # ylims = np.array([(-lim, lim) for lim in ista["crange"]])

        ylims = plt_utils.node_type_collection_ax_lims_per_batch(
            {key: value.sum(-1) for key, value in ista["source_current"].items()},
            offset=0.2,
        )

        contour = ista["input"][:, :, ista["input"].shape[-1] // 2]

        self.n_samples, self.frames = ista["input"].shape[:2]
        num_axes = len(ista["source_current"]) + 3
        gridheight = 3
        gridwidth = int(np.ceil(num_axes / gridheight))
        self.fig, self.axes, _ = plt_utils.get_axis_grid(
            gridheight=gridheight,
            gridwidth=gridwidth,
            figsize=figsize,
            hspace=0.4,
        )

        for ax in self.axes[-3:]:
            plt_utils.rm_spines(
                ax,
                spines=("top", "right", "bottom", "left"),
                visible=False,
                rm_xticks=True,
                rm_yticks=True,
            )

        animations = []
        animations.append(
            ActivationPlusTrace(
                ista["input"],
                dt=dt,
                vmin=0,
                vmax=1,
                fig=self.fig,
                ax=self.axes[0],
                title="input",
                ylabel="center intensity",
                fontsize=fontsize,
            )
        )
        tau = time_constants.get(target, "")
        animations.append(
            ActivationPlusTrace(
                ista["target"],
                dt=dt,
                cranges=np.abs(ista["target"]).max(axis=(1, 2)),
                cmap=cm.get_cmap("seismic"),
                label=f"τ={tau*1000:.2G}ms" if tau else "",
                fig=self.fig,
                ax=self.axes[1],
                title=f"target act.: {target}",
                ylabel="central activation",
                fontsize=fontsize,
                trace_mode="center",
                contour=contour
                # ylims=ylims,
            )
        )
        animations.append(
            ActivationPlusTrace(
                ista["summed_activity"],
                dt=dt,
                cranges=np.abs(ista["summed_activity"]).max(axis=(1, 2)),
                cmap=cm.get_cmap("seismic"),
                label="",
                fig=self.fig,
                ax=self.axes[2],
                title=f"source currents",
                ylabel="summed currents",
                fontsize=fontsize,
                trace_mode="sum",
                ylims=ylims,
                contour=contour,
            )
        )

        source_types = set(ista["source_current"].keys())
        ordered_source_types, _ = utils.order_nodes_list(source_types)
        for i, source_type in enumerate(ordered_source_types):
            tau = time_constants.get(source_type, "")
            animations.append(
                ActivationPlusTrace(
                    ista["source_current"][source_type],
                    dt=dt,
                    cranges=ista["crange"],
                    cmap=cm.get_cmap("seismic"),
                    label="",
                    fig=self.fig,
                    ax=self.axes[3 + i],
                    title=f"src currents: {source_type}" + f" (τ={tau*1000:.2G}ms)"
                    if tau
                    else "",
                    ylabel="summed currents",
                    fontsize=fontsize,
                    trace_mode="sum",
                    ylims=ylims,
                    contour=contour,
                )
            )
        self.animations = animations
        self.update = False
        self.batch_sample = batch_sample
        super().__init__(ista["path"], self.fig)


class ActivationGridPlusTraces_v2(AnimationCollector):
    """Hex-scatter animation grid for displaying the input, source and target
        activations for a particular target neuron type.

    Args:
        stimulus: (#samples, #frames, #hexals)
        source_currents: postsynaptic currents per source. Dict[str, array (#samples, #frames, #hexals (RFs))]
        target_currents: postsynaptic current of target. array (#samples, #frames, #hexals)
        responses: response of target. array (#samples, #frames, #hexals)
    """

    def __init__(
        self,
        target_type,
        stimulus,
        source_currents,  # source: w * relu(sources.activities)
        target_currents,  # sum_sources(w * relu(sources.activities))
        responses,  # exp(- sum_source(w * relu(sources.activities))/ tau) + bias
        dt=1,
        figsize=[22, 13],
        batch_sample=0,
        fontsize=9,
        gridheight=3,
        hspace=0.4,
        wspace=0.1,
        trace_vertical_ratio=0.3,
        path=None,
    ):

        global_trace_ylims = plt_utils.node_type_collection_ax_lims_per_batch(
            {
                **{key: value.sum(-1) for key, value in source_currents.items()},
                **{"target_currents": target_currents.sum(-1)},
            },
            offset=0.1,
        )

        global_cmap_lims = np.abs(
            plt_utils.node_type_collection_ax_lims_per_batch(
                source_currents, offset=0.05
            )
        ).max(axis=1)

        contour = stimulus[:, :, stimulus.shape[-1] // 2]

        self.n_samples, self.frames = stimulus.shape[:2]
        # to also plot the stimulus, the target responses and the summed inputs
        num_axes = len(source_currents) + 3
        gridwidth = int(np.ceil(num_axes / gridheight))
        self.fig, self.axes, _ = plt_utils.get_axis_grid(
            gridheight=gridheight,
            gridwidth=gridwidth,
            figsize=figsize,
            hspace=hspace,
            wspace=wspace,
        )

        #         # to remove all spines except for the stimulus, target, and summed inputs
        #         for ax in self.axes[-3:]:
        #             plt_utils.rm_spines(ax, spines=('top', 'right', 'bottom', 'left'),
        #                                 visible=False, rm_xticks=True, rm_yticks=True)

        animations = []
        animations.append(
            ActivationPlusTrace(
                stimulus,
                dt=dt,
                vmin=0,
                vmax=1,
                fig=self.fig,
                ax=self.axes[0],
                title="stimulus",
                ylabel="center intensity (a.u.)",
                fontsize=fontsize,
                labelxy=(-0.25, 1.15),
                ratio=trace_vertical_ratio,
                label="",
            )
        )
        # to normalize colors over time and hexals making the magnitudes
        # comparable within stimuli samples
        # midpoint is 0
        responses_cranges = np.abs(responses).max(axis=(1, 2))
        animations.append(
            ActivationPlusTrace(
                responses,
                dt=dt,
                cranges=responses_cranges,
                cmap=cm.get_cmap("seismic"),
                label="",
                fig=self.fig,
                ax=self.axes[1],
                title=f"{target_type} responses",
                ylabel="center voltage (a.u.)",
                fontsize=fontsize,
                trace_mode="center",
                contour=contour,
                # ylims=ylims,,
                ratio=trace_vertical_ratio,
            )
        )
        target_currents_crange = np.abs(target_currents).max(axis=(1, 2))
        animations.append(
            ActivationPlusTrace(
                target_currents,
                dt=dt,
                cranges=target_currents_crange,
                cmap=cm.get_cmap("seismic"),
                label="",
                fig=self.fig,
                ax=self.axes[2],
                title=f"summed postsynaptic currents\n(home column)",
                ylabel="summed currents (a.u.)",
                fontsize=fontsize,
                trace_mode="sum",
                ylims=global_trace_ylims,
                contour=contour,
                ratio=trace_vertical_ratio,
            )
        )

        source_types = set(source_currents.keys())
        ordered_source_types, _ = utils.order_nodes_list(source_types)
        for i, source_type in enumerate(ordered_source_types):
            source_current = source_currents[source_type]
            #             source_crange = np.abs(source_current).max(axis=(1, 2))
            animations.append(
                ActivationPlusTrace(
                    source_current,
                    dt=dt,
                    cranges=global_cmap_lims,
                    cmap=cm.get_cmap("seismic"),
                    label="",
                    fig=self.fig,
                    ax=self.axes[3 + i],
                    title=f"postsynaptic current {source_type}",
                    ylabel="summed currents (a.u.)",
                    fontsize=fontsize,
                    trace_mode="sum",
                    ylims=global_trace_ylims,
                    contour=contour,
                    ratio=trace_vertical_ratio,
                )
            )
        for ax in self.axes[3 + i :]:
            ax.remove()
        self.animations = animations
        self.update = False
        self.batch_sample = batch_sample
        super().__init__(path, self.fig)


class CentralActivity(Animation):
    """Bar chart of the activity of all central nodes.

    Args:
        tnn (DataWrap): trained network datawrap instance.
        batch_type (str): displayed batch, assuming that recordings for it exist.
            E.g. 'tracked_train_batch', 'validation', 'full_val'.
        fig (Figure): existing Figure instance or None.
        ax (Axis): existing Axis instance or None.
        batch_sample (int): batch sample to start from. Defaults to 0.
        update (bool): whether to update the canvas after an animation step.
            Must be False if this animation is composed with others.
            Defaults to False.
        label (str): label of the animation. Defaults to 'Sample: {}\nFrame:{}',
            which is formatted with the current sample and frame number per frame.
        labelxy (tuple): normalized x and y location of the label. Defaults to
            (0, 1), i.e. top-left corner.

    Kwargs:
        passed to ~dvs.plots.plots.activity_bars.


    Note: activity is assumed to be stored as h5 file in the Datawrap under
        tnn[batch_type].network_states.nodes.activity.

    TODO: can be deprecated in favor of more generic Bars animation.
    """

    def __init__(
        self,
        tnn,
        batch_type="tracked_train_batch",
        fig=None,
        ax=None,
        batch_sample=0,
        update=False,
        label="Sample: {}\nFrame: {}",
        labelxy=(0, 1),
        **kwargs,
    ):
        super().__init__()
        self.fig = fig
        self.ax = ax
        self.tnn = tnn
        self.batch_sample = batch_sample
        self.kwargs = kwargs
        self.update = update
        self.label = label
        self.labelxy = labelxy
        self.label_text = None
        self.batch_type = batch_type
        self.activity = self.tnn[self.batch_type].network_states.nodes.activity[:]
        self.n_samples, self.frames = self.activity.shape[0:2]

        self.central_nodes = None
        path = tnn.path if tnn is not None else None
        super().__init__(path, self.fig)

    def init(self, frame=0):
        nodes = utils.order_nodes(self.tnn.ctome.nodes.to_df())
        self.central_nodes = nodes[(nodes.u == 0) & (nodes.v == 0)]
        activity = self.activity[
            self.batch_sample, frame, self.central_nodes.index
        ].squeeze()
        self.fig, self.ax, self.label_text = plots.activity_bars(
            self.central_nodes.type,
            activity,
            fig=self.fig,
            ax=self.ax,
            labelxy=self.labelxy,
            label=self.label.format(self.batch_sample, frame),
            **self.kwargs,
        )

    def animate(self, frame):
        texts = self.ax.texts

        activity = self.activity[
            self.batch_sample, frame, self.central_nodes.index
        ].squeeze()
        invalid = np.ma.masked_invalid(activity).mask
        activity[invalid] = 0.0

        for i, a in enumerate(activity):
            self.ax.patches[i].set_height(a)
            if invalid[i]:
                texts[i].set_text("NaN")
            else:
                texts[i].set_text("")

        self.ax.set_ylim(*plt_utils.get_lims(activity, 0.1))

        if self.label:
            self.label_text.set_text(self.label.format(self.batch_sample, frame))

        if self.update:
            self.update_figure()


class InputHexScatter(AnimationCollector):
    """Hex-scatter animations for input and single activation.

    Args:
        input (tensor): hexagonal input of shape (#samples, #frames, #hexals).
        activation (tensor): hexagonal activation of particular neuron type
            (#samples, #frames, #hexals).
        crange_input (tuple): optional color range of the input.
        batch_sample (int): batch sample to start from. Defaults to 0.
        cbound (float): absolute color bound for the activation.
    """

    def __init__(
        self,
        input,
        activation,
        crange_input=(None, None),
        batch_sample=0,
        cbound=0,
    ):
        self.input = input
        self.activation = activation
        self.crange_input = crange_input
        self.update = False
        self.n_samples, self.frames = self.activation.shape[:2]
        self.fig, self.axes, (gw, gh) = plt_utils.get_axis_grid(
            range(2), figsize=[9, 3]
        )
        animations = []
        input = utils.to_numpy(self.input)
        if not all([r is None for r in self.crange_input]):
            vmin, vmax = self.crange_input
        else:
            vmin, vmax = 0, np.std(input) * 3
        animations.append(
            HexScatter(
                input,
                vmin=vmin,
                vmax=vmax,
                edgecolor="k",
                fig=self.fig,
                ax=self.axes[0],
                title="input",
                labelxy=(-0.1, 1),
            )
        )

        activation = utils.to_numpy(self.activation)

        self.cbound = cbound
        if self.cbound != 0:
            cbound = np.abs(activation).max()
            vmin, vmax = -cbound, cbound
        else:
            vmin, vmax = -self.cbound, self.cbound

        animations.append(
            HexScatter(
                self.activation,
                vmin=vmin,
                vmax=vmax,
                fig=self.fig,
                ax=self.axes[1],
                cmap=cm.get_cmap("seismic"),
                title="activation",
                label="",
            )
        )

        self.animations = animations
        self.batch_sample = batch_sample
        super().__init__(None, self.fig)


class LayerActivityGrid(Animation):
    """Hex-scatter animations for activation arranged on a network graph layout.

    Note: either initialized with a trained network datawrap instance and
        a batch_type and optional activity_type or with a ready-made activity
        array.

    Args:
        tnn (DataWrap): trained network datawrap instance.
        activity (array): activation of all neurons
            of shape (#samples, #frames, #neurons).
        activity_type (str): activity_type allows to refer to differently named
            activity arrays belonging to the tnn. E.g. activity_argmax_T4.
        rectify (bool): whether to rectify the activity. Defaults to False.
        batch_type (str): displayed batch, assuming that recordings for it exist.
            E.g. 'tracked_train_batch', 'validation', 'full_val', 'gratings', 'flashes'.
        ax_label (str): label displayed next to each ax. Defaults to
            "<neuron_type>, σ/μ: <val>".
        batch_sample (int): batch sample to start from. Defaults to 0.
        update (bool): whether to update the canvas after an animation step.
            Must be False if this animation is composed with others.
            Defaults to False.
        color_norm_per (str): how to normalize the colors.
            One of 'video', 'batch' or 'frame'.
        figsize (list): size of initialied figure if no fig instance given.
            Defaults to [20, 10].
        label (str): label of the animation. Defaults to 'Sample: {}\nFrame:{}',
            which is formatted with the current sample and frame number per frame.
        cmap (colormap): colormap for the hex-scatter. Defaults to
            cm.get_cmap("binary_r") (greyscale).
        hspace (float): height-space between axes.
        wspace (float): width-space between axes.
        region_spacing (float): see ~dvs.plots.plots._network_graph_ax_scatter.
        labelxy (tuple): normalized x and y location of the label. Defaults to
            (0, 1), i.e. top-left corner.
        titlepad (float): ax title padding. #TODO: deprecated?
        fontsize (float): fontsize. Defaults to 10.
        fig (Figure): existing Figure instance or None.
    """

    def __init__(
        self,
        ctome,
        activity,
        rectify=False,
        batch_type="validation",
        ax_label="{}",
        batch_sample=0,
        update=False,
        color_norm_per="video",
        figsize=[5, 2],
        label="Sample: {}\nFrame: {}",
        cmap=cm.get_cmap("binary_r"),
        hspace=0.0,
        wspace=0.0,
        region_spacing=2,
        labelxy=(0, 0.9),
        titlepad=1,
        fontsize=5,
        fig=None,
        **kwargs,
    ):
        plt.rc("axes", titlepad=titlepad)
        self.ctome = ctome
        self.batch_sample = batch_sample
        self.kwargs = kwargs
        self.update = update
        self.label = label
        self.labelxy = labelxy
        self.fontsize = fontsize
        self.cmap = cmap
        self.batch_type = batch_type
        self.n_samples, self.frames = activity.shape[:2]
        self.ax_label = ax_label
        self.rectify = rectify

        self.nodes = self.ctome.nodes.to_df()
        self.edges = self.ctome.edges.to_df()
        self.layout = utils.nodes_edges_utils.layout

        self.neuron_types = utils.nodes_edges_utils.order_nodes_list(
            ctome.unique_node_types[:].astype(str)
        )[0]
        self.fig, self.axes = plots._network_graph_ax_scatter(
            self.neuron_types,
            self.nodes,
            self.edges,
            self.layout,
            figsize=figsize,
            fig=fig,
            region_spacing=region_spacing,
            wspace=wspace,
            hspace=hspace,
        )

        self.activity = utils.activity_utils.LayerActivity(
            activity,
            ctome,
            keepref=True,
        )

        if color_norm_per == "video":
            self.sms = np.zeros([len(self.neuron_types)], dtype=object)
            for j, nt in enumerate(self.neuron_types):
                vmin, vmax = self.activity[nt].min(), self.activity[nt].max()
                self.sms[j] = plt_utils.get_scalarmapper(
                    cmap=cmap, vmin=vmin, vmax=vmax
                )[0]

        if color_norm_per == "batch":
            # Store one scalarmapper per batch samples and node type.
            self.sms = np.zeros([self.n_samples, len(self.neuron_types)], dtype=object)
            for i in range(self.n_samples):
                for j, nt in enumerate(self.neuron_types):
                    vmin, vmax = (
                        self.activity[nt][i].min(),
                        self.activity[nt][i].max(),
                    )
                    self.sms[i, j] = plt_utils.get_scalarmapper(
                        cmap=cmap, vmin=vmin, vmax=vmax
                    )[0]

        self.color_norm_per = color_norm_per
        path = None
        super().__init__(path, self.fig)

    def init(self, frame=0):
        self.labels = []
        for i, nt in enumerate(self.neuron_types):
            nodes = self.nodes[self.nodes.type == nt]
            u, v = nodes.u.values, nodes.v.values
            activity = self.activity[nt][self.batch_sample, frame]
            if self.rectify:
                activity = np.maximum(activity, 0)
            std, mean = activity.std(), activity.mean()
            cv = std / mean if len(activity) * np.abs(mean) > 1e-6 else 0

            if self.color_norm_per == "video":
                sm = self.sms[i]
            elif self.color_norm_per == "frame":
                vmin, vmax = activity.min(), activity.max()
                sm = plt_utils.get_scalarmapper(cmap=self.cmap, vmin=vmin, vmax=vmax)[0]
            elif self.color_norm_per == "batch":
                sm = self.sms[self.batch_sample, i]
            else:
                sm = None

            _, _, (label, _) = plots.hex_scatter(
                u,
                v,
                activity,
                fill=True,
                alpha=1,
                labelxy=self.labelxy,
                fig=self.fig,
                ax=self.axes[i],
                cmap=self.cmap,
                scalarmapper=sm,
                label=self.ax_label.format(nt, cv),
                cbar=False,
                fontsize=self.fontsize,
                **self.kwargs,
            )
            # self.sms.append(sm)
            self.labels.append(label)

        self.fig_label = self.fig.text(
            self.labelxy[0],
            self.labelxy[1],
            self.label.format(self.batch_sample, frame),
            transform=self.fig.transFigure,
            ha="left",
            va="center",
            fontsize=self.fontsize,
        )

    def animate(self, frame):
        for i, nt in enumerate(self.neuron_types):
            activity = self.activity[nt][self.batch_sample, frame]
            if self.rectify:
                activity = np.maximum(activity, 0)
            std, mean = activity.std(), activity.mean()
            cv = std / mean if len(activity) * np.abs(mean) > 1e-6 else 0

            if self.color_norm_per == "video":
                sm = self.sms[i]
            elif self.color_norm_per == "frame":
                vmin, vmax = activity.min(), activity.max()
                sm = plt_utils.get_scalarmapper(cmap=self.cmap, vmin=vmin, vmax=vmax)[0]
            elif self.color_norm_per == "batch":
                sm = self.sms[self.batch_sample, i]
            else:
                sm = None

            activity_rgba = sm.to_rgba(activity)
            for j, patch in enumerate(
                self.axes[i].patches[self.axes[i]._valid_patches_start_index :]
            ):
                patch.set_color(activity_rgba[j])
            self.labels[i].set_text(self.ax_label.format(nt, cv))

        if self.label:
            self.fig_label.set_text(self.label.format(self.batch_sample, frame))

        if self.update:
            self.update_figure()
