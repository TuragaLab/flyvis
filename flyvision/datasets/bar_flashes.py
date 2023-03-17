# ---- TwoBarFlashes


@root(root_dir)
class TwoBarFlashWrap(Directory):
    class Config:
        boxfilter: dict
        t_stim = ([0.04, 0.160],)
        t_pre = (0.5,)
        dt = (1 / 76,)
        angles: list
        offsets = ([-3, -2, -1, 1, 2, 3],)
        widths = ([1, 2, 3, 4],)
        dynamic_range = ([0, 1],)
        filter_type: str = "median"
        hex_sample: bool = True

    def __init__(self, config: Config):
        boxfilter = BoxEye(config.boxfilter)

        dynamic_range = np.array(config.dynamic_range)
        background = dynamic_range.sum() / 2
        intensities = dynamic_range.tolist()

        offsets = list(zip(config.offsets[:-1], config.offsets[1:]))
        samples = dict(
            t_stim=config.t_stim,
            angle=config.angles,
            stim_width=config.widths,
            offsets=offsets,
            intensities=intensities,
        )

        values = list(product(*(v for v in samples.values())))
        sequence = []  # samples, n_frames, width, height
        for t_stim, angle, width, (os_start, os_end), intensity in tqdm(
            values, desc="TwoBarFlash"
        ):
            sequence.append(
                get_bar_flash_seq(
                    boxfilter,
                    offsets=[os_start, os_end],
                    t_stim=t_stim,
                    t_pre=config.t_pre,
                    dt=config.dt,
                    hex_sample=True,
                    angle=angle,
                    n_bars=1,
                    stim_width=width,
                    background=0.5,
                    bar=intensity,
                    ftype=config.filter_type,
                )
            )

        self.flashes = np.array(sequence)


class TwoBarFlashes(SequenceDataset):
    """
    #TODO: docstring
    """

    augment = False
    n_sequences = 0
    dt = None
    framerate = None
    t_pre = 0.0
    t_post = 0.0

    def __init__(
        self,
        boxfilter=dict(extent=15, kernel_size=13),
        angles=list(range(0, 360, 30)),
        offsets=[-3, -2, -1, 1, 2, 3],
        widths=[1, 2, 3, 4],
        background=0.5,
        intensities=[0, 1],
        t_stim=[0.04, 0.160],
        t_pre=0.5,
        dt=1 / 76,
        tnn=None,
    ):
        self.boxfilter = BoxEye(boxfilter)
        self.dt = dt
        self.background = background

        offsets = list(zip(offsets[:-1], offsets[1:]))
        self.samples = Namespace(
            angle=angles,
            offsets=offsets,
            stim_width=widths,
            intensities=intensities,
            t_stim=t_stim,
        )

        self.values = np.array(list(product(*(v for v in self.samples.values()))))

        self.theta = np.array(self.samples.angle)
        self.tnn = tnn
        # if self.tnn is not None:
        #     self.config = tnn[chkpt_type].config
        # else:
        #     self.config = self.gratings_wrap.config

    def get(self, angle, offset, width, intensity, t_stim):
        """
        #TODO: docstring
        """
        # breakpoint()
        return get_bar_flash_seq(
            self.boxfilter,
            offsets=offset,
            t_stim=t_stim,
            t_pre=0.5,
            dt=self.dt,
            hex_sample=True,
            angle=angle,
            n_bars=1,
            stim_width=width,
            background=self.background,
            bar=intensity,
            ftype="median",
        )

    def arg_df(self):
        keys = self.samples.keys()
        values = self.values
        return pd.DataFrame(values, columns=keys)

    def __len__(self):
        return len(self.values)

    def item(self, key):
        """
        #TODO: docstring
        """
        return eval(
            "dict(angle={}, offset={}, width={}, intensity={}, t_stim={})".format(
                *self.values[key]
            )
        )

    def get_item(self, key):
        """
        #TODO: docstring
        """
        return torch.Tensor(self.get(*self.values[key]))

    def __repr__(self):
        return repr(self.samples)


def get_bar_flash(
    boxfilter,
    hex_sample=True,
    angle=0,
    n_bars=1,
    stim_width=1,
    background=0,
    bar=1,
    offset=0,
    stim_height=None,
    non_overlapping=False,
    ftype="median",
):
    """Returns single bar flash offset by 'offset' columns.

    Args:
        boxfilter (HexBoxFilter)
        hex_sample (bool): whether to return cartesian or hexagonal gratings. Defaults to True.
        stim_width (int): width of the bars in receptor columns.
        angle (int): 0 to 360 degree.
        background (float): value of the background. 1 is white, 0 is black.
        bar (float): value of the bar. 1 is white, 0 is black.
        n_bars (int): number of bars.
        offset (int): offset from center. Columnar if 'non_overlapping=False',
            else in units of the stimulus width.
        stim_height (int): height of the bars in receptor columns.
        non_overlapping (bool): whether offset is in units of columns or of
            stimulus width. Defaults to True, i.e. columns.

    Returns:
        (array): shape n_hexals
    """
    orientation_angle = angle % 180
    col_extent = boxfilter.conf.extent

    assert stim_width * boxfilter.conf.kernel_size >= 1

    # Periodicity allows offsets outisde of the lattice.
    if non_overlapping is True:
        # Periodic on the grid divided by the stimulus width.
        offset += col_extent // stim_width
        offset %= 2 * (col_extent // stim_width) + 1
        offset -= col_extent // stim_width
    else:
        # Periodic on the columnar grid per columns.
        offset += col_extent
        offset %= 2 * col_extent + 1
        offset -= col_extent

    def rotate_sample(stim):
        #  Rotate bars.
        cartesian = Image.fromarray((255 * stim).astype("uint8"))
        cartesian = cartesian.rotate(angle, Image.NEAREST, False, None)
        cartesian = np.array(cartesian).astype(float) / 255.0
        if not hex_sample:
            return cartesian
        return boxfilter.sample(cartesian, ftype=ftype)

    min_frame_size = (
        boxfilter.min_frame_size.cpu().numpy()
        if isinstance(boxfilter.min_frame_size, torch.Tensor)
        else boxfilter.min_frame_size
    )

    # # The kernel size in x is 2 pixels more than that in y.
    # # Taking the bigger one to reduce artifacts.
    # y, x = np.array(list(boxfilter._receptor_centers())).T

    # if orientation_angle <= 45 or orientation_angle >= 135:
    #     kernel_size = int(x[len(x)//2+1] * 2)
    # else:
    #     kernel_size = int(y[-2])
    kernel_size = boxfilter.conf.kernel_size

    # Raw stimuli dimensions.
    padding = (
        11 * kernel_size,
        11 * kernel_size,
    )  # An even multiplicator leads to artifacts.
    height, width = min_frame_size + padding

    # Bar properties.
    #  Width in pixels.
    bar_width = np.floor(stim_width * kernel_size).astype(int)
    bar_height = (
        height
        if stim_height is None
        else np.floor(stim_height * kernel_size).astype(int)
    )
    height_slice = slice(int((height - bar_height) / 2), int((height + bar_height) / 2))
    #  Regular spacing between bars.
    bar_spacing = int(round(width / n_bars - bar_width))

    #  Horizontal center in pixel coords.
    hcenter = width // 2
    # Init raw stimuli.
    _stim = np.ones([height, width]) * background
    for i in range(n_bars):
        #  Fill background with bars.
        width_slice = slice(
            i * bar_width + i * bar_spacing, (i + 1) * bar_width + i * bar_spacing
        )
        _stim[height_slice, width_slice] = bar  # intensity value
    #  Cut off after last bar spacing so that there's no overhang.
    _stim = _stim[:, 0 : n_bars * bar_width + n_bars * bar_spacing]
    #  Move the first bar to the center minus one stimulus widths.
    if non_overlapping:
        _loc = hcenter + (offset * stim_width - stim_width // 2) * kernel_size
    else:
        _loc = hcenter + (offset - stim_width // 2) * kernel_size
    _loc -= _loc % kernel_size
    _stim = np.roll(_stim, int(_loc), axis=1)
    sample = rotate_sample(_stim)

    return sample


# def get_cartesian_bars(img_height, img_width, bar_width, bar_height, bar_spacing,
#                        n_bars, bar_intensity, background_intensity):
#     height_slice = slice(int((img_height - bar_height) / 2),
#                          int((img_height + bar_height) / 2))

#     img = np.ones([img_height, img_width]) * background
#     for i in range(n_bars):
#         #  Fill background with bars.
#         width_slice = slice(i * bar_width + i * bar_spacing,
#                             (i + 1) * bar_width + i * bar_spacing)
#         img[height_slice, width_slice] = bar
#         #  Cut off after last bar spacing so that there's no overhang.
#         img = img[:, 0:n_bars * bar_width + n_bars * bar_spacing]


def get_bar_flash_seq(
    boxfilter,
    offsets,
    t_stim=0.16,
    t_pre=0.5,
    dt=1 / 76,
    hex_sample=True,
    angle=0,
    n_bars=1,
    stim_width=1,
    stim_height=None,
    background=0.5,
    non_overlapping=False,
    t_between=0,
    t_post=0,
    bar=1,
    ftype="median",
):
    pre_frames = int(round(t_pre / dt))
    stim_frames = int(round(t_stim / dt))
    between_frames = int(round(t_between / dt))
    post_frames = int(round(t_post / dt))
    flashes = []
    if pre_frames:
        flashes.append(np.ones([pre_frames, boxfilter.hexals]) * background)
    for i, offset in enumerate(offsets):
        flash = get_bar_flash(
            boxfilter,
            hex_sample=hex_sample,
            angle=angle,
            n_bars=n_bars,
            stim_width=stim_width,
            background=background,
            bar=bar,
            non_overlapping=non_overlapping,
            offset=offset,
            stim_height=stim_height,
            ftype=ftype,
        )
        flashes.append(flash[None].repeat(stim_frames, axis=0))
        if between_frames and i < len(offsets) - 1:
            flashes.append(np.ones([between_frames, boxfilter.hexals]) * background)
    if post_frames:
        flashes.append(np.ones([post_frames, boxfilter.hexals]) * background)
    return np.concatenate(flashes, axis=0)
