"""Functions to calculate figure sizes for plotting."""
from dataclasses import dataclass
import flyvision


def figsize_from_n_items(
    n_panels: int = None,
    max_figure_height_cm: float = 22,
    panel_height_cm: float = 3,
    max_figure_width_cm: float = 18,
    panel_width_cm: float = 3.6,
    dw_cm=0.1,
):
    # number of columns takes precedence
    n_columns = int(max_figure_width_cm / panel_width_cm)
    n_rows = 1
    while n_columns * n_rows < n_panels:
        n_rows += 1
    return fit_panel_size(
        n_rows,
        n_columns,
        max_figure_height_cm,
        panel_height_cm,
        max_figure_width_cm,
        panel_width_cm,
        dw_cm,
    )


@dataclass
class FigsizeCM:
    n_rows: int
    n_columns: int
    height: float
    width: float
    pad: float = 0.5

    @property
    def inches_wh(self):
        return cm_to_inch(
            self.width + self.pad, self.height + self.pad
        )

    @property
    def panel_height_cm(self):
        return self.height / self.n_rows

    @property
    def panel_width_cm(self):
        return self.width / self.n_columns

    def axis_grid(
        self,
        projection=None,
        as_matrix=False,
        fontsize=5,
        wspace=0.1,
        hspace=0.3,
        alpha=1,
        unmask_n=None,
    ):
        fig, axes, _ = flyvision.plots.plt_utils.get_axis_grid(
            gridwidth=self.n_columns,
            gridheight=self.n_rows,
            figsize=self.inches_wh,
            projection=projection,
            as_matrix=as_matrix,
            fontsize=fontsize,
            wspace=wspace,
            hspace=hspace,
            alpha=alpha,
            unmask_n=unmask_n,
        )
        return fig, axes


def figure_size_cm(
    n_panel_rows: int,
    n_panel_columns: int,
    max_figure_height_cm: float = 22,
    panel_height_cm: float = 3,
    max_figure_width_cm: float = 18,
    panel_width_cm: float = 3.6,
    allow_rearranging: bool = True,
):

    width = n_panel_columns * panel_width_cm
    height = n_panel_rows * panel_height_cm
    n_panels = n_panel_rows * n_panel_columns

    # decrease individual panel size
    if width > max_figure_width_cm and height > max_figure_height_cm:
        raise ValueError("not realizable under given size constraints")
    # change layout
    elif width > max_figure_width_cm and allow_rearranging:
        # decrease incrementally to stay close to original laytout
        n_panel_columns = n_panel_columns - 1
        while n_panel_columns * n_panel_rows < n_panels:
            n_panel_rows += 1
        return figure_size_cm(
            n_panel_rows=n_panel_rows,
            n_panel_columns=n_panel_columns,
            max_figure_height_cm=max_figure_height_cm,
            panel_height_cm=panel_height_cm,
            max_figure_width_cm=max_figure_width_cm,
            panel_width_cm=panel_width_cm,
        )
    elif height > max_figure_height_cm and allow_rearranging:
        # decrease incrementally to stay close to original laytout
        n_panel_rows = n_panel_rows - 1
        while n_panel_columns * n_panel_rows < n_panels:
            n_panel_columns += 1
        return figure_size_cm(
            n_panel_rows=n_panel_rows,
            n_panel_columns=n_panel_columns,
            max_figure_height_cm=max_figure_height_cm,
            panel_height_cm=panel_height_cm,
            max_figure_width_cm=max_figure_width_cm,
            panel_width_cm=panel_width_cm,
        )
    elif not allow_rearranging and (
        width > max_figure_width_cm or height > max_figure_height_cm
    ):
        raise ValueError("not realizable under given size constraints")

    return FigsizeCM(n_panel_rows, n_panel_columns, height, width)


def fit_panel_size(
    n_panel_rows: int,
    n_panel_columns: int,
    max_figure_height_cm: float = 22,
    panel_height_cm: float = 3,
    max_figure_width_cm: float = 18,
    panel_width_cm: float = 3.6,
    dw_cm=0.1,
    allow_rearranging=True,
):

    ratio = panel_width_cm / panel_height_cm

    try:
        return figure_size_cm(
            n_panel_rows,
            n_panel_columns,
            max_figure_height_cm,
            panel_height_cm,
            max_figure_width_cm,
            panel_width_cm,
            allow_rearranging=allow_rearranging,
        )
    except ValueError:
        new_panel_width_cm = panel_width_cm - dw_cm
        new_panel_height_cm = new_panel_width_cm / ratio
        return fit_panel_size(
            n_panel_rows,
            n_panel_columns,
            max_figure_height_cm,
            new_panel_height_cm,
            max_figure_width_cm,
            new_panel_width_cm,
            dw_cm,
            allow_rearranging=allow_rearranging,
        )


def cm_to_inch(*args):
    if len(args) == 1:
        width = args[0][0]
        height = args[0][1]
    elif len(args) == 2:
        width = args[0]
        height = args[1]
    return width / 2.54, height / 2.54