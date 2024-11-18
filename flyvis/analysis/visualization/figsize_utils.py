"""Functions to calculate figure sizes for plotting."""

from dataclasses import dataclass
from typing import Tuple, Union

from . import plt_utils


def figsize_from_n_items(
    n_panels: int,
    max_figure_height_cm: float = 22,
    panel_height_cm: float = 3,
    max_figure_width_cm: float = 18,
    panel_width_cm: float = 3.6,
    dw_cm: float = 0.1,
) -> "FigsizeCM":
    """
    Calculate figure size based on the number of panels.

    Args:
        n_panels: Number of panels in the figure.
        max_figure_height_cm: Maximum figure height in centimeters.
        panel_height_cm: Height of each panel in centimeters.
        max_figure_width_cm: Maximum figure width in centimeters.
        panel_width_cm: Width of each panel in centimeters.
        dw_cm: Decrement width in centimeters for panel size adjustment.

    Returns:
        FigsizeCM: Calculated figure size.
    """
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
    """
    Represents figure size in centimeters.

    Attributes:
        n_rows: Number of rows in the figure.
        n_columns: Number of columns in the figure.
        height: Height of the figure in centimeters.
        width: Width of the figure in centimeters.
        pad: Padding in centimeters.
    """

    n_rows: int
    n_columns: int
    height: float
    width: float
    pad: float = 0.5

    @property
    def inches_wh(self) -> Tuple[float, float]:
        """Convert width and height to inches."""
        return cm_to_inch(self.width + self.pad, self.height + self.pad)

    @property
    def panel_height_cm(self) -> float:
        """Calculate panel height in centimeters."""
        return self.height / self.n_rows

    @property
    def panel_width_cm(self) -> float:
        """Calculate panel width in centimeters."""
        return self.width / self.n_columns

    def axis_grid(
        self,
        projection: Union[str, None] = None,
        as_matrix: bool = False,
        fontsize: int = 5,
        wspace: float = 0.1,
        hspace: float = 0.3,
        alpha: float = 1,
        unmask_n: Union[int, None] = None,
    ) -> Tuple:
        """
        Create an axis grid for the figure.

        Args:
            projection: Type of projection for the axes.
            as_matrix: Whether to return axes as a matrix.
            fontsize: Font size for the axes.
            wspace: Width space between subplots.
            hspace: Height space between subplots.
            alpha: Alpha value for the axes.
            unmask_n: Number of axes to unmask.

        Returns:
            Tuple containing the figure and axes.
        """
        fig, axes, _ = plt_utils.get_axis_grid(
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
) -> FigsizeCM:
    """
    Calculate figure size in centimeters.

    Args:
        n_panel_rows: Number of panel rows.
        n_panel_columns: Number of panel columns.
        max_figure_height_cm: Maximum figure height in centimeters.
        panel_height_cm: Height of each panel in centimeters.
        max_figure_width_cm: Maximum figure width in centimeters.
        panel_width_cm: Width of each panel in centimeters.
        allow_rearranging: Whether to allow rearranging panels.

    Returns:
        FigsizeCM: Calculated figure size.

    Raises:
        ValueError: If the figure size is not realizable under given constraints.
    """
    width = n_panel_columns * panel_width_cm
    height = n_panel_rows * panel_height_cm
    n_panels = n_panel_rows * n_panel_columns

    if width > max_figure_width_cm and height > max_figure_height_cm:
        raise ValueError("Not realizable under given size constraints")
    elif width > max_figure_width_cm and allow_rearranging:
        n_panel_columns -= 1
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
        n_panel_rows -= 1
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
        raise ValueError("Not realizable under given size constraints")

    return FigsizeCM(n_panel_rows, n_panel_columns, height, width)


def fit_panel_size(
    n_panel_rows: int,
    n_panel_columns: int,
    max_figure_height_cm: float = 22,
    panel_height_cm: float = 3,
    max_figure_width_cm: float = 18,
    panel_width_cm: float = 3.6,
    dw_cm: float = 0.1,
    allow_rearranging: bool = True,
) -> FigsizeCM:
    """
    Fit panel size to figure constraints.

    Args:
        n_panel_rows: Number of panel rows.
        n_panel_columns: Number of panel columns.
        max_figure_height_cm: Maximum figure height in centimeters.
        panel_height_cm: Height of each panel in centimeters.
        max_figure_width_cm: Maximum figure width in centimeters.
        panel_width_cm: Width of each panel in centimeters.
        dw_cm: Decrement width in centimeters for panel size adjustment.
        allow_rearranging: Whether to allow rearranging panels.

    Returns:
        FigsizeCM: Fitted figure size.
    """
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


def cm_to_inch(*args: Union[Tuple[float, float], float]) -> Tuple[float, float]:
    """
    Convert centimeters to inches.

    Args:
        *args: Either a tuple of (width, height) or separate width and height values.

    Returns:
        Tuple of width and height in inches.
    """
    if len(args) == 1:
        width, height = args[0]
    elif len(args) == 2:
        width, height = args
    else:
        raise ValueError("Invalid number of arguments")
    return width / 2.54, height / 2.54
