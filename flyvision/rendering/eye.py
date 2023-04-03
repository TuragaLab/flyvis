"""'Transduction' of cartesian pixels to hexals on a regular hexagonal lattice.
"""
from typing import Iterator, Tuple

import numpy as np
from numpy.typing import NDArray

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from itertools import product

from flyvision.plots.plt_utils import init_plot, rm_spines
from flyvision.rendering.utils import median

__all__ = ["BoxEye"]

# ----- BoxEye -----------------------------------------------------------------


class BoxEye:
    """BoxFilter to produce an array of hexals matching the photoreceptor array.

    Args:
        extent: Radius, in number of receptors, of the hexagonal array
        kernel_size: Photon collection radius, in pixels

    Attributes:
        extent: Radius, in number of receptors, of the hexagonal array
        kernel_size: Photon collection radius, in pixels
        receptor_centers: Tensor of shape (hexals, 2) containing the y, x
            coordinates of the hexal centers
        hexals: Number of hexals in the array
        min_frame_size: Minimum frame size to contain the hexal array
        pad: Padding to apply to the frame before convolution
        conv (nn.Conv2d): Convolutional box filter to apply to the frame


    """

    def __init__(self, extent: int = 15, kernel_size: int = 13):
        self.extent = extent
        self.kernel_size = kernel_size
        self.receptor_centers = (
            torch.Tensor([*self._receptor_centers()]).long().to(torch.tensor(0).device)
        )
        self.hexals = len(self.receptor_centers)
        # The rest of kernel_size distance from outer centers to the border
        # is taken care of by the padding of the convolution object.
        self.min_frame_size = (
            self.receptor_centers.max(dim=0).values
            - self.receptor_centers.min(dim=0).values
            + 1
        )
        self._set_filter()

        pad = (self.kernel_size - 1) / 2
        self.pad = (
            int(np.ceil(pad)),
            int(np.floor(pad)),
            int(np.ceil(pad)),
            int(np.floor(pad)),
        )

    def _receptor_centers(self) -> Iterator[Tuple[float, float]]:
        """
        Generate receptive field centers for the population.

        Generates the hexagonal slice which scales with extent and pixels.
        """
        n = self.extent
        d = self.kernel_size
        for u in range(-n, n + 1):
            v_min = max(-n, -n - u)
            v_max = min(n, n - u)
            for v in range(v_min, v_max + 1):
                # y = -d * v
                # x = 2 / np.sqrt(3) * d * (u + v/2)
                y = d * (
                    u + v / 2
                )  # - d * v # either must be negative or origin must be upper
                x = d * v  # 2 / np.sqrt(3) * d * (u + v / 2)
                yield y, x
                # xs.append()
                # ys.append()

    def _set_filter(self) -> None:
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, padding=0)
        self.conv.weight.data /= self.conv.weight.data
        self.conv.bias.data.fill_(0)  # if not self.requires_grad else None
        self.conv.weight.requires_grad = False  # self.requires_grad
        self.conv.bias.requires_grad = False  # self.requires_grad

    def __call__(
        self, sequence: torch.Tensor, ftype: str = "mean", hex_sample: bool = True
    ) -> torch.Tensor:
        """Applies a box kernel to all frames in a sequence.

        Args:
            sequence: cartesian movie sequences of shape
                (samples, frames, height, width).
            ftype: filter type, 'mean', 'sum' or 'median'.
            hex_sample: if False, returns filtered cartesian sequences.
                Defaults to True.

        Returns
            torch.Tensor: shape (samples, frames, 1, hexals)
        """
        samples, frames, height, width = sequence.shape

        if not isinstance(sequence, torch.cuda.FloatTensor):
            # auto-moving to GPU in case default tensor is cuda but passed
            # sequence is not for convenience
            sequence = torch.Tensor(sequence)

        if (self.min_frame_size > torch.tensor([height, width])).any():
            # to rescale to the minimum frame size
            sequence = ttf.resize(sequence, self.min_frame_size.tolist())
            height, width = sequence.shape[2:]

        def _convolve():
            # convole each sample sequentially to avoid gpu memory issues
            def conv(x):
                return self.conv(x.unsqueeze(1))

            return torch.cat(
                tuple(map(conv, torch.unbind(F.pad(sequence, self.pad), dim=0))), dim=0
            )

        if ftype == "mean":
            out = _convolve() / self.kernel_size**2
        elif ftype == "sum":
            out = _convolve()
        elif ftype == "median":
            out = median(sequence, self.kernel_size)
        else:
            raise ValueError("ftype must be 'sum', 'mean', or 'median." f"Is {ftype}.")

        if hex_sample is True:
            return self.hex_sample(out).reshape(samples, frames, 1, -1)

        return out.reshape(samples, frames, height, width)

    def hex_sample(self, sequence: torch.Tensor) -> torch.Tensor:
        """Sample receptor locations from a sequence of cartesian frames.

        Args:
            sequence: cartesian movie sequences of shape (samples, frames, height, width).

        Returns:
            torch.Tensor: shape (samples, frames, 1, hexals)

        Note: resizes the sequence to the minimum frame size if necessary.
        """
        h, w = sequence.shape[2:]
        if (self.min_frame_size > torch.tensor([h, w])).any():
            sequence = ttf.resize(sequence, self.min_frame_size.tolist())
            h, w = sequence.shape[2:]
        c = self.receptor_centers + torch.tensor([h // 2, w // 2])
        out = sequence[:, :, c[:, 0], c[:, 1]]
        return out.view(*sequence.shape[:2], 1, -1)

    def sample(self, img, ftype="mean") -> torch.Tensor:
        """Sample individual frames.

        Args:
            img: a single frame of shape (height, width).
            ftype: filter type, 'mean', 'sum' or 'median'.

        Returns:
            tensor of shape (hexals,)
        """
        _type = np.asarray if isinstance(img, np.ndarray) else torch.Tensor
        _device = "cpu" if not isinstance(img, torch.cuda.FloatTensor) else "cuda"
        return _type(
            self(img[None, None, :, :], ftype=ftype, hex_sample=True)
            .squeeze()
            .to(_device)
        )

    def illustrate(
        self,
    ):
        """Illustrate the receptive field centers and the hexagonal sampling."""
        figsize = [2, 2]
        fontsize = 5
        y_hc, x_hc = np.array(list(self._receptor_centers())).T

        height, width = self.min_frame_size.cpu().numpy()
        x_img, y_img = np.array(
            list(
                product(
                    np.arange(-width / 2, width / 2),
                    np.arange(-height / 2, height / 2),
                )
            )
        ).T

        r = np.sqrt(2) * self.kernel_size / 2

        vertices = []
        angles = [45, 135, 225, 315, 405]
        for _y_c, _x_c in zip(y_hc, x_hc):
            _vertices = []
            for i, angle in enumerate(angles):
                offset = r * np.exp(np.radians(angle) * 1j)
                _vertices.append([_y_c + offset.real, _x_c + offset.imag])
            vertices.append(_vertices)
        vertices = np.transpose(vertices, (1, 2, 0))

        fig, ax = init_plot(figsize=figsize, fontsize=fontsize)
        ax.scatter(x_hc, y_hc, color="#00008B", zorder=1, s=0.5)
        ax.scatter(x_img, y_img, color="#34ebd6", s=0.1, zorder=0)

        for h in range(len(x_hc)):
            for i in range(4):
                y1, x1 = vertices[i, :, h]  # x1, y1: (n_hexagons)
                y2, x2 = vertices[i + 1, :, h]
                ax.plot([x1, x2], [y1, y2], c="black", lw=0.25)

        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-height / 2, height / 2)
        rm_spines(ax)
        # fig.tight_layout()
        return fig
