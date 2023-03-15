"""'Transduction' of cartesian pixels to hexals on a regular hexagonal lattice.
"""
from typing import Iterator, Tuple

import numpy as np
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
        extent: int
        "Radius, in number of receptors, of the hexagonal array"
        kernel_size: int
        "Photon collection radius, in pixels"

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

    def _set_filter(self):
        self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, padding=0)
        self.conv.weight.data /= self.conv.weight.data
        self.conv.bias.data.fill_(0)  # if not self.requires_grad else None
        self.conv.weight.requires_grad = False  # self.requires_grad
        self.conv.bias.requires_grad = False  # self.requires_grad

    def __call__(self, sequence, ftype="mean", hex_sample=True):
        """Applies a box kernel to all frames in a sequence.

        Arguments:
            sequence (torch.Tensor): shape (samples, frames, height, width)
            ftype: filter type, 'mean' or 'median'.
        """
        samples, frames, height, width = sequence.shape

        if not isinstance(sequence, torch.cuda.FloatTensor):
            sequence = torch.Tensor(sequence)

        if (self.min_frame_size > torch.tensor([height, width])).any():
            sequence = ttf.resize(sequence, self.min_frame_size.tolist())

        if ftype == "mean":

            sequence = F.pad(sequence, self.pad)

            def conv(x):
                return self.conv(x.unsqueeze(1))

            out = torch.cat(tuple(map(conv, torch.unbind(sequence, dim=0))), dim=0)

            out /= self.kernel_size**2

        elif ftype == "sum":

            sequence = F.pad(sequence, self.pad)

            def conv(x):
                return self.conv(x.unsqueeze(1))

            out = torch.cat(tuple(map(conv, torch.unbind(sequence, dim=0))), dim=0)

        elif ftype == "median":
            out = median(sequence, self.kernel_size)

        else:
            raise ValueError("ftype must be 'sum', 'mean', or 'median." f"Is {ftype}.")

        if hex_sample is True:
            out = self.hex_sample(out)
            return out.reshape(samples, frames, -1)

        return out.reshape(samples, frames, height, width)

    def hex_sample(self, sequence):
        h, w = sequence.shape[-2:]
        c = self.receptor_centers + torch.tensor([h // 2, w // 2])
        out = sequence[:, :, c[:, 0], c[:, 1]]
        return out

    def sample(self, img, ftype="mean"):
        """Sample individual frames."""
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
        fig.tight_layout()
        return fig
