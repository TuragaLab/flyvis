"""Modules for decoding the DMN activity."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as nnf
from datamate import Namespace
from torch import nn

from flyvis import device
from flyvis.connectome import ConnectomeFromAvgFilters
from flyvis.utils.activity_utils import LayerActivity
from flyvis.utils.hex_utils import get_hex_coords
from flyvis.utils.nn_utils import n_params

logging = logging.getLogger(__name__)

__all__ = ["ActivityDecoder", "DecoderGAVP", "init_decoder"]


class ActivityDecoder(nn.Module):
    """
    Base class for decoding DMN activity.

    Args:
        connectome: Connectome directory with output_cell_types.

    Attributes:
        dvs_channels (LayerActivity): Dictionary of DVS channels.
        num_parameters (NumberOfParams): Number of parameters in the model.
        u (torch.Tensor): u-coordinates of hexagonal grid.
        v (torch.Tensor): v-coordinates of hexagonal grid.
        H (int): Height of the hexagonal grid.
        W (int): Width of the hexagonal grid.
    """

    dvs_channels: Union[Dict[str, torch.Tensor], LayerActivity]

    def __init__(self, connectome: ConnectomeFromAvgFilters):
        super().__init__()
        self.dvs_channels = LayerActivity(None, connectome, use_central=False)
        self.num_parameters = n_params(self)
        radius = connectome.config.extent
        self.u, self.v = get_hex_coords(radius)
        self.u -= self.u.min()
        self.v -= self.v.min()
        self.H, self.W = self.u.max() + 1, self.v.max() + 1

    def forward(self, activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the ActivityDecoder.

        Args:
            activity: Tensor of shape (n_samples, n_frames, n_cells).

        Returns:
            Dictionary of tensors with shape
            (n_samples, n_frames, output_cell_types, n_hexals).
        """
        self.dvs_channels.update(activity)
        return self.dvs_channels


class GlobalAvgPool(nn.Module):
    """Returns the average over the last dimension."""

    def forward(self, x):
        return x.mean(dim=-1)


class Conv2dConstWeight(nn.Conv2d):
    """
    PyTorch's Conv2d layer with optional constant weight initialization.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        const_weight: Optional constant value for weight initialization.
            If None, the standard PyTorch initialization is used.
        stride: Stride of the convolution.
        padding: Padding added to input.
        **kwargs: Additional keyword arguments for Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        const_weight: Optional[float] = None,
        stride: int = 1,
        padding: int = 0,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )
        if const_weight is not None and self.weight is not None:
            self.weight.data.fill_(const_weight)
        if const_weight is not None and self.bias is not None:
            self.bias.data.fill_(const_weight)


class Conv2dHexSpace(Conv2dConstWeight):
    """
    Convolution with regularly, hexagonally shaped filters (in cartesian map storage).

    Reference to map storage:
    https://www.redblobgames.com/grids/hexagons/#map-storage

    Info:
        kernel_size must be odd!

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        const_weight: Optional constant value for weight initialization.
            If None, the standard PyTorch initialization is used.
        stride: Stride of the convolution.
        padding: Padding added to input.
        **kwargs: Additional keyword arguments for Conv2d.

    Attributes:
        mask: Mask for hexagonal convolution.
        _filter_to_hex: Whether to apply hexagonal filter.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        const_weight: Optional[float] = 1e-3,
        stride: int = 1,
        padding: int = 0,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            const_weight,
            stride=stride,
            padding=padding,
            **kwargs,
        )

        if not kernel_size % 2:
            raise ValueError(f"{kernel_size} is even. Must be odd.")
        if kernel_size > 1:
            u, v = get_hex_coords(kernel_size // 2)
            u -= u.min()
            v -= v.min()
            mask = np.zeros(tuple(self.weight.shape))
            mask[:, :, u, v] = 1
            self.mask = torch.tensor(mask, device="cpu")
            self.weight.data.mul_(self.mask.to(device))
            self._filter_to_hex = True
        else:
            self._filter_to_hex = False

    def filter_to_hex(self):
        """Apply hexagonal filter to weights."""
        self.weight.data.mul_(self.mask.to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Conv2dHexSpace layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after hexagonal convolution.
        """
        if self._filter_to_hex:
            self.filter_to_hex()
        return super().forward(x)


class DecoderGAVP(ActivityDecoder):
    """
    Fully convolutional decoder with optional global average pooling.

    Args:
        connectome: Connectome directory.
        shape: List of channel sizes for each layer.
        kernel_size: Size of the convolutional kernel.
        p_dropout: Dropout probability.
        batch_norm: Whether to use batch normalization.
        n_out_features: Number of output features.
        const_weight: Constant value for weight initialization.
        normalize_last: Whether to normalize the last layer.
        activation: Activation function to use.

    Attributes:
        _out_channels: Number of output channels before reshaping.
        out_channels: Total number of output channels.
        n_out_features: Number of output features.
        base: Base convolutional layers.
        decoder: Decoder convolutional layers.
        head: Head layers for global average pooling.
        normalize_last: Whether to normalize the last layer.
        num_parameters: Number of parameters in the model.
    """

    def __init__(
        self,
        connectome: ConnectomeFromAvgFilters,
        shape: List[int],
        kernel_size: int,
        p_dropout: float = 0.5,
        batch_norm: bool = True,
        n_out_features: Optional[int] = None,
        const_weight: Optional[float] = None,
        normalize_last: bool = True,
        activation: str = "Softplus",
    ):
        super().__init__(connectome)
        p = int((kernel_size - 1) / 2)
        in_channels = len(connectome.output_cell_types)
        out_channels = shape[-1]
        self._out_channels = out_channels
        self.out_channels = (
            out_channels * n_out_features if n_out_features is not None else out_channels
        )
        self.n_out_features = n_out_features

        self.base = []
        for c in shape[:-1]:
            if c == 0:
                continue
            self.base.append(
                Conv2dHexSpace(
                    in_channels,
                    c,
                    kernel_size,
                    const_weight=const_weight,
                    padding=p,
                )
            )
            if batch_norm:
                self.base.append(nn.BatchNorm2d(c))
            self.base.append(getattr(nn, activation)())
            if p_dropout:
                self.base.append(nn.Dropout(p_dropout))
            in_channels = c
        self.base = nn.Sequential(*self.base)

        self.decoder = []
        if len(self.base) == 0 and batch_norm:
            self.decoder.append(nn.BatchNorm2d(in_channels))
        self.decoder.append(
            Conv2dHexSpace(
                in_channels,
                self.out_channels + 1 if normalize_last else self.out_channels,
                kernel_size,
                const_weight=const_weight,
                padding=p,
            )
        )
        self.decoder = nn.Sequential(*self.decoder)

        self.n_out_features = n_out_features
        self.head = []
        if n_out_features is not None:
            self.head.append(GlobalAvgPool())
        self.head = nn.Sequential(*self.head)

        self.normalize_last = normalize_last

        self.num_parameters = n_params(self)
        logging.info(f"Initialized decoder with {self.num_parameters} parameters.")
        logging.info(repr(self))

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DecoderGAVP.

        Args:
            activity: Input activity tensor.

        Returns:
            Decoded output tensor.
        """
        self.dvs_channels.update(activity)
        # Ensure that the outputs of the dvs-model are rectified potentials.
        x = nnf.relu(self.dvs_channels.output)

        # (n_frames, #samples, #outputneurons, n_hexals)
        n_samples, n_frames, in_channels, n_hexals = x.shape

        # Store hexals in square map.
        # (n_frames, #samples, #outputneurons, H, W)
        x_map = torch.zeros([n_samples, n_frames, in_channels, self.H, self.W])
        x_map[:, :, :, self.u, self.v] = x

        # Concatenate actual batch dimension with the frame dimension.
        # torch.flatten(x_map, 0, 1)  # (#samples*n_frames, #outputneurons, H, W)
        x_map = x_map.view(-1, in_channels, self.H, self.W)

        # Run decoder.
        # (n_frames*#samples, out_channels + 1, H, W)
        out = self.decoder(self.base(x_map))

        if self.normalize_last:
            # Do some normalization with the additional channel.
            # (n_frames*#samples, out_channels, H, W)
            out = out[:, : self.out_channels] / (
                nnf.softplus(out[:, self.out_channels :]) + 1
            )

        # Bring back into shape: # (#samples, n_frames, out_channels, n_hexals)
        out = out.view(n_samples, n_frames, self.out_channels, self.H, self.W)[
            :, :, :, self.u, self.v
        ]

        if self.n_out_features is not None:
            out = self.head(out).view(
                n_samples, n_frames, self._out_channels, self.n_out_features
            )

        return out


def init_decoder(
    decoder_config: Namespace, connectome: ConnectomeFromAvgFilters
) -> nn.Module:
    """
    Initialize a decoder based on the provided configuration.

    Args:
        decoder_config: Configuration for the decoder.
        connectome: Connectome directory.

    Returns:
        Initialized decoder module.
    """
    decoder_config = decoder_config.deepcopy()
    _type = decoder_config.pop("type")
    decoder_type = globals()[_type]
    decoder_config.update(dict(connectome=connectome))
    return decoder_type(**decoder_config)
