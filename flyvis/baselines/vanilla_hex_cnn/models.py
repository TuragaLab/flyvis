import torch
from torch import nn
from torch.nn.init import constant_, kaiming_normal_

from flyvis.task.decoder import Conv2dHexSpace
from flyvis.utils.hex_utils import get_hex_coords

__all__ = [
    "VanillaHexCNNBaseline",
    "VanillaHexCNNBaselineHexSpace",
]


def conv_block(
    batch_norm: bool,
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    nonlinearity: str = "ELU",
) -> nn.Module:
    if nonlinearity is not None:
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
                getattr(nn, nonlinearity)(),
            )
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            getattr(nn, nonlinearity)(),
        )
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=True,
    )


def conv_block_hex(
    batch_norm: bool,
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    nonlinearity: str = "ELU",
) -> nn.Module:
    conv = Conv2dHexSpace(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        const_weight=None,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=not batch_norm,
    )
    if nonlinearity is not None:
        if batch_norm:
            return nn.Sequential(
                conv,
                nn.BatchNorm2d(out_planes),
                getattr(nn, nonlinearity)(),
            )
        return nn.Sequential(conv, getattr(nn, nonlinearity)())
    return conv


class RegularHexToCartesianMap(nn.Module):
    """Translate regular hex-lattice inputs to cartesian maps."""

    def __init__(self, extent: int = 15):
        super().__init__()
        u, v = get_hex_coords(extent)
        u = u - u.min()
        v = v - v.min()
        self.H = int(u.max() + 1)
        self.W = int(v.max() + 1)
        self.register_buffer("u", torch.tensor(u, dtype=torch.long), persistent=False)
        self.register_buffer("v", torch.tensor(v, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples, n_frames, in_channels, _ = x.shape
        x_map = torch.zeros(
            [n_samples, n_frames, in_channels, self.H, self.W],
            dtype=x.dtype,
            device=x.device,
        )
        x_map[:, :, :, self.u, self.v] = x
        return x_map.squeeze(dim=2)


class CartesianMapToRegularHex(nn.Module):
    """Translate cartesian maps back to regular hex-lattice layout."""

    def __init__(self, extent: int = 15):
        super().__init__()
        u, v = get_hex_coords(extent)
        u = u - u.min()
        v = v - v.min()
        self.H = int(u.max() + 1)
        self.W = int(v.max() + 1)
        self.register_buffer("u", torch.tensor(u, dtype=torch.long), persistent=False)
        self.register_buffer("v", torch.tensor(v, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_samples, n_channels = x.shape[:2]
        n_frames = 1
        x = x.view(n_samples, n_frames, n_channels, self.H, self.W)
        return x[:, :, :, self.u, self.v]


class VanillaHexCNNBaseline(nn.Module):
    """Vanilla hex CNN baseline (~400k parameters in the L_structured variant)."""

    def __init__(
        self,
        n_frames: int = 4,
        conv_kernel_sizes=None,
        conv_strides=None,
        batchNorm: bool = True,
        shape=None,
        nonlinearity: str = "ELU",
    ):
        if shape is None:
            shape = [64, 32, 16, 8, 2]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [1, 3, 3, 3, 5]
        if conv_strides is None:
            conv_strides = [1, 1, 1, 1, 1]
        super().__init__()

        self.to_cartesian = RegularHexToCartesianMap(extent=15)
        conv = [
            conv_block(
                batch_norm=batchNorm,
                in_planes=n_frames,
                out_planes=shape[0],
                kernel_size=conv_kernel_sizes[0],
                stride=conv_strides[0],
                nonlinearity=nonlinearity,
            )
        ]
        in_planes = shape[0]
        for i, out_planes in enumerate(shape[1:-1]):
            conv.append(
                conv_block(
                    batch_norm=batchNorm,
                    in_planes=in_planes,
                    out_planes=out_planes,
                    kernel_size=conv_kernel_sizes[i + 1],
                    stride=conv_strides[i + 1],
                    nonlinearity=nonlinearity,
                )
            )
            in_planes = out_planes
        conv.append(
            conv_block(
                batch_norm=batchNorm,
                in_planes=in_planes,
                out_planes=shape[-1],
                kernel_size=conv_kernel_sizes[-1],
                stride=conv_strides[-1],
                nonlinearity=None,
            )
        )

        self.conv = nn.Sequential(*conv)
        self.to_hex = CartesianMapToRegularHex(extent=15)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                kaiming_normal_(module.weight, 0.1)
                if module.bias is not None:
                    constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                constant_(module.weight, 1)
                constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_hex(self.conv(self.to_cartesian(x)))

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]


class VanillaHexCNNBaselineHexSpace(nn.Module):
    """Vanilla hex CNN baseline with Conv2dHexSpace filters."""

    def __init__(
        self,
        n_frames: int = 4,
        conv_kernel_sizes=None,
        conv_strides=None,
        batchNorm: bool = True,
        shape=None,
        nonlinearity: str = "ELU",
    ):
        if shape is None:
            shape = [64, 32, 16, 8, 2]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [1, 3, 3, 3, 5]
        if conv_strides is None:
            conv_strides = [1, 1, 1, 1, 1]
        super().__init__()

        self.to_cartesian = RegularHexToCartesianMap(extent=15)
        conv = [
            conv_block_hex(
                batch_norm=batchNorm,
                in_planes=n_frames,
                out_planes=shape[0],
                kernel_size=conv_kernel_sizes[0],
                stride=conv_strides[0],
                nonlinearity=nonlinearity,
            )
        ]
        in_planes = shape[0]
        for i, out_planes in enumerate(shape[1:-1]):
            conv.append(
                conv_block_hex(
                    batch_norm=batchNorm,
                    in_planes=in_planes,
                    out_planes=out_planes,
                    kernel_size=conv_kernel_sizes[i + 1],
                    stride=conv_strides[i + 1],
                    nonlinearity=nonlinearity,
                )
            )
            in_planes = out_planes
        conv.append(
            conv_block_hex(
                batch_norm=batchNorm,
                in_planes=in_planes,
                out_planes=shape[-1],
                kernel_size=conv_kernel_sizes[-1],
                stride=conv_strides[-1],
                nonlinearity=None,
            )
        )

        self.conv = nn.Sequential(*conv)
        self.to_hex = CartesianMapToRegularHex(extent=15)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                kaiming_normal_(module.weight, 0.1)
                if module.bias is not None:
                    constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                constant_(module.weight, 1)
                constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_hex(self.conv(self.to_cartesian(x)))

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]

