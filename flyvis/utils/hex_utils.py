"""Utility function for operations on hexagonal lattices."""

from numbers import Number
from typing import Iterable, Literal, Tuple

import numpy as np
import torch
from matplotlib import colormaps as cm
from numpy.typing import NDArray

import flyvis
from flyvis.utils.tensor_utils import matrix_mask_by_sub


def get_hex_coords(extent: int, astensor: bool = False) -> Tuple[NDArray, NDArray]:
    """Construct hexagonal coordinates for a regular hex-lattice with extent.

    Args:
        extent: Integer radius of hexagonal lattice. 0 returns the single
            center coordinate.
        astensor: If True, returns torch.Tensor, else np.array.

    Returns:
        A tuple containing:
            u: Hex-coordinates in u-direction.
            v: Hex-coordinates in v-direction.

    Note:
        Will return `get_num_hexals(extent)` coordinates.

    See Also:
        https://www.redblobgames.com/grids/hexagons/#range-coordinate
    """
    u = []
    v = []
    for q in range(-extent, extent + 1):
        for r in range(max(-extent, -extent - q), min(extent, extent - q) + 1):
            u.append(q)
            v.append(r)
    if astensor:
        return torch.tensor(u, dtype=torch.long), torch.tensor(v, dtype=torch.long)
    return np.array(u), np.array(v)


def hex_to_pixel(
    u: NDArray,
    v: NDArray,
    size: float = 1,
    mode: Literal["default", "flat", "pointy"] = "default",
) -> Tuple[NDArray, NDArray]:
    """Returns pixel coordinates from hex coordinates.

    Args:
        u: Hex-coordinates in u-direction.
        v: Hex-coordinates in v-direction.
        size: Size of hexagon.
        mode: Coordinate system convention.

    Returns:
        A tuple containing:
            x: Pixel-coordinates in x-direction.
            y: Pixel-coordinates in y-direction.

    See Also:
        https://www.redblobgames.com/grids/hexagons/#hex-to-pixel
    """
    if isinstance(u, list) and isinstance(v, list):
        u = np.array(u)
        v = np.array(v)
    if mode == "default":
        return 3 / 2 * v, -np.sqrt(3) * (u + v / 2)
    elif mode == "flat":
        return (3 / 2 * u) * size, (np.sqrt(3) / 2 * u + np.sqrt(3) * v) * size
    elif mode == "pointy":
        return (np.sqrt(3) * u + np.sqrt(3) / 2 * v) * size, (3 / 2 * v) * size
    else:
        raise ValueError(f"{mode} not recognized.")


def hex_rows(
    n_rows: int,
    n_columns: int,
    eps: float = 0.1,
    mode: Literal["pointy", "flat"] = "pointy",
) -> Tuple[NDArray, NDArray]:
    """Return a hex grid in pixel coordinates.

    Args:
        n_rows: Number of rows.
        n_columns: Number of columns.
        eps: Small offset to avoid overlapping hexagons.
        mode: Orientation of hexagons.

    Returns:
        A tuple containing:
            x: X-coordinates of hexagon centers.
            y: Y-coordinates of hexagon centers.
    """
    u = []
    v = []
    for r in range(n_rows):
        for c in range(n_columns):
            u.append(c)
            v.append(r)
    u = np.array(u)
    v = np.array(v)
    x, y = hex_to_pixel(u, v, mode=mode)
    x += eps
    y += eps
    return x, y


def pixel_to_hex(
    x: NDArray,
    y: NDArray,
    size: float = 1,
    mode: Literal["default", "flat", "pointy"] = "default",
) -> Tuple[NDArray, NDArray]:
    """Returns hex coordinates from pixel coordinates.

    Args:
        x: Pixel-coordinates in x-direction.
        y: Pixel-coordinates in y-direction.
        size: Size of hexagon.
        mode: Coordinate system convention.

    Returns:
        A tuple containing:
            u: Hex-coordinates in u-direction.
            v: Hex-coordinates in v-direction.

    See Also:
        https://www.redblobgames.com/grids/hexagons/#hex-to-pixel
    """
    if mode == "default":
        return -x / 3 - y / np.sqrt(3), 2 / 3 * x
    elif mode == "flat":
        return (2 / 3 * x) / size, (-1 / 3 * x + np.sqrt(3) / 3 * y) / size
    elif mode == "pointy":
        return (np.sqrt(3) / 3 * x - 1 / 3 * y) / size, (2 / 3 * y) / size
    else:
        raise ValueError(f"{mode} not recognized.")


def pad_to_regular_hex(
    u: NDArray,
    v: NDArray,
    values: NDArray,
    extent: int,
    value: float = np.nan,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Pad hexals with coordinates to a regular hex lattice.

    Args:
        u: U-coordinate of hexal.
        v: V-coordinate of hexal.
        values: Value of hexal with arbitrary shape but last axis
            must match the hexal dimension.
        extent: Extent of regular hex grid to pad to.
        value: The pad value.

    Returns:
        A tuple containing:
            u_padded: Padded u-coordinate.
            v_padded: Padded v-coordinate.
            values_padded: Padded value.

    Note:
        The canonical use case here is to pad a filter, receptive field, or
        postsynaptic current field for visualization.

    Example:
        ```python
        u = np.array([1, 0, -1, 0, 1, 2])
        v = np.array([-2, -1, 0, 0, 0, 0])
        values = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9])
        hexals = pad_to_regular_hex(u, v, values, 6)
        hex_scatter(*hexals, edgecolor='k', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ```
    """
    u_padded, v_padded = flyvis.utils.hex_utils.get_hex_coords(extent)
    slices = tuple()
    if len(values.shape) > 1:
        values_padded = np.ones([*values.shape[:-1], len(u_padded)]) * value
        for _ in range(len(values.shape[:-1])):
            slices += (slice(None),)
    else:
        values_padded = np.ones([len(u_padded)]) * value
    index = flyvis.utils.tensor_utils.where_equal_rows(
        np.stack((u, v), axis=1), np.stack((u_padded, v_padded), axis=1)
    )
    slices += (index,)
    values_padded[slices] = values
    return u_padded, v_padded, values_padded


def max_extent_index(u: NDArray, v: NDArray, max_extent: int) -> NDArray:
    """Returns a mask to constrain u and v axial-hex-coordinates by max_extent.

    Args:
        u: Hex-coordinates in u-direction.
        v: Hex-coordinates in v-direction.
        max_extent: Maximal extent.

    Returns:
        Boolean mask.
    """
    return (
        (-max_extent <= u)
        & (u <= max_extent)
        & (-max_extent <= v)
        & (v <= max_extent)
        & (-max_extent <= u + v)
        & (u + v <= max_extent)
    )


def get_num_hexals(extent: int) -> int:
    """Returns the absolute number of hexals in a hexagonal grid with extent.

    Args:
        extent: Extent of hex-lattice.

    Returns:
        Number of hexals.

    Note:
        Inverse of get_hextent.
    """
    return 1 + 3 * extent * (extent + 1)


def get_hextent(num_hexals: int) -> int:
    """Computes the hex-lattice extent from the number of hexals.

    Args:
        num_hexals: Number of hexals.

    Returns:
        Extent of hex-lattice.

    Note:
        Inverse of get_num_hexals.
    """

    return np.floor(np.sqrt(num_hexals / 3)).astype("int")


def sort_u_then_v(
    u: NDArray, v: NDArray, values: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    """Sorts u, v, and values by u and then v.

    Args:
        u: U-coordinate of hexal.
        v: V-coordinate of hexal.
        values: Value of hexal.

    Returns:
        A tuple containing:
            u: Sorted u-coordinate of hexal.
            v: Sorted v-coordinate of hexal.
            values: Sorted value of hexal.
    """
    index = np.lexsort((v, u))
    return u[index], v[index], values[index]


def sort_u_then_v_index(u: NDArray, v: NDArray) -> NDArray:
    """Index to sort u, v by u and then v.

    Args:
        u: U-coordinate of hexal.
        v: V-coordinate of hexal.

    Returns:
        Index to sort u and v.
    """
    return np.lexsort((v, u))


def get_extent(u: NDArray, v: NDArray, astype: type = int) -> int:
    """Returns extent (integer distance to origin) of arbitrary u, v coordinates.

    Args:
        u: U-coordinate of hexal.
        v: V-coordinate of hexal.
        astype: Type to cast to.

    Returns:
        Extent of hex-lattice.

    Note:
        If u and v are arrays, returns the maximum extent.

    See Also:
        https://www.redblobgames.com/grids/hexagons/#distances
    """
    if isinstance(u, Number) and isinstance(v, Number):
        u, v = np.array((u,)), np.array((v,))
    uv = np.stack((u, v), 1)
    extent = (
        abs(0 - uv[:, 0]) + abs(0 + 0 - uv[:, 0] - uv[:, 1]) + abs(0 - uv[:, 1])
    ) / 2
    return np.max(extent).astype(astype)


def crop_to_extent(
    u: NDArray, v: NDArray, color: NDArray, max_extent: int
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Crop hexagonal grid data to a specified maximum extent.

    Args:
        u: Array of hex coordinates in u direction.
        v: Array of hex coordinates in v direction.
        color: Array of values associated with each (u, v) coordinate.
        max_extent: Maximum extent to crop the hexagonal grid to.

    Returns:
        Tuple of cropped u, v, and color arrays.
    """
    extent_condition = (
        (-max_extent <= u)
        & (u <= max_extent)
        & (-max_extent <= v)
        & (v <= max_extent)
        & (-max_extent <= u + v)
        & (u + v <= max_extent)
    )
    return u[extent_condition], v[extent_condition], color[extent_condition]


# -- Experimental explicit hex-datastructures ----------------------------------


class Hexal:
    """Hexal representation containing u, v, z coordinates and value.

    Attributes:
        u: Coordinate in u principal direction (0 degree axis).
        v: Coordinate in v principal direction (60 degree axis).
        z: Coordinate in z principal direction (-60 degree axis).
        value: 'Hexal' value.
        u_stride: Stride in u-direction.
        v_stride: Stride in v-direction.
    """

    def __init__(
        self, u: int, v: int, value: float = np.nan, u_stride: int = 1, v_stride: int = 1
    ):
        self.u = u
        self.v = v
        self.z = -(u + v)
        self.value = value
        self.u_stride = u_stride
        self.v_stride = v_stride

    def __repr__(self):
        return "Hexal(u={}, v={}, value={}, u_stride={}, v_stride={})".format(
            self.u, self.v, self.value, self.u_stride, self.v_stride
        )

    def __eq__(self, other):
        """Compares coordinates (not values)."""
        if isinstance(other, Hexal):
            return all((self.u == other.u, self.v == other.v))
        elif isinstance(other, Iterable):
            return np.array([self == h for h in other])

    def __add__(self, other):
        """Adds u and v coordinates, while keeping the value of the left hexal."""
        if isinstance(other, Hexal):
            return Hexal(self.u + other.u, self.v + other.v, self.value)
        elif isinstance(other, Iterable):
            return np.array([self + h for h in other])

    def __mul__(self, other):
        """Multiplies values, while preserving coordinates."""
        if isinstance(other, Hexal):
            return Hexal(self.u, self.v, self.value * other.value)
        elif isinstance(other, Iterable):
            return np.array([self * h for h in other])
        else:
            return Hexal(self.u, self.v, self.value * other)

    def eq_val(self, other):
        """Compares the values, not the coordinates."""
        if isinstance(other, Hexal):
            return self.value == other.value
        elif isinstance(other, Iterable):
            return np.array([self.eq_val(h) for h in other])

    # ----- Neighbour identification

    @property
    def east(self):
        return Hexal(self.u + self.u_stride, self.v, 0)

    @property
    def north_east(self):
        return Hexal(self.u, self.v + self.v_stride, 0)

    @property
    def north_west(self):
        return Hexal(self.u - self.u_stride, self.v + self.v_stride, 0)

    @property
    def west(self):
        return Hexal(self.u - self.u_stride, self.v, 0)

    @property
    def south_west(self):
        return Hexal(self.u, self.v - self.v_stride, 0)

    @property
    def south_east(self):
        return Hexal(self.u + self.u_stride, self.v - self.v_stride, 0)

    def neighbours(self):
        """Returns 6 neighbours sorted CCW, starting from east."""
        return (
            self.east,
            self.north_east,
            self.north_west,
            self.west,
            self.south_west,
            self.south_east,
        )

    def is_neighbour(self, other):
        """Evaluates if other is a neighbour."""
        neighbours = self.neighbours()
        if isinstance(other, Hexal):
            return other in neighbours
        elif isinstance(other, Iterable):
            return np.array([self.neighbour(h) for h in other])

    @staticmethod
    def unit_directions():
        """Returns the six unit directions."""
        return HexArray(Hexal(0, 0, 0).neighbours())

    def neighbour(self, angle):
        neighbours = np.array(self.neighbours())
        angles = np.array([h.angle(signed=True) for h in neighbours])
        distance = (angles - angle) % np.pi
        index = np.argsort(distance)
        return HexArray(neighbours[index[:2]])

    def direction(self, angle):
        neighbours = HexArray(self.neighbour(angle))
        angles = np.array([h.angle(signed=True) for h in neighbours])
        distance = (angles - angle) % np.pi
        index = np.argsort(distance)
        return HexArray(self.unit_directions()[index[:2]])

    # ----- Geometric methods

    def interp(self, other, t):
        """Interpolates towards other.

        Args:
            other (Hexal)
            t (float): interpolation step, 0<t<1.

        Returns:
            Hexal
        """

        def hex_round(u, v):
            z = -(u + v)
            ru = round(u)
            rv = round(v)
            rz = round(z)
            u_diff = abs(ru - u)
            v_diff = abs(rv - v)
            z_diff = abs(rz - z)
            if u_diff > v_diff and u_diff > z_diff:
                ru = -rv - rz
            elif v_diff > z_diff:
                rv = -ru - rz
            return ru, rv

        uprime, vprime = (
            self.u + (other.u - self.u) * t,
            self.v + (other.v - self.v) * t,
        )
        uprime, vprime = hex_round(uprime, vprime)
        return Hexal(uprime, vprime, 0)

    def angle(self, other=None, non_negative=False):
        """
        Returns the angle to other or the origin.

        Args:
            other (Hexal)
            non_negative (bool): add 2pi if angle is negative.
                Default: False.

        Returns:
            float: angle in radians.
        """

        def _angle(p1, p2):
            """Counter clockwise angle from p1 to p2.

            Returns:
                float: angle in [0, np.pi]
            """
            dot = p1[0] * p2[0] + p1[1] * p2[1]
            det = p1[0] * p2[1] - p1[1] * p2[0]
            angle = np.arctan2(det, dot)
            return angle

        x, y = self._to_pixel(self.u, self.v)
        theta = np.arctan2(y, x)
        if other is not None:
            xother, yother = self._to_pixel(other.u, other.v)
            theta = _angle([x, y], [xother, yother])
        if non_negative:
            theta += 2 * np.pi if theta < 0 else 0
        return theta

    def distance(self, other=None):
        """Returns the columnar distance between to hexals."""
        if other is not None:
            return int(
                (
                    abs(self.u - other.u)
                    + abs(self.u + self.v - other.u - other.v)
                    + abs(self.v - other.v)
                )
                / 2
            )
        return int((abs(self.u) + abs(self.u + self.v) + abs(self.v)) / 2)

    @staticmethod
    def _to_pixel(u, v, scale=1):
        """Converts to pixel coordinates."""
        return hex_to_pixel(u, v, scale)


class HexArray(np.ndarray):
    """Flat array holding Hexal's as elements.

    Can be constructed with:
        HexArray(hexals: Iterable, values: Optional[np.nan])
        HexArray(u: Iterable, v: Iterable, values: Optional[np.nan])
    """

    def __new__(cls, hexals=None, u=None, v=None, values=0):
        if isinstance(hexals, Iterable):
            u = np.array([h.u for h in hexals])
            v = np.array([h.v for h in hexals])
            values = np.array([h.value for h in hexals])
        if not isinstance(values, Iterable):
            values = np.ones_like(u) * values
        u, v = HexArray.sort(u, v)
        hexals = np.array(
            [Hexal(_u, _v, _val) for _u, _v, _val in zip(u, v, values)],
            dtype=Hexal,
        ).view(cls)
        return hexals

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __eq__(self, other):
        if isinstance(other, Hexal):
            return other == self
        else:
            return super().__eq__(other)

    def __getitem__(self, key):
        if isinstance(key, HexArray):
            mask = self.where_hexarray(key)
            return self[mask]
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, slice) and key == slice(None):
            self.values = value
        elif isinstance(key, HexArray):
            mask = self.where_hexarray(key)
            super().__setitem__(mask, value)
        else:
            super().__setitem__(key, value)

    def where_hexarray(self, hexarray):
        return matrix_mask_by_sub(
            np.stack((hexarray.u, hexarray.v), axis=0).T,
            np.stack((self.u, self.v), axis=0).T,
        )

    @staticmethod
    def sort(u, v):
        sort_index = np.lexsort((v, u))
        u = u[sort_index]
        v = v[sort_index]
        return u, v

    @staticmethod
    def get_extent(hexals=None, u=None, v=None, center=Hexal(0, 0, 0)):
        """Returns the columnar extent."""
        from numbers import Number

        if isinstance(u, Number) and isinstance(v, Number):
            h = Hexal(u, v, 0)
            return h.distance(center)
        else:
            ha = HexArray(hexals, u, v)
            distance = max([h.distance(center) for h in ha])
            return distance

    @property
    def u(self):
        return np.array([h.u for h in self])

    @property
    def v(self):
        return np.array([h.v for h in self])

    @property
    def values(self):
        return np.array([h.value for h in self])

    @values.setter
    def values(self, values):
        for h, val in zip(self, values):
            h.value = val

    @property
    def extent(self):
        return super().get_extent(self)

    def with_stride(self, u_stride=None, v_stride=None):
        """Returns a sliced instance obeying strides in u- and v-direction."""
        new = []
        for u, v, _ in zip(self.u, self.v, self.values):
            new.append(u % u_stride == 0 and v % v_stride == 0)
        return self[np.array(new)]

    def where(self, value):
        """Returns a mask of where values are equal to the given one.

        Note: value can be np.nan.
        """
        return np.isclose(self.values, value, rtol=0, atol=0, equal_nan=True)

    def fill(self, value):
        """Fills the values with the given one."""
        for h in self:
            h.value = value

    def to_pixel(self, scale=1, mode="default"):
        """Converts to pixel coordinates."""
        return hex_to_pixel(self.u, self.v, scale, mode=mode)

    def plot(self, figsize=[3, 3], fill=True):
        """Plots values in regular hexagonal lattice.

        Meant for debugging.
        """
        u = np.array([h.u for h in self])
        v = np.array([h.v for h in self])
        color = np.array([h.value for h in self])
        return flyvis.plots.hex_scatter(
            u,
            v,
            color,
            fill=fill,
            cmap=cm.get_cmap("binary"),
            edgecolor="black",
            figsize=figsize,
        )


class HexLattice(HexArray):
    """Flat array of Hexals.

    Args:
        extent: Extent of the regular hexagon grid.
        hexals: Existing hexals to initialize with.
        center: Center hexal of the lattice.
        u_stride: Stride in u-direction.
        v_stride: Stride in v-direction.
    """

    def __new__(
        cls,
        extent=15,
        hexals=None,
        center=Hexal(0, 0, 0),
        u_stride=1,
        v_stride=1,
    ):
        if isinstance(hexals, Iterable):
            hexals = HexArray(hexals=hexals)
            u = np.array([h.u for h in hexals])
            v = np.array([h.v for h in hexals])
            extent = extent or super().get_extent(hexals, center=center)
            lattice = HexLattice(
                extent=extent,
                center=center,
                u_stride=u_stride,
                v_stride=v_stride,
            )
            for h in lattice:
                if h in hexals:
                    h.value = hexals[h == hexals][0].value
        else:
            u, v = flyvis.utils.hex_utils.get_hex_coords(extent)
            u += center.u
            v += center.v
            values = [np.nan for _ in range(len(u))]  # np.ones_like(u) * np.nan
            lattice = []
            for _u, _v, _val in zip(u, v, values):
                if _u % u_stride == 0 and _v % v_stride == 0:
                    lattice.append(Hexal(_u, _v, _val, u_stride, v_stride))
            lattice = np.array(lattice, dtype=Hexal).view(cls)
        return lattice

    @property
    def center(self):
        return self[len(self) // 2]

    @property
    def extent(self):
        return super().get_extent(self, center=self.center)

    # ----- Geometry

    def circle(self, radius=None, center=Hexal(0, 0, 0), as_lattice=False):
        """Draws a circle in hex coordinates.

        Args:
            radius: Radius in columns of the circle.
            center: Center of the circle.
            as_lattice: Returns the circle on a constrained regular lattice.
        """
        lattice = HexLattice(extent=max(radius or 0, self.extent), center=center)
        radius = radius or self.extent
        circle = []
        for _, h in enumerate(lattice):
            distance = center.distance(h)
            if distance == radius:
                h.value = 1
                circle.append(h)
        if as_lattice:
            return HexLattice(hexals=circle)
        return HexArray(hexals=circle)

    @staticmethod
    def filled_circle(radius=None, center=Hexal(0, 0, 0), as_lattice=False):
        """Draws a circle in hex coordinates.

        Args:
            radius: Radius in columns of the circle.
            center: Center of the circle.
            as_lattice: Returns the circle on a constrained regular lattice.
        """
        lattice = HexLattice(extent=radius or 0, center=center)
        radius = radius
        circle = []
        for _, h in enumerate(lattice):
            distance = center.distance(h)
            if distance <= radius:
                h.value = 1
                circle.append(h)
        if as_lattice:
            return HexLattice(hexals=circle)
        return HexArray(hexals=circle)

    def hull(self):
        """Returns the hull of the regular lattice."""
        return self.circle(radius=self.extent, center=self.center)

    def _line_span(self, angle):
        """Returns two points spanning a line with given angle wrt. origin.

        Args:
            angle: In [0, np.pi]

        Returns:
            HexArray
        """
        # To offset the line by simple addition of the offset,
        # radius=2 * self.extent spans the line in ways that each valid offset
        # can be added.
        distant_hull = self.ring(radius=2 * self.extent)
        angles = np.array([h.angle(signed=True) for h in distant_hull])
        distance = (angles - angle) % np.pi
        index = np.argsort(distance)
        span = distant_hull[index[0:2]]
        for h in span:
            h.value = 1
        return HexArray(hexals=span)

    def line(self, angle, center=Hexal(0, 0, 1), as_lattice=False):
        """Returns a line on a HexLattice or HexArray.

        Args:
            angle: In [0, np.pi]
            center: Midpoint of the line
            as_lattice: Returns the ring on a constrained regular lattice.

        Returns:
            HexArray or constrained HexLattice
        """
        line_span = self._line_span(angle)
        distance = line_span[0].distance(line_span[1])
        line = []
        for i in range(distance + 1):
            _next = line_span[0].interp(line_span[1], 1 / distance * i)
            line.append(_next)
        for h in line:
            h.value = 1
        if as_lattice:
            return HexLattice(extent=self.extent, hexals=center + line)
        return HexArray(hexals=center + line)

    def _get_neighbour_indices(self, index):
        _neighbours = self[index].neighbours()
        neighbours = ()
        for n in _neighbours:
            valid = self == n
            if valid.any():
                neighbours += (np.where(valid)[0][0],)
        return neighbours

    def valid_neighbours(self):
        neighbours = ()
        for i in range(len(self)):
            neighbours += (self._get_neighbour_indices(i),)
        return neighbours


class LatticeMask:
    """Boolean masks for lattice dimension.

    Args:
        extent: Extent of the hexagonal lattice.
        u_stride: Stride in u-direction.
        v_stride: Stride in v-direction.
    """

    def __init__(self, extent: int = 15, u_stride: int = 1, v_stride: int = 1):
        self._lattice = HexLattice(extent=extent, u_stride=u_stride, v_stride=v_stride)

    @property
    def center(self):
        return self._lattice.center == self._lattice

    @property
    def center_east(self):
        return self._lattice.center.east == self._lattice

    @property
    def center_north_east(self):
        return self._lattice.center.north_east == self._lattice

    @property
    def center_north_west(self):
        return self._lattice.center.north_west == self._lattice

    @property
    def center_west(self):
        return self._lattice.center.west == self._lattice

    @property
    def center_south_west(self):
        return self._lattice.center.south_west == self._lattice

    @property
    def center_south_east(self):
        return self._lattice.center.south_east == self._lattice
