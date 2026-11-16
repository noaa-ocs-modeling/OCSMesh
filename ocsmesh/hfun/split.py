"""Splitting large rasters into windows for the size function collector.

A raster larger than ``split_size`` pixels is cut into windows. Each
window is computed on its *core* plus a *halo* around it, and only the
core is kept: shapes detected on the window (channels, contours) are
clipped to the core, and the window's size function only provides
sizes inside its core when the inputs are composited. The cores of a
raster's windows tile it exactly, so every pixel is kept once.

The halo gives each core the neighbourhood its refinements look at: a
channel is only recognised when the whole channel width is visible, and
a constraint rate spreads from pixels up to its reach away. Without a
halo the window edges themselves look like channels (narrow strips of
water cut off by the edge) and rates stop at the edges.
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.windows
import shapely
from pyproj import CRS, Transformer
from rasterio.transform import array_bounds
from shapely import ops
from shapely.geometry import (
    GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon,
    box)

from ocsmesh.raster import Raster


@dataclass(frozen=True)
class Core:
    """Pixels of a parent raster grid that one window keeps.

    ``cols``/``rows`` are half-open pixel ranges on the parent grid,
    whose affine ``transform``, ``shape`` (height, width) and ``crs``
    are stored so that a worker can test points without the file.
    """

    transform: tuple
    shape: Tuple[int, int]
    crs: str
    cols: Tuple[int, int]
    rows: Tuple[int, int]

    @property
    def affine(self) -> rasterio.Affine:
        return rasterio.Affine(*self.transform[:6])

    def bounds(self) -> Tuple[float, float, float, float]:
        """West, south, east, north of the core (in ``crs``)."""
        c0, c1 = self.cols
        r0, r1 = self.rows
        west, south, east, north = array_bounds(
            r1 - r0, c1 - c0,
            self.affine * rasterio.Affine.translation(c0, r0))
        return (min(west, east), min(south, north),
                max(west, east), max(south, north))

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean per point (in ``crs``): is its parent pixel in the core?

        Points up to half a pixel outside the parent raster count as its
        edge pixel, as the composite footprints do, so the cores of a
        raster tile its whole extent with no gap and no overlap.
        """
        col, row = ~self.affine * (np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        col, row = np.asarray(col), np.asarray(row)
        height, width = self.shape
        inside = ((col >= -0.5) & (col < width + 0.5)
                  & (row >= -0.5) & (row < height + 0.5))
        col = np.clip(np.floor(col), 0, width - 1)
        row = np.clip(np.floor(row), 0, height - 1)
        return (inside
                & (col >= self.cols[0]) & (col < self.cols[1])
                & (row >= self.rows[0]) & (row < self.rows[1]))

    def contains_lonlat(self, lonlat: np.ndarray) -> np.ndarray:
        """``contains`` for EPSG:4326 points."""
        x, y = lonlat[:, 0], lonlat[:, 1]
        crs = CRS.from_user_input(self.crs)
        if not crs.equals(CRS.from_epsg(4326)):
            x, y = Transformer.from_crs(
                'EPSG:4326', crs, always_xy=True).transform(x, y)
        return self.contains(x, y)


def _ranges(n_pixels: int, n_parts: int) -> List[Tuple[int, int]]:
    """``n_parts`` consecutive half-open ranges covering ``n_pixels``."""
    edges = np.linspace(0, n_pixels, n_parts + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def plan_cores(grid: Raster, native: Raster, split_size: int
               ) -> List[Core]:
    """Cores for ``grid``, or an empty list when it needs no split.

    ``split_size`` is the largest window side in pixels of ``native``
    (the DEM at its own resolution, where detection is costly); the
    cores are laid out on ``grid`` (the size function's pixels) so that
    each grid pixel belongs to exactly one core.
    """
    n_x = math.ceil(native.src.width / split_size)
    n_y = math.ceil(native.src.height / split_size)
    n_x = min(n_x, grid.src.width)
    n_y = min(n_y, grid.src.height)
    if n_x * n_y <= 1:
        return []
    transform = tuple(grid.src.transform)
    shape = (grid.src.height, grid.src.width)
    crs = grid.crs.to_wkt()
    return [Core(transform, shape, crs, cols, rows)
            for rows in _ranges(grid.src.height, n_y)
            for cols in _ranges(grid.src.width, n_x)]


def halo_bounds(core: Core, raster: Raster, halo_m: float
                ) -> Tuple[float, float, float, float]:
    """Bounds of ``core`` grown by ``halo_m`` metres (in the core's CRS).

    The growth is measured with ``raster``'s pixel size, so it is
    rounded up to whole pixels of that raster.
    """
    px_x_m, px_y_m = raster.pixel_size_m()
    # 2% margin: in a geographic CRS the pixel width shrinks away from
    # the raster's middle latitude.
    n_x = math.ceil(1.02 * halo_m / px_x_m) if halo_m > 0 else 0
    n_y = math.ceil(1.02 * halo_m / px_y_m) if halo_m > 0 else 0
    west, south, east, north = core.bounds()
    dx, dy = abs(raster.dx), abs(raster.dy)
    return (west - n_x * dx, south - n_y * dy,
            east + n_x * dx, north + n_y * dy)


def write_window(raster: Raster, bounds, out_path: str) -> Optional[str]:
    """Write the pixels of ``raster`` within ``bounds`` (its CRS) to a
    new GeoTIFF; None when they miss the raster.

    The window is snapped outward to whole pixels and limited to the
    raster, so the copy is on the same pixel grid.
    """
    src = raster.src
    win = rasterio.windows.from_bounds(*bounds, transform=src.transform)
    win = win.round_offsets(op='floor').round_lengths(op='ceil')
    full = rasterio.windows.Window(0, 0, src.width, src.height)
    try:
        win = win.intersection(full)
    except rasterio.errors.WindowError:
        return None
    if win.width < 1 or win.height < 1:
        return None
    win = rasterio.windows.Window(
        int(win.col_off), int(win.row_off), int(win.width), int(win.height))
    profile = src.profile.copy()
    profile.update(
        driver='GTiff', width=win.width, height=win.height,
        transform=src.window_transform(win))
    for key in ('blockxsize', 'blockysize', 'tiled', 'compress',
                'interleave'):
        profile.pop(key, None)
    if win.width >= 256 and win.height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        for band in range(1, src.count + 1):
            dst.write(src.read(band, window=win), band)
    return out_path


def _parts_of_type(geom, kinds):
    """The parts of ``geom`` that are of one of ``kinds``."""
    if geom.is_empty:
        return []
    if isinstance(geom, kinds):
        return [geom]
    if hasattr(geom, 'geoms'):
        return [p for g in geom.geoms for p in _parts_of_type(g, kinds)]
    return []


def clip_to_core(shape, shape_crs, core: Core):
    """``shape`` clipped to ``core``, of the same kind (polygons stay
    polygons, lines stay lines); None when nothing is left."""
    if shape is None or shape.is_empty:
        return None
    core_poly = box(*core.bounds())
    crs = CRS.from_user_input(core.crs)
    shape_crs = CRS.from_user_input(shape_crs)
    if not shape_crs.equals(crs):
        transformer = Transformer.from_crs(crs, shape_crs, always_xy=True)
        core_poly = ops.transform(transformer.transform,
                                  core_poly.segmentize(
                                      max(core_poly.length / 400, 1e-9)))
    if not shape.is_valid:
        # Detection with a simplification tolerance can give
        # self-touching polygons, which GEOS cannot intersect.
        shape = shapely.make_valid(shape)
    clipped = shape.intersection(core_poly)
    if isinstance(shape, (Polygon, MultiPolygon)):
        parts = _parts_of_type(clipped, (Polygon,))
        return MultiPolygon(parts) if parts else None
    if isinstance(shape, (LineString, MultiLineString)):
        parts = _parts_of_type(clipped, (LineString,))
        return MultiLineString(parts) if parts else None
    if isinstance(shape, GeometryCollection):
        return None
    return None if clipped.is_empty else clipped
