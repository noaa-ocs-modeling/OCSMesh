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

import itertools
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

from ocsmesh import utils
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


def _core_poly(core: Core, shape_crs):
    """The core as a polygon in ``shape_crs``."""
    core_poly = box(*core.bounds())
    crs = CRS.from_user_input(core.crs)
    shape_crs = CRS.from_user_input(shape_crs)
    if not shape_crs.equals(crs):
        transformer = Transformer.from_crs(crs, shape_crs, always_xy=True)
        core_poly = ops.transform(transformer.transform,
                                  core_poly.segmentize(
                                      max(core_poly.length / 400, 1e-9)))
    return core_poly


def clip_channels_to_core(shape, shape_crs, core: Core):
    """Detected channels clipped to ``core``, polygon by polygon.

    Returns ``(pieces, labels)``: the MultiPolygon of the clipped pieces
    and, for each of its polygons, the index of the detected polygon it
    comes from (``drop_small_channels`` keeps the pieces of one detected
    polygon together); ``(None, None)`` when nothing is left. Each
    polygon is repaired on its own when invalid: the repair can split
    one into polygons that only meet at a point.
    """
    if shape is None or shape.is_empty:
        return None, None
    parts = np.array(_parts_of_type(shape, (Polygon,)), dtype=object)
    if len(parts) == 0:
        return None, None
    invalid = ~shapely.is_valid(parts)
    if invalid.any():
        parts[invalid] = shapely.make_valid(parts[invalid])
    clipped = shapely.intersection(parts, _core_poly(core, shape_crs))
    pieces, labels = [], []
    for k, geom in enumerate(clipped):
        for piece in _parts_of_type(geom, (Polygon,)):
            pieces.append(piece)
            labels.append(k)
    if not pieces:
        return None, None
    return MultiPolygon(pieces), np.array(labels)


def clip_to_core(shape, shape_crs, core: Core):
    """``shape`` clipped to ``core``, of the same kind (polygons stay
    polygons, lines stay lines); None when nothing is left."""
    if shape is None or shape.is_empty:
        return None
    core_poly = _core_poly(core, shape_crs)
    # The kind of shape decides what to keep, taken before the repair
    # below: make_valid often returns a GeometryCollection (polygons
    # plus the lines of collapsed spikes).
    polygonal = isinstance(shape, (Polygon, MultiPolygon))
    lineal = isinstance(shape, (LineString, MultiLineString))
    if isinstance(shape, GeometryCollection) and not (polygonal or lineal):
        return None
    if not shape.is_valid:
        # Detected channels can have spikes, self-touching rings and
        # nested holes, which GEOS cannot intersect.
        shape = shapely.make_valid(shape)
    clipped = shape.intersection(core_poly)
    if polygonal:
        parts = _parts_of_type(clipped, (Polygon,))
        return MultiPolygon(parts) if parts else None
    if lineal:
        parts = _parts_of_type(clipped, (LineString,))
        return MultiLineString(parts) if parts else None
    return None if clipped.is_empty else clipped


def _area_m2(parts, crs, bounds):
    """Area of each polygon of ``parts`` (in ``crs``) in square metres,
    measured in the UTM zone of ``bounds`` when ``crs`` is geographic
    (as ``Raster.get_channels`` measures them)."""
    utm = utils.estimate_bounds_utm(bounds, crs)
    if utm is None:
        return shapely.area(parts)
    transformer = Transformer.from_crs(crs, utm, always_xy=True)

    def to_utm(coords):
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.area(shapely.transform(parts, to_utm))


def _components(n: int, pairs) -> np.ndarray:
    """Connected component (its smallest member) of each of ``n`` items
    linked by ``pairs``."""
    parent = np.arange(n)

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(k) for k in range(n)])


def _shared_side(a: Core, b: Core, eps: float):
    """The edge cores ``a`` and ``b`` share: (axis along it, its
    coordinate, the side of ``a``), or None (not neighbours, or only a
    corner in common). Axis 0 is x (a south/north edge), 1 is y."""
    aw, as_, ae, an = a.bounds()
    bw, bs, be, bn = b.bounds()
    for coord, side in ((ae, 'E'), (aw, 'W')):
        other = bw if side == 'E' else be
        if abs(coord - other) < eps and min(an, bn) - max(as_, bs) > eps:
            return 1, coord, side
    for coord, side in ((an, 'N'), (as_, 'S')):
        other = bs if side == 'N' else bn
        if abs(coord - other) < eps and min(ae, be) - max(aw, bw) > eps:
            return 0, coord, side
    return None


def _on_side(part, core: Core, side: str, eps: float):
    """Extent of ``part`` along one edge of its core: the intervals,
    along the edge, of the part within ``eps`` of it."""
    w, s, e, n = core.bounds()
    rect = {'W': (w - eps, s, w + eps, n), 'E': (e - eps, s, e + eps, n),
            'S': (w, s - eps, e, s + eps), 'N': (w, n - eps, e, n + eps)}[side]
    axis = 1 if side in 'WE' else 0
    strip = shapely.clip_by_rect(part, *rect)
    return [(p.bounds[axis], p.bounds[axis + 2])
            for p in shapely.get_parts(strip) if not p.is_empty]


def _seam_pairs(parts, owner, cores):
    """Pairs of pieces that are the two sides of one channel cut by a
    core edge: in neighbouring cores and sharing a stretch of their
    common edge, however short (a channel may cross it through a narrow
    neck), beyond rounding. Pieces that only meet at a corner, or in
    the same core, are not paired here, as separate parts of one
    unsplit detection stay apart."""
    core = cores[0]
    px = min(abs(core.affine.a), abs(core.affine.e))
    eps = 0.01 * px      # clipping puts both sides on one line, up to rounding
    tree = shapely.STRtree(parts)
    left, right = tree.query(parts, predicate='dwithin', distance=eps)
    cache = {}

    def along(k, side):
        if (k, side) not in cache:
            cache[(k, side)] = _on_side(parts[k], cores[k], side, eps)
        return cache[(k, side)]

    opposite = {'E': 'W', 'W': 'E', 'N': 'S', 'S': 'N'}
    for a, b in zip(left, right):
        if a >= b or owner[a] == owner[b]:
            continue
        shared = _shared_side(cores[a], cores[b], eps)
        if shared is None:
            continue
        side = shared[2]
        contact = sum(max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
                      for lo_a, hi_a in along(a, side)
                      for lo_b, hi_b in along(b, opposite[side]))
        if contact > eps:
            yield a, b


def drop_small_channels(items, width: float) -> list:
    """Channels of split rasters with the small pieces dropped.

    ``Raster.get_channels`` drops channel pieces no larger than
    ``utils.channel_min_area(width)``. A window only sees part of its
    raster, so a channel that crosses a window edge is cut into pieces
    that may each be smaller than that while the whole channel is not.
    Windows therefore detect with ``drop_small=False``; this joins the
    clipped pieces that continue each other across the edge between two
    cores of one raster (they share a stretch of it) and drops those
    whose joined area is too small.

    ``items`` are ``(shape, crs, core, labels)``, one per detected shape,
    as ``clip_channels_to_core`` returns them: the polygons of ``shape``
    with the same label come from one detected polygon and stay
    together (labels None: each polygon on its own). Shapes with no
    core (unsplit rasters) are returned unchanged. Returns the shapes in
    the same order, None where nothing is left.
    """
    out = [item[0] for item in items]
    groups = {}
    for i, (shape, _, core, _) in enumerate(items):
        if core is not None and shape is not None:
            key = (core.transform, core.shape, core.crs)
            groups.setdefault(key, []).append(i)
    min_area = utils.channel_min_area(width)
    for idxs in groups.values():
        crs = CRS.from_user_input(items[idxs[0]][1])
        parts, owner, label = [], [], []
        for i in idxs:
            shape_parts = list(shapely.get_parts(items[i][0]))
            labels = items[i][3]
            if labels is None or len(labels) != len(shape_parts):
                labels = np.arange(len(shape_parts))
            for part, lab in zip(shape_parts, labels):
                if isinstance(part, Polygon) and not part.is_empty:
                    parts.append(part)
                    owner.append(i)
                    label.append(int(lab))
        if not parts:
            continue
        parts = np.array(parts, dtype=object)
        owner = np.array(owner)
        bounds = shapely.total_bounds(parts)
        area = _area_m2(parts, crs, tuple(bounds))

        cores = [items[i][2] for i in owner]
        first = {}
        same_polygon = []
        for k, key in enumerate(zip(owner, label)):
            if key in first:
                same_polygon.append((first[key], k))
            else:
                first[key] = k
        roots = _components(len(parts), itertools.chain(
            same_polygon, _seam_pairs(parts, owner, cores)))
        total = np.bincount(roots, weights=area, minlength=len(parts))
        keep = total[roots] > min_area

        for i in idxs:
            kept = list(parts[keep & (owner == i)])
            out[i] = MultiPolygon(kept) if kept else None
    return out
