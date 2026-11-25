"""Splitting large rasters for channel and contour detection.

Detecting channels and contours is the costly part of the size function
on a large DEM, and with one task per raster a large raster keeps one
worker busy while the others wait. With ``split_size`` the detection
runs on windows of such rasters instead, as separate tasks. Only the
detection is split: the size function of every raster is computed
whole, from the shapes detected on all windows, as without splitting.

Each window is the raster's detection grid (the DEM at the definition's
resolution) cut to a *core* plus a *halo* around it, and only the shapes
in the core are kept. The cores of a raster tile its grid, so every
pixel is detected once, and every window pixel is a pixel of the whole
raster's grid. The halo gives the core the neighbourhood detection looks
at: a channel is only recognised when its whole width is visible, and
without a halo the window edges themselves look like channels.

Channel detection drops channels below a size. A window only sees part
of a channel that crosses its edge, so windows keep their small pieces
and ``drop_small_channels`` joins the pieces of all windows before
applying the cutoff.
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
from rasterio import warp
from rasterio.transform import array_bounds, from_bounds
from shapely import ops
from shapely.geometry import (
    GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon,
    box)

from ocsmesh import utils, geodesy
from ocsmesh.raster import Raster, _as_resampling


OPTIMUM = 'optimum'
# split_size='optimum': the window size used on the New Jersey test domain
# (CUDEM tiles of about 8,100 x 8,100 pixels at 3 m, cut 2 x 2), where
# channel detection matched the unsplit run 2.2 times faster with MPI.
OPTIMUM_SPLIT_SIZE = 4100
OPTIMUM_WARNING = (
    "split_size='optimum' uses the window size found on the New Jersey "
    "test domain (4100 pixels, CUDEM 3 m tiles of about 8,100 pixels). "
    "Other domains or DEMs may need another: check the run, or set "
    "split_size yourself.")


@dataclass(frozen=True)
class Grid:
    """A raster's pixel grid: affine ``transform`` (as a tuple), ``shape``
    (height, width) and ``crs`` (WKT)."""

    transform: tuple
    shape: Tuple[int, int]
    crs: str

    @property
    def affine(self) -> rasterio.Affine:
        return rasterio.Affine(*self.transform[:6])

    @classmethod
    def of(cls, raster: Raster) -> 'Grid':
        """The grid of ``raster``."""
        return cls(tuple(raster.src.transform),
                   (raster.src.height, raster.src.width),
                   raster.crs.to_wkt())

    @classmethod
    def resampled(cls, raster: Raster, scale: float) -> 'Grid':
        """The grid ``Raster.resample(scale)`` gives ``raster``."""
        src = raster.src
        width = max(1, int(round(src.width * scale)))
        height = max(1, int(round(src.height * scale)))
        return cls(tuple(from_bounds(*src.bounds, width, height)),
                   (height, width), raster.crs.to_wkt())

    def pixel_size_m(self) -> Tuple[float, float]:
        """Approximate pixel size ``(x, y)`` in metres, as
        ``Raster.pixel_size_m`` gives it."""
        a = self.affine
        crs = CRS.from_user_input(self.crs)
        if crs.is_geographic:
            height = self.shape[0]
            lat = math.radians(a.f + a.e * height / 2)
            return (abs(a.a) * 111_320.0 * math.cos(lat),
                    abs(a.e) * 110_540.0)
        unit = crs.axis_info[0].unit_conversion_factor or 1.0
        return abs(a.a) * unit, abs(a.e) * unit


@dataclass(frozen=True)
class Core:
    """Pixels of a raster's detection grid that one window keeps:
    half-open ranges ``cols`` and ``rows`` of ``grid``."""

    grid: Grid
    cols: Tuple[int, int]
    rows: Tuple[int, int]

    @property
    def affine(self) -> rasterio.Affine:
        return self.grid.affine

    @property
    def crs(self) -> str:
        return self.grid.crs

    def bounds(self) -> Tuple[float, float, float, float]:
        """West, south, east, north of the core (in ``crs``)."""
        c0, c1 = self.cols
        r0, r1 = self.rows
        west, south, east, north = array_bounds(
            r1 - r0, c1 - c0,
            self.affine * rasterio.Affine.translation(c0, r0))
        return (min(west, east), min(south, north),
                max(west, east), max(south, north))


@dataclass(frozen=True)
class Window:
    """What a detection task reads: ``core`` grown by ``halo`` pixels
    (limited to the grid), from the raster at ``source`` (a file), whose
    pixels are reprojected onto the grid with ``resampling`` (None: the
    source is on the grid already and its pixels are copied)."""

    source: str
    core: Core
    halo: Tuple[int, int]
    resampling: Optional[str] = None

    def pixels(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Column and row ranges of the window on the grid."""
        height, width = self.core.grid.shape
        hx, hy = self.halo
        c0, c1 = self.core.cols
        r0, r1 = self.core.rows
        return ((max(0, c0 - hx), min(width, c1 + hx)),
                (max(0, r0 - hy), min(height, r1 + hy)))


def _ranges(n_pixels: int, n_parts: int) -> List[Tuple[int, int]]:
    """``n_parts`` consecutive half-open ranges covering ``n_pixels``."""
    edges = np.linspace(0, n_pixels, n_parts + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def plan_cores(grid: Grid, native: Raster, split_size: int) -> List[Core]:
    """Cores of ``grid``, or an empty list when the raster needs no
    split.

    ``split_size`` is the largest window side in pixels of ``native``
    (the DEM at its own resolution); the cores are laid out on ``grid``
    (the detection grid) so that each of its pixels is in one core.
    """
    height, width = grid.shape
    n_x = min(math.ceil(native.src.width / split_size), width)
    n_y = min(math.ceil(native.src.height / split_size), height)
    if n_x * n_y <= 1:
        return []
    return [Core(grid, cols, rows)
            for rows in _ranges(height, n_y)
            for cols in _ranges(width, n_x)]


def halo_pixels(grid: Grid, halo_m: float) -> Tuple[int, int]:
    """``halo_m`` metres in pixels of ``grid`` along x and y, rounded up.

    A 2% margin covers the pixel width of a geographic grid shrinking
    away from its middle latitude.
    """
    if halo_m <= 0:
        return 0, 0
    px_x, px_y = grid.pixel_size_m()
    return math.ceil(1.02 * halo_m / px_x), math.ceil(1.02 * halo_m / px_y)


def write_window(window: Window, out_path: str) -> str:
    """Write the pixels of ``window`` to a GeoTIFF; returns its path.

    Every pixel is the pixel of the whole raster's detection grid: copied
    from the source when it is on that grid, otherwise reprojected from
    it onto the grid the way ``Raster.resample`` reprojects the whole
    raster.
    """
    (c0, c1), (r0, r1) = window.pixels()
    width, height = c1 - c0, r1 - r0
    transform = window.core.affine * rasterio.Affine.translation(c0, r0)
    with rasterio.open(window.source) as src:
        profile = src.profile.copy()
        for key in ('blockxsize', 'blockysize', 'tiled', 'compress',
                    'interleave'):
            profile.pop(key, None)
        profile.update(driver='GTiff', width=width, height=height,
                       transform=transform)
        if width >= 256 and height >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with rasterio.open(out_path, 'w', **profile) as dst:
            for band in range(1, src.count + 1):
                if window.resampling is None:
                    data = src.read(band, window=rasterio.windows.Window(
                        c0, r0, width, height))
                else:
                    nodata = src.nodatavals[band - 1]
                    data = np.full((height, width),
                                   nodata if nodata is not None else 0,
                                   dtype=src.dtypes[band - 1])
                    warp.reproject(
                        source=rasterio.band(src, band), destination=data,
                        src_transform=src.transform, src_crs=src.crs,
                        src_nodata=nodata, dst_transform=transform,
                        dst_crs=src.crs, dst_nodata=nodata,
                        resampling=_as_resampling(window.resampling))
                dst.write(data, band)
    return out_path


def to_raster_coords(geom, window: Window):
    """Contour coordinates found on ``window`` as the whole raster gives
    them.

    ``Raster.get_contour`` places pixel ``i`` of a raster ``n`` pixels
    wide at ``x0 + i * dx * n / (n - 1)`` (``Raster.get_x``: from the
    first pixel's outer edge to the last one's), so the position of a
    pixel depends on the raster's extent. This maps the window's
    positions to the whole raster's, so that contours detected on
    windows are those of the whole raster. (For a raster contoured in
    chunks, ``chunk_size``, the positions also depend on the chunks and
    are left as found.)
    """
    if geom is None or geom.is_empty:
        return geom
    (c0, c1), (r0, r1) = window.pixels()
    height, width = window.core.grid.shape
    a = window.core.affine

    def axis(origin, step, offset, n_window, n_whole):
        if n_window < 2 or n_whole < 2:
            return lambda v: v
        win_step = step * n_window / (n_window - 1)
        whole_step = step * n_whole / (n_whole - 1)
        win_origin = origin + offset * step

        def mapped(v):
            index = (v - win_origin) / win_step + offset
            return origin + index * whole_step
        return mapped

    map_x = axis(a.c, a.a, c0, c1 - c0, width)
    map_y = axis(a.f, a.e, r0, r1 - r0, height)

    def to_whole(coords):
        return np.column_stack([map_x(coords[:, 0]), map_y(coords[:, 1])])

    return shapely.transform(geom, to_whole)


def merge_line_pieces(pieces, grid_size: float):
    """Lines cut into ``pieces`` (by core edges) merged back.

    The two sides of a cut are computed on different windows and meet
    up to rounding, so the pieces are snapped to ``grid_size`` (far
    below a pixel) first. Clipping a line adds a vertex where it
    crosses a core edge; merged lines leave these out again, so that
    they have the vertices of the line before it was cut
    (``add_feature`` partitions lines by vertex count).
    """
    lines = [p for piece in pieces
             for p in _parts_of_type(piece, (LineString,))]
    if not lines:
        return None
    lines = [p for line in shapely.set_precision(
                 np.array(lines, dtype=object), grid_size)
             for p in _parts_of_type(line, (LineString,))]
    cuts = {tuple(c) for line in lines
            for c in (line.coords[0], line.coords[-1])}
    merged = shapely.line_merge(shapely.union_all(lines))
    out = []
    for line in _parts_of_type(merged, (LineString,)):
        coords = list(line.coords)
        keep = [coords[0]] + [c for c in coords[1:-1]
                              if tuple(c) not in cuts] + [coords[-1]]
        out.append(LineString(keep))
    return MultiLineString(out)


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
        # Lines that only touch the core leave points, or lines of no
        # length after rounding.
        parts = [p for p in _parts_of_type(clipped, (LineString,))
                 if p.length > 0]
        return MultiLineString(parts) if parts else None
    return None if clipped.is_empty else clipped


def _area_m2(parts, crs, bounds, metric_crs=None):
    """Area of each polygon of ``parts`` (in ``crs``) in square metres,
    measured in ``metric_crs`` when it is a CRS, else in the UTM zone of
    ``bounds`` when ``crs`` is geographic (as ``Raster.get_channels``
    measures them)."""
    utm = geodesy.local_metric_crs(bounds, crs, metric_crs)
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


def drop_small_channels(items, width: float, metric_crs=None) -> list:
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
    the same order, None where nothing is left. Areas are measured in
    ``metric_crs`` when it is a CRS, else in the UTM zone of each
    raster's pieces.
    """
    out = [item[0] for item in items]
    groups = {}
    for i, (shape, _, core, _) in enumerate(items):
        if core is not None and shape is not None:
            key = core.grid
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
        area = _area_m2(parts, crs, tuple(bounds), metric_crs)

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
