"""Metric coordinate reference systems for size functions and meshing.

OCSMesh measures sizes, distances, slopes and areas in metres. For
geographic inputs it projects to a metric CRS first. ``metric_crs``
chooses that CRS:

``'auto'`` (default)
    UTM, with the zone at the centre of the domain, when the worst
    size error of UTM over the domain is at most ``UTM_MAX_ERROR``
    (0.5%), so small and medium domains are projected as before.
    Otherwise the conformal projection fitted to the domain with the
    smallest worst error: transverse Mercator centred on the domain,
    Lambert conformal conic with standard parallels at 1/6 and 5/6 of
    the domain's latitudes, or oblique stereographic centred on the
    domain. Conformal projections keep angles, so mesh elements are
    not stretched; only their size is scaled, and the scale factor is
    set so that the error is the same at both extremes.
``'utm'``
    Always UTM at the centre of the domain (the behaviour before
    ``metric_crs`` existed).
any CRS (``'EPSG:3338'``, a PROJ string, a ``pyproj.CRS``)
    That CRS, for everything: the domain-wide steps and the tile-local
    ones (distances to features, slopes, channel widths and areas).

With ``'auto'`` and ``'utm'``, tile-local steps keep the UTM zone of
the tile or window they work on, which is the most accurate locally.

Longitudes may be in -180..180 or 0-360; the centre of the domain is
taken on the shortest arc that holds all its longitudes, so domains
crossing 180° get the right UTM zone and projection.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import shapely
from pyproj import CRS, Proj
import utm


_logger = logging.getLogger(__name__)

AUTO = 'auto'
UTM = 'utm'

# 'auto' keeps UTM when its worst size error over the domain is at
# most this (0.5%)
UTM_MAX_ERROR = 0.005

# Points used to measure projection errors
_N_SAMPLE = 2000

MetricCRS = Union[str, CRS, None]


def check_metric_crs(metric_crs: MetricCRS) -> Union[str, CRS]:
    """``metric_crs`` validated: ``'auto'``, ``'utm'`` or a projected
    ``CRS`` with metre units. ``None`` gives ``'auto'``."""

    if metric_crs is None:
        return AUTO
    if isinstance(metric_crs, str) and metric_crs.lower() in (AUTO, UTM):
        return metric_crs.lower()
    try:
        crs = CRS.from_user_input(metric_crs)
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise ValueError(
            f"metric_crs must be 'auto', 'utm' or a CRS, got "
            f"{metric_crs!r}") from err
    if not crs.is_projected:
        raise ValueError(
            f"metric_crs must be a projected CRS (sizes are in metres), "
            f"got {crs.name}")
    unit = crs.axis_info[0].unit_name if crs.axis_info else 'metre'
    if unit not in ('metre', 'meter'):
        raise ValueError(
            f"metric_crs must have metre units, {crs.name} has {unit}")
    return crs


def explicit(metric_crs: MetricCRS) -> Optional[CRS]:
    """The CRS of an explicit ``metric_crs``, or None for 'auto'/'utm'."""

    checked = check_metric_crs(metric_crs)
    return checked if isinstance(checked, CRS) else None


def wrap_lon(lon, lon_min: float = -180.0):
    """Longitudes put in ``[lon_min, lon_min + 360)``."""

    return (np.asarray(lon, dtype=float) - lon_min) % 360.0 + lon_min


def lon_arc(lon) -> Tuple[float, float]:
    """``(west, east)``: the shortest arc holding all ``lon``.

    ``west`` is in -180..180 and ``east`` in ``west``..``west + 360``,
    so an arc crossing 180° has ``east > 180`` (e.g. 170, 200).
    """

    lon = np.unique(wrap_lon(np.asarray(lon, dtype=float).ravel()))
    if lon.size == 0:
        raise ValueError("no longitudes")
    if lon.size == 1:
        return float(lon[0]), float(lon[0])
    # The largest gap between consecutive longitudes (around the
    # circle) is outside the domain.
    gaps = np.diff(np.append(lon, lon[0] + 360.0))
    k = int(np.argmax(gaps))
    west, east = lon[(k + 1) % lon.size], lon[k]
    if east < west:
        east += 360.0
    return float(west), float(east)


def lon_center(lon) -> float:
    """Centre of the shortest arc holding all ``lon``, in -180..180."""

    west, east = lon_arc(lon)
    return float(wrap_lon((west + east) / 2.0))


def bounds_center(bounds) -> Tuple[float, float]:
    """``(lon, lat)`` centre of geographic ``bounds``, with longitudes
    in -180..180 or 0-360 (``x0 <= x1``); the longitude in -180..180."""

    x0, y0, x1, y1 = bounds
    return float(wrap_lon((x0 + x1) / 2.0)), (y0 + y1) / 2.0


def utm_crs_at(lon: float, lat: float, ellps: str = 'WGS84') -> CRS:
    """The UTM CRS of the zone holding ``(lon, lat)``; ``lon`` may be
    in 0-360. Norway and Svalbard zones as ``utm.from_latlon``."""

    _, _, number, _ = utm.from_latlon(lat, float(wrap_lon(lon)))
    # PyProj 3.2.1 throws error if letter is provided
    return CRS(proj='utm', zone=f'{number}', south=lat < 0, ellps=ellps)


def ellps_of(crs) -> str:
    """PROJ ellipsoid name of geographic ``crs`` (WGS84 or GRS80)."""

    name = CRS.from_user_input(crs).ellipsoid.name
    return {'GRS 1980': 'GRS80', 'WGS 84': 'WGS84'}.get(name, 'WGS84')


def sample_points(lon, lat, n: int = _N_SAMPLE):
    """At most ``n`` of the points, the extreme ones included."""

    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    ok = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[ok], lat[ok]
    if lon.size <= n:
        return lon, lat
    idx = set(np.linspace(0, lon.size - 1, n).astype(int).tolist())
    unwrapped = _unwrap(lon)
    for arr in (unwrapped, lat):
        idx.update((int(np.argmin(arr)), int(np.argmax(arr))))
    idx = np.array(sorted(idx))
    return lon[idx], lat[idx]


def _unwrap(lon):
    """Longitudes continuous on the shortest arc holding them."""

    west, _ = lon_arc(lon)
    return wrap_lon(lon, west)


def scale_error(crs, lon, lat, rescale: bool = True) -> Tuple[float, float]:
    """``(error, k0)``: the worst size error in any direction of ``crs``
    over the points, as a fraction (0.01 is 1%). Points the CRS cannot
    project give inf.

    ``rescale=True``: after scaling by the best constant ``k0`` (for the
    fitted projections, which are then built with that ``k0``).
    ``rescale=False``: as the CRS is (``k0`` = 1), e.g. for UTM, whose
    scale factor is fixed.
    """

    factors = Proj(CRS.from_user_input(crs)).get_factors(
        wrap_lon(lon), np.asarray(lat, dtype=float), errcheck=False)
    a = np.asarray(factors.tissot_semimajor, dtype=float)
    b = np.asarray(factors.tissot_semiminor, dtype=float)
    if a.size == 0 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return np.inf, 1.0
    hi, lo = a.max(), b.min()
    if not rescale:
        return float(max(hi - 1.0, 1.0 - lo)), 1.0
    return float((hi - lo) / (hi + lo)), float(2.0 / (hi + lo))


def _candidates(lon, lat, ellps):
    """``{name: PROJ string}`` of the conformal projections fitted to
    the points (scale factor 1; it is set afterwards)."""

    west, east = lon_arc(lon)
    lon_0 = float(wrap_lon((west + east) / 2.0))
    south, north = float(np.min(lat)), float(np.max(lat))
    lat_0 = (south + north) / 2.0
    tail = f'+ellps={ellps} +units=m +no_defs'
    cands = {
        'TM centred': f'+proj=tmerc +lat_0={lat_0} +lon_0={lon_0} +k_0=1 {tail}',
        'stereographic centred':
            f'+proj=stere +lat_0={lat_0} +lon_0={lon_0} +k_0=1 {tail}',
    }
    lat_1 = south + (north - south) / 6.0
    lat_2 = north - (north - south) / 6.0
    # A cone needs its standard parallels off the equator's mirror
    # image (lat_1 = -lat_2 is a cylinder)
    if abs(lat_1 + lat_2) > 1.0:
        cands['LCC fitted'] = (
            f'+proj=lcc +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} '
            f'+lon_0={lon_0} +k_0=1 {tail}')
    return cands


def pick_metric_crs(lon, lat, metric_crs: MetricCRS = AUTO,
                    utm_crs: Optional[CRS] = None,
                    geo_crs='EPSG:4326') -> CRS:
    """The metric CRS for a domain given by points (geographic).

    Parameters
    ----------
    lon, lat : array-like
        Points of the domain (e.g. mesh nodes or polygon vertices) in
        ``geo_crs``; longitudes in -180..180 or 0-360.
    metric_crs : 'auto', 'utm' or CRS, default='auto'
        See the module documentation.
    utm_crs : CRS or None
        The UTM CRS to use for 'utm' and to try first for 'auto'. When
        None, the zone at the centre of the points.
    geo_crs : CRS, default='EPSG:4326'
        The CRS of the points.
    """

    checked = check_metric_crs(metric_crs)
    if isinstance(checked, CRS):
        return checked
    ellps = ellps_of(geo_crs)
    lon, lat = sample_points(lon, lat)
    if lon.size == 0:
        raise ValueError("no points to choose a metric CRS from")
    if utm_crs is None:
        west, east = lon_arc(lon)
        utm_crs = utm_crs_at((west + east) / 2.0,
                             (np.min(lat) + np.max(lat)) / 2.0, ellps)
    if checked == UTM:
        return utm_crs

    utm_err, _ = scale_error(utm_crs, lon, lat, rescale=False)
    if utm_err <= UTM_MAX_ERROR:
        _logger.debug(
            f"metric CRS: UTM ({utm_crs.srs}), max size error "
            f"{100 * utm_err:.2f}%")
        return utm_crs

    best = ('UTM', utm_crs, utm_err)
    for name, proj in _candidates(lon, lat, ellps).items():
        err, k0 = scale_error(proj, lon, lat)
        if err < best[2]:
            best = (name, CRS.from_proj4(
                proj.replace('+k_0=1', f'+k_0={k0:.10f}')), err)
    name, crs, err = best
    _logger.info(
        f"metric CRS: {name} ({crs.srs}), max size error "
        f"{100 * err:.2f}% (UTM: {100 * utm_err:.2f}%)")
    return crs


def estimate_bounds_utm(bounds, crs='EPSG:4326') -> Optional[CRS]:
    """UTM CRS at the centre of ``bounds`` (in ``crs``), or None when
    ``crs`` is not geographic (its coordinates are metric already).

    Longitudes may be in -180..180 or 0-360; bounds of rasters stored
    south-up or east-to-west (``y1 < y0``, ``x1 < x0``) are sorted
    first. Raises ``ValueError`` for bounds that cannot be
    longitudes and latitudes, or that span more than 180° of
    longitude (no UTM zone fits; pass ``metric_crs``), instead of
    leaving the caller to compute in degrees.
    """

    in_crs = CRS.from_user_input(crs)
    if not in_crs.is_geographic:
        return None
    x0, x1 = sorted((float(bounds[0]), float(bounds[2])))
    y0, y1 = sorted((float(bounds[1]), float(bounds[3])))
    if not (-90.0 <= y0 <= y1 <= 90.0 and -180.0 <= x0 and x1 <= 360.0):
        raise ValueError(
            f"Bounds {tuple(bounds)} are not longitudes and latitudes in "
            f"-180..180 or 0-360 x -90..90, although the CRS "
            f"({in_crs.name}) is geographic; check the CRS of the input.")
    if x1 - x0 > 180.0:
        raise ValueError(
            f"Bounds {tuple(bounds)} span {x1 - x0:.1f}° of longitude: no "
            f"UTM zone fits. Pass metric_crs (e.g. metric_crs='auto' with "
            f"the domain's points, or a CRS such as 'EPSG:3338').")
    lon, lat = bounds_center((x0, y0, x1, y1))
    return utm_crs_at(lon, lat, ellps_of(in_crs))


def local_metric_crs(bounds, crs, metric_crs: MetricCRS = None
                     ) -> Optional[CRS]:
    """The metric CRS of a tile-local step on ``bounds`` (in ``crs``):
    an explicit ``metric_crs``, else the UTM zone of ``bounds`` (None
    when ``crs`` is metric already)."""

    crs_explicit = explicit(metric_crs)
    if crs_explicit is not None:
        return crs_explicit
    return estimate_bounds_utm(bounds, crs)


# ---------------------------------------------------------------------
# Longitude conventions: shapes and points against rasters
# ---------------------------------------------------------------------
#
# Inputs may use -180..180 or 0-360 longitudes, and PROJ returns
# -180..180 from any transformation to geographic coordinates. Before a
# geographic comparison (clip, mask, within, reach) one side is shifted
# by a multiple of 360 to the copy nearest the other.


def wrap_lon_near(lon, ref_lon: float):
    """Longitudes shifted by multiples of 360 into ``ref_lon ± 180``.

    Longitudes already there are returned exactly as they are (no
    floating point change), so points on a shape's edge stay on it.
    """

    lon = np.asarray(lon, dtype=float)
    return lon + 360.0 * np.round((ref_lon - lon) / 360.0)


def _shift_x(geom, dx: float):
    if dx == 0.0:
        return geom
    return shapely.transform(geom, lambda xy: xy + np.array([dx, 0.0]))


def align_lon(geoms, ref_lon: float) -> list:
    """Each geometry of ``geoms`` shifted by a multiple of 360 so that
    the centre of its bounds is within ``ref_lon ± 180`` (the copy
    nearest ``ref_lon``). Empty geometries are returned as they are."""

    geoms = list(geoms)
    if not geoms:
        return geoms
    bounds = shapely.bounds(np.asarray(geoms, dtype=object))
    cx = (bounds[:, 0] + bounds[:, 2]) / 2.0
    ok = np.isfinite(cx)
    shift = np.zeros(len(geoms))
    shift[ok] = np.round(
        (wrap_lon_near(cx[ok], ref_lon) - cx[ok]) / 360.0) * 360.0
    return [_shift_x(g, float(s)) for g, s in zip(geoms, shift)]


def align_geometry(geom, ref_lon: float):
    """``geom`` with each of its parts shifted by a multiple of 360 to
    the copy nearest ``ref_lon`` (see ``align_lon``): a shape given
    in either convention, or split at 180, meets a raster in its own."""

    if geom is None or geom.is_empty:
        return geom
    parts = list(shapely.get_parts(geom))
    aligned = align_lon(parts, ref_lon)
    if all(a is p for a, p in zip(aligned, parts)):
        return geom
    if len(aligned) == 1:
        return aligned[0]
    kinds = {g.geom_type for g in aligned}
    multi = {'Polygon': shapely.MultiPolygon,
             'LineString': shapely.MultiLineString,
             'Point': shapely.MultiPoint}.get(
                 kinds.pop() if len(kinds) == 1 else None,
                 shapely.GeometryCollection)
    return multi(aligned)


def bounds_lon_center(bounds) -> float:
    """Longitude at the centre of ``bounds`` (west, south, east, north;
    rasters stored east-to-west give west > east)."""

    west, east = sorted((float(bounds[0]), float(bounds[2])))
    return (west + east) / 2.0
