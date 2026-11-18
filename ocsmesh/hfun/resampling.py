"""DEM resampling for raster size functions.

A ``resampling_factor`` is a fraction of the size function's ``hmin``:
with ``hmin=50`` m a factor of ``0.25`` asks for 12.5 m pixels. It is set
on the collector, for the grid the sizes are computed on, and optionally
on a refinement (``add_contour``, ``add_channel``,
``add_subtidal_flow_limiter``, ``add_constant_value``) for the DEM that
refinement is computed from:

``None``
    On the collector: no resampling (the DEM's own pixels, the
    default). On a refinement: use the collector's value.
``'native'``
    On a refinement: the DEM's own pixels, whatever the collector uses.
a positive number
    Pixels of ``factor * hmin`` metres. DEMs are never resampled to
    pixels finer than their own.
``'optimum'``
    On the collector: the factors found on the New Jersey test domain
    (``OPTIMUM_GRID`` for the grid, ``OPTIMUM_REFINEMENTS`` for each
    refinement that does not set its own). A refinement's own factor
    still wins.
"""

import logging
import math
from typing import Optional, Union

from rasterio.enums import Resampling

from ocsmesh.raster import Raster


_logger = logging.getLogger(__name__)

NATIVE = 'native'
OPTIMUM = 'optimum'

# The factors resampling_factor='optimum' uses: the coarsest that kept the
# size function within the 95% rule on the New Jersey test domain (CUDEM
# 3 m tiles, hmin=50 m, hmax=3000 m), with hmin/2 as the coarsest grid.
# (factor, method); method None is the collector's.
OPTIMUM_GRID = (0.5, 'average')
OPTIMUM_REFINEMENTS = {
    # sizes on the grid, channels detected on finer pixels
    'channel': (0.125, None),
    # contours on the grid's pixels, keeping the deepest value
    'contour': (0.5, 'min'),
    # Not covered: no setting kept the limiter's sizes. Slopes on the
    # DEM's own pixels are the closest, but the sizes still end on the
    # coarse grid (see the TODO in HfunRaster.refine_on), so adding a
    # flow limiter to a resampled grid warns.
    'flow_limiter': (NATIVE, None),
    # depth bands on the grid
    'constant_value': (None, None),
}
OPTIMUM_WARNING = (
    "resampling_factor='optimum' uses the factors found on the New Jersey "
    "test domain (CUDEM 3 m tiles, hmin=50 m, hmax=3000 m). They may not "
    "suit other domains, DEMs or hmin/hmax: check the resulting mesh, or "
    "set the factors yourself. Flow limiters are not covered: leave "
    "resampling_factor unset when using add_subtidal_flow_limiter.")

Factor = Union[float, str, None]


def check_factor(factor: Factor, name: str = 'resampling_factor',
                 allow_native: bool = True) -> Factor:
    """Validate a resampling factor; returns it as float, 'native' or None."""

    if factor is None:
        return None
    if isinstance(factor, str):
        if allow_native and factor == NATIVE:
            return NATIVE
        expected = "a positive number, 'native' or None" if allow_native \
            else "a positive number or None"
        raise ValueError(f"{name} must be {expected}, got {factor!r}")
    value = float(factor)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, got {factor!r}")
    return value


def optimum_defaults(kind: str, factor: Factor,
                     method: Optional[str]) -> tuple:
    """``(factor, method)`` of a refinement of ``kind`` (a key of
    ``OPTIMUM_REFINEMENTS``) under ``resampling_factor='optimum'``: its
    own when given, the optimum otherwise."""

    if factor is not None:
        return factor, method
    opt_factor, opt_method = OPTIMUM_REFINEMENTS[kind]
    return opt_factor, method if method is not None else opt_method


def check_method(method: Union[str, Resampling, None],
                 name: str = 'resampling_method'
                 ) -> Optional[str]:
    """Validate a resampling method; returns its name or None."""

    if method is None:
        return None
    if isinstance(method, Resampling):
        return method.name
    if method not in Resampling.__members__:
        raise ValueError(
            f"{name} must be one of {sorted(Resampling.__members__)}, "
            f"got {method!r}")
    return str(method)


def resolve_factor(own: Factor, inherited: Optional[float]
                   ) -> Optional[float]:
    """Factor a refinement uses: ``None`` for the DEM's own pixels."""

    if own is None:
        return inherited
    if own == NATIVE:
        return None
    return float(own)


def target_resolution(factor: Optional[float],
                      hmin: Optional[float]) -> Optional[float]:
    """Pixel size in metres for ``factor``, or None for native pixels."""

    if factor is None:
        return None
    if hmin is None:
        raise ValueError("resampling_factor needs hmin to be set")
    return factor * hmin


def scaling_for(raster: Raster, res_m: Optional[float]) -> Optional[float]:
    """``Raster.resample`` scaling factor that gives ``res_m`` pixels.

    None when the raster is already that coarse (DEMs are never
    resampled to finer pixels).
    """

    if res_m is None:
        return None
    px, py = raster.pixel_size_m()
    current = math.sqrt(px * py)
    scale = current / res_m
    # A pixel within 1% of the target is kept as it is.
    if scale >= 0.99:
        return None
    return scale


def resampled_copy(raster: Raster, res_m: Optional[float],
                   method: str) -> Raster:
    """``raster`` resampled to ``res_m`` metres, as a new Raster.

    Returns ``raster`` itself when no resampling is needed. The copy is
    written to its own temporary file; ``raster`` is not modified.
    """

    scale = scaling_for(raster, res_m)
    if scale is None:
        return raster
    copy = Raster(str(raster.tmpfile), crs=raster.crs,
                  chunk_size=raster.chunk_size, overlap=raster.overlap)
    copy.resample(scale, method)
    _logger.info(
        "Resampled %s to %.1f m pixels (%s, %d x %d)",
        raster.path, res_m, method, copy.src.width, copy.src.height)
    return copy
