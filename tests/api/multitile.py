"""Shared multi-tile test fixture.

Builds a small synthetic DEM stack that has the features real coastal
stacks have and that the single-raster tests never exercise:

* a coarse background tile covering everything (lowest priority),
* two medium tiles that overlap each other,
* two fine tiles with nodata margins along their edges,
* one tile that is entirely nodata but whose footprint intersects the AOI,
* an AOI polygon (``base_shape``) that clips through every tile.

All valid pixels describe the same bathymetry, so any difference between
execution modes, or between tiles where they overlap, comes from the code.

The helpers are MPI-aware: under ``mpiexec`` every rank must call them
collectively (constructing an ``HfunCollector`` broadcasts its work
directory), and only rank 0 receives results.
"""

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon

NODATA = -99999.0
HMIN = 50.0
HMAX = 3000.0

# (value, lower_bound, upper_bound) for HfunCollector.add_constant_value
DEPTH_BANDS = (
    (3000.0, None, -30.0),
    (2500.0, -30.0, -25.0),
    (2000.0, -25.0, -20.0),
    (1500.0, -20.0, -15.0),
    (1000.0, -15.0, None),
)


def bathymetry(x, y):
    """Depth -40 m in the west rising to +5 m east of x = 0.5 (N-S coast)."""
    del y
    return (-40.0 + 45.0 * np.clip(x / 0.5, 0.0, 1.2)).astype('float32')


@dataclass
class TileSpec:
    name: str
    res: float
    bounds: Sequence[float]          # west, south, east, north (EPSG:4326)
    nodata_border_px: int = 0
    all_nodata: bool = False
    declare_nodata: bool = True


@dataclass
class MultiTileStack:
    """Paths are in priority order: the last one wins where tiles overlap."""
    paths: List[str]
    specs: List[TileSpec]
    aoi: Polygon
    tmpdir: Path
    by_name: Dict[str, str] = field(default_factory=dict)

    def without(self, *names: str) -> List[str]:
        return [p for s, p in zip(self.specs, self.paths) if s.name not in names]


def default_specs(scale: float = 1.0, include_empty: bool = True,
                  background_declares_nodata: bool = True) -> List[TileSpec]:
    specs = [
        TileSpec('background', 0.004 * scale, (0.00, 0.00, 0.80, 0.60),
                 declare_nodata=background_declares_nodata),
        TileSpec('medium_a', 0.001 * scale, (0.05, 0.05, 0.42, 0.55), 3),
        TileSpec('medium_b', 0.001 * scale, (0.40, 0.05, 0.75, 0.55), 3),
        TileSpec('fine_a', 0.0004 * scale, (0.30, 0.20, 0.52, 0.45), 5),
        TileSpec('fine_b', 0.0004 * scale, (0.50, 0.20, 0.70, 0.45), 5),
    ]
    if include_empty:
        # footprint touches the AOI near its north-east edge, every pixel nodata
        specs.append(TileSpec('empty', 0.001 * scale, (0.60, 0.45, 0.80, 0.60),
                              all_nodata=True))
    return specs


def default_aoi() -> Polygon:
    """Circle that cuts through every tile of :func:`default_specs`."""
    return Point(0.42, 0.30).buffer(0.27)


def write_tile(path: Path, spec: TileSpec,
               depth_fn: Callable = bathymetry) -> Path:
    x0, y0, x1, y1 = spec.bounds
    width = max(1, int(round((x1 - x0) / spec.res)))
    height = max(1, int(round((y1 - y0) / spec.res)))
    rows, cols = np.mgrid[0:height, 0:width]
    x = x0 + (cols + 0.5) * spec.res
    y = y1 - (rows + 0.5) * spec.res
    if spec.all_nodata:
        z = np.full((height, width), NODATA, dtype='float32')
    else:
        z = depth_fn(x, y).astype('float32')
        b = spec.nodata_border_px
        if b:
            z[:b, :] = NODATA
            z[-b:, :] = NODATA
            z[:, :b] = NODATA
            z[:, -b:] = NODATA
    profile = {
        'driver': 'GTiff', 'width': width, 'height': height, 'count': 1,
        'dtype': 'float32', 'crs': 'EPSG:4326',
        'transform': from_origin(x0, y1, spec.res, spec.res),
    }
    if spec.declare_nodata:
        profile['nodata'] = NODATA
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(z, 1)
    return path


def build_stack(out_dir: Optional[Path] = None, scale: float = 1.0,
                include_empty: bool = True,
                background_declares_nodata: bool = True,
                depth_fn: Callable = bathymetry) -> MultiTileStack:
    """Write the stack on rank 0 and share it with every MPI rank."""
    comm = mpi_comm()
    rank = comm.Get_rank() if comm is not None else 0
    specs = default_specs(scale, include_empty, background_declares_nodata)
    tmpdir = None
    if rank == 0:
        tmpdir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix='multitile_'))
        tmpdir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            write_tile(tmpdir / f'{spec.name}.tif', spec, depth_fn)
    if comm is not None:
        tmpdir = comm.bcast(tmpdir, root=0)
    paths = [str(tmpdir / f'{s.name}.tif') for s in specs]
    return MultiTileStack(paths=paths, specs=specs, aoi=default_aoi(),
                          tmpdir=tmpdir,
                          by_name={s.name: p for s, p in zip(specs, paths)})


def cleanup(stack: MultiTileStack) -> None:
    comm = mpi_comm()
    if comm is not None:
        comm.Barrier()
        if comm.Get_rank() != 0:
            return
    shutil.rmtree(stack.tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------
# MPI helpers
# --------------------------------------------------------------------------

def mpi_comm():
    """COMM_WORLD when running under an MPI launcher with >1 rank, else None."""
    from ocsmesh.mpi import _is_mpi_active, _get_mpi
    if not _is_mpi_active():
        return None
    return _get_mpi().COMM_WORLD


def is_manager() -> bool:
    comm = mpi_comm()
    return comm is None or comm.Get_rank() == 0


def available_modes() -> List[str]:
    """Execution modes testable in this process."""
    return ['serial', 'parallel'] + (['mpi'] if mpi_comm() is not None else [])


# --------------------------------------------------------------------------
# Building collectors and comparing modes
# --------------------------------------------------------------------------

def add_depth_bands(hfun) -> None:
    for value, lower, upper in DEPTH_BANDS:
        kwargs = {'value': value}
        if lower is not None:
            kwargs['lower_bound'] = lower
        if upper is not None:
            kwargs['upper_bound'] = upper
        hfun.add_constant_value(**kwargs)


def make_hfun(paths: Sequence[str], mode: str, aoi: Optional[Polygon] = None,
              method: str = 'exact', nprocs: Optional[int] = None,
              configure: Optional[Callable] = None):
    """Collector for ``paths`` in ``mode`` with optional refinements.

    ``nprocs`` defaults to 1 for serial/mpi and 2 for parallel.
    """
    import warnings
    from ocsmesh import Hfun
    if nprocs is None:
        nprocs = 2 if mode == 'parallel' else 1
    kwargs = {'hmin': HMIN, 'hmax': HMAX, 'nprocs': nprocs, 'method': method}
    if aoi is not None:
        kwargs.update(base_shape=aoi, base_shape_crs='EPSG:4326')
    hfun = Hfun(list(paths), **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hfun.execution_mode = mode
    if configure is not None:
        configure(hfun)
    return hfun


def refined_rasters(paths, mode, aoi=None, configure=None, nprocs=None):
    """Apply the refinements only (exact method) and return the raster arrays.

    Returns a list of 2D arrays (one per raster size function) on rank 0,
    None on MPI worker ranks.
    """
    hfun = make_hfun(paths, mode, aoi=aoi, configure=configure, nprocs=nprocs)
    hfun._apply_features()  # pylint: disable=protected-access
    if not is_manager():
        return None
    return [h.get_values().copy()
            for h in hfun._hfun_list  # pylint: disable=protected-access
            if hasattr(h, 'get_values')]


def meshdata_stats(paths, mode, aoi=None, configure=None, method='exact',
                   nprocs=None):
    """Run meshdata() and return summary stats on rank 0 (None on workers)."""
    hfun = make_hfun(paths, mode, aoi=aoi, configure=configure, method=method,
                     nprocs=nprocs)
    md = hfun.meshdata()
    if md is None:
        return None
    values = np.asarray(md.values, dtype=float).ravel()
    return {'nodes': int(len(md.coords)), 'min': float(values.min()),
            'max': float(values.max()), 'mean': float(values.mean()),
            'meshdata': md}


def compare_rasters(reference: List[np.ndarray], other: List[np.ndarray]) -> str:
    """Empty string if identical, otherwise a short description of the difference."""
    if len(reference) != len(other):
        return f'{len(reference)} vs {len(other)} raster size functions'
    problems = []
    for i, (a, b) in enumerate(zip(reference, other)):
        if a.shape != b.shape:
            problems.append(f'raster {i}: shape {a.shape} vs {b.shape}')
        elif not np.array_equal(a, b):
            n = int(np.sum(a != b))
            problems.append(f'raster {i}: {n} of {a.size} cells differ')
    return '; '.join(problems)
