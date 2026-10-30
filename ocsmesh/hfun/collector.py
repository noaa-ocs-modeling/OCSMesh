"""This module defines size function collector.
`
Size function collector objects accepts a list of valid basic `Hfun`
inputs and creates an object that merges the results of all the
other types of size functions, e.g. mesh-based and raster-based.

Notes
-----
This enables the user to define size on multiple DEM and mesh
without having to worry about the details of merging the output size
functions defined on each DEM or mesh.
"""
import os
import shutil
import gc
import pickle
import logging
import warnings
import tempfile
import traceback
from pathlib import Path
from time import time, sleep
from multiprocessing import Pool
from copy import deepcopy
from typing import (
    Union, Sequence, List, Tuple, Iterable, Any, Optional, Callable, Dict)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import numpy.typing as npt
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import (
    MultiLineString,
    LineString,
    MultiPolygon,
    Polygon,
    GeometryCollection,
)
import shapely
from shapely import ops
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import rasterio

from ocsmesh import utils
from ocsmesh.utils import effective_cpu_count
from ocsmesh.internal import MeshData
from ocsmesh.hfun.base import BaseHfun
from ocsmesh.hfun.raster import HfunRaster
from ocsmesh.hfun.mesh import HfunMesh
from ocsmesh.mesh.mesh import Mesh, EuclideanMesh2D
from ocsmesh.mesh.base import BaseMesh
from ocsmesh.raster import Raster, get_iter_windows
from ocsmesh.features.contour import Contour
from ocsmesh.features.patch import Patch
from ocsmesh.features.linefeature import LineFeature
from ocsmesh.features.channel import Channel
from ocsmesh.features.constraint import (
    TopoConstConstraint,
    TopoFuncConstraint,
    CourantNumConstraint,
    RegionConstraint,
    _default_topo_func,
)

CanCreateSingleHfun = Union[Raster, EuclideanMesh2D]
CanCreateMultipleHfun = Iterable[Union[CanCreateSingleHfun, str]]
CanCreateHfun = Union[CanCreateSingleHfun, CanCreateMultipleHfun]

SizeFuncList = Sequence[Union[HfunRaster, HfunMesh]]

RASTER_CONSTR = (
    TopoConstConstraint,
    TopoFuncConstraint,
    CourantNumConstraint,
)

_logger = logging.getLogger(__name__)

# pylint: disable=wrong-import-position
from ocsmesh.mpi import (
    MPIExecutor,
    _get_mpi,
    _is_mpi_active,
    _is_mpi_env_detected,
    _configure_mpi_environment,
)


class _RefinementContourInfoCollector:
    """Collection for contour refinement specification

    Accumulates information about the specified contour refinements
    to be applied later when computing the return size function.
    Provides interator interface for looping over the collection items.
    """

    def __init__(self) -> None:
        self._contours_info = {}

    def add(self, contour_defn: Contour, **size_info: Any) -> None:
        """Add contour specification to the collection

        Parameters
        ----------
        contour_defn : Contour
            The level at which contour lines need to be extracted.
        size_info : dict
            Information related to contour application such as
            target size, rate, etc.

        Returns
        -------
        None
        """

        self._contours_info[contour_defn] = size_info

    def __len__(self) -> int:
        return len(self._contours_info)

    def __iter__(self) -> Tuple[Contour, dict]:
        """Iterator method for this collection object

        Yields
        ------
        defn : Contour
            Contour definition added to the collection.
        info : dict
            Dictionary of contour refinement specifications.
        """

        for defn, info in self._contours_info.items():
            yield defn, info




class _RefinementContourCollector:
    """Collection for extracted refinement contours

    Extracts and stores on the disk the contours specified by the user.
    Provides interator interface for looping over the collection items.
    """

    def __init__(
            self,
            contours_info: _RefinementContourInfoCollector
            ) -> None:
        """Initialize the collection object with empty output list.

        Parameters
        ----------
        contours_info : _RefinementContourInfoCollector
            Handle to the collection of user specified contours
            specification.
        """

        self._contours_info = contours_info
        self._container: List[Union[Tuple, None]] = []

    def calculate(
            self,
            source_list: Iterable[HfunRaster],
            out_path: Union[Path, str]
            ) -> None:
        """Extract specified contours and store on disk in `out_path`.

        Parameters
        ----------
        source_list : list of HfunRaster
            List of raster size functions from which the contours
            must be calculated.
        out_path : path-like
            Path for storing calculated contours and their crs data.

        Returns
        -------
        None
        """

        out_dir = Path(out_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        file_counter = 0
        pid = os.getpid()
        self._container.clear()
        for contour_defn, size_info in self._contours_info:
            if not contour_defn.has_source:
                # Attach the sources to a copy: the definition must stay
                # sourceless so that a later run uses the rasters of that
                # run. A shallow copy would share (and fill) its list.
                contour_defn = deepcopy(contour_defn)
                for source in source_list:
                    contour_defn.add_source(source)

            for contour, crs in contour_defn.iter_contours():
                file_counter = file_counter + 1
                self._container.append(_write_shape_file(
                    contour, crs, size_info,
                    out_dir / f"contour_{pid}_{file_counter}.feather",
                    out_dir / f"crs_{pid}_{file_counter}.json"))


    @property
    def files(self) -> List[Tuple[Path, Path]]:
        return list(self._container)

    def set_files(self, files: List[Tuple[Path, Path]]) -> None:
        """Use shape files detected elsewhere (e.g. by workers)."""
        self._container[:] = files


    def __iter__(self) -> gpd.GeoDataFrame:
        """Iterator method for this collection object

        Yields
        ------
        gpd.GeoDataFrame
            Data from containing the extracted contour line and the
            CRS information.
        """

        for raster_data in self._container:
            feather_path, crs_path = raster_data
            gdf = gpd.read_feather(feather_path)
            with open(crs_path) as fp:
                gdf.set_crs(CRS.from_json(fp.read()))
            yield gdf




class _ConstantValueContourInfoCollector:
    """Collection for constant value refinement specification"""

    def __init__(self) -> None:
        self._contours_info = {}

    def add(self,
            src_idx: Optional[Sequence[int]],
            contour_defn0: Contour,
            contour_defn1: Contour,
            value: float
            ) -> None:
        """Add a new fixed-value refinement specification to the spec.

        Parameters
        ----------
        src_idx : tuple of int or None
            Indices of sources (indexed based on all `HfunCollector`
            **not** just rasters) on which constant value refinement
            must be applied.
        contour_defn0 : Contour
            Lower bound of region to apply constant value refinement.
        contour_defn1 : Contour
            Upper bound of region to apply constant value refinement.
        value : float
            Fixed-value to be applied as refinement.

        Returns
        -------
        None
        """

        srcs = tuple(src_idx) if src_idx is not None else None
        self._contours_info[
                (srcs, contour_defn0, contour_defn1)] = value

    def __len__(self) -> int:
        return len(self._contours_info)

    def __iter__(self) -> Tuple[Tuple[Sequence[int], Contour, Contour], dict]:
        """Iterator method for this collection object

        Yields
        ------
        defn : Tuple[Sequence[int], Contour, Contour]
            The lower and upper bound contours definitions provided
            by the user as well as the list of source.
        info : dict
            Dictionary of specifications for constant value refinement.
        """

        for defn, info in self._contours_info.items():
            yield defn, info



class _RefinementShapeInfoCollector:
    """Collection for shape (patch or line) refinement specifications"""

    def __init__(self) -> None:
        self._shape_info = {}

    def add(self,
            shape_defn: Union[Patch, LineFeature],
            **size_info: Any
        ) -> None:
        """Add shape refinement specifications to the collection

        Parameters
        ----------
        shape_defn : Patch or LineFeature
            Shape of the region to apply the refinement during
            application.
        size_info : dict
            Information related to shape application such as
            target size, rate, etc.

        Returns
        -------
        None
        """

        self._shape_info[shape_defn] = size_info

    def __len__(self) -> int:
        return len(self._shape_info)

    def __iter__(self) -> Tuple[Union[Patch, LineFeature], dict]:
        """Iterator method for this collection object

        Yields
        ------
        defn : Patch or LineFeature
            Object representing the shape of refinement
        info : dict
            Dictionary of specifications for shape refinement.
        """

        for defn, info in self._shape_info.items():
            yield defn, info



class _FlowLimiterInfoCollector:
    """Collection for subtidal flow limiter refinement spec."""

    def __init__(self) -> None:
        self._flow_lim_info = []

    def add(self,
            src_idx: Optional[Sequence[int]],
            hmin: float,
            hmax: float,
            upper_bound: float,
            lower_bound: float
            ) -> None:
        """Add subtidal flow limiter refinement spec to the collection.

        Parameters
        ----------
        src_idx : tuple of int or None
            Indices of sources (indexed based on all `HfunCollector`
            **not** just rasters) on which subtidal flow limiter
            refinement must be applied.
        hmin : float
            Minimum mesh size to be applied based on limiter
            calculations.
        hmax : float
            Maximum mesh size to be applied based on limiter
            calculations.
        upper_bound : float
            Elevation upper bound of the area that the limiter
            refinement is applied.
        lower_bound : float
            Elevation lower bound of the area that the limiter
            refinement is applied.

        Returns
        -------
        None
        """

        srcs = tuple(src_idx) if src_idx is not None else None
        self._flow_lim_info.append(
                (src_idx, hmin, hmax, upper_bound, lower_bound))

    def __len__(self) -> int:
        return len(self._flow_lim_info)

    def __iter__(self) -> Tuple[Sequence[int], float, float, float, float]:
        """Iterator method for this collection object

        Yields
        ------
        src_idx : tuple of int
            Similar to `add`
        hmin : float
            Similar to `add`
        hmax : float
            Similar to `add`
        ub : float
            Similar to `add`
        lb : float
            Similar to `add`
        """

        for src_idx, hmin, hmax, ub, lb in self._flow_lim_info:
            yield src_idx, hmin, hmax, ub, lb



class _ChannelRefineInfoCollector:
    """Collection for channel refinement specifications"""

    def __init__(self) -> None:
        self._ch_info_dict = {}

    def add(self,
            channel_defn: Channel,
            **size_info: Any
            ) -> None:
        """Add channel refinement spec to the collection.

        Parameters
        ----------
        channel_defn : Channel
            Definition of channel detection specification.
        size_info : dict
            Information related to channel refinement application
            such as target size, rate, etc.

        Returns
        -------
        None
        """

        self._ch_info_dict[channel_defn] = size_info

    def __len__(self) -> int:
        return len(self._ch_info_dict)

    def __iter__(self) -> Tuple[Channel, dict]:
        """Iterator method for this collection object

        Yields
        ------
        defn : Channel
            Similar to `channel_defn` in `add`
        info : dict
            Similar to `size_info` in `add`
        """

        for defn, info in self._ch_info_dict.items():
            yield defn, info


class _ChannelRefineCollector:
    """Collection for extracted refinement channels"""

    def __init__(self, channels_info) -> None:
        """Initialize the collection object with empty output list.

        Parameters
        ----------
        channels_info : _ChannelRefineInfoCollector
            Handle to the collection of user specified channel
            refinement specification.
        """

        self._channels_info = channels_info
        self._container: List[Union[Tuple, None]] = []

    def calculate(
            self,
            source_list,
            out_path
            ) -> None:
        """Extract specified channels and store on disk in `out_path`.

        Parameters
        ----------
        source_list : list of HfunRaster
            List of raster size functions from which the channels
            must be calculated.
        out_path : path-like
            Path for storing calculated channels and their crs data.

        Returns
        -------
        None
        """

        out_dir = Path(out_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        file_counter = 0
        pid = os.getpid()
        self._container.clear()
        for channel_defn, size_info in self._channels_info:
            if not channel_defn.has_source:
                # Attach the sources to a copy: the definition must stay
                # sourceless so that a later run uses the rasters of that
                # run. A shallow copy would share (and fill) its list.
                channel_defn = deepcopy(channel_defn)
                for source in source_list:
                    channel_defn.add_source(source)

            for channels, crs in channel_defn.iter_channels():
                file_counter = file_counter + 1
                self._container.append(_write_shape_file(
                    channels, crs, size_info,
                    out_dir / f"channels_{pid}_{file_counter}.feather",
                    out_dir / f"crs_{pid}_{file_counter}.json"))

    @property
    def files(self) -> List[Tuple[Path, Path]]:
        return list(self._container)

    def set_files(self, files: List[Tuple[Path, Path]]) -> None:
        """Use shape files detected elsewhere (e.g. by workers)."""
        self._container[:] = files

    def __iter__(self):
        """Iterator method for this collection object

        Yields
        ------
        gpd.GeoDataFrame
            Data from containing the extracted channel polygons and
            the CRS information.
        """

        for raster_data in self._container:
            feather_path, crs_path = raster_data
            gdf = gpd.read_feather(feather_path)
            with open(crs_path) as fp:
                gdf.set_crs(CRS.from_json(fp.read()))
            yield gdf


class _ConstraintInfoCollector:
    """Collection for the applied constraints"""

    def __init__(self) -> None:
        self._constraints_info = []

    def add(self,
            src_idx: Optional[Sequence[int]],
            constraint: Union[TopoConstConstraint, TopoFuncConstraint]
            ) -> None:
        """Add size function constraint spec to the collection.

        Parameters
        ----------
        src_idx : tuple of int or None
            Indices of sources (indexed based on all `HfunCollector`
            **not** just rasters) on which constant value refinement
            must be applied.
        constraint : TopoConstConstraint or TopoFuncConstraint
            The constraint definition object.

        Returns
        -------
        None
        """

        srcs = tuple(src_idx) if src_idx is not None else None
        self._constraints_info.append((srcs, constraint))

    def __len__(self) -> int:
        return len(self._constraints_info)

    def __iter__(self) -> Union[TopoConstConstraint, TopoFuncConstraint]:
        """Iterator method for this collection object

        Yields
        ------
        TopoConstConstraint or TopoFuncConstraint
            The constraint object provided in `add`
        """

        yield from self._constraints_info


    def get_constraints(self, hfun, in_idx, per_hfun=True):
        is_raster = isinstance(hfun, HfunRaster)
        constraint_list = []
        for src_idx, constraint_defn in self:
            if per_hfun and src_idx is not None and in_idx not in src_idx:
                continue

            if isinstance(constraint_defn, RASTER_CONSTR) and not is_raster:
                continue

            constraint_list.append(constraint_defn)
        return constraint_list


    def apply(self, hfun_list, per_hfun=True):
        for in_idx, hfun in enumerate(hfun_list):
            constraint_list = self.get_constraints(hfun, in_idx, per_hfun)

            if constraint_list:
                hfun.apply_constraints(constraint_list)


def _refine_on_rebuilt_hfun(task: dict, apply) -> dict:
    """Run one refinement task on a size function rebuilt from files.

    Raster handles cannot be pickled, so a task only carries paths: the
    worker rebuilds the HfunRaster from the size function file
    (``hfun_input_path``) over the topography (``topo_input_path``),
    calls ``apply(hfun)`` and saves the result to ``output_path``. Errors
    are returned, not raised, so the coordinator can report every
    failed task at once.
    """

    try:
        worker_hfun = HfunRaster(
            raster=Raster(task['topo_input_path']),
            hmin=task['global_hmin'],
            hmax=task['global_hmax'],
            verbosity=0,
            initial_value=task['hfun_input_path'])
        apply(worker_hfun)
        worker_hfun.save(task['output_path'])
        return {
            'status': 'success',
            'original_index': task['original_index'],
            'output_path': task['output_path'],
        }
    except Exception:  # pylint: disable=broad-exception-caught
        return {
            'status': 'error',
            'original_index': task['original_index'],
            'error': traceback.format_exc(),
        }


def _flow_limiter_task_worker(task: dict):
    """Apply the flow limiters in ``task['limiter_params']``."""

    def apply(hfun):
        for params in task['limiter_params']:
            hfun.add_subtidal_flow_limiter(
                hmin=params['hmin'],
                hmax=params['hmax'],
                lower_bound=params['zmin'],
                upper_bound=params['zmax'])

    return _refine_on_rebuilt_hfun(task, apply)


def _const_val_task_worker(task: dict):
    """Apply the constant values in ``task['const_val_rules']``."""

    def apply(hfun):
        for rule in task['const_val_rules']:
            hfun.add_constant_value(
                value=rule['value'],
                lower_bound=rule['lower_bound'],
                upper_bound=rule['upper_bound'])

    return _refine_on_rebuilt_hfun(task, apply)


def _constraints_task_worker(task: dict):
    """Apply the constraint objects in ``task['constraint_list']``."""

    return _refine_on_rebuilt_hfun(
        task, lambda hfun: hfun.apply_constraints(task['constraint_list']))


def _user_shapes_task_worker(task: dict):
    """Apply user-provided shapes (patches or line features).

    The coordinator resolves each definition to a ``(shape, CRS,
    size_info)`` triple and places it in the task; ``task['method_name']``
    (``add_patch`` or ``add_feature``) and ``task['shape_kwarg']`` select
    the HfunRaster method. ``size_info`` holds only ``expansion_rate``
    and ``target_size``.
    """

    def apply(hfun):
        apply_shape = getattr(hfun, task['method_name'])
        for shape, shape_crs, size_info in task['shapes']:
            if not shape_crs.equals(hfun.crs):
                transformer = Transformer.from_crs(
                    shape_crs, hfun.crs, always_xy=True)
                shape = ops.transform(transformer.transform, shape)
            # use_threads=True: _ThreadPool uses threads, not child
            # processes, so it is legal inside a daemon Pool worker
            # (add_pool_args would otherwise fall back to sequential).
            apply_shape(**{
                task['shape_kwarg']: shape,
                'nprocs': task.get('worker_nprocs', 1),
                'use_threads': True,
                **size_info,
            })

    return _refine_on_rebuilt_hfun(task, apply)


def _replay_shapes_task_worker(task: dict, method_name: str, shape_kwarg: str):
    """Replay shapes the coordinator extracted to feather files.

    ``method_name`` is the HfunRaster method to call (``add_feature``
    for contours, ``add_patch`` for channels) and ``shape_kwarg`` the
    name that method uses for the geometry argument.
    """

    def apply(hfun):
        apply_shape = getattr(hfun, method_name)
        for feather_path, crs_path in task['shape_files']:
            gdf = gpd.read_feather(feather_path)
            with open(crs_path) as fp:
                # Read the CRS the same way the collector does, so serial
                # and parallel see identical input.
                gdf = gdf.set_crs(
                    CRS.from_json(fp.read()), allow_override=True)

            # Built once per file: every row shares the source CRS.
            transformer = None
            if not gdf.crs.equals(hfun.crs):
                _logger.info("Reprojecting feature...")
                transformer = Transformer.from_crs(
                    gdf.crs, hfun.crs, always_xy=True)

            for row in gdf.itertuples():
                _logger.debug(row)
                shape = row.geometry
                # Checked before any CRS work: reprojecting an empty
                # GeometryCollection raises.
                if isinstance(shape, GeometryCollection):
                    continue
                if transformer is not None:
                    shape = ops.transform(transformer.transform, shape)
                # use_threads=True: the heavy work (NumPy, cKDTree, GEOS)
                # releases the GIL, and threads need no pickling.
                apply_shape(**{
                    shape_kwarg: shape,
                    'expansion_rate': row.expansion_rate,
                    'target_size': row.target_size,
                    'nprocs': task.get('worker_nprocs', 1),
                    'use_threads': True,
                })

    return _refine_on_rebuilt_hfun(task, apply)


def _write_shape_file(shape, crs, size_info, feather_path, crs_path):
    """Write one detected shape and its CRS; returns the two paths."""

    gpd.GeoDataFrame(
        {'geometry': [shape],
         'expansion_rate': size_info['expansion_rate'],
         'target_size': size_info['target_size'],
        },
        crs=crs).to_feather(feather_path)
    gc.collect()
    with open(crs_path, 'w') as fp:
        fp.write(crs.to_json())
    return feather_path, crs_path


def _detect_shapes_task_worker(task: dict):
    """Detect one contour or channel definition on one raster.

    The definition is sourceless; the worker attaches the clipped
    topography and writes each detected shape the way the collectors'
    ``calculate`` does, for the coordinator to collect in order.
    """

    try:
        raster = Raster(task['topo_path'], chunk_size=task['chunk_size'],
                        overlap=task['overlap'])
        defn = deepcopy(task['defn'])
        defn.add_source(raster)
        shapes = (defn.iter_contours() if task['op'] == 'contour_detect'
                  else defn.iter_channels())
        files = [
            _write_shape_file(shape, crs, task['size_info'],
                              f"{task['prefix']}_{k}.feather",
                              f"{task['prefix']}_{k}.json")
            for k, (shape, crs) in enumerate(shapes)]
        return {
            'status': 'success',
            'original_index': task['original_index'],
            'files': files,
        }
    except Exception:  # pylint: disable=broad-exception-caught
        return {
            'status': 'error',
            'original_index': task['original_index'],
            'error': traceback.format_exc(),
        }


def _contours_task_worker(task: dict):
    """Apply contour refinements to a single HfunRaster."""
    return _replay_shapes_task_worker(task, 'add_feature', 'feature')


def _channels_task_worker(task: dict):
    """Apply channel refinements to a single HfunRaster."""
    return _replay_shapes_task_worker(task, 'add_patch', 'multipolygon')


def _meshdata_4326(hfun, kwargs: dict) -> MeshData:
    """``hfun.meshdata(**kwargs)`` as a copy in EPSG:4326."""

    try:
        meshdata = deepcopy(hfun.meshdata(**kwargs))
    except TypeError:
        # HfunMesh.meshdata() takes no kwargs such as stride
        meshdata = deepcopy(hfun.meshdata())
    # If no CRS info, we assume EPSG:4326
    if hasattr(meshdata, "crs"):
        dst_crs = CRS.from_user_input("EPSG:4326")
        if meshdata.crs != dst_crs:
            utils.reproject(meshdata, dst_crs)
    return meshdata


def _meshdata_task_worker(task: dict):
    """Compute one entry's meshdata in a worker and save it as ``.npz``.

    ``task['type']`` is ``'raster'`` (the HfunRaster is rebuilt from
    file paths, raster handles cannot be pickled) or ``'mesh'`` (the
    pickled HfunMesh is in ``task['hfun_obj']``). Overlaps are resolved
    later, in priority order, on the coordinator.
    """

    original_index = task['original_index']
    try:
        if task['type'] == 'raster':
            worker_hfun = HfunRaster(
                raster=Raster(task['topo_path']),
                hmin=task['hmin'],
                hmax=task['hmax'],
                verbosity=0,
                initial_value=task['hfun_input_path'])
        elif task['type'] == 'mesh':
            worker_hfun = task['hfun_obj']
        else:
            raise ValueError(f"Unknown task type: {task['type']}")
        meshdata = _meshdata_4326(worker_hfun, task.get('meshdata_kwargs', {}))

        # np.savez appends .npz to output_path
        np.savez(
            task['output_path'],
            coords=meshdata.coords,
            tria=meshdata.tria if meshdata.tria is not None else np.array([]),
            quad=meshdata.quad if meshdata.quad is not None else np.array([]),
            values=meshdata.values,
            crs=np.array(str(meshdata.crs) if meshdata.crs else ""))

        # Flush before returning: on Lustre/GPFS the coordinator may
        # otherwise not see the file right after np.savez returns.
        actual_path = str(task['output_path']) + '.npz'
        try:
            with open(actual_path, 'rb') as _fh:
                os.fsync(_fh.fileno())
        except OSError:
            pass  # fsync not supported on all filesystems — best effort
        if not os.path.exists(actual_path) or os.path.getsize(actual_path) == 0:
            raise RuntimeError(
                f"Worker: npz file missing or empty after savez+fsync: "
                f"{actual_path}")

        return {
            'status': 'success',
            'original_index': original_index,
            'output_path': actual_path
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {
            'status': 'error',
            'original_index': original_index,
            'op': task.get('op'),
            'error': repr(e),
            'traceback': traceback.format_exc(),
        }


def _load_meshdata_npz(npz_path: str, retries: int = 30,
                       delay: float = 1.0) -> MeshData:
    """Load (and remove) a ``.npz`` written by ``_meshdata_task_worker``.

    On a shared filesystem the file can show up on the coordinator a
    little after the worker wrote it, hence the retries.
    """

    for attempt in range(retries):
        if os.path.exists(npz_path):
            break
        _logger.debug(
            f"Stage 2: waiting for {npz_path} "
            f"(attempt {attempt + 1}/{retries})")
        sleep(delay)
    else:
        raise FileNotFoundError(
            f"Stage 2: worker output not visible after "
            f"{retries * delay:.1f}s: {npz_path}")

    # The context manager closes the file before it is removed (Windows)
    with np.load(npz_path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    crs_str = str(data['crs'])
    meshdata = MeshData(
        coords=data['coords'],
        tria=data['tria'] if data['tria'].size > 0 else None,
        quad=data['quad'] if data['quad'].size > 0 else None,
        values=data['values'],
        crs=CRS.from_user_input(crs_str) if crs_str else None)
    try:
        os.remove(npz_path)
    except OSError:
        _logger.debug(f"Could not remove temp file {npz_path}")
    return meshdata


def _task_topo_path(hfun: HfunRaster) -> str:
    """Topography file a worker must pair with ``hfun.tmpfile``.

    Workers rebuild an ``HfunRaster`` from two files: the size function
    (``hfun.tmpfile``) and the topography it is computed from. When the
    collector clipped the input (``base_shape``/``base_mesh``), the size
    function is on the clipped grid, so the topography must be the clipped
    working copy (``raster.tmpfile``), not the original DEM (``raster.path``).
    Using the original made every depth-based refinement read elevations from
    shifted pixels. ``tmpfile`` falls back to the original path when the
    raster was never modified.
    """
    return str(hfun.raster.tmpfile)


class _Footprint:
    """Where one size function provides sizes, for compositing.

    Raster size functions cover the pixels where their (clipped)
    topography has valid data, so a higher-priority tile's nodata margin
    or clipped-away corner no longer hides a lower-priority tile there
    (that bbox-based rule left strips of the wrong size along tile
    edges). Other size functions cover their mesh bounding box.
    """

    def __init__(self, topo_path=None, bbox=None):
        self._topo_path = None if topo_path is None else str(topo_path)
        self._bbox = bbox
        if self._topo_path is not None:
            with rasterio.open(self._topo_path) as src:
                self._crs = src.crs
                self._transform = src.transform
                self._shape = (src.height, src.width)

    @classmethod
    def of(cls, hfun, meshdata_4326: MeshData) -> '_Footprint':
        if isinstance(hfun, HfunRaster):
            return cls(topo_path=_task_topo_path(hfun))
        return cls(bbox=Mesh(meshdata_4326).get_bbox(crs='EPSG:4326'))

    def covers(self, lonlat: np.ndarray) -> np.ndarray:
        """Boolean per EPSG:4326 point: is it inside this footprint?"""
        if self._topo_path is None:
            return shapely.intersects_xy(self._bbox, lonlat[:, 0], lonlat[:, 1])
        x, y = lonlat[:, 0], lonlat[:, 1]
        if not CRS.from_user_input(self._crs).equals(CRS.from_epsg(4326)):
            x, y = Transformer.from_crs(
                'EPSG:4326', self._crs, always_xy=True).transform(x, y)
        inv = ~self._transform
        col, row = inv * (np.asarray(x), np.asarray(y))
        col, row = np.asarray(col), np.asarray(row)
        height, width = self._shape
        # Size function nodes lie on the raster edge too (and a round trip
        # through UTM moves them by ~1e-7 deg): up to half a pixel outside
        # counts as the edge pixel.
        inside = ((col >= -0.5) & (col < width + 0.5)
                  & (row >= -0.5) & (row < height + 0.5))
        col = np.clip(np.floor(col), 0, width - 1).astype(np.int64)
        row = np.clip(np.floor(row), 0, height - 1).astype(np.int64)
        result = np.zeros(len(lonlat), dtype=bool)
        if not inside.any():
            return result
        r0, r1 = row[inside].min(), row[inside].max() + 1
        c0, c1 = col[inside].min(), col[inside].max() + 1
        with rasterio.open(self._topo_path) as src:
            mask = src.read_masks(
                1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
        result[inside] = mask[row[inside] - r0, col[inside] - c0] > 0
        return result


def _keep_nodes(meshdata: MeshData, keep: np.ndarray) -> MeshData:
    """Subset of ``meshdata`` with the nodes in ``keep`` (and the
    elements whose nodes are all kept)."""
    new_index = np.full(len(keep), -1, dtype=np.int64)
    new_index[keep] = np.arange(int(keep.sum()))
    elems = {}
    for name in ('tria', 'quad'):
        old = np.asarray(getattr(meshdata, name))
        if old.size:
            old = old[np.all(keep[old], axis=1)]
            elems[name] = new_index[old]
        else:
            elems[name] = None
    values = np.asarray(meshdata.values)
    return MeshData(coords=np.asarray(meshdata.coords)[keep],
                    tria=elems['tria'], quad=elems['quad'],
                    values=values[keep], crs=meshdata.crs)


def _has_valid_data(raster: Raster) -> bool:
    """True if any pixel of band 1 is not nodata (stops at the first)."""
    for window in raster.iter_windows():
        if raster.src.read_masks(1, window=window).any():
            return True
    return False


def _kwargs_key(kwargs: dict) -> tuple:
    """Hashable, order-independent key for meshdata() kwargs."""
    return tuple(sorted((k, repr(v)) for k, v in kwargs.items()))


def _raise_on_failures(kind: str, results: list) -> None:
    """Fail fast on worker errors from a local ``Pool``.

    Skipping a failed task would leave a size function quietly missing
    its refinement (or a tile missing from the output), which is worse
    than stopping. ``MPIExecutor.run`` applies the same rule in 'mpi' mode.
    """
    failures = [r for r in results if r['status'] == 'error']
    if failures:
        msgs = [f"  idx {r['original_index']}: {r['error']}"
                for r in failures]
        raise RuntimeError(
            f"{len(failures)} {kind} worker(s) failed:\n"
            + "\n".join(msgs))


# Register collector-specific MPI operations
MPIExecutor.register_op('meshdata', _meshdata_task_worker)
MPIExecutor.register_op('contours', _contours_task_worker)
MPIExecutor.register_op('channels', _channels_task_worker)
MPIExecutor.register_op('contour_detect', _detect_shapes_task_worker)
MPIExecutor.register_op('channel_detect', _detect_shapes_task_worker)
MPIExecutor.register_op('flow_limiter', _flow_limiter_task_worker)
MPIExecutor.register_op('const_val', _const_val_task_worker)
MPIExecutor.register_op('constraints', _constraints_task_worker)
MPIExecutor.register_op('patch', _user_shapes_task_worker)
MPIExecutor.register_op('linefeature', _user_shapes_task_worker)



class HfunCollector(BaseHfun):
    """Define size function based on multiple inputs of different types

    Attributes
    ----------

    Methods
    -------
    meshdata()
        Return mesh sizes interpolated on an size-optimized
        unstructured mesh
    add_topo_bound_constraint(...)
        Add size fixed-per-point value constraint to the area
        bounded by specified bounds with expansion/contraction
        rate `rate` specified. This refinement is only applied on
        rasters with specified indices. The index is w.r.t the
        full input list for collector object creation.
    add_topo_func_constraint(upper_bound=np.inf, lower_bound=-np.inf,
                             value_type='min', rate=0.01)
        Add size value constraint based on function of depth/elevation
        to the area bounded by specified bounds with the expansion or
        contraction rate `rate` specified. This constraint is only
        applied on rasters with specified indices. The index is w.r.t
        the full input list for collector object creation.
    add_region_constraint(...)
        Add a constraint on size based on a specified region
    add_patch(...)
        Add a region of fixed size refinement with optional expansion
        rate for points outside the region to achieve smooth size
        transition.
    add_contour(...)
        Add refinement based on contour lines auto-extrcted from the
        underlying raster data. The size is calculated based on the
        specified `rate`, `target_size` and the distance from the
        extracted feature line. For refinement contours are extracted
        only from raster inputs, but are applied on all input size
        function bases.
    add_channel(...)
        Add refinement for auto-detected narrow domain regions.
        Optionally use an expansion rate for points outside detected
        narrow regions for smooth size transition.
    add_subtidal_flow_limiter(...)
        Add mesh size refinement based on the value as well as
        gradient of the topography within the region between
        specified by lower and upper bound on topography.
        This refinement is only applied on rasters with specified
        indices. The index counts raster inputs only, in the order
        they were passed to the collector.
    add_constant_value(...)
        Add fixed size mesh refinement in the region specified by
        upper and lower bounds on topography.  This refinement is
        only applied on rasters with specified indices. The index
        counts raster inputs only, in the order they were passed to
        the collector.

    Notes
    -----
    All the refinements and constraints of this collector size
    function are applied lazily. That means the size values are **not**
    evaluated at the time they are called. Instead the effect of
    all these refinements and constraints on the size is calculated
    when `meshdata()` method is called.

    Two distinct algorithms are implemented for storing the size
    function values during evaluation and before creating the
    "background mesh" on which sizes are specified. Currently
    the difference between algorithms are only due to how raster
    inputs to the collector are processed. The **exact** algorithm
    is more computationally expensive; it processes all refinements
    on the original rasters and applies the indivitually on all
    of those individual rasters; this results in exponential time
    calculation of contours or raster features as the extracted
    features on all rasters must be applied on all rasters. Application
    of features usually involve distance calculation using a tree
    and can be very expensive when many rasters-features are involved.
    The **fast** approach is less exact and can use more memory, but
    it is much faster. The approach it takes is to still extract
    the raster features individually but then apply it to a lower
    resolution large raster that covers all the input rasters.
    """

    def __init__(
            self,
            in_list: CanCreateMultipleHfun,
            base_mesh: Optional[Mesh] = None,
            hmin: Optional[float] = None,
            hmax: Optional[float] = None,
            nprocs: Optional[int] = None,
            verbosity: int = 0,
            method: Literal['exact', 'fast'] = 'exact',
            base_as_hfun: bool = True,
            base_shape: Optional[Union[Polygon, MultiPolygon]] = None,
            base_shape_crs: Union[str, CRS] = 'EPSG:4326',
            execution_mode: Optional[
                Literal['serial', 'parallel', 'mpi']] = None,
            ) -> None:
        """Initialize a collector size function object

        Parameters
        ----------
        in_list : CanCreateMultipleHfun
        base_mesh : Mesh or None, default=None
        hmin : float or None, default=None
        hmax : float or None, default=None
        nprocs : int or None, default=None
        verbosity : int, default=0
        method : {'exact', 'fast'}, default='exact
        base_as_hfun : bool, default=True
        base_shape: Polygon or MultiPolygon or None, default=None
        base_shape_crs: str or CRS, default='EPSG:4326'
        execution_mode : {'serial', 'parallel', 'mpi'} or None, default=None
            How refinements and meshdata are computed (see the
            ``execution_mode`` property). Under an MPI launcher, 'mpi' or
            None makes the ranks share one collector (rank 0 holds the
            size functions; construction is collective), while 'serial'
            or 'parallel' gives every rank its own independent collector.
        """

        # NOTE: Input Hfuns and their Rasters can get modified

        # Under an MPI launcher the ranks share this collector unless a
        # per-rank mode is given: rank 0 owns the size functions, and the
        # other ranks only run the tasks it sends.
        self._rank_independent = execution_mode in ('serial', 'parallel')
        self._shared_ranks = _is_mpi_active() and not self._rank_independent

        # Add a persistent working directory for this instance's outputs
        # CRITICAL: ranks sharing the collector must share the SAME
        # _work_dir path. tempfile.mkdtemp() creates a DIFFERENT
        # directory on each rank, so rank 0 creates it and broadcasts it.
        if self._shared_ranks:
            _mpi = _get_mpi()
            _comm = _mpi.COMM_WORLD
            if _comm.Get_rank() == 0:
                _work_dir_path = tempfile.mkdtemp(
                    prefix='hfun_collector_',
                    dir=os.environ.get('TMPDIR', tempfile.gettempdir()))
            else:
                _work_dir_path = None
            self._work_dir = _comm.bcast(_work_dir_path, root=0)
            # Workers must create the directory if it does not exist yet
            # (bcast is fast but mkdtemp only runs on rank 0)
            os.makedirs(self._work_dir, exist_ok=True)
        else:
            self._work_dir = tempfile.mkdtemp(
                prefix='hfun_collector_',
                dir=os.environ.get('TMPDIR', tempfile.gettempdir()))
        # Only rank 0 (manager) deletes _work_dir: all MPI ranks share it,
        # and rank 0 still reads files workers wrote there. Decided here
        # because __del__ can run at interpreter shutdown, when importing
        # ocsmesh.mpi fails.
        self._owns_work_dir = (not self._shared_ranks
                               or MPIExecutor.is_manager())
        self._creator_pid = os.getpid()
        # meshdata() cache: (key, result). The key holds _state_version,
        # which every add_* call bumps, and the meshdata() kwargs, so a
        # cached result is reused only for the same inputs. MPI workers
        # cache their None too: a repeated call must not re-enter the
        # collective dispatch on workers while rank 0 returns the cache.
        self._state_version = 0
        self._meshdata_cache = None
        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = effective_cpu_count() if nprocs == -1 else nprocs

        self._applied = False
        self._size_info = {'hmin': hmin, 'hmax': hmax}
        self._nprocs = nprocs
        self._hfun_list = []
        self._method = method

        self._base_shape = base_shape
        self._base_shape_crs = CRS.from_user_input(base_shape_crs)
        self._base_as_hfun = base_as_hfun

        # NOTE: Base mesh has to have a crs otherwise HfunMesh throws
        # exception
        self._base_mesh = None
        if base_mesh:
            self._base_mesh = HfunMesh(base_mesh)
            if self._base_as_hfun:
                self._base_mesh.size_from_mesh()

        self._contour_info_coll = _RefinementContourInfoCollector()
        self._contour_coll = _RefinementContourCollector(
                self._contour_info_coll)

        self._const_val_contour_coll = _ConstantValueContourInfoCollector()

        self._refine_patch_info_coll = _RefinementShapeInfoCollector()
        self._refine_line_info_coll = _RefinementShapeInfoCollector()

        self._flow_lim_coll = _FlowLimiterInfoCollector()

        self._ch_info_coll = _ChannelRefineInfoCollector()
        self._channels_coll = _ChannelRefineCollector(
                self._ch_info_coll)

        self._constraint_info_coll = _ConstraintInfoCollector()

        self._type_chk(in_list)

        # TODO: Interpolate max size on base mesh basemesh?
        #
        # TODO: CRS considerations

        # Ranks sharing the collector: only rank 0 builds (and clips) the
        # raster size functions, which avoids N x M temporary-file writes
        # on the shared filesystem. Workers rebuild a raster from file
        # paths inside each task instead.
        is_mpi_worker = self._shared_ranks and not MPIExecutor.is_manager()

        for in_item in in_list:
            # Add supports(ext) to each hfun type?

            if isinstance(in_item, BaseHfun):
                hfun = in_item
            # pylint: disable=R0801
            elif isinstance(in_item, Raster):
                # Workers skip HfunRaster construction to avoid blank
                # raster writes on the shared filesystem (multi-node).
                # Workers reconstruct HfunRaster from file paths inside
                # the MPI worker functions instead.
                if is_mpi_worker or not self._clip_to_base(in_item):
                    continue
                hfun = HfunRaster(in_item, **self._size_info)

            elif isinstance(in_item, EuclideanMesh2D):
                hfun = HfunMesh(in_item)

            elif isinstance(in_item, (str, Path)):
                in_item = str(in_item)
                if in_item.endswith('.tif'):
                    # See comment above — same guard for str/Path input.
                    if is_mpi_worker:
                        continue
                    raster = Raster(in_item)
                    if not self._clip_to_base(raster):
                        continue
                    hfun = HfunRaster(raster, **self._size_info)

                elif in_item.endswith(
                        ('.14', '.grd', '.gr3', '.msh', '.2dm')):
                    mesh = Mesh.open(in_item)
                    hfun = HfunMesh(mesh)

                else:
                    raise TypeError("Input file extension not supported!")

            self._hfun_list.append(hfun) # pylint: disable=E0606

        if execution_mode is not None:
            self.execution_mode = execution_mode


    def _clip_to_base(self, raster: Raster) -> bool:
        """Clip ``raster`` to the base shape (or base mesh bbox).

        Returns False when the raster must be skipped: it does not
        intersect the base shape, or it holds no valid data there. A
        tile that is all nodata gives no sizes, and its refinements
        (nothing detected) used to flatten the size function to hmin.
        """
        if self._base_shape:
            clip_shape = self._base_shape
            if not self._base_shape_crs.equals(raster.crs):
                transformer = Transformer.from_crs(
                    self._base_shape_crs, raster.crs, always_xy=True)
                clip_shape = ops.transform(transformer.transform, clip_shape)
        elif self._base_mesh:
            clip_shape = self._base_mesh.mesh.get_bbox(crs=raster.crs)
        else:
            clip_shape = None

        if clip_shape is not None:
            try:
                raster.clip(clip_shape)
            except ValueError as err:
                # This raster does not intersect shape
                _logger.debug(err)
                return False

        if not _has_valid_data(raster):
            warnings.warn(
                f"Raster {raster.path} has no valid data"
                f"{' inside the base shape' if clip_shape is not None else ''}"
                "; it is not used for the size function.")
            return False
        return True

    def __del__(self):
        # Attributes may be missing if __init__ failed part way.
        if (getattr(self, '_owns_work_dir', False)
                and hasattr(self, '_work_dir')
                and hasattr(self, '_creator_pid')
                and os.getpid() == self._creator_pid
                and os.path.exists(self._work_dir)):
            shutil.rmtree(self._work_dir, ignore_errors=True)



    def meshdata(self, **kwargs) -> MeshData:
        """Interpolates mesh size functions on an unstructred mesh

        Parameters
        ----------
        **kwargs : dict
            Arguments passed down to the underlying Hfun classes (e.g.
            mesh_engine='gmsh', stride=...).
        """
        # Reuse the result while neither the refinements nor the kwargs
        # changed (MeshDriver calls meshdata() again after the user did).
        key = (self._state_version, _kwargs_key(kwargs))
        if self._meshdata_cache is not None and self._meshdata_cache[0] == key:
            return deepcopy(self._meshdata_cache[1])

        if self.execution_mode == 'mpi':
            result = self._meshdata_mpi_pipeline(**kwargs)
        else:
            result = self._meshdata_pipeline(**kwargs)

        # Keep a private copy: callers may modify the returned object
        # (e.g. utils.reproject works in place).
        self._meshdata_cache = (key, deepcopy(result))
        return result

    def _current_meshdata(self) -> Optional[MeshData]:
        """meshdata() for MeshDriver: reuse the cached result whatever
        kwargs produced it, as long as no refinement was added since.

        Decided on every rank alike (MPI workers cached None), so the
        workers enter the collective dispatch only if rank 0 does.
        """
        cache = self._meshdata_cache
        if cache is not None and cache[0][0] == self._state_version:
            return deepcopy(cache[1])
        return self.meshdata()

    def _invalidate(self) -> None:
        """Mark the refinements as not applied and the cache as stale."""
        self._applied = False
        self._state_version += 1

    def _meshdata_mpi_pipeline(self, **kwargs) -> MeshData:
        """Full meshdata pipeline — all ranks call collectively.

        Only MPIExecutor.run() is collective (all ranks participate).
        All other stages are rank-0-only for now, with comments marking
        future MPI candidates. Workers skip straight to the run() call,
        then return None.
        """
        is_manager = MPIExecutor.is_manager()

        if self._method == 'exact':
            self._apply_features()

            with tempfile.TemporaryDirectory() as temp_dir:
                # ── COLLECTIVE: all ranks participate here ──
                # MPIExecutor.run() inside this call handles the
                # rank split (rank 0 dispatches, workers recv).
                hfun_path_list = self._calculate_and_write_hfun_to_disk(
                    temp_dir, **kwargs
                )

                # Workers got [] from run() — nothing left to do.
                if not hfun_path_list:
                    return None

                # TODO(mpi): _get_hfun_composite reads .2dm files and
                # vstacks coordinates. Could be distributed in the future.
                # For now, rank 0 only.
                composite_hfun = self._get_hfun_composite(hfun_path_list)

        elif self._method == 'fast':
            # TODO(mpi): fast method not yet MPI-enabled.
            # For now, rank 0 only. Workers return None.
            if not is_manager:
                return None

            with tempfile.TemporaryDirectory() as temp_dir:
                rast = self._create_big_raster(temp_dir)
                hfun = self._apply_features_fast(rast)
                composite_hfun = self._get_hfun_composite_fast(
                    hfun, **kwargs
                )
                del rast
                del hfun

        else:
            raise ValueError(
                f"Invalid method specified: {self._method}"
            )

        return composite_hfun

    def _meshdata_pipeline(self, **kwargs) -> MeshData:
        # Ranks sharing the collector: only rank 0 holds the rasters, so
        # only rank 0 runs the serial/parallel pipelines.
        if self._shared_ranks and not MPIExecutor.is_manager():
            return None

        # Just dummy object
        composite_hfun = MeshData([[0,0]])

        if self._method == 'exact':
            self._apply_features()

            with tempfile.TemporaryDirectory() as temp_dir:
                # Pass kwargs if needed
                hfun_path_list = self._calculate_and_write_hfun_to_disk(temp_dir, **kwargs)
                composite_hfun = self._get_hfun_composite(hfun_path_list)

        elif self._method == 'fast':

            with tempfile.TemporaryDirectory() as temp_dir:
                rast = self._create_big_raster(temp_dir)
                hfun = self._apply_features_fast(rast)
                # Pass kwargs (like mesh_engine) down to the composite generator
                composite_hfun = self._get_hfun_composite_fast(hfun, **kwargs)

                del rast
                del hfun

        else:
            raise ValueError(f"Invalid method specified: {self._method}")

        return composite_hfun


    def add_topo_bound_constraint(
            self,
            value: Union[float, npt.NDArray[np.float32]],
            upper_bound: float = np.inf,
            lower_bound: float = -np.inf,
            value_type: Literal['min', 'max'] = 'min',
            rate: float = 0.01,
            source_index: Union[List[int], int, None] = None
            ) -> None:
        """Add a fixed-value or fixed-matrix constraint.

        Add a fixed-value or fixed-matrix constraint to the region
        of the size function specified by lower and upper elevation
        of the underlying DEM. Optionally a `rate` can be specified
        to relax the constraint gradually outside the bounds.

        Parameters
        ----------
        value : float or array-like
            A single fixed value or array of values to be used for
            mesh size if condition is not met based on `value_type`.
            In case of an array the dimensions must match or be
            broadcastable to the raster grid.
        upper_bound : float, default=np.inf
            Maximum elevation to cut off the region where the
            constraint needs to be applied
        lower_bound : float, default=-np.inf
            Minimum elevation to cut off the region where the
            constraint needs to be applied
        value_type : {'min', 'max'}, default='min'
            Type of contraint. If 'min', it means the mesh size
            should not be smaller than the specified `value` at each
            point.
        rate : float, default=0.01
            Rate of relaxation of constraint outside the region defined
            by `lower_bound` and `upper_bound`.
        source_index : int or list of ints or None, default=None
            The index of entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None
        """

        self._invalidate()

        constraint_defn = TopoConstConstraint(
            value, upper_bound, lower_bound, value_type, rate)

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]
        self._constraint_info_coll.add(source_index, constraint_defn)


    def add_topo_func_constraint(
            self,
            func: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
                = _default_topo_func,
            upper_bound: float = np.inf,
            lower_bound: float = -np.inf,
            value_type: Literal['min', 'max'] = 'min',
            rate: float = 0.01,
            source_index: Union[List[int], int, None] = None
            ) -> None:
        """Add constraint based on a function of the topography

        Add a constraint based on the provided function `func` of
        the topography and apply to the region of the size function
        specified by lower and upper elevation of the underlying DEM.
        Optionally a `rate` can be specified to relax the constraint
        gradually outside the bounds.

        Parameters
        ----------
        func : callable
            A function to be applied on the topography to acquire the
            values to be used for mesh size if condition is not met
            based on `value_type`.
        upper_bound : float, default=np.inf
            Maximum elevation to cut off the region where the
            constraint needs to be applied
        lower_bound : float, default=-np.inf
            Minimum elevation to cut off the region where the
            constraint needs to be applied
        value_type : {'min', 'max'}, default='min'
            Type of contraint. If 'min', it means the mesh size
            should not be smaller than the value calculated from the
            specified `func` at each point.
        rate : float, default=0.01
            Rate of relaxation of constraint outside the region defined
            by `lower_bound` and `upper_bound`.
        source_index : int or list of ints or None, default=None
            The index of entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None
        """

        self._invalidate()

        constraint_defn = TopoFuncConstraint(
            func, upper_bound, lower_bound, value_type, rate)

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]
        self._constraint_info_coll.add(source_index, constraint_defn)


    def add_courant_num_constraint(
            self,
            upper_bound: float = 0.9,
            lower_bound: Optional[float] = None,
            timestep: float = 150,
            wave_amplitude: float = 2,
            source_index: Union[List[int], int, None] = None
            ) -> None:
        """Add constraint based on approximated Courant number bounds


        Parameters
        ----------
        upper_bound : float, default=0.9
            Maximum Courant number to allow on this mesh size function
        lower_bound : float or None, default=None
            Minimum Courant number to allow on this mesh size function
        timestep : float
            Timestep size (:math:`seconds`) to
        wave_amplitude : float, default=2
            Free surface elevation (:math:`meters`) from the reference
            (i.e. wave height)
        source_index : int or list of ints or None, default=None
            The index of entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None
        """

        self._invalidate()

        # TODO: Validate conflicting constraints, right now last one wins
        if upper_bound is None and lower_bound is None:
            raise ValueError("Both upper and lower Courant bounds can NOT be None!")

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]

        if upper_bound is not None:
            self._constraint_info_coll.add(
                source_index,
                CourantNumConstraint(
                    value=upper_bound,
                    timestep=timestep,
                    wave_amplitude=wave_amplitude,
                    value_type='min'
                )
            )
        if lower_bound is not None:
            self._constraint_info_coll.add(
                source_index,
                CourantNumConstraint(
                    value=lower_bound,
                    timestep=timestep,
                    wave_amplitude=wave_amplitude,
                    value_type='min'
                )
            )


    def add_region_constraint(
            self,
            value: Union[float, npt.NDArray[np.float32]],
            shape: Union[Polygon, MultiPolygon],
            crs: Union[CRS, str] = None,
            value_type: Literal['min', 'max'] = 'min',
            rate: float = 0.01,
            source_index: Union[List[int], int, None] = None
            ) -> None:

        """Add a value contraint for the points in specified region

        Add a fixed-value constraint to the region of the size function
        specified by the input shape.  Optionally a `rate` can be
        specified to relax the constraint gradually outside the bounds.

        Parameters
        ----------
        value : float or array-like
            A single fixed value to be used for
            mesh size if condition is not met based on `value_type`.
        shape: MultiPolygon or Polygon
            Region of specified constraint
        crs: CRS or None, default=None
            The CRS of the input shape
        value_type : {'min', 'max'}, default='min'
            Type of contraint. If 'min', it means the mesh size
            should not be smaller than the specified `value` at each
            point.
        rate : float, default=0.01
            Rate of relaxation of constraint outside the specified region
        source_index: int or list of ints or None, default=None
            The index of input hfun source. If `None` all inputs
            are used.

        Returns
        -------
        None
        """
        self._constraint_info_coll.add(
            source_index,
            RegionConstraint(
                value=value,
                shape=shape,
                crs=crs,
                value_type=value_type,
                rate=rate
            )
        )

        self._invalidate()


    def add_contour(
            self,
            level: Union[List[float], float, None] = None,
            expansion_rate: float = 0.01,
            target_size: Optional[float] = None,
            contour_defn: Optional[Contour] = None
            ) -> None:
        """Add refinement for auto extracted contours

        Add refinement for the contour lines extracted based on
        level or levels specified by `level`. The refinement
        is relaxed with `expansion_rate` and distance from the
        extracted contour lines. Contours are extracted only from the
        raster inputs, but are applied on all inputs passed to the
        collector constructor.

        Parameters
        ----------
        level : float or list of floats or None, default=None
            Level(s) at which contour lines should be extracted.
        expansion_rate : float, default=0.01
            Rate to use for expanding refinement with distance away
            from the extracted contours.
        target_size : float or None, default=None
            Target size to use on the extracted contours and expand
            from with distance.
        contour_defn : Contour or None, default=None
            Contour definition objects which defines contour extraction
            specification from rasters.

        Returns
        -------
        None
        """

        # Always lazy
        self._invalidate()

        levels = []
        if isinstance(level, (list, tuple)):
            levels.extend(level)
        else:
            levels.append(level)


        contour_defns = []
        if contour_defn is None:
            for lvl in levels:
                contour_defns.append(Contour(level=lvl))

        elif not isinstance(contour_defn, Contour):
            raise TypeError(
                f"Contour definition must be of type {Contour} not"
                f" {type(contour_defn)}!")

        elif level is not None:
            msg = "Level is ignored since a contour definition is provided!"
            warnings.warn(msg)
            _logger.info(msg)

        else:
            contour_defns.append(contour_defn)

        for ctr_dfn in contour_defns:
            self._contour_info_coll.add(
                ctr_dfn,
                expansion_rate=expansion_rate,
                target_size=target_size)

    def add_channel(
            self,
            level: float = 0,
            width: float = 1000,
            target_size: float = 200,
            expansion_rate: Optional[float] = None,
            tolerance: Optional[float] = None,
            channel_defn: Optional[Channel] = None
            ) -> None:
        """Add refinement for auto detected channels

        Automatically detects narrow regions in the domain and apply
        refinement size with an expanion rate (if provided) outside
        the detected area.

        Parameters
        ----------
        level : float, default=0
            High water mark at which domain is extracted for narrow
            region or channel calculations.
        width : float, default=1000
            The cut-off width for channel detection.
        target_size : float, default=200
            Target size to use on the detected channels and expand
            from with distance.
        expansion_rate : float or None, default=None
            Rate to use for expanding refinement with distance away
            from the detected channels.
        tolerance : float or None, default=None
            Tolerance to use for simplifying the polygon extracted
            from DEM data. If `None` don't simplify.
        channel_defn : Contour or None
            Channel definition objects which defines channel
            extraction specification from rasters.

        Returns
        -------
        None
        """

        # Always lazy
        self._invalidate()

        # Even a tolerance of 1 for simplifying polygon for channel
        # calculations is much faster than no simplification. 50
        # is much faster than 1. The reason is in simplify we don't
        # preserve topology
        if channel_defn is None:
            channel_defn = Channel(
                level=level, width=width, tolerance=tolerance)

        elif not isinstance(channel_defn, Channel):
            raise TypeError(
                f"Channel definition must be of type {Channel} not"
                f" {type(channel_defn)}!")

        self._ch_info_coll.add(
            channel_defn,
            expansion_rate=expansion_rate,
            target_size=target_size)


    def add_subtidal_flow_limiter(
            self,
            hmin: Optional[float] = None,
            hmax: Optional[float] = None,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None,
            source_index: Union[List[int], int, None] = None
            ) -> None:
        """Add mesh refinement based on topography.

        Calculates a pre-defined function of topography to use
        as values for refinement. The function values are cut off
        using `hmin` and `hmax` and is applied to the region bounded
        by `lower_bound` and `upper_bound`.

        Parameters
        ----------
        hmin : float or None, default=None
            Minimum mesh size in the refinement
        hmax : float or None, default=None
            Maximum mesh size in the refinement
        lower_bound : float or None, default=None
            Lower limit of the cut-off elevation for region to apply
            the fixed `value`.
        upper_bound : float or None, default=None
            Higher limit of the cut-off elevation for region to apply
            the fixed `value`.
        source_index : int or list of ints or None, default=None
            The index of raster entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None

        See Also
        --------
        hfun.raster.HfunRaster.add_subtidal_flow_limiter :
        """

        self._invalidate()

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]

        # TODO: Checks on hmin/hmax, etc?

        self._flow_lim_coll.add(
            source_index,
            hmin=hmin,
            hmax=hmax,
            upper_bound=upper_bound,
            lower_bound=lower_bound)


    def add_constant_value(
            self,
            value: float,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None,
            source_index: Union[List[int], int, None] =None):
        """Add refinement of fixed value in the region specified by bounds.

        Apply fixed value mesh size refinement to the region specified
        by bounds (if provided) or the whole domain.

        Parameters
        ----------
        value : float
            Fixed value to use for refinement size
        lower_bound : float or None, default=None
            Lower limit of the cut-off elevation for region to apply
            the fixed `value`.
        upper_bound : float or None, default=None
            Higher limit of the cut-off elevation for region to apply
            the fixed `value`.
        source_index : int or list of ints or None, default=None
            The index of raster entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None
        """

        self._invalidate()

        contour_defn0 = None
        contour_defn1 = None
        if lower_bound is not None and not np.isinf(lower_bound):
            contour_defn0 = Contour(level=lower_bound)
        if upper_bound is not None and not np.isinf(upper_bound):
            contour_defn1 = Contour(level=upper_bound)

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]
        self._const_val_contour_coll.add(
            source_index, contour_defn0, contour_defn1, value)


    def add_patch(
            self,
            shape: Union[MultiPolygon, Polygon, None] = None,
            patch_defn: Optional[Patch] = None,
            shapefile: Union[None, str, Path] = None,
            expansion_rate: Optional[float] = None,
            target_size: Optional[float] = None,
            ) -> None:
        """Add refinement as a region of fixed size with an optional rate

        Add a refinement based on a region specified by `shape`,
        `patch_defn` or `shapefile`.  The fixed `target_size`
        refinement can be expanded outside the region specified by the
        shape if `expansion_rate` is provided.

        Parameters
        ----------
        shape : MultiPolygon or Polygon or None, default=None
            Shape of the region to use specified `target_size` for
            refinement. Only one of `shape`, `patch_defn` or `shapefile`
            must be specified.
        patch_defn : Patch or None, default=None
            Shape of the region to use specified `target_size` for
            refinement. Only one of `shape`, `patch_defn` or `shapefile`
            must be specified.
        shapefile : None or str or Path, default=None
            Shape of the region to use specified `target_size` for
            refinement. Only one of `shape`, `patch_defn` or `shapefile`
            must be specified.
        expansion_rate : float or None, default=None
            Optional rate to use for expanding refinement outside
            the specified shape in `multipolygon`.
        target_size : float or None, default=None
            Fixed target size of mesh to use for refinement in
            `multipolygon`

        Returns
        -------
        None
        """

        # "shape" should be in 4326 CRS. For shapefile or patch_defn
        # CRS info is included

        self._invalidate()

        if not patch_defn:
            if shape:
                patch_defn = Patch(shape=shape)

            elif shapefile:
                patch_defn = Patch(shapefile=shapefile)

        self._refine_patch_info_coll.add(
            patch_defn,
            expansion_rate=expansion_rate,
            target_size=target_size)

    def add_feature(
            self,
            shape: Union[MultiLineString, LineString, None] = None,
            line_defn: Optional[LineFeature] = None,
            shapefile: Union[None, str, Path] = None,
            expansion_rate: float = 0.01,
            target_size: Optional[float] = None,
            crs: Union[CRS, str, int] = 4326
            ) -> None:
        """Add refinement as a region of fixed size with an optional rate

        Add a refinement based on lines specified by `shape`,
        `line_defn` or `shapefile`.  The fixed `target_size`
        refinement is expanded by the `expansion_rate`.

        Parameters
        ----------
        shape : MultiLineString or LineString or None, default=None
            Shape of the region to use specified `target_size` for
            refinement. Only one of `shape`, `line_defn` or `shapefile`
            must be specified.
        line_defn : LineFeature or None, default=None
            Shape of the region to use specified `target_size` for
            refinement. Only one of `shape`, `line_defn` or `shapefile`
            must be specified.
        shapefile : None or str or Path, default=None
            Shape of the region to use specified `target_size` for
            refinement. Only one of `shape`, `line_defn` or `shapefile`
            must be specified.
        expansion_rate : float, default=0.01
            Rate to use for expanding refinement away from
            the specified shape
        target_size : float or None, default=None
            Fixed target size of mesh to use for refinement in
            `multipolygon`
        crs : CRS, default 4326
            The CRS of the input `shape`

        Returns
        -------
        None
        """

        self._invalidate()

        if not line_defn:
            if shape:
                line_defn = LineFeature(
                    shape=shape, shape_crs=CRS.from_user_input(crs))

            elif shapefile:
                line_defn = LineFeature(shapefile=shapefile)

        self._refine_line_info_coll.add(
            line_defn,
            expansion_rate=expansion_rate,
            target_size=target_size)


    @staticmethod
    def _type_chk(input_list: List[Any]) -> None:
        """Checks the if the input types are supported for size function

        Checks if size function collector supports handling
        size functions created from the input types.

        Parameters
        ----------
        input_list : List[Any]

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the any of the inputs are not supported
        """

        valid_types = (str, Path, Raster, BaseMesh, HfunRaster, HfunMesh)
        if not all(isinstance(item, valid_types) for item in input_list):
            raise TypeError(
                f'Input list items must be of type'
                f' {", ".join(str(i) for i in valid_types)},'
                f' or a derived type.')

    def _apply_features(self) -> None:
        """Internal: apply all specified refinements and constraints

        Apply all specified refinements and constrains for the exact
        algorithm.

        Parameters
        ----------

        Returns
        -------
        None

        See Also
        --------
        _apply_features_fast :
        """

        if not self._applied:
            # MPI-Aware Refinements (All ranks must execute for collective dispatch)
            self._apply_contours()

            # MPI-Aware Refinements (All ranks must participate for collective dispatch)
            self._apply_flow_limiters()
            self._apply_const_val()

            # MPI-Aware Refinements (All ranks must participate for collective dispatch)
            self._apply_linefeatures()
            self._apply_patch()

            # MPI-Aware Refinements (All ranks must execute for collective dispatch)
            self._apply_channels()

            # MPI-Aware Refinements (All ranks must participate for collective dispatch)
            self._apply_constraints()

        self._applied = True


    def _apply_constraints(self) -> None:
        """Internal: dispatch constraint application to serial or parallel.

        Dispatches to either ``_apply_constraints_serial`` or
        ``_apply_constraints_parallel`` based on the current
        ``execution_mode``.  All MPI ranks call this collectively so
        that ``MPIExecutor.run()`` inside ``_apply_constraints_parallel``
        can coordinate work across ranks.

        Returns
        -------
        None

        See Also
        --------
        _apply_constraints_serial :
        _apply_constraints_parallel :
        _apply_constraints_fast :
        """

        if self._method == 'fast':
            raise NotImplementedError(
                "This function does not support fast hfun method")

        # Every rank holds the same definitions, so all ranks skip together
        # and no collective dispatch is left waiting.
        if len(self._constraint_info_coll) == 0:
            return
        if (self._can_distribute_refinements()
                and not self._constraints_picklable()):
            # Every rank holds the same constraints, so all ranks take
            # this branch and none waits in a collective dispatch.
            warnings.warn(
                "A size function constraint cannot be pickled (e.g. a"
                " lambda passed to add_topo_func_constraint), so the"
                " constraints are applied serially on the coordinator."
                " Define the function at module level to distribute them.")
            if self.execution_mode == 'mpi' and not MPIExecutor.is_manager():
                return
            self._apply_constraints_serial()
        elif self._can_distribute_refinements():
            _logger.info("Applying constraints using PARALLEL method.")
            self._apply_constraints_parallel()
        else:
            _logger.info("Applying constraints using SERIAL method.")
            self._apply_constraints_serial()


    def _constraints_picklable(self) -> bool:
        """True when every constraint can be sent to a worker."""

        try:
            pickle.dumps([c for _, c in self._constraint_info_coll])
        except (pickle.PicklingError, AttributeError, TypeError):
            return False
        return True

    def _apply_constraints_serial(self) -> None:
        """Internal: apply specified constraints serially.

        Delegates to ``_ConstraintInfoCollector.apply()`` which iterates
        over each hfun and applies matching constraints one-by-one.

        Returns
        -------
        None
        """

        self._constraint_info_coll.apply(self._hfun_list)


    def _apply_constraints_parallel(self) -> None:
        """Internal: apply the constraints through worker tasks.

        One task per raster with constraints; the constraint objects
        travel in the task (``_apply_constraints`` checked they pickle).
        Mesh size functions keep no constraints here, as before.
        """

        tasks = []
        if self._is_coordinator():
            for in_idx, hfun in enumerate(self._hfun_list):
                if not isinstance(hfun, HfunRaster):
                    continue
                constraint_list = self._constraint_info_coll.get_constraints(
                    hfun, in_idx, per_hfun=True)
                if constraint_list:
                    tasks.append(self._raster_task(
                        'constraints', in_idx, hfun,
                        constraint_list=constraint_list))
        self._run_raster_tasks('constraints', tasks, 'constraint')

    def _apply_contours(self, apply_to: Optional[SizeFuncList] = None) -> None:
        """Internal: dispatch contour application to serial or parallel.

        Parameters
        ----------
        apply_to : SizeFuncList or None, default=None
            Size functions on which contours must be applied. If `None`
            all inputs are used to apply the calculated contours.

        Returns
        -------
        None

        See Also
        --------
        _apply_contours_serial :
        _apply_shape_refinements :
        """

        # Every rank holds the same definitions, so all ranks skip together
        # and no collective dispatch is left waiting.
        if len(self._contour_info_coll) == 0:
            return
        if self._can_distribute_refinements():
            _logger.info("Applying contours using PARALLEL method.")
            self._apply_shape_refinements('contours', apply_to)
        else:
            _logger.info("Applying contours using SERIAL method.")
            self._apply_contours_serial(apply_to)


    def _apply_contours_serial(
            self,
            apply_to: Optional[SizeFuncList] = None
            ) -> None:
        """Internal: apply specified contours one size function at a time.

        Parameters
        ----------
        apply_to : SizeFuncList or None, default=None
            Size functions on which contours must be applied. If `None`
            all inputs are used to apply the calculated contours.

        Returns
        -------
        None
        """

        # TODO: Consider CRS before applying to different hfuns
        #
        # NOTE: for parallelization make sure a single hfun is NOT
        # passed to multiple processes

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        apply_to = self._default_targets(apply_to)

        with tempfile.TemporaryDirectory() as temp_path:
            with Pool(processes=self._nprocs) as p:
                # Contours are ONLY extracted from raster sources
                self._contour_coll.calculate(raster_hfun_list, temp_path)
                counter = 0
                for hfun in apply_to:
                    for gdf in self._contour_coll:
                        for row in gdf.itertuples():
                            _logger.debug(row)
                            shape = row.geometry
                            if isinstance(shape, GeometryCollection):
                                continue
                            # NOTE: CRS check is done AFTER
                            # GeometryCollection check because
                            # gdf.to_crs results in an error in case
                            # of empty GeometryCollection
                            if not gdf.crs.equals(hfun.crs):
                                _logger.info("Reprojecting feature...")
                                transformer = Transformer.from_crs(
                                    gdf.crs, hfun.crs, always_xy=True)
                                shape = ops.transform(
                                        transformer.transform, shape)
                            counter = counter + 1
                            hfun.add_feature(**{
                                'feature': shape,
                                'expansion_rate': row.expansion_rate,
                                'target_size': row.target_size,
                                'pool': p
                            })
            p.join()


    # collector attribute, HfunRaster method, geometry kwarg name
    _SHAPE_REFINEMENTS = {
        'contours': ('_contour_coll', 'add_feature', 'feature'),
        'channels': ('_channels_coll', 'add_patch', 'multipolygon'),
    }
    # kind: (definition collection, detection op)
    _SHAPE_DETECTION = {
        'contours': ('_contour_info_coll', 'contour_detect'),
        'channels': ('_ch_info_coll', 'channel_detect'),
    }

    def _detect_shapes(self, kind, raster_hfun_list, shape_dir):
        """Detect the contours or channels on the rasters, distributed.

        One task per definition and raster; the shape files are returned
        in the order serial detection (``calculate``) writes them, so
        the refinements are replayed identically. Collective under MPI:
        worker ranks pass empty arguments and get an empty list.
        """

        coll = getattr(self, self._SHAPE_REFINEMENTS[kind][0])
        info_attr, op = self._SHAPE_DETECTION[kind]
        info = getattr(self, info_attr)
        # Definitions given their own sources are detected on those
        # sources, on the coordinator. Every rank holds the same
        # definitions, so every rank takes the same branch.
        if any(defn.has_source for defn, _ in info):
            if not self._is_coordinator():
                return []
            coll.calculate(raster_hfun_list, shape_dir)
            return coll.files

        tasks = []
        if self._is_coordinator():
            for d_idx, (defn, size_info) in enumerate(info):
                for r_idx, hfun in enumerate(raster_hfun_list):
                    tasks.append({
                        'op': op,
                        'original_index': len(tasks),
                        'defn': defn,
                        'size_info': size_info,
                        'topo_path': _task_topo_path(hfun),
                        'chunk_size': hfun.raster.chunk_size,
                        'overlap': hfun.raster.overlap,
                        'prefix': os.path.join(
                            shape_dir, f'{kind}_{d_idx}_{r_idx}'),
                    })
        results = self._dispatch_refinement_tasks(
            op, tasks, f'{kind} detection')
        if results is None:
            return []  # MPI worker rank
        files = [f for r in sorted(results, key=lambda r: r['original_index'])
                 for f in r['files']]
        coll.set_files(files)
        return files

    def _can_distribute_refinements(self) -> bool:
        """True when contours/channels may go through the task path.

        The 'fast' method builds a throw-away big raster that is not in
        ``self._hfun_list``, so the task path (which writes results back
        by list position) cannot be used for it.
        """

        if self._method == 'fast':
            return False
        if self.execution_mode == 'mpi':
            # nprocs sizes the intra-rank Pool; tile distribution is
            # across ranks, so nprocs == 1 is still a valid MPI run.
            return True
        return self.execution_mode == 'parallel' and self._nprocs > 1

    def _default_targets(self, apply_to=None) -> list:
        """``apply_to``, or by default the base mesh (if used as a size
        function), the mesh size functions and then the rasters."""

        if apply_to is not None:
            return apply_to
        mesh_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunMesh)]
        if self._base_mesh and self._base_as_hfun:
            mesh_hfun_list.insert(0, self._base_mesh)
        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        return [*mesh_hfun_list, *raster_hfun_list]

    def _split_refinement_targets(self, apply_to):
        """Split size functions into distributable rasters and the rest.

        Returns ``(raster_hfun_list, {index: hfun}, [hfun, ...])``.
        """

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        apply_to = self._default_targets(apply_to)

        # apply_to can hold objects that are NOT in self._hfun_list (the
        # base mesh is one), so we match by object identity instead of by
        # position. Anything we cannot look up, or that is not a raster,
        # stays on the coordinator.
        # A dict also means the same hfun listed twice becomes one task,
        # which keeps two workers from writing the same output file.
        idx_by_id = {id(h): i for i, h in enumerate(self._hfun_list)}
        parallel_targets = {}
        serial_targets = []
        for hfun in apply_to:
            idx = idx_by_id.get(id(hfun))
            if idx is not None and isinstance(hfun, HfunRaster):
                parallel_targets[idx] = hfun
            else:
                serial_targets.append(hfun)

        return raster_hfun_list, parallel_targets, serial_targets

    def _is_coordinator(self) -> bool:
        """True on the process that owns the size functions: rank 0
        under MPI, the only process otherwise."""

        return self.execution_mode != 'mpi' or MPIExecutor.is_manager()

    def _raster_task(self, op: str, idx: int, hfun: HfunRaster,
                     **payload) -> dict:
        """Task refining ``self._hfun_list[idx]`` in a worker (see
        ``_refine_on_rebuilt_hfun``); ``payload`` is op specific."""

        return {
            'op': op,
            'original_index': idx,
            'hfun_input_path': hfun.tmpfile,
            'topo_input_path': _task_topo_path(hfun),
            'output_path': os.path.join(
                self._work_dir, f'{op}_result_{idx}.tif'),
            'global_hmin': hfun._hmin,  # pylint: disable=W0212
            'global_hmax': hfun._hmax,  # pylint: disable=W0212
            # MPI ranks size their own thread pool from their affinity
            'worker_nprocs': -1 if self.execution_mode == 'mpi' else 1,
            **payload,
        }

    def _rules_per_raster(self, rules_for) -> Dict[int, list]:
        """``{slot in _hfun_list: rules}`` for the rasters with rules.

        ``rules_for(raster_idx)`` lists the rules of the raster_idx-th
        raster: user ``source_index`` values count rasters only, while
        the result goes back to the raster's slot in ``_hfun_list``
        (a mesh size function earlier in the list would otherwise be
        overwritten).
        """

        rules = {}
        raster_idx = -1
        for hfun_idx, hfun in enumerate(self._hfun_list):
            if not isinstance(hfun, HfunRaster):
                continue
            raster_idx += 1
            raster_rules = rules_for(raster_idx)
            if raster_rules:
                rules[hfun_idx] = raster_rules
        return rules

    def _run_raster_tasks(self, kind: str, tasks: list,
                          label: Optional[str] = None) -> bool:
        """Run refinement tasks and put the refined rasters back.

        Collective under MPI: every rank calls it, worker ranks with an
        empty task list. Returns False on MPI worker ranks.
        """

        results = self._dispatch_refinement_tasks(kind, tasks, label)
        if results is None:
            return False
        self._integrate_refinement_results(kind, results)
        return True

    def _dispatch_refinement_tasks(self, kind, tasks, label=None):
        """Run prepared refinement tasks via MPI ranks or a local Pool.

        Returns the result dicts on the coordinator, or ``None`` on an
        MPI worker rank. ``MPIExecutor.run()`` is collective, so every
        rank must reach it — including ranks holding no tasks.
        ``label`` names the stage in the error raised when workers fail
        (default ``kind``).
        """

        if self.execution_mode == 'mpi':
            raw_results = MPIExecutor.run(
                tasks, work_dir=self._work_dir, fail_fast=True)
            if raw_results is None:
                return None
            return [raw_results[idx] for idx in sorted(raw_results)]

        if not tasks:
            return []

        _logger.info(
            f"Start parallel execution for {len(tasks)} {kind} tasks")
        # Cap workers at the number of tasks: spawning more workers
        # than tasks wastes spawn time and memory with idle processes.
        n_workers = min(self._nprocs, len(tasks))
        worker = MPIExecutor.get_op(kind)
        if worker is None:
            raise ValueError(
                f"No worker registered for op {kind!r}. "
                f"Call MPIExecutor.register_op({kind!r}, fn) at module load time.")
        with Pool(processes=n_workers) as p:
            results = p.map(worker, tasks)
        _logger.info("Parallel execution finished.")
        _raise_on_failures(label or kind, results)
        return results

    def _integrate_refinement_results(self, kind, results):
        """Replace refined size functions with the worker output files."""

        for result in results:
            idx = result['original_index']
            original_hfun = self._hfun_list[idx]
            _logger.info(
                f"Update HfunCollector with {kind} raster at idx {idx}.")
            self._hfun_list[idx] = HfunRaster(
                raster=original_hfun.raster,
                hmin=original_hfun.hmin,
                hmax=original_hfun.hmax,
                verbosity=original_hfun.verbosity,
                initial_value=result['output_path']
            )

    def _apply_shape_refinements(
            self,
            kind: str,
            apply_to: Optional[SizeFuncList] = None
            ) -> None:
        """Internal: apply contours or channels through the task path.

        Uses the same 3-phase pattern as ``_apply_flow_limiters_parallel``:

        1. **Preparation** — extract the shapes once on the coordinator,
           then build one pickleable task dict per raster size function.
        2. **Execution** — the tasks go to ``Pool.map()`` in 'parallel'
           mode, or to ``MPIExecutor.run()`` in 'mpi' mode, where each
           rank refines a different raster.
        3. **Integration** — replace ``self._hfun_list`` entries with new
           ``HfunRaster`` objects built from the worker output files.

        Mesh size functions and the base mesh are handled on the
        coordinator, because only rasters can be rebuilt from a file path
        inside a worker.

        Parameters
        ----------
        kind : {'contours', 'channels'}
            Which refinement collection to apply.
        apply_to : SizeFuncList or None, default=None
            Size functions on which the refinement must be applied. If
            `None` all inputs are used.

        Returns
        -------
        None
        """

        coll_attr, method_name, shape_kwarg = self._SHAPE_REFINEMENTS[kind]
        coll = getattr(self, coll_attr)
        # On worker ranks the coordinator's task list is the one that
        # counts, so they skip preparation and go straight to the
        # collective dispatch.
        is_coordinator = self._is_coordinator()

        tasks = []
        serial_targets = []
        raster_hfun_list = []
        shape_dir = None
        try:
            if is_coordinator:
                # Phase 1: PREPARATION (Coordinator)
                (raster_hfun_list, parallel_targets,
                 serial_targets) = self._split_refinement_targets(apply_to)

                # The worker rebuilds a bare HfunRaster, so any constraint
                # already attached to the original would be silently
                # dropped. Safe today because _apply_features runs
                # contours/channels first and constraints last.
                if any(h._constraints  # pylint: disable=W0212
                       for h in parallel_targets.values()):
                    raise RuntimeError(
                        f"_apply_shape_refinements({kind!r}) cannot run "
                        f"after constraints have been added; refinements "
                        f"must be applied first.")

                # Shapes go under _work_dir rather than a local temp dir:
                # an MPI job spreads over nodes, and /tmp is not shared
                # between them.
                shape_dir = os.path.join(self._work_dir, kind)
                os.makedirs(shape_dir, exist_ok=True)

            # Shapes are ONLY extracted from raster sources (collective)
            shape_file_list = self._detect_shapes(
                kind, raster_hfun_list, shape_dir)

            if is_coordinator:
                tasks = [
                    self._raster_task(kind, idx, hfun,
                                      shape_files=shape_file_list)
                    for idx, hfun in parallel_targets.items()
                ]
                if not tasks:
                    _logger.info(f"No {kind} tasks to execute.")

            # Phase 2 and 3: EXECUTION and INTEGRATION
            if not self._run_raster_tasks(kind, tasks):
                # MPI worker rank — the coordinator owns the results.
                return

            # Mesh size functions and the base mesh, done here. This has to
            # happen before shape_dir is deleted, because iterating the
            # collection re-reads the feather files from disk.
            if serial_targets:
                with Pool(processes=self._nprocs) as p:
                    for hfun in serial_targets:
                        apply_shape = getattr(hfun, method_name)
                        for gdf in coll:
                            for row in gdf.itertuples():
                                _logger.debug(row)
                                shape = row.geometry
                                if isinstance(shape, GeometryCollection):
                                    continue
                                # NOTE: CRS check is done AFTER
                                # GeometryCollection check because
                                # gdf.to_crs results in an error in case
                                # of empty GeometryCollection
                                if not gdf.crs.equals(hfun.crs):
                                    _logger.info("Reprojecting feature...")
                                    transformer = Transformer.from_crs(
                                        gdf.crs, hfun.crs, always_xy=True)
                                    shape = ops.transform(
                                            transformer.transform, shape)
                                apply_shape(**{
                                    shape_kwarg: shape,
                                    'expansion_rate': row.expansion_rate,
                                    'target_size': row.target_size,
                                    'pool': p
                                })
        finally:
            if shape_dir is not None:
                shutil.rmtree(shape_dir, ignore_errors=True)


    def _apply_channels(self, apply_to: Optional[SizeFuncList] = None) -> None:
        """Internal: dispatch channel application to serial or parallel.

        Parameters
        ----------
        apply_to : SizeFuncList or None, default=None
            Size functions on which channels must be applied. If `None`
            all inputs are used to apply the calculated channels.

        Returns
        -------
        None

        See Also
        --------
        _apply_channels_serial :
        _apply_shape_refinements :
        """

        # Every rank holds the same definitions, so all ranks skip together
        # and no collective dispatch is left waiting.
        if len(self._ch_info_coll) == 0:
            return
        if self._can_distribute_refinements():
            _logger.info("Applying channels using PARALLEL method.")
            self._apply_shape_refinements('channels', apply_to)
        else:
            _logger.info("Applying channels using SERIAL method.")
            self._apply_channels_serial(apply_to)


    def _apply_channels_serial(
            self,
            apply_to: Optional[SizeFuncList] = None
            ) -> None:
        """Internal: apply specified channel refinements.

        Parameters
        ----------
        apply_to : SizeFuncList or None, default=None
            Size functions on which channels must be applied. If `None`
            all inputs are used to apply the calculated channels.

        Returns
        -------
        None
        """

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        apply_to = self._default_targets(apply_to)

        with tempfile.TemporaryDirectory() as temp_path:
            # Channels are ONLY extracted from raster sources
            self._channels_coll.calculate(raster_hfun_list, temp_path)
            counter = 0
            with Pool(processes=self._nprocs) as p:
                for hfun in apply_to:
                    for gdf in self._channels_coll:
                        for row in gdf.itertuples():
                            _logger.debug(row)
                            shape = row.geometry
                            if isinstance(shape, GeometryCollection):
                                continue
                            # NOTE: CRS check is done AFTER
                            # GeometryCollection check because
                            # gdf.to_crs results in an error in case
                            # of empty GeometryCollection
                            if not gdf.crs.equals(hfun.crs):
                                _logger.info("Reprojecting feature...")
                                transformer = Transformer.from_crs(
                                    gdf.crs, hfun.crs, always_xy=True)
                                shape = ops.transform(
                                        transformer.transform, shape)
                            counter = counter + 1
                            hfun.add_patch(**{
                                'multipolygon': shape,
                                'expansion_rate': row.expansion_rate,
                                'target_size': row.target_size,
                                'pool': p
                            })


    @property
    def execution_mode(self) -> str:
        """
        Gets the current execution mode for refinements ('serial' or 'parallel').
        Defaults to 'serial' if not previously set.
        """
        # Lazy Initialization: If the attribute doesn't exist yet,
        # create it here with the default value.
        if not hasattr(self, '_execution_mode'):
            self._execution_mode = 'serial'
        return self._execution_mode


    @execution_mode.setter
    def execution_mode(self, mode: str) -> None:
        """
        Sets the execution mode for refinements.

        Parameters
        ----------
        mode : str
            The desired mode. Must be 'serial', 'parallel', or 'mpi'.
        """
        if mode not in ['serial', 'parallel', 'mpi']:
            raise ValueError(
                "Execution mode must be 'serial', 'parallel', or 'mpi'"
            )

        if (mode == 'mpi' and _is_mpi_active()
                and getattr(self, '_rank_independent', False)):
            # Its ranks built separate size function lists and work dirs.
            raise ValueError(
                "This collector was created with a per-rank execution mode;"
                " create it with execution_mode='mpi' (or None) to use MPI.")

        if mode == 'mpi':
            if not _is_mpi_env_detected():
                # Not running under mpiexec/srun — can't use MPI.
                warnings.warn(
                    "MPI mode requested but no MPI environment detected "
                    "(no mpiexec/srun). Falling back to 'parallel' mode.",
                    UserWarning
                )
                mode = 'parallel'
            elif _get_mpi() is None:
                # Under MPI launcher but mpi4py not installed.
                warnings.warn(
                    "mpi4py is not installed. Falling back to 'parallel' "
                    "mode. Install mpi4py for MPI support: "
                    "pip install ocsmesh[mpi]",
                    UserWarning
                )
                mode = 'parallel'
            elif not _is_mpi_active():
                # mpi4py available but only 1 rank — pointless.
                warnings.warn(
                    "MPI mode requested but only 1 rank detected. "
                    "Falling back to 'parallel' mode.",
                    UserWarning
                )
                mode = 'parallel'
            else:
                _configure_mpi_environment()

        if mode == 'parallel' and (self._nprocs is None or self._nprocs <= 1):
            warnings.warn(
                f"Execution mode set to 'parallel' but nprocs is {self._nprocs}. "
            "Execution will fall back to serial. Set nprocs > 1 for parallel."
            )

        self._execution_mode = mode

        # Worker ranks never use the list in MPI mode (rank 0 sends them
        # file paths); drop the mesh size functions they still hold.
        if (mode == 'mpi'
                and _is_mpi_active()
                and not MPIExecutor.is_manager()):
            self._hfun_list = []



    def _apply_flow_limiters(self) -> None:
        """
        Dispatches to either the serial or parallel implementation based on
        the current execution mode.
        """
        # Every rank holds the same definitions, so all ranks skip together
        # and no collective dispatch is left waiting.
        if len(self._flow_lim_coll) == 0:
            return
        if self._can_distribute_refinements():
            _logger.info("Applying flow limiters using PARALLEL method.")
            self._apply_flow_limiters_parallel()
        else:
            _logger.info("Applying flow limiters using SERIAL method.")
            self._apply_flow_limiters_serial()


    def _apply_flow_limiters_serial(self) -> None:
        """Internal: apply specified sub tidal flow limiter refinements

        Applies specified subtidal flow limiter refinements for
        the exact algorithm.

        Parameters
        ----------

        Returns
        -------
        None

        See Also
        --------
        _apply_flow_limiters_fast :
        """

        if self._method == 'fast':
            raise NotImplementedError(
                "This function does not suuport fast hfun method")

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        for in_idx, hfun in enumerate(raster_hfun_list):
            for src_idx, hmin, hmax, zmax, zmin in self._flow_lim_coll:
                if src_idx is not None and in_idx not in src_idx:
                    continue
                if hmin is None:
                    hmin = self._size_info['hmin']
                if hmax is None:
                    hmax = self._size_info['hmax']
                hfun.add_subtidal_flow_limiter(hmin, hmax, zmin, zmax)


    def _apply_flow_limiters_parallel(self) -> None:
        """Internal: apply the flow limiters through worker tasks."""

        if self._method == 'fast':
            raise NotImplementedError(
                "This function is part of the 'exact' method and doesnt"
                " support 'fast' mode.")

        tasks = []
        if self._is_coordinator():
            def rules_for(raster_idx):
                return [
                    {'hmin': hmin if hmin is not None
                             else self._size_info.get('hmin'),
                     'hmax': hmax if hmax is not None
                             else self._size_info.get('hmax'),
                     'zmin': zmin,
                     'zmax': zmax}
                    for src_idx, hmin, hmax, zmax, zmin in self._flow_lim_coll
                    if src_idx is None or raster_idx in src_idx]

            tasks = [
                self._raster_task('flow_limiter', idx, self._hfun_list[idx],
                                  limiter_params=rules)
                for idx, rules in self._rules_per_raster(rules_for).items()]
        self._run_raster_tasks('flow_limiter', tasks, 'flow limiter')

    def _apply_const_val(self) -> None:
        """
        Dispatches to either the serial or parallel implementation based on
        the current execution mode.
        """
        # Every rank holds the same definitions, so all ranks skip together
        # and no collective dispatch is left waiting.
        if len(self._const_val_contour_coll) == 0:
            return
        if self._can_distribute_refinements():
            _logger.info("Applying constant values using PARALLEL method.")
            self._apply_const_val_parallel()
        else:
            _logger.info("Applying constant values using SERIAL method.")
            self._apply_const_val_serial()


    def _apply_const_val_serial(self):
        """Internal: apply specified constant value refinements.

        Applies constant value refinements for the exact algorithm.

        Returns
        -------
        None

        See Also
        --------
        _apply_const_val_fast :
        """

        if self._method == 'fast':
            raise NotImplementedError(
                "This function does not suuport fast hfun method")

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        for in_idx, hfun in enumerate(raster_hfun_list):
            for (src_idx, ctr0, ctr1), const_val in self._const_val_contour_coll:
                if src_idx is not None and in_idx not in src_idx:
                    continue
                level0 = None
                level1 =  None
                if ctr0 is not None:
                    level0 = ctr0.level
                if ctr1 is not None:
                    level1 = ctr1.level
                hfun.add_constant_value(const_val, level0, level1)


    def _apply_const_val_parallel(self):
        """Internal: apply the constant values through worker tasks."""

        if self._method == 'fast':
            raise NotImplementedError(
                "This function does not suuport fast hfun method")

        tasks = []
        if self._is_coordinator():
            def rules_for(raster_idx):
                return [
                    {'value': const_val,
                     'lower_bound': ctr0.level if ctr0 else None,
                     'upper_bound': ctr1.level if ctr1 else None}
                    for (src_idx, ctr0, ctr1), const_val
                    in self._const_val_contour_coll
                    if src_idx is None or raster_idx in src_idx]

            tasks = [
                self._raster_task('const_val', idx, self._hfun_list[idx],
                                  const_val_rules=rules)
                for idx, rules in self._rules_per_raster(rules_for).items()]
        self._run_raster_tasks('const_val', tasks, 'constant value')

    # kind: (definition collection, resolver, HfunRaster method, its
    # geometry keyword)
    _USER_SHAPES = {
        'patch': ('_refine_patch_info_coll', 'get_multipolygon',
                  'add_patch', 'multipolygon'),
        'linefeature': ('_refine_line_info_coll', 'get_multiline',
                        'add_feature', 'feature'),
    }

    def _apply_patch(self, apply_to: Optional[SizeFuncList] = None) -> None:
        self._apply_user_shapes('patch', apply_to)

    def _apply_linefeatures(self, apply_to=None):
        self._apply_user_shapes('linefeature', apply_to)

    def _apply_user_shapes(
            self,
            kind: str,
            apply_to: Optional[SizeFuncList] = None
            ) -> None:
        """Internal: apply the patches or line features.

        Parameters
        ----------
        kind : {'patch', 'linefeature'}
            Which refinement collection to apply.
        apply_to : SizeFuncList or None, default=None
            Size functions on which the shapes must be applied. If
            `None` all inputs are used.

        Returns
        -------
        None
        """

        coll_attr, resolver, method_name, shape_kwarg = \
            self._USER_SHAPES[kind]
        coll = getattr(self, coll_attr)
        # Every rank holds the same definitions, so all ranks skip together
        # and no collective dispatch is left waiting.
        if len(coll) == 0:
            return

        if not self._can_distribute_refinements():
            _logger.info(f"Applying {kind} refinements using SERIAL method.")
            with Pool(processes=self._nprocs) as p:
                for hfun in self._default_targets(apply_to):
                    for defn, size_info in coll:
                        shape, crs = getattr(defn, resolver)()
                        if hfun.crs != crs:
                            transformer = Transformer.from_crs(
                                crs, hfun.crs, always_xy=True)
                            shape = ops.transform(
                                    transformer.transform, shape)
                        getattr(hfun, method_name)(
                            **{shape_kwarg: shape}, pool=p, **size_info)
            return

        _logger.info(f"Applying {kind} refinements using PARALLEL method.")
        tasks = []
        resolved_shapes = []
        serial_targets = []
        if self._is_coordinator():
            # Shapely geometries pickle, so they travel in the task.
            _, parallel_targets, serial_targets = \
                self._split_refinement_targets(apply_to)
            for defn, size_info in coll:
                shape, shape_crs = getattr(defn, resolver)()
                resolved_shapes.append((shape, shape_crs, size_info))
            tasks = [
                self._raster_task(kind, idx, hfun, shapes=resolved_shapes,
                                  method_name=method_name,
                                  shape_kwarg=shape_kwarg)
                for idx, hfun in parallel_targets.items()]

        if not self._run_raster_tasks(kind, tasks):
            return  # MPI worker rank

        # HfunMesh / base-mesh targets cannot be rebuilt from a file in a
        # worker, so they always stay on the coordinator.
        if serial_targets:
            with Pool(processes=self._nprocs) as p:
                for hfun in serial_targets:
                    for shape, shape_crs, size_info in resolved_shapes:
                        if not shape_crs.equals(hfun.crs):
                            transformer = Transformer.from_crs(
                                shape_crs, hfun.crs, always_xy=True)
                            shape = ops.transform(transformer.transform, shape)
                        getattr(hfun, method_name)(
                            **{shape_kwarg: shape}, pool=p, **size_info)


    def _calculate_and_write_hfun_to_disk(
            self,
            out_path: Union[str, Path],
            **kwargs
            ) -> List[Union[str, Path]]:
        """Internal: write individual size function output mesh to file

        Calculate the interpolated on-mesh size function from each
        individual input, clip overlaps based on priority, and
        write the results to disk for later combining.

        The meshdata of each input (stage 1) is computed in place
        (serial), in a Pool (parallel) or on the MPI ranks (mpi).
        Overlaps are then resolved in priority order on the
        coordinator (stage 2, ``_write_hfun_stage2``).

        Parameters
        ----------
        out_path : path-like
            The path of the (temporary) directory to which mesh size
            functions must be written.
        **kwargs : dict
            Arguments to pass to the hfun.meshdata() method (e.g. stride).

        Returns
        -------
        list of path-like
            List of individual file path for mesh size function of
            each input; empty on MPI worker ranks.
        """

        # Last user input item has the highest priority (its nodes
        # are kept), so process in reverse order
        hfun_list = self._hfun_list[::-1]
        if self._base_mesh and self._base_as_hfun:
            hfun_list = [*hfun_list, self._base_mesh]

        distributed = (self.execution_mode == 'mpi'
                       or (self.execution_mode == 'parallel'
                           and self._nprocs > 1))
        if not distributed:
            _logger.info("Calculate & Writing hfun to disk using SERIAL method.")
            return self._write_hfun_stage2(
                out_path, hfun_list,
                (_meshdata_4326(hfun, kwargs) for hfun in hfun_list))

        _logger.info(
            "Calculate & Writing hfun to disk using "
            f"{self.execution_mode.upper()} method.")
        # Stage 1: one meshdata task per entry. MPI is collective: all
        # ranks build and dispatch the same list, workers get None back.
        pid = os.getpid()
        tasks = []
        for loop_idx, hfun in enumerate(hfun_list):
            task = {
                'op': 'meshdata',
                'original_index': loop_idx,
                # np.savez appends .npz
                'output_path': os.path.join(
                    self._work_dir, f"meshdata_stage1_{pid}_{loop_idx}"),
                'meshdata_kwargs': kwargs,
            }
            if isinstance(hfun, HfunRaster):
                task.update(type='raster', topo_path=_task_topo_path(hfun),
                            hfun_input_path=hfun.tmpfile,
                            hmin=hfun._hmin,  # pylint: disable=W0212
                            hmax=hfun._hmax)  # pylint: disable=W0212
            else:
                # HfunMesh and other types are picklable
                task.update(type='mesh', hfun_obj=deepcopy(hfun))
            tasks.append(task)

        results = self._dispatch_refinement_tasks('meshdata', tasks)
        if results is None:
            return []  # MPI worker rank
        npz_paths = {r['original_index']: r['output_path'] for r in results}
        # Loaded one at a time, so only one entry is in memory at once
        meshdata_list = (_load_meshdata_npz(npz_paths[i])
                         for i in range(len(hfun_list)))
        return self._write_hfun_stage2(out_path, hfun_list, meshdata_list)

    @staticmethod
    def _resolve_overlaps(meshdata_hfun, hfun, footprints):
        """Keep the nodes where ``hfun`` provides sizes and no
        higher-priority input (``footprints``) does.

        Returns the clipped meshdata and the footprint of ``hfun``.
        ``meshdata_hfun`` must be in EPSG:4326.
        """
        footprint = _Footprint.of(hfun, meshdata_hfun)
        lonlat = np.asarray(meshdata_hfun.coords)
        if lonlat.shape[0] == 0:
            return meshdata_hfun, footprint
        keep = (footprint.covers(lonlat) if isinstance(hfun, HfunRaster)
                else np.ones(len(lonlat), dtype=bool))
        for higher in footprints:
            keep &= ~higher.covers(lonlat)
        if not keep.all():
            meshdata_hfun = _keep_nodes(meshdata_hfun, keep)
        return meshdata_hfun, footprint

    def _write_hfun_stage2(
            self,
            out_path: Union[str, Path],
            hfun_list: list,
            meshdata_list: Iterable[MeshData],
            ) -> List[Path]:
        """Stage 2: resolve overlaps in priority order and write .2dm.

        ``hfun_list`` is in priority order (highest first) and
        ``meshdata_list`` yields the matching EPSG:4326 meshdata.
        Each entry keeps the nodes no higher-priority entry covers,
        clamped to hmin/hmax.
        """

        out_dir = Path(out_path)
        path_list = []
        pid = os.getpid()
        footprints = []  # higher-priority entries already written
        hmin = self._size_info['hmin']
        hmax = self._size_info['hmax']
        for hfun, meshdata_hfun in zip(hfun_list, meshdata_list):
            _logger.info("Resolving overlaps with higher-priority inputs...")
            meshdata_hfun, footprint = self._resolve_overlaps(
                meshdata_hfun, hfun, footprints)
            if len(meshdata_hfun.coords) == 0:
                _logger.debug("Hfun ignored due to overlap")
                continue

            if hmin:
                meshdata_hfun.values[meshdata_hfun.values < hmin] = hmin
            if hmax:
                meshdata_hfun.values[meshdata_hfun.values > hmax] = hmax

            footprints.append(footprint)
            file_path = out_dir / f'hfun_{pid}_{len(path_list) + 1}.2dm'
            _logger.info(f'write mesh {len(path_list) + 1} to file...')
            Mesh(meshdata_hfun).write(file_path, format='2dm')
            path_list.append(file_path)
            _logger.info('Done writing 2dm file.')
            gc.collect()
        return path_list


    def _get_hfun_composite(
            self,
            hfun_path_list: List[Union[str, Path]]
            ) -> MeshData:
        """Internal: combine the size functions written to disk

        Combine the size functions written to disk from the list of
        input files `hfun_path_list`. This is used for `exact` method.

        Parameters
        ----------
        hfun_path_list : list of path-like

        Retruns
        -------
        MeshData
            The combined size function interpolated on an optimized
            mesh.

        See Also
        --------
        _get_hfun_composite_fast :
        """

        collection = []
        _logger.info('Reading 2dm hfun files...')
        start = time()
        for path in hfun_path_list:
            collection.append(Mesh.open(path, crs='EPSG:4326'))
        _logger.info(f'Reading 2dm hfun files took {time()-start}.')

        # NOTE: Overlaps are taken care of in the write stage

        coord = []
        index = []
        value = []
        offset = 0
        for hfun in collection:
            index.append(hfun.triangles + offset)
            coord.append(hfun.coord)
            value.append(hfun.value.reshape(-1, 1))
            offset += hfun.coord.shape[0]

        composite_hfun = MeshData(
                coords=np.vstack(coord),
                tria=np.vstack(index),
                values=np.vstack(value),
                crs=CRS.from_user_input("EPSG:4326")
        )

        # NOTE: In the end we need to return in a CRS that
        # uses meters as units. UTM based on the center of
        # the bounding box of the hfun is used
        # Up until now all calculation was in EPSG:4326
        utils.project_to_utm(composite_hfun)

        return composite_hfun


    def _create_big_raster(self, out_path: Union[str, Path]) -> Raster:
        """Internal: create a large raster covering all input rasters.

        Parameters
        ----------
        out_path : path-like
            Path of the (tempoerary) directory to which the large
            raster needs to be written

        Returns
        -------
        Raster
            Lower resolution raster covering all input rasters.
        """

        out_dir = Path(out_path)
        out_rast = out_dir / 'big_raster.tif'

        rast_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        if len(rast_hfun_list) == 0:
            return None

        all_bounds = []
        n_cell_lim = 0
        for hfun_in in rast_hfun_list:
            n_cell_lim = max(
                hfun_in.raster.src.shape[0]
                    * hfun_in.raster.src.shape[1],
                n_cell_lim)
            all_bounds.append(
                    hfun_in.get_bbox(crs='EPSG:4326').bounds)
        # 3 is just a arbitray tolerance for memory limit calculations
        n_cell_lim = n_cell_lim * self._nprocs / 3
        all_bounds = np.array(all_bounds)

        x0, y0 = np.min(all_bounds[:, [0, 1]], axis=0)
        x1, y1 = np.max(all_bounds[:, [2, 3]], axis=0)

        utm_crs = utils.estimate_bounds_utm(
                (x0, y0, x1, y1), "EPSG:4326")
        assert utm_crs is not None
        transformer = Transformer.from_crs(
                'EPSG:4326', utm_crs, always_xy=True)

        # If it's the full earth, then the spaces are at least 1/2 deg
        xs = np.linspace(x0, x1, 720)
        ys = np.linspace(y0, y1, 720)
        coords = [[x0, y] for y in ys]
        coords.extend([[x, y1] for x in xs])
        coords.extend([[x1, y] for y in reversed(ys)])
        coords.extend([[x, y0] for x in reversed(xs)])
        poly_epsg4326 = Polygon(np.array(coords))
        poly_utm = ops.transform(transformer.transform, poly_epsg4326)
        x0, y0, x1, y1 = poly_utm.bounds

        worst_res = 0
        for hfun_in in rast_hfun_list:
            bnd1 = hfun_in.get_bbox(crs=utm_crs).bounds
            dim1 = np.max([bnd1[2] - bnd1[0], bnd1[3] - bnd1[1]])
            bnd2 = hfun_in.get_bbox(crs='EPSG:4326').bounds
            dim2 = np.max([bnd2[2] - bnd2[0], bnd2[3] - bnd2[1]])
            ratio = dim1 / dim2
            pixel_size_x = hfun_in.raster.src.transform[0] * ratio
            pixel_size_y = -hfun_in.raster.src.transform[4] * ratio

            worst_res = np.max([worst_res, pixel_size_x, pixel_size_y])

        # TODO: What if no hmin? -> use smallest raster res!
        g_hmin = self._size_info['hmin']
        res = np.max([g_hmin / 2, worst_res])
        _logger.info(
                f"Spatial resolution"
                f" chosen: {res}, worst: {worst_res}")
        shape0 = int(np.ceil(abs(x1 - x0) / res))
        shape1 = int(np.ceil(abs(y1 - y0) / res))

        approx =  int(np.sqrt(n_cell_lim))
        window_size = None #default of OCSMesh.raster.Raster
        mem_lim = 0 # default of rasterio
        if approx < max(shape0, shape1):
            window_size = np.min([shape0, shape1, approx])
            # Memory limit in MB
            mem_lim = n_cell_lim * np.float32(1).itemsize / 10e6


        # NOTE: Upper-left vs lower-left origin
        # (this only works for upper-left)
        transform = from_origin(x0 - res / 2, y1 + res / 2, res, res)

        rast_profile = {
                'driver': 'GTiff',
                'dtype': np.float32,
                'width': shape0,
                'height': shape1,
                'crs': utm_crs,
                'transform': transform,
                'count': 1,
        }
        with rasterio.open(str(out_rast), 'w', **rast_profile) as dst:
            # For places where raster is DEM is not provided it's
            # assumed deep ocean for contouring purposes
            if window_size is not None:
                write_wins = get_iter_windows(
                    shape0, shape1, chunk_size=window_size)
                for win in write_wins:
                    z = np.full((win.width, win.height), -99999, dtype=np.float32)
                    dst.write(z, 1, window=win)

            else:
                z = np.full((shape0, shape1), -99999, dtype=np.float32)
                dst.write(z, 1)
                del z


            # Reproject if needed (for now only needed if constant
            # value levels or subtidal limiters are added)
            for in_idx, hfun in enumerate(rast_hfun_list):
                ignore = True
                for (src_idx, _, _), _ in self._const_val_contour_coll:
                    if src_idx is None or in_idx in src_idx:
                        ignore = False
                        break
                for src_idx, _, _, _, _ in self._flow_lim_coll:
                    if src_idx is None or in_idx in src_idx:
                        ignore = False
                        break
                for src_idx, _ in self._constraint_info_coll:
                    if src_idx is None or in_idx in src_idx:
                        ignore = False
                        break
                if ignore:
                    continue

                # NOTE: Last one implicitely has highest priority in
                # case of overlap
                reproject(
                    source=rasterio.band(hfun.raster.src, 1),
                    destination=rasterio.band(dst, 1),
                    resampling=Resampling.nearest,
                    init_dest_nodata=False, # To avoid overwrite
                    num_threads=self._nprocs,
                    warp_mem_limit=mem_lim)



        return Raster(out_rast, chunk_size=window_size)

    def _apply_features_fast(self, big_raster: HfunRaster):
        """Internal: apply all specified refinements and constraints

        Apply all specified refinements and constrains for the fast
        algorithm.

        Parameters
        ----------
        big_raster : HfunRaster
            The lower-resolution large raster that covers all the
            input rasters.

        Returns
        -------
        None

        See Also
        --------
        _apply_features :
        """

        # NOTE: Caching applied doesn't work here since we apply
        # everything on a temporary big raster
        rast_hfun_list = []
        hfun_rast = None
        if big_raster:
            hfun_rast = HfunRaster(big_raster, **self._size_info)
            rast_hfun_list.append(hfun_rast)



        mesh_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunMesh)]
        if self._base_mesh and self._base_as_hfun:
            mesh_hfun_list.insert(0, self._base_mesh)

        # Mesh hfun parts are still stateful
        self._apply_contours([*mesh_hfun_list, *rast_hfun_list])
        if hfun_rast:
            # In fast method we only have big raster if any
            self._apply_flow_limiters_fast(hfun_rast)
            self._apply_const_val_fast(hfun_rast)
        # Mesh hfun parts are still stateful
        self._apply_linefeatures([*mesh_hfun_list, *rast_hfun_list])
        self._apply_patch([*mesh_hfun_list, *rast_hfun_list])
        self._apply_channels([*mesh_hfun_list, *rast_hfun_list])

        if hfun_rast:
            self._apply_constraints_fast(hfun_rast)


        return hfun_rast

    def _apply_flow_limiters_fast(self, big_hfun: HfunRaster) -> None:
        """Internal: apply specified sub tidal flow limiter refinements

        Applies specified subtidal flow limiter refinements for
        the fast algorithm.

        Parameters
        ----------
        big_hfun : HfunRaster
            The lower-resolution large raster that covers all the
            input rasters.

        Returns
        -------
        None

        See Also
        --------
        _apply_flow_limiters :
        """

        for src_idx, hmin, hmax, zmax, zmin in self._flow_lim_coll:
            # TODO: Account for source index
            if hmin is None:
                hmin = self._size_info['hmin']
            if hmax is None:
                hmax = self._size_info['hmax']

            # To avoid sharp gradient where no raster is projected
            if zmin is None:
                zmin = -99990
            else:
                zmin = max(zmin, -99990)

            big_hfun.add_subtidal_flow_limiter(hmin, hmax, zmin, zmax)

    def _apply_const_val_fast(self, big_hfun):
        """Internal: apply specified constant value refinements.

        Applies constant value refinements for the fast algorithm.

        Returns
        -------
        None

        See Also
        --------
        _apply_const_val :
        """

        for (src_idx, ctr0, ctr1), const_val in self._const_val_contour_coll:
            # TODO: Account for source index
            level0 = None
            level1 =  None
            if ctr0 is not None:
                level0 = ctr0.level
            if ctr1 is not None:
                level1 = ctr1.level
            big_hfun.add_constant_value(const_val, level0, level1)


    def _apply_constraints_fast(self, big_hfun: HfunRaster) -> None:
        """Internal: apply specified constraints.

        Apply specified constraints for the fast algorithm.

        Parameters
        ----------

        Returns
        -------
        None

        See Also
        --------
        _apply_constraints
        """

        # TODO: Account for source index
        self._constraint_info_coll.apply([big_hfun], per_hfun=False)


    def _get_hfun_composite_fast(self, big_hfun, **kwargs) -> MeshData:
        """Internal: combine the size function functions for fast method

        Combine the size functions of the large raster with non-raster
        inputs. This is used for `fast` method.

        Parameters
        ----------
        big_hfun : HfunRaster
            The single large raster based size function covering all
            input rasters.

        Retruns
        -------
        MeshData
            The combined size function interpolated on an optimized
            mesh.

        See Also
        --------
        _get_hfun_composite
        """

        # In fast method all DEM hfuns have more priority than all other inputs
        dem_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        nondem_hfun_list = [
            i for i in self._hfun_list if not isinstance(i, HfunRaster)]

        epsg4326 = CRS.from_user_input("EPSG:4326")

        dem_box_list = []
        for hfun in dem_hfun_list:
            dem_box_list.append(hfun.get_bbox(crs=epsg4326))

        index = []
        coord = []
        value = []
        offset = 0

        # Calculate multipoly and clip big hfun
        big_cut_shape = None
        if big_hfun:
            dem_gdf = gpd.GeoDataFrame(
                    geometry=dem_box_list, crs=epsg4326)
            big_cut_shape = dem_gdf.union_all()

            #  (e.g. mesh_engine='gmsh')
            big_meshdata = big_hfun.meshdata(**kwargs)

            if big_meshdata.crs is not None:
                if not epsg4326.equals(big_meshdata.crs):
                    utils.reproject(big_meshdata, epsg4326)

            big_meshdata = utils.clip_mesh_by_shape(
                big_meshdata,
                big_cut_shape,
                use_box_only=False,
                fit_inside=False)

            if big_meshdata.tria is not None:
                index.append(big_meshdata.tria + offset)

            coord.append(big_meshdata.coords)
            value.append(big_meshdata.values)
            offset = offset + coord[-1].shape[0]

        hfun_list = nondem_hfun_list[::-1]
        if self._base_mesh and self._base_as_hfun:
            hfun_list = [*nondem_hfun_list[::-1], self._base_mesh]

        nondem_shape_list = []
        for hfun in hfun_list:
            nondem_meshdata = deepcopy(hfun.meshdata())
            if hasattr(nondem_meshdata, "crs"):
                if not epsg4326.equals(nondem_meshdata.crs):
                    utils.reproject(nondem_meshdata, epsg4326)

            nondem_shape = utils.get_mesh_polygons(hfun.mesh.meshdata)
            if not epsg4326.equals(hfun.crs):
                transformer = Transformer.from_crs(
                    hfun.crs, epsg4326, always_xy=True)
                nondem_shape = ops.transform(
                        transformer.transform, nondem_shape)

            # In fast method all DEM hfuns have more priority than all
            # other inputs
            if big_cut_shape:
                nondem_meshdata = utils.clip_mesh_by_shape(
                    nondem_meshdata,
                    big_cut_shape,
                    use_box_only=False,
                    fit_inside=True,
                    inverse=True)

            for ishp in nondem_shape_list:
                nondem_meshdata = utils.clip_mesh_by_shape(
                    nondem_meshdata,
                    ishp,
                    use_box_only=False,
                    fit_inside=True,
                    inverse=True)

            nondem_shape_list.append(nondem_shape)

            if nondem_meshdata.tria is not None:
                index.append(nondem_meshdata.tria + offset)
            coord.append(nondem_meshdata.coords)
            value.append(nondem_meshdata.values)
            offset += coord[-1].shape[0]

        # Stack arrays
        final_coords = np.vstack(coord) if coord else np.empty((0, 2))
        final_values = np.vstack(value) if value else np.empty((0, 1))
        # Handle case where some inputs might not have triangles (points only)
        if index:
            final_tria = np.vstack(index)
        else:
            final_tria = None

        composite_hfun = MeshData(
            coords=final_coords,
            tria=final_tria,
            values=final_values,
            crs=epsg4326
        )

        hmin = self._size_info['hmin']
        hmax = self._size_info['hmax']
        if hmin:
            composite_hfun.values[composite_hfun.values < hmin] = hmin
        if hmax:
            composite_hfun.values[composite_hfun.values > hmax] = hmax

        utils.project_to_utm(composite_hfun)

        return composite_hfun
