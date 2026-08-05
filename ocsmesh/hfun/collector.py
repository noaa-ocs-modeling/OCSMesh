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
import sys
import shutil
import gc
import logging
import warnings
import tempfile
import traceback
from pathlib import Path
from time import time
from multiprocessing import Pool, cpu_count
from copy import copy, deepcopy
from typing import (
    Union, Sequence, List, Tuple, Iterable, Any, Optional, Callable)
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
from shapely import ops
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import rasterio

from ocsmesh import utils
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

# TODO: review if we can make this cleaner
# MPI support is optional
_MPI = None
_MPI_IMPORT_ATTEMPTED = False


def _get_mpi():
    """Lazy-import mpi4py.MPI. Returns the module or None."""
    global _MPI, _MPI_IMPORT_ATTEMPTED
    if not _MPI_IMPORT_ATTEMPTED:
        _MPI_IMPORT_ATTEMPTED = True
        try:
            from mpi4py import MPI
            _MPI = MPI
        except ImportError:
            pass
    return _MPI


def _get_mpi_comm():
    """Return MPI.COMM_WORLD if available, else None."""
    MPI = _get_mpi()
    if MPI is None:
        return None
    return MPI.COMM_WORLD


def _is_mpi_active():
    """Check if we're running under an MPI launcher with >1 rank."""
    comm = _get_mpi_comm()
    if comm is None:
        return False
    try:
        return comm.Get_size() > 1
    except Exception:
        return False

# NOTE: multiprocessing.set_start_method('spawn') is handled in
# ocsmesh/__init__.py (before any import can initialize the context).
# Thread pinning (_MPI_THREAD_PIN_VARS) is also handled there.
# The _configure_mpi_environment() function below is a fallback
# for direct imports that bypass __init__.py.

def _configure_mpi_environment():
    # TODO: Review if this is necessary. It is a safety net for now.
    """Safety net for MPI + numerical library thread pinning.

    The primary thread pinning happens in ``ocsmesh/__init__.py``
    (before NumPy is imported). This function serves as a fallback
    for edge cases where collector.py is imported directly without
    going through the package ``__init__``.
    """
    from ocsmesh import _MPI_THREAD_PIN_VARS
    for var in _MPI_THREAD_PIN_VARS:
        os.environ.setdefault(var, '1')


class MPITaskRunner:
    """Dynamic manager/worker MPI task runner.

    Encapsulates all MPI dispatch logic. Called identically by all
    ranks — zero rank checks in user code.

    User script::

        from ocsmesh.hfun.collector import MPITaskRunner

        runner = MPITaskRunner()

        def main():
            hfun = Hfun(raster_list)
            hfun.execution_mode = 'mpi'
            return hfun.meshdata()

        result = runner.run(main)  # All ranks call this

    Features
    --------
    - Dynamic on-demand scheduling (fast ranks pull more work)
    - Rank 0 = dedicated coordinator
    - Soft-fail: worker exceptions → structured error dicts,
      worker stays alive for more tasks
    - Global excepthook prevents zombie cloud/HPC processes
    - Individual TAG_STOP per worker (no collective shutdown)
    - Sequential fallback when size == 1 or no MPI

    Adding a new MPI operation requires:
      1. Write a `_*_task_worker(task)` function
      2. Register it in `MPITaskRunner._worker_registry()`: `{'my_op': _my_op_task_worker}`
    """

    # Message tags for the manager/worker protocol.
    # Using a dict keeps them grouped and discoverable.
    _TAGS = {
        'TASK':   1,   # rank 0 → worker : here is a task dict
        'RESULT': 2,   # worker → rank 0 : task succeeded (result dict)
        'ERROR':  3,   # worker → rank 0 : task raised / structured failure
        'STOP':   4,   # rank 0 → worker : no more work, leave the loop
    }

    @staticmethod
    def _worker_registry():
        """Map operation name → worker function.

        Built lazily (at call time) rather than at import time so it can
        reference worker functions defined later in this module without a
        forward-reference NameError. Extend this dict to add new
        MPI-enabled operations.
        """
        return {
            'meshdata': _meshdata_task_worker,
            'check_shared_fs': _check_shared_fs_task_worker,
        }

    @staticmethod
    def install_mpi_excepthook():
        """Override ``sys.excepthook`` to abort all MPI ranks on uncaught exception.

        When an uncaught exception propagates to the top of the stack on any
        rank, this hook writes a traceback to stderr and calls
        ``comm.Abort(1)`` to kill ALL processes across ALL nodes immediately.
        This prevents zombie cloud/HPC instances that keep running while
        blocked in a ``recv()``.

        Only triggers on truly uncaught exceptions; normal task failures are
        caught by the worker loop and returned as structured error dicts.
        """
        MPI = _get_mpi()
        if MPI is None:
            return

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        def _mpi_excepthook(exctype, value, tb):
            sys.stderr.write(
                f"\n[CRITICAL] Uncaught exception on Rank {rank}:\n"
            )
            traceback.print_exception(exctype, value, tb, file=sys.stderr)
            sys.stderr.flush()
            comm.Abort(1)

        sys.excepthook = _mpi_excepthook

    def __init__(self):
        """Initialize MPI state and install global safety net."""
        comm = _get_mpi_comm()
        if comm is not None:
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        # Install global safety net for multi-rank jobs
        if self.comm is not None and self.size > 1:
            self.install_mpi_excepthook()

    # ── Public API ─────────────────────────────────────────────────

    def run(self, user_fn):
        """Execute user_fn on Rank 0, worker loop on all other ranks.

        All ranks call this identically. Rank 0 runs *user_fn* (which
        internally calls :meth:`dispatch` for MPI work). Workers enter
        the recv/execute/send loop and exit when Rank 0 finishes.

        Workers are always released in the ``finally`` block — even if
        *user_fn* raises an exception.

        Parameters
        ----------
        user_fn : callable
            The function to run on Rank 0. Must accept no arguments.

        Returns
        -------
        any
            Return value of *user_fn* on Rank 0, ``None`` on workers.
        """
        # ── Safety check ──
        if not callable(user_fn):
            raise TypeError(
                f"MPITaskRunner.run() expects a callable, "
                f"got {type(user_fn).__name__}"
            )

        # Single rank or no MPI: just run directly
        if self.comm is None or self.size == 1:
            return user_fn()

        if self.rank == 0:
            try:
                return user_fn()
            finally:
                # Always release workers, even on exception
                self._shutdown_workers()
        else:
            self._run_worker()
            return None

    def dispatch(self, tasks):
        """Rank-0-only: stream tasks dynamically to idle workers.

        Seeds one task per worker, then refills each worker as soon as
        it returns a result — a central work queue with on-demand
        assignment. Returns the list of result/error dicts (order is
        NOT the task order; use ``'original_index'`` to reassociate).

        Rank 0 does not compute any task here (dedicated coordinator).

        Parameters
        ----------
        tasks : list of dict
            Self-describing task dicts, each with an ``'op'`` key.

        Returns
        -------
        list of dict
            One result (or structured error) dict per task.
        Notes
        -----
        Must be called from within the callable passed to :meth:`run`.
        Calling this method standalone without ``run()`` will cause workers
        to hang permanently — ``run()`` is responsible for both entering
        the worker recv loop and guaranteeing shutdown via its ``finally``
        block.
        """
        # ── Safety checks ──
        if self.comm is None:
            raise RuntimeError("mpi4py is not available")
        if self.rank != 0:
            raise RuntimeError(
                "dispatch() must only be called by Rank 0."
            )
        if not isinstance(tasks, list):
            raise TypeError(
                f"dispatch() expects a list of task dicts, "
                f"got {type(tasks).__name__}"
            )

        # Degenerate case: no workers (size == 1)
        # Run tasks locally so single-rank runs stay correct.
        if self.size == 1:
            return self._run_tasks_locally(tasks)

        results = []
        task_iter = iter(tasks)
        inflight = 0  # tasks currently being processed by workers

        # ── Seed: give each worker one task to start ──
        for worker in range(1, self.size):
            task = next(task_iter, None)
            if task is None:
                break  # fewer tasks than workers
            self.comm.send(task, dest=worker, tag=self._TAGS['TASK'])
            inflight += 1

        # ── Refill: as each worker reports back, hand it the next ──
        MPI = _get_mpi()
        while inflight > 0:
            status = MPI.Status()
            message = self.comm.recv(
                source=MPI.ANY_SOURCE,
                tag=MPI.ANY_TAG,
                status=status,
            )
            worker = status.Get_source()
            inflight -= 1
            results.append(message)

            next_task = next(task_iter, None)
            if next_task is not None:
                self.comm.send(
                    next_task, dest=worker, tag=self._TAGS['TASK'])
                inflight += 1

        return results

    def verify_shared_filesystem(self, work_dir: Union[str, Path]):
        """Rank 0 only: verify all workers can access and write to work_dir.

        On multi-node HPC jobs, default temporary directories (like /tmp)
        live on node-local disks. If workers are on separate physical nodes
        from Rank 0, intermediate file operations will fail. This method
        dispatches lightweight test tasks to confirm cross-node read visibility
        and write permissions before any computationally expensive work begins.

        Parameters
        ----------
        work_dir : path-like
            The working directory path to test across all worker ranks.

        Raises
        ------
        RuntimeError
            If any worker rank fails to read from or write to `work_dir`.
        """
        if self.comm is None or self.size <= 1:
            return
        if self.rank != 0:
            raise RuntimeError(
                "verify_shared_filesystem() must only be called by Rank 0."
            )

        work_dir_str = str(work_dir)
        pid = os.getpid()
        _logger.info(
            f"Verifying shared filesystem visibility across "
            f"{self.size - 1} worker rank(s)..."
        )
        rank0_test_file = os.path.join(
            work_dir_str, f".mpi_fs_check_{pid}.tmp"
        )
        with open(rank0_test_file, 'w') as f:
            f.write("rank0_ok")

        try:
            check_tasks = [
                {
                    'op': 'check_shared_fs',
                    'original_index': idx,
                    'work_dir': work_dir_str,
                    'test_read_file': rank0_test_file,
                    'worker_rank': idx + 1,
                }
                for idx in range(self.size - 1)
            ]
            check_results = self.dispatch(check_tasks)
        finally:
            if os.path.exists(rank0_test_file):
                try:
                    os.remove(rank0_test_file)
                except OSError:
                    pass

        check_failures = [
            res for res in check_results
            if not isinstance(res, dict) or res.get('status') == 'error'
        ]
        if check_failures:
            first_err = (
                check_failures[0].get('error', repr(check_failures[0]))
                if isinstance(check_failures[0], dict)
                else repr(check_failures[0])
            )
            wrk = (
                check_failures[0].get('worker_rank', '?')
                if isinstance(check_failures[0], dict)
                else '?'
            )
            raise RuntimeError(
                f"\n[MPI Shared Filesystem Error] Worker rank {wrk} failed "
                f"to access or write to working directory "
                f"'{work_dir_str}':\n  {first_err}\n\n"
                f"On multi-node HPC/cluster jobs, intermediate files must "
                f"be created on a shared parallel filesystem (such as "
                f"Lustre, GPFS, or NFS) visible to all nodes, rather than "
                f"node-local /tmp.\n\n"
                f"--> ACTION REQUIRED: Set the TMPDIR environment variable "
                f"to point to your shared network filesystem before "
                f"running OCSMesh:\n"
                f"    export TMPDIR=/scratch/$USER\n"
                f"    # or export TMPDIR=/lustre/..."
            )
        _logger.info("Shared filesystem check passed across all worker ranks.")

    # ── Internal ───────────────────────────────────────────────────

    def _run_worker(self):
        """Worker recv/execute/send loop (point-to-point).

        Each non-zero rank calls this ONCE. The worker blocks in a
        ``recv`` until rank 0 either hands it a task (``TAG_TASK``) or
        tells it to stop (``TAG_STOP``). After running a task it sends
        the result back and loops again.

        A blocking ``recv`` is the correct idle-wait primitive: the
        worker sleeps inside MPI without busy-polling and is woken only
        when a message addressed to it arrives.
        """
        # ── Safety checks ──
        if self.rank == 0:
            raise RuntimeError(
                "_run_worker() must only be called by worker ranks."
            )

        MPI = _get_mpi()
        registry = self._worker_registry()
        _logger.debug(
            f"Rank {self.rank}: entering point-to-point worker loop"
        )

        while True:
            # Block until rank 0 sends us something. ANY_TAG lets a
            # single recv distinguish a task from a stop signal.
            status = MPI.Status()
            message = self.comm.recv(
                source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == self._TAGS['STOP']:
                _logger.debug(
                    f"Rank {self.rank}: received STOP, leaving loop"
                )
                break

            if tag != self._TAGS['TASK']:
                # Defensive: unknown control tag. Report and keep
                # serving so a protocol slip does not silently hang.
                _logger.warning(
                    f"Rank {self.rank}: unexpected tag {tag!r}; "
                    f"ignoring message"
                )
                continue

            task = message
            op = task.get('op') if isinstance(task, dict) else None
            worker_fn = registry.get(op)

            if worker_fn is None:
                # Unknown operation → structured error, worker stays.
                self.comm.send(
                    {
                        'status': 'error',
                        'original_index': (
                            task.get('original_index', -1)
                            if isinstance(task, dict) else -1),
                        'op': op,
                        'worker_rank': self.rank,
                        'error': f"No worker registered for op "
                                 f"{op!r}",
                        'traceback': '',
                    },
                    dest=0, tag=self._TAGS['ERROR'],
                )
                continue

            try:
                result = worker_fn(task)
                # Some worker functions catch their own exceptions
                # and return {'status': 'error', ...}. Route those
                # through the error channel too.
                if (isinstance(result, dict)
                        and result.get('status') == 'error'):
                    result.setdefault('worker_rank', self.rank)
                    self.comm.send(
                        result, dest=0, tag=self._TAGS['ERROR'])
                else:
                    self.comm.send(
                        result, dest=0, tag=self._TAGS['RESULT'])
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Task blew up — NOT an MPI rank failure. Catch it,
                # report it, keep the worker available for more tasks.
                self.comm.send(
                    {
                        'status': 'error',
                        'original_index': (
                            task.get('original_index', -1)
                            if isinstance(task, dict) else -1),
                        'op': op,
                        'worker_rank': self.rank,
                        'error': repr(exc),
                        'traceback': traceback.format_exc(),
                    },
                    dest=0, tag=self._TAGS['ERROR'],
                )

        _logger.debug(
            f"Rank {self.rank}: worker loop exited cleanly"
        )

    def _shutdown_workers(self):
        """Send TAG_STOP to each worker individually.

        Individual sends (not a collective) mean each worker exits
        independently — if one already crashed, the others still
        receive their stop signal cleanly.
        """
        _logger.debug(
            f"Rank 0: sending STOP to {self.size - 1} worker(s)"
        )
        for worker in range(1, self.size):
            self.comm.send(None, dest=worker, tag=self._TAGS['STOP'])

    def _run_tasks_locally(self, tasks):
        """Fallback for single-rank: run tasks without MPI."""
        registry = self._worker_registry()
        results = []
        for task in tasks:
            fn = registry.get(task.get('op'))
            if fn is None:
                results.append({
                    'status': 'error',
                    'original_index': task.get('original_index', -1),
                    'error': f"No worker registered for op "
                             f"{task.get('op')!r}",
                })
                continue
            try:
                results.append(fn(task))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                results.append({
                    'status': 'error',
                    'original_index': task.get('original_index', -1),
                    'error': repr(exc),
                    'traceback': traceback.format_exc(),
                })
        return results


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
                # Copy so that in case of a 2nd run the no-source
                # contour still gets all current sources
                contour_defn = copy(contour_defn)
                for source in source_list:
                    contour_defn.add_source(source)

            for contour, crs in contour_defn.iter_contours():
                file_counter = file_counter + 1
                feather_path = out_dir / f"contour_{pid}_{file_counter}.feather"
                crs_path = out_dir / f"crs_{pid}_{file_counter}.json"
                gpd.GeoDataFrame(
                    { 'geometry': [contour],
                      'expansion_rate': size_info['expansion_rate'],
                      'target_size': size_info['target_size'],
                    },
                    crs=crs).to_feather(feather_path)
                gc.collect()
                with open(crs_path, 'w') as fp:
                    fp.write(crs.to_json())
                self._container.append((feather_path, crs_path))


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
                # Copy so that in case of a 2nd run the no-source
                # channel still gets all current sources
                channel_defn = copy(channel_defn)
                for source in source_list:
                    channel_defn.add_source(source)

            for channels, crs in channel_defn.iter_channels():
                file_counter = file_counter + 1
                feather_path = out_dir / f"channels_{pid}_{file_counter}.feather"
                crs_path = out_dir / f"crs_{pid}_{file_counter}.json"
                gpd.GeoDataFrame(
                    { 'geometry': [channels],
                      'expansion_rate': size_info['expansion_rate'],
                      'target_size': size_info['target_size'],
                    },
                    crs=crs).to_feather(feather_path)
                gc.collect()
                with open(crs_path, 'w') as fp:
                    fp.write(crs.to_json())
                self._container.append((feather_path, crs_path))

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


def _flow_limiter_task_worker(task: dict):
    """
    A self-contained worker that operates ONLY on file paths.
    """

    # 1. Unpack the simple, pickleable task description
    original_index = task['original_index']
    hfun_input_path = task['hfun_input_path']
    topo_input_path = task['topo_input_path']
    output_path = task['output_path']
    global_hmin = task['global_hmin']
    global_hmax = task['global_hmax']
    limiter_params_list = task['limiter_params']

    # Let's initialize directly from the input path.

    # 2. Create the necessary Raster and HfunRaster instances INSIDE the worker.
    topo_raster = Raster(topo_input_path)
    worker_hfun = HfunRaster(
        raster=topo_raster,
        hmin=global_hmin,
        hmax=global_hmax,
        verbosity=0,
        initial_value=hfun_input_path
    )

    # 3. Apply all the required flow limiter refinements.
    #    Each call will modify the worker_hfun's internal state (_tmpfile).
    for params in limiter_params_list:
        worker_hfun.add_subtidal_flow_limiter(
            hmin=params['hmin'],
            hmax=params['hmax'],
            lower_bound=params['zmin'],
            upper_bound=params['zmax']
        )

    # 4. Explicitly save the final state of the worker's
    #    object to the designated output path.
    worker_hfun.save(output_path)

    # 5. The work is done. Return the simple result dictionary.
    return {
        'status': 'success',
        'original_index': original_index,
        'output_path': output_path
    }


def _const_val_task_worker(task: dict):
    """
    A self-contained worker for applying constant value refinements.
    It reads the state from the previous step, applies its rules, and saves
    a new file with the result.
    """

    # 1. Unpack the simple, pickleable task description
    original_index = task['original_index']
    hfun_input_path = task['hfun_input_path']
    topo_input_path = task['topo_input_path']
    output_path = task['output_path']
    global_hmin = task['global_hmin']
    global_hmax = task['global_hmax']
    const_val_rules = task['const_val_rules']

    # 2. Create the necessary Raster and HfunRaster instances INSIDE the worker.
    #    This is the "Wrap Existing Painting" mode.
    topo_raster = Raster(topo_input_path)
    worker_hfun = HfunRaster(
        raster=topo_raster,
        hmin=global_hmin,
        hmax=global_hmax,
        verbosity=0,
        initial_value=hfun_input_path
    )

    # 3. Apply all the required constant value refinements for this raster.
    for rule in const_val_rules:
        worker_hfun.add_constant_value(
            value=rule['value'],
            lower_bound=rule['lower_bound'],
            upper_bound=rule['upper_bound']
        )

    # 4. Explicitly save the final state of the worker's object to the
    #    designated output path.
    worker_hfun.save(output_path)

    # 5. The work is done. Return a simple result dictionary.
    return {
        'status': 'success',
        'original_index': original_index,
        'output_path': output_path
    }


def _constraints_task_worker(task: dict):
    """
    A self-contained worker for applying constraint objects to a single
    HfunRaster.

    This worker reconstructs the HfunRaster from file paths inside the
    child process, applies pickleable constraint objects via
    ``HfunRaster.apply_constraints()``, and saves the result to disk.
    """

    # 1. Unpack the simple, pickleable task description
    original_index = task['original_index']
    hfun_input_path = task['hfun_input_path']
    topo_input_path = task['topo_input_path']
    output_path = task['output_path']
    global_hmin = task['global_hmin']
    global_hmax = task['global_hmax']
    constraint_list = task['constraint_list']

    # 2. Create the necessary Raster and HfunRaster instances INSIDE the worker.
    topo_raster = Raster(topo_input_path)
    worker_hfun = HfunRaster(
        raster=topo_raster,
        hmin=global_hmin,
        hmax=global_hmax,
        verbosity=0,
        initial_value=hfun_input_path
    )

    # 3. Apply the constraint objects using the existing HfunRaster method.
    #    This handles windowed processing, hmin/hmax clamping, etc.
    worker_hfun.apply_constraints(constraint_list)

    # 4. Save the final state to the designated output path.
    worker_hfun.save(output_path)

    # 5. Return a simple result dictionary.
    return {
        'status': 'success',
        'original_index': original_index,
        'output_path': output_path
    }


def _meshdata_task_worker(task: dict):
    """Worker that calls hfun.meshdata() on a single hfun entry.

    Supports two task types:
    - ``'raster'``: Reconstructs an ``HfunRaster`` from file paths
      (raster data cannot be pickled directly).
    - ``'mesh'``: Uses a pre-pickled ``HfunMesh`` object directly
      (HfunMesh is fully picklable).

    Runs the expensive triangulation/interpolation pipeline inside a
    worker process and serializes the raw MeshData result to disk as
    ``.npz``.  Overlap clipping is NOT performed here — that requires
    sequential bounding-box accumulation and is handled by Stage 2 in
    the coordinator.
    """

    original_index = task['original_index']
    output_path = task['output_path']
    meshdata_kwargs = task.get('meshdata_kwargs', {})
    task_type = task.get('type', 'raster')

    try:
        if task_type == 'raster':
            # Reconstruct HfunRaster in the worker process
            topo_raster = Raster(task['topo_path'])
            worker_hfun = HfunRaster(
                raster=topo_raster,
                hmin=task['hmin'],
                hmax=task['hmax'],
                verbosity=0,
                initial_value=task['hfun_input_path']
            )
            meshdata_result = worker_hfun.meshdata(**meshdata_kwargs)

        elif task_type == 'mesh':
            # HfunMesh is picklable — use the object directly
            worker_hfun = task['hfun_obj']
            try:
                meshdata_result = deepcopy(worker_hfun.meshdata(**meshdata_kwargs))
            except TypeError:
                # HfunMesh.meshdata() doesn't accept kwargs like stride
                meshdata_result = deepcopy(worker_hfun.meshdata())
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Reproject to EPSG:4326 (same as serial path)
        if hasattr(meshdata_result, "crs"):
            dst_crs = CRS.from_user_input("EPSG:4326")
            if meshdata_result.crs != dst_crs:
                utils.reproject(meshdata_result, dst_crs)

        # Serialize to .npz for fast coordinator pickup
        tria_data = (meshdata_result.tria
                     if meshdata_result.tria is not None
                     else np.array([]))
        quad_data = (meshdata_result.quad
                     if meshdata_result.quad is not None
                     else np.array([]))
        crs_str = (str(meshdata_result.crs)
                   if meshdata_result.crs else "")
        np.savez(
            output_path,
            coords=meshdata_result.coords,
            tria=tria_data,
            quad=quad_data,
            values=meshdata_result.values,
            crs=np.array(crs_str)
        )

        return {
            'status': 'success',
            'original_index': original_index,
            'output_path': str(output_path) + '.npz'
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {
            'status': 'error',
            'original_index': original_index,
            'op': task.get('op'),
            'error': repr(e),
            'traceback': traceback.format_exc(),
        }


def _check_shared_fs_task_worker(task: dict):
    """Worker task that validates cross-node shared filesystem access.

    Verifies that this worker rank can both read a file created by Rank 0
    in `work_dir` (proving cross-node read visibility) and create a new
    file in `work_dir` (proving write permissions).
    """
    original_index = task.get('original_index', -1)
    work_dir = task['work_dir']
    test_read_file = task['test_read_file']
    worker_rank = task.get('worker_rank', -1)

    try:
        # 1. Check read visibility of Rank 0's test file
        if not (os.path.exists(test_read_file) and os.access(test_read_file, os.R_OK)):
            raise FileNotFoundError(
                f"Rank {worker_rank} cannot see/read Rank 0 test file: {test_read_file}"
            )

        # 2. Check write permission by creating and removing a temporary test file
        worker_test_file = os.path.join(
            work_dir, f".mpi_fs_write_test_rank_{worker_rank}_{os.getpid()}.tmp"
        )
        with open(worker_test_file, 'w') as f:
            f.write(f"write_ok_from_rank_{worker_rank}")
        if os.path.exists(worker_test_file):
            os.remove(worker_test_file)
        else:
            raise FileNotFoundError(
                f"Rank {worker_rank} wrote file {worker_test_file} but cannot see it."
            )

        return {
            'status': 'success',
            'original_index': original_index,
            'worker_rank': worker_rank,
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {
            'status': 'error',
            'original_index': original_index,
            'op': task.get('op'),
            'worker_rank': worker_rank,
            'error': repr(e),
            'traceback': traceback.format_exc(),
        }



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
        indices. The index is w.r.t the full input list for collector
        object creation.
    add_constant_value(...)
        Add fixed size mesh refinement in the region specified by
        upper and lower bounds on topography.  This refinement is
        only applied on rasters with specified indices. The index is
        w.r.t the full input list for collector object creation.

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
            base_shape_crs: Union[str, CRS] = 'EPSG:4326'
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
        """

        # NOTE: Input Hfuns and their Rasters can get modified

         # Add a persistent working directory for this instance's outputs
        self._work_dir = tempfile.mkdtemp(prefix='hfun_collector_')
        # TODO: Prove this fix is needed
        self._creator_pid = os.getpid()
        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs

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

        for in_item in in_list:
            # Add supports(ext) to each hfun type?

            if isinstance(in_item, BaseHfun):
                hfun = in_item
            # pylint: disable=R0801
            elif isinstance(in_item, Raster):
                if self._base_shape:
                    clip_shape = self._base_shape
                    if not self._base_shape_crs.equals(in_item.crs):
                        transformer = Transformer.from_crs(
                            self._base_shape_crs, in_item.crs, always_xy=True)
                        clip_shape = ops.transform(
                                transformer.transform, clip_shape)
                    try:
                        in_item.clip(clip_shape)
                    except ValueError as err:
                        # This raster does not intersect shape
                        _logger.debug(err)
                        continue

                elif self._base_mesh:
                    try:
                        in_item.clip(self._base_mesh.mesh.get_bbox(crs=in_item.crs))
                    except ValueError as err:
                        # This raster does not intersect shape
                        _logger.debug(err)
                        continue

                hfun = HfunRaster(in_item, **self._size_info)

            elif isinstance(in_item, EuclideanMesh2D):
                hfun = HfunMesh(in_item)

            elif isinstance(in_item, (str, Path)):
                in_item = str(in_item)
                if in_item.endswith('.tif'):
                    raster = Raster(in_item)
                    if self._base_shape:
                        clip_shape = self._base_shape
                        if not self._base_shape_crs.equals(raster.crs):
                            transformer = Transformer.from_crs(
                                self._base_shape_crs, raster.crs, always_xy=True)
                            clip_shape = ops.transform(
                                    transformer.transform, clip_shape)
                        try:
                            in_item.clip(clip_shape)
                        except ValueError as err:
                            # This raster does not intersect shape
                            _logger.debug(err)
                            continue

                    elif self._base_mesh:
                        try:
                            raster.clip(self._base_mesh.mesh.get_bbox(crs=raster.crs))
                        except ValueError as err:
                            # This raster does not intersect shape
                            _logger.debug(err)
                            continue

                    hfun = HfunRaster(raster, **self._size_info)

                elif in_item.endswith(
                        ('.14', '.grd', '.gr3', '.msh', '.2dm')):
                    mesh = Mesh.open(in_item)
                    hfun = HfunMesh(mesh)

                else:
                    raise TypeError("Input file extension not supported!")

            self._hfun_list.append(hfun) # pylint: disable=E0606


    def __del__(self):
        # TODO: Prove this fix is needed
        if (hasattr(self, '_work_dir')
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

        self._applied = False

        constraint_defn = TopoConstConstraint(
            value, upper_bound, lower_bound, value_type, rate)

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]
        self._constraint_info_coll.add(source_index, constraint_defn)


    def add_topo_func_constraint(
            self,
            func: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
                = lambda i: i / 2.0,
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

        self._applied = False

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

        self._applied = False

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
        self._applied = False

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

        self._applied = False

        # Always lazy
        self._applied = False

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

        self._applied = False

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

        self._applied = False

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

        self._applied = False

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
            crs: CRS = 4326
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

        self._applied = False

        if not line_defn:
            if shape:
                line_defn = LineFeature(shape=shape, shape_crs=crs)

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
            self._apply_contours()
            self._apply_flow_limiters()
            self._apply_const_val()
            self._apply_linefeatures()
            self._apply_patch()
            self._apply_channels()
            self._apply_constraints()

        self._applied = True


    def _apply_constraints(self) -> None:
        """Internal: dispatch constraint application to serial or parallel.

        Dispatches to either ``_apply_constraints_serial`` or
        ``_apply_constraints_parallel`` based on the current
        ``execution_mode``.  When any ``TopoFuncConstraint`` is present
        (which stores an unpickleable lambda), the parallel path falls
        back to serial automatically with a logged warning.

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

        if self.execution_mode == 'parallel' and self._nprocs > 1:
            # TopoFuncConstraint stores a lambda which cannot be pickled
            # for multiprocessing.Pool — fall back to serial in that case.
            has_func_constraint = any(
                isinstance(c, TopoFuncConstraint)
                for _, c in self._constraint_info_coll
            )
            if has_func_constraint:
                warnings.warn(
                    "TopoFuncConstraint contains a callable that cannot "
                    "be pickled for parallel execution. Falling back to "
                    "serial for _apply_constraints().",
                    UserWarning
                )
                self._apply_constraints_serial()
            else:
                _logger.info("Applying constraints using PARALLEL method.")
                self._apply_constraints_parallel()
        else:
            _logger.info("Applying constraints using SERIAL method.")
            self._apply_constraints_serial()


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
        """Internal: apply specified constraints in parallel.

        Uses the same 3-phase pattern as ``_apply_flow_limiters_parallel``:

        1. **Preparation** — build one pickleable task dict per raster,
           containing file paths + the list of applicable ``Constraint``
           objects.
        2. **Execution** — ``Pool.map()`` sends tasks to
           ``_constraints_task_worker`` processes.
        3. **Integration** — replace ``self._hfun_list`` entries with new
           ``HfunRaster`` objects built from the worker output files.

        Notes
        -----
        This method must **not** be called when any constraint is a
        ``TopoFuncConstraint`` because its internal lambda is not
        pickleable.  The dispatcher ``_apply_constraints()`` handles
        this check.

        Returns
        -------
        None
        """

        # Phase 1: PREPARATION
        tasks = []
        hfuns_to_process = {}

        for in_idx, hfun in enumerate(self._hfun_list):
            if not isinstance(hfun, HfunRaster):
                continue

            constraint_list = self._constraint_info_coll.get_constraints(
                hfun, in_idx, per_hfun=True
            )

            if constraint_list:
                hfuns_to_process[in_idx] = {
                    'hfun': hfun,
                    'constraints': constraint_list
                }


        # Same style as other functions
        # (Can be refactored to be a shared function that
        # accept needed parameters to make code cleaner)
        # in another PR
        for in_idx, data in hfuns_to_process.items():
            hfun = data['hfun']
            hfun_input_path = hfun.tmpfile
            topo_input_path = hfun._raster.path  # pylint: disable=W0212

            output_path = os.path.join(
                self._work_dir, f"constraints_result_{in_idx}.tif")

            task = {
                'original_index': in_idx,
                'hfun_input_path': hfun_input_path,
                'topo_input_path': topo_input_path,
                'output_path': output_path,
                'global_hmin': hfun._hmin,   # pylint: disable=W0212
                'global_hmax': hfun._hmax,   # pylint: disable=W0212
                'constraint_list': data['constraints']
            }
            tasks.append(task)

        if not tasks:
            _logger.info("No constraint tasks to execute.")
            return

        # Phase 2: EXECUTION
        _logger.info(
            f"Start parallel execution for {len(tasks)} constraint tasks")
        with Pool(processes=self._nprocs) as p:
            results = p.map(_constraints_task_worker, tasks)
        _logger.info("Parallel execution finished.")

        # Same style as other functions
        # (Can be refactored to be a shared function that
        # accept needed parameters to make code cleaner)
        # in another PR
        # Phase 3: INTEGRATION
        new_hfun_objects = {}
        for result in results:
            if result['status'] == 'error':
                _logger.error(
                    "Constraint worker %s failed: %s",
                    result['original_index'],
                    result['error'])
                continue

            idx = result['original_index']
            output_path = result['output_path']
            original_hfun = self._hfun_list[idx]

            new_hfun_objects[idx] = HfunRaster(
                raster=original_hfun.raster,
                hmin=original_hfun.hmin,
                hmax=original_hfun.hmax,
                verbosity=original_hfun.verbosity,
                initial_value=output_path
            )

        for idx, new_hfun in new_hfun_objects.items():
            _logger.info(
                f"Updating HfunCollector with constraint result at idx {idx}.")
            self._hfun_list[idx] = new_hfun


    def _apply_contours(self, apply_to: Optional[SizeFuncList] = None) -> None:
        """Internal: apply specified constraints.

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
        if apply_to is None:
            mesh_hfun_list = [
                i for i in self._hfun_list if isinstance(i, HfunMesh)]
            if self._base_mesh and self._base_as_hfun:
                mesh_hfun_list.insert(0, self._base_mesh)
            apply_to = [*mesh_hfun_list, *raster_hfun_list]

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

            # hfun objects cause issue with pickling
            # -> cannot be passed to pool
#            with Pool(processes=self._nprocs) as p:
#                p.starmap(
#                    _apply_contours_worker,
#                    [(hfun, self._contour_coll, self._nprocs)
#                     for hfun in apply_to])

    def _apply_channels(self, apply_to: Optional[SizeFuncList] = None) -> None:
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
        if apply_to is None:
            mesh_hfun_list = [
                i for i in self._hfun_list if isinstance(i, HfunMesh)]
            if self._base_mesh and self._base_as_hfun:
                mesh_hfun_list.insert(0, self._base_mesh)
            apply_to = [*mesh_hfun_list, *raster_hfun_list]

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

        if mode == 'mpi':
            if _get_mpi() is None:
                warnings.warn(
                    "mpi4py is not installed. Falling back to 'parallel' "
                    "mode. Install mpi4py for MPI support: "
                    "pip install ocsmesh[mpi]",
                    UserWarning
                )
                mode = 'parallel'
            elif not _is_mpi_active():
                warnings.warn(
                    "MPI mode requested but no MPI environment detected. "
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


    def _apply_flow_limiters(self) -> None:
        """
        Dispatches to either the serial or parallel implementation based on
        the current execution mode.
        """
        if self.execution_mode == 'parallel' and self._nprocs > 1:
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
        """
        Applies specified subtidal flow limiter refinements in parallel using a
        robust file-based worker pattern.

        This method coordinates the parallel processing by:
        1.  Preparing simple, pickleable 'task' dictionaries for each HfunRaster
            that needs modification.
        2.  Distributing these tasks to a pool of worker processes.
        3.  Integrating the results by creating new HfunRaster objects from the
            output files generated by the workers.
        """
        if self._method == 'fast':
            raise NotImplementedError(
            "This function is part of the 'exact' method and doesnt support 'fast' mode."
            )

        # Phase 1: PREPARATION (Coordinator)
        # This phase gathers all the work that needs to be done and packages it
        # into a list of simple, pickleable tasks for the worker processes.

        tasks = []
        hfuns_to_process = {}

        # First, group all applicable refinement rules by the HfunRaster they apply to.
        # This is more efficient than creating a separate task for every single rule.

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        for in_idx, hfun in enumerate(raster_hfun_list):
            limiter_rules_for_this_hfun = []
            for src_idx, hmin, hmax, zmax, zmin in self._flow_lim_coll:
                # Check if the rule applies to this specific hfun instance
                if src_idx is None or in_idx in src_idx:
                    limiter_rules_for_this_hfun.append({
                    'hmin': hmin if hmin is not None else self._size_info.get('hmin'),
                    'hmax': hmax if hmax is not None else self._size_info.get('hmax'),
                    'zmin': zmin,
                    'zmax': zmax
                    })

            # If any rules were found, mark this HfunRaster for processing.
            if limiter_rules_for_this_hfun:
                hfuns_to_process[in_idx] = {
                    'hfun': hfun,
                    'rules': limiter_rules_for_this_hfun
                }

        # Now, create the simple task dictionaries that can be sent to the pool.
        for in_idx, data in hfuns_to_process.items():
            hfun = data['hfun']

            # Determine the correct input file. If this raster was already processed
            # by another step (e.g., _apply_constraints), use that output file.
            # Otherwise, use the original hfun path.
            hfun_input_path = hfun.tmpfile


            # The path to the original, unmodified topography/DEM data.
            topo_input_path = hfun._raster.path # pylint: disable=W0212

            # Define a unique output path in our persistent working directory.
            output_path = os.path.join(self._work_dir,
                                       f"flow_limiter_result_{in_idx}.tif")

            task = {
                'original_index': in_idx,
                'hfun_input_path': hfun_input_path,
                'topo_input_path': topo_input_path,
                'output_path': output_path,
                'global_hmin': hfun._hmin, # pylint: disable=W0212
                'global_hmax': hfun._hmax, # pylint: disable=W0212
                'limiter_params': data['rules']
            }
            tasks.append(task)


        # If no tasks were generated, there's nothing to do.
        if not tasks:
            _logger.info("No flow limiter tasks to execute.")
            return

        # Phase 2: EXECUTION (Distribute to Laborers)
        # This phase sends the prepared tasks to a pool of worker processes
        # and waits for them to complete the heavy computational work.

        _logger.info(f"Start parallel execution for {len(tasks)} flow limiter tasks")
        with Pool(processes=self._nprocs) as p:
            results = p.map(_flow_limiter_task_worker, tasks)
        _logger.info("Parallel execution finished.")

        # Phase 3: INTEGRATION (Process Results)
        # This phase takes the results from the workers (which are just file paths)
        # and updates the main HfunCollector's state with the new, processed data.

        new_hfun_objects = {}
        for result in results:
            if result['status'] == 'error':
                _logger.error(
                    f"Worker failed for HfunRaster at index "
                    f"{result['original_index']}: {result['error']}"
                )
                continue

            idx = result['original_index']
            output_path = result['output_path']
            original_hfun = self._hfun_list[idx]
            # Create a new, updated HfunRaster instance to replace the old one.
            new_hfun_objects[idx] = HfunRaster(
                raster=original_hfun.raster,
                hmin=original_hfun.hmin,
                hmax=original_hfun.hmax,
                verbosity=original_hfun.verbosity,
                initial_value=output_path
            )

        # Finally, update the main list with the new objects.
        for idx, new_hfun in new_hfun_objects.items():
            _logger.info(f"Updating HfunCol. state with processed raster for idx {idx}.")
            self._hfun_list[idx] = new_hfun


    def _apply_const_val(self) -> None:
        """
        Dispatches to either the serial or parallel implementation based on
        the current execution mode.
        """
        if self.execution_mode == 'parallel' and self._nprocs > 1:
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

       # Phase 1: PREPARATION (Coordinator)
        tasks = []
        hfuns_to_process = {}

        # Group all applicable constant value rules by the HfunRaster they apply to.

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        for in_idx, hfun in enumerate(raster_hfun_list):
            rules_for_this_hfun = []
            for (src_idx, ctr0, ctr1), const_val in self._const_val_contour_coll:
                if src_idx is None or in_idx in src_idx:
                    rules_for_this_hfun.append({
                        'value': const_val,
                        'lower_bound': ctr0.level if ctr0 else None,
                        'upper_bound': ctr1.level if ctr1 else None
                    })

            if rules_for_this_hfun:
                hfuns_to_process[in_idx] = { 'hfun': hfun,
                                            'rules': rules_for_this_hfun }

        # Now, create the simple task dictionaries for the pool.
        for in_idx, data in hfuns_to_process.items():
            hfun = data['hfun']
            # Determine the correct input file.If this raster was already processed
            hfun_input_path = hfun.tmpfile
            topo_input_path = hfun._raster.path # pylint: disable=W0212

            output_path = os.path.join(self._work_dir,
                                       f"const_val_result_{in_idx}.tif")

            task = {
                'original_index': in_idx,
                'hfun_input_path': hfun_input_path,
                'topo_input_path': topo_input_path,
                'output_path': output_path,
                'global_hmin': hfun._hmin, # pylint: disable=W0212
                'global_hmax': hfun._hmax, # pylint: disable=W0212
                'const_val_rules': data['rules']
            }
            tasks.append(task)

        if not tasks:
            _logger.info("No constant value tasks to execute.")
            return


            # Phase 2: EXECUTION
        _logger.info(f"Start parallel execution for {len(tasks)} const. value tasks")
        with Pool(processes=self._nprocs) as p:
            results = p.map(_const_val_task_worker, tasks)
        _logger.info("Parallel execution finished.")

        # Phase 3: INTEGRATION
        new_hfun_objects = {}
        for result in results:
            if result['status'] == 'error':
                _logger.error("Val worker %s failed: %s",
                              result['original_index'],
                              result['error'])
                continue

            idx = result['original_index']
            output_path = result['output_path']
            original_hfun = self._hfun_list[idx]

            # Create the new HfunRaster, ensuring we don't wipe the worker's data.
            new_hfun_objects[idx] = HfunRaster(
            raster=original_hfun.raster, # Pass the original topography Raster object
            hmin=original_hfun.hmin,
            hmax=original_hfun.hmax,
            verbosity=original_hfun.verbosity,
            initial_value=output_path # Pass the path to the file created by the worker
            )

        # Finally, update the main list with the newly created objects.
        for idx, new_hfun in new_hfun_objects.items():
            _logger.info(f"Update HfunCollector with const_val raster at idx {idx}.")
            self._hfun_list[idx] = new_hfun


    def _apply_patch(self, apply_to: Optional[SizeFuncList] = None) -> None:
        """Internal: apply the specified patch refinements.

        Parameters
        ----------
        apply_to : SizeFuncList or None, default=None
            Size functions on which patches must be applied. If `None`
            all inputs are used to apply the patches.

        Returns
        -------
        None
        """

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        if apply_to is None:
            mesh_hfun_list = [
                i for i in self._hfun_list if isinstance(i, HfunMesh)]
            if self._base_mesh and self._base_as_hfun:
                mesh_hfun_list.insert(0, self._base_mesh)
            apply_to = [*mesh_hfun_list, *raster_hfun_list]

        with Pool(processes=self._nprocs) as p:
            for hfun in apply_to:
                for patch_defn, size_info in self._refine_patch_info_coll:
                    shape, crs = patch_defn.get_multipolygon()
                    if hfun.crs != crs:
                        transformer = Transformer.from_crs(
                            crs, hfun.crs, always_xy=True)
                        shape = ops.transform(
                                transformer.transform, shape)

                    hfun.add_patch(
                            shape, pool=p, **size_info)


    def _apply_linefeatures(self, apply_to: Optional[SizeFuncList] = None) -> None:
        """Internal: apply the specified line feature refinements.

        Parameters
        ----------
        apply_to : SizeFuncList or None, default=None
            Size functions on which line features must be applied. If `None`
            all inputs are used to apply the line features.

        Returns
        -------
        None
        """

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        if apply_to is None:
            mesh_hfun_list = [
                i for i in self._hfun_list if isinstance(i, HfunMesh)]
            if self._base_mesh and self._base_as_hfun:
                mesh_hfun_list.insert(0, self._base_mesh)
            apply_to = [*mesh_hfun_list, *raster_hfun_list]

        # TODO: Parallelize
        with Pool(processes=self._nprocs) as p:
            for hfun in apply_to:
                for lineftr_defn, size_info in self._refine_line_info_coll:
                    shape, crs = lineftr_defn.get_multiline()
                    if hfun.crs != crs:
                        transformer = Transformer.from_crs(
                            crs, hfun.crs, always_xy=True)
                        shape = ops.transform(
                                transformer.transform, shape)

                    hfun.add_feature(
                        feature=shape,
                        pool=p,
                        **size_info
                    )


    def _calculate_and_write_hfun_to_disk(
            self,
            out_path: Union[str, Path],
            **kwargs
            ) -> List[Union[str, Path]]:
        """Internal: write individual size function output mesh to file

        Calculate the interpolated on-mesh size function from each
        individual input, clip overlaps based on priority, and
        write the results to disk for later combining.

        Dispatches to serial or parallel based on ``execution_mode``.

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
            each input.
        """
        if self.execution_mode == 'mpi':
            _logger.info("Calculate & Writing hfun to disk using MPI method.")
            return self._calculate_and_write_hfun_to_disk_mpi(out_path, **kwargs)
        elif self.execution_mode == 'parallel' and self._nprocs > 1:
            _logger.info("Calculate & Writing hfun to disk using PARALLEL method.")
            return self._calculate_and_write_hfun_to_disk_parallel(out_path, **kwargs)
        else:
            _logger.info("Calculate & Writing hfun to disk using SERIAL method.")
            return self._calculate_and_write_hfun_to_disk_serial(out_path, **kwargs)


    def _calculate_and_write_hfun_to_disk_serial(
            self,
            out_path: Union[str, Path],
            **kwargs
            ) -> List[Union[str, Path]]:
        """Serial path for writing hfun to disk.

        This is the original implementation extracted verbatim.

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
            each input.
        """

        out_dir = Path(out_path)
        path_list = []
        file_counter = 0
        pid = os.getpid()
        bbox_list = []

        hfun_list = self._hfun_list[::-1]
        if self._base_mesh and self._base_as_hfun:
            hfun_list = [*self._hfun_list[::-1], self._base_mesh]

        # Last user input item has the highest priority (its trias
        # are not dropped) so process in reverse order
        for hfun in hfun_list:
            # TODO: Calling meshdata() on HfunMesh more than once causes
            # issue right now due to change in crs of internal Mesh
            try:
                meshdata_hfun = deepcopy(hfun.meshdata(**kwargs))
            except TypeError:
            # If a specific hfun type (like HfunMesh) doesn't support the args,
            # ignore them
            # This protects against passing 'stride' to a Mesh-based hfun which
            # might not need it
                meshdata_hfun = deepcopy(hfun.meshdata())

            # If no CRS info, we assume EPSG:4326
            if hasattr(meshdata_hfun, "crs"):
                dst_crs = CRS.from_user_input("EPSG:4326")
                if meshdata_hfun.crs != dst_crs:
                    utils.reproject(meshdata_hfun, dst_crs)

            # Get all previous bbox and clip to resolve overlaps
            # removing all tria that have NODE in bbox because it's
            # faster and so we can resolve all overlaps
            _logger.info("Removing bounds from hfun mesh...")
            for ibox in bbox_list:
                meshdata_hfun = utils.clip_mesh_by_shape(
                    meshdata_hfun,
                    ibox,
                    use_box_only=True,
                    fit_inside=True,
                    inverse=True)

            if len(meshdata_hfun.coords) == 0:
                _logger.debug("Hfun ignored due to overlap")
                continue

            # Check meshdata_hfun.value against hmin & hmax
            hmin = self._size_info['hmin']
            hmax = self._size_info['hmax']
            if hmin:
                meshdata_hfun.values[meshdata_hfun.values < hmin] = hmin
            if hmax:
                meshdata_hfun.values[meshdata_hfun.values > hmax] = hmax

            mesh = Mesh(meshdata_hfun)
            bbox_list.append(mesh.get_bbox(crs="EPSG:4326"))
            file_counter = file_counter + 1
            _logger.info(f'write mesh {file_counter} to file...')
            file_path = out_dir / f'hfun_{pid}_{file_counter}.2dm'
            mesh.write(file_path, format='2dm')
            path_list.append(file_path)
            _logger.info('Done writing 2dm file.')
            del mesh
            gc.collect()
        return path_list


    def _calculate_and_write_hfun_to_disk_parallel(
            self,
            out_path: Union[str, Path],
            **kwargs
            ) -> List[Union[str, Path]]:
        """Two-stage parallel path for writing hfun to disk.

        Stage 1 (Parallel): Workers call ``hfun.meshdata()``
        independently for each hfun entry and serialize results
        as ``.npz`` files.  All workers run simultaneously — this is
        where the expensive triangulation/interpolation cost lives.
        ``HfunRaster`` entries are reconstructed from file paths in
        the worker; ``HfunMesh`` entries are pickled and sent directly.

        Stage 2 (Sequential): The coordinator loads results in
        priority order, clips each against accumulated bounding
        boxes, clamps hmin/hmax, and writes the final ``.2dm``
        files.  This stage is fast (array operations only).

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
            each input.
        """

        out_dir = Path(out_path)
        path_list = []
        file_counter = 0
        pid = os.getpid()
        bbox_list = []

        hfun_list = self._hfun_list[::-1]
        if self._base_mesh and self._base_as_hfun:
            hfun_list = [*self._hfun_list[::-1], self._base_mesh]

        # ========== STAGE 1: PARALLEL meshdata() ==========
        tasks = []
        for loop_idx, hfun in enumerate(hfun_list):
            # NOTE: np.savez appends .npz automatically; output_path here does not
            # include the extension. The returned path in the result dict adds it.
            npz_path = os.path.join(
                self._work_dir,
                f"meshdata_stage1_{pid}_{loop_idx}"
            )
            if isinstance(hfun, HfunRaster):
                task = {
                    'type': 'raster',
                    'original_index': loop_idx,
                    'topo_path': hfun._raster.path,
                    'hfun_input_path': hfun.tmpfile,
                    'output_path': npz_path,
                    'hmin': hfun._hmin,
                    'hmax': hfun._hmax,
                    'meshdata_kwargs': kwargs
                }
            else:
                # HfunMesh and other types are picklable —
                # send the object directly to the worker
                task = {
                    'type': 'mesh',
                    'original_index': loop_idx,
                    'hfun_obj': deepcopy(hfun),
                    'output_path': npz_path,
                    'meshdata_kwargs': kwargs
                }
            tasks.append(task)

        # Run parallel meshdata for ALL entries
        stage1_results = {}
        if tasks:
            _logger.info(
                f"Stage 1: Launching {len(tasks)} parallel "
                f"meshdata() calls with {self._nprocs} workers"
            )
            with Pool(processes=self._nprocs) as p:
                results = p.map(_meshdata_task_worker, tasks)
            _logger.info("Stage 1: All meshdata() calls complete.")

            for result in results:
                if result['status'] == 'error':
                    _logger.error(
                        f"meshdata worker failed for loop index "
                        f"{result['original_index']}: {result['error']}"
                    )
                    continue
                stage1_results[result['original_index']] = \
                    result['output_path']

        # ========== STAGE 2: SEQUENTIAL overlap clip + write ==========
        _logger.info("Stage 2: Sequential overlap clipping and .2dm write")
        for loop_idx in range(len(hfun_list)):
            # Load the meshdata from Stage 1 result
            if loop_idx in stage1_results:
                npz_path = stage1_results[loop_idx]
                data = np.load(npz_path, allow_pickle=False)
                coords = data['coords'].copy()
                tria_raw = data['tria']
                tria = tria_raw.copy() if tria_raw.size > 0 else None
                quad_raw = data['quad']
                quad = quad_raw.copy() if quad_raw.size > 0 else None
                values = data['values'].copy()
                crs_str = str(data['crs'])
                crs = (CRS.from_user_input(crs_str)
                       if crs_str else None)

                # Necessary for windows : Close the NpzFile handle before deleting
                data.close()
                del data
                meshdata_hfun = MeshData(
                    coords=coords, tria=tria,
                    quad=quad, values=values, crs=crs
                )
                # Clean up intermediate .npz — data is in memory now
                try:
                    os.remove(npz_path)
                except OSError:
                    _logger.debug(
                        f"Could not remove temp file {npz_path}"
                    )
            else:
                # Worker failed for this index — skip
                continue

            # Clip against all previously-accumulated bounding boxes
            _logger.info("Removing bounds from hfun mesh...")
            for ibox in bbox_list:
                meshdata_hfun = utils.clip_mesh_by_shape(
                    meshdata_hfun,
                    ibox,
                    use_box_only=True,
                    fit_inside=True,
                    inverse=True)

            if len(meshdata_hfun.coords) == 0:
                _logger.debug("Hfun ignored due to overlap")
                continue

            # Check meshdata_hfun.value against hmin & hmax
            hmin = self._size_info['hmin']
            hmax = self._size_info['hmax']
            if hmin:
                meshdata_hfun.values[
                    meshdata_hfun.values < hmin] = hmin
            if hmax:
                meshdata_hfun.values[
                    meshdata_hfun.values > hmax] = hmax

            mesh = Mesh(meshdata_hfun)
            bbox_list.append(mesh.get_bbox(crs="EPSG:4326"))
            file_counter = file_counter + 1
            _logger.info(f'write mesh {file_counter} to file...')
            file_path = out_dir / f'hfun_{pid}_{file_counter}.2dm'
            mesh.write(file_path, format='2dm')
            path_list.append(file_path)
            _logger.info('Done writing 2dm file.')
            del mesh
            gc.collect()

        return path_list

     # TODO: refactor this with the parallel method instead of duplicated code.
    def _calculate_and_write_hfun_to_disk_mpi(
            self,
            out_path: Union[str, Path],
            **kwargs
            ) -> List[Union[str, Path]]:
        """MPI path for writing hfun to disk (dynamic send/recv).

        Same two-stage design as the parallel variant, but Stage 1 uses
        MPITaskRunner.dispatch() to dynamically stream tasks to idle
        workers via point-to-point send/recv. The existing
        _meshdata_task_worker is reused unchanged.

        Stage 1 (MPI, dynamic): Rank 0 streams meshdata tasks to
           workers on demand. Each worker reads its input from the
           shared filesystem and writes a .npz result back.

        Stage 2 (Sequential, Rank 0 only): Load .npz results in
           priority order, clip overlaps, clamp hmin/hmax, write .2dm.

        Error policy: if ANY task fails, a single aggregated
        RuntimeError is raised and Stage 2 is skipped; we never
        silently emit a partial mesh built from only the surviving
        tiles. Worker shutdown is handled by MPITaskRunner.run()'s
        finally block, not here.

        Shared-filesystem requirement: workers exchange paths to .npz
        files under self._work_dir. On a multi-node job that directory
        MUST live on a filesystem visible to every node (e.g.
        Lustre/GPFS/NFS).

        Parameters
        ----------
        out_path : path-like
            Directory for final .2dm output files.
        **kwargs : dict
            Arguments for hfun.meshdata() (e.g. stride).

        Returns
        -------
        list of path-like
            List of .2dm file paths.
        """

        out_dir = Path(out_path)
        path_list = []
        file_counter = 0
        pid = os.getpid()
        bbox_list = []

        hfun_list = self._hfun_list[::-1]
        if self._base_mesh and self._base_as_hfun:
            hfun_list = [*self._hfun_list[::-1], self._base_mesh]

        # ========== STAGE 1: MPI dynamic meshdata() ==========
        # Build tasks — same as parallel variant, plus 'op' key so
        # the worker loop knows which function to call.
        tasks = []
        for loop_idx, hfun in enumerate(hfun_list):
            # NOTE: np.savez appends .npz automatically; output_path here does not
            # include the extension. The returned path in the result dict adds it.
            npz_path = os.path.join(
                self._work_dir,
                f"meshdata_stage1_{pid}_{loop_idx}"
            )
            if isinstance(hfun, HfunRaster):
                task = {
                    'op': 'meshdata',
                    'type': 'raster',
                    'original_index': loop_idx,
                    'topo_path': hfun._raster.path,
                    'hfun_input_path': hfun.tmpfile,
                    'output_path': npz_path,
                    'hmin': hfun._hmin,
                    'hmax': hfun._hmax,
                    'meshdata_kwargs': kwargs
                }
            else:
                # HfunMesh and other types are picklable —
                # send the object directly to the worker
                task = {
                    'op': 'meshdata',
                    'type': 'mesh',
                    'original_index': loop_idx,
                    'hfun_obj': deepcopy(hfun),
                    'output_path': npz_path,
                    'meshdata_kwargs': kwargs
                }
            tasks.append(task)

        # ── Dynamic point-to-point dispatch (Rank 0 only) ──
        # Workers are in MPITaskRunner._run_worker() — they receive
        # tasks individually via recv() and return results via send().
        # Shutdown is handled by runner.run()'s finally block.

        #TODO: All the MPI code to be refactored and reused between all MPI method.
        runner = MPITaskRunner()

        # ── Pre-flight validation: verify shared filesystem across ranks ──
        runner.verify_shared_filesystem(self._work_dir)

        _logger.info(
            f"Stage 1 (MPI/dynamic): streaming {len(tasks)} meshdata() "
            f"tasks to {max(runner.size - 1, 1)} worker rank(s)"
        )

        results = runner.dispatch(tasks)
        _logger.info("Stage 1 (MPI/dynamic): all meshdata() tasks returned.")

        # ── Aggregate results & enforce fail-fast error policy ──
        stage1_results = {}
        failures = []
        for result in results:
            if not isinstance(result, dict) or result.get('status') == 'error':
                failures.append(result)
                idx = (result.get('original_index', -1)
                       if isinstance(result, dict) else -1)
                err = (result.get('error') if isinstance(result, dict)
                       else repr(result))
                wrk = (result.get('worker_rank', '?')
                       if isinstance(result, dict) else '?')
                _logger.error(
                    f"meshdata task failed for loop index {idx} "
                    f"(worker rank {wrk}): {err}"
                )
                continue
            stage1_results[result['original_index']] = result['output_path']

        if failures:
            # Do NOT proceed to Stage 2 with a partial result set —
            # that would produce a mesh silently missing tiles.
            summary = "; ".join(
                f"idx={f.get('original_index', -1)} "
                f"rank={f.get('worker_rank', '?')} "
                f"err={f.get('error', 'unknown')}"
                for f in failures if isinstance(f, dict)
            )
            raise RuntimeError(
                f"{len(failures)} of {len(tasks)} MPI meshdata task(s) "
                f"failed; aborting without writing output. "
                f"Details: {summary}"
            )

        # ========== STAGE 2: SEQUENTIAL overlap clip + write ==========
        # Identical to the parallel variant.
        _logger.info("Stage 2: Sequential overlap clipping and .2dm write")
        for loop_idx in range(len(hfun_list)):
            if loop_idx in stage1_results:
                npz_path = stage1_results[loop_idx]
                data = np.load(npz_path, allow_pickle=False)
                coords = data['coords'].copy()
                tria_raw = data['tria']
                tria = tria_raw.copy() if tria_raw.size > 0 else None
                quad_raw = data['quad']
                quad = quad_raw.copy() if quad_raw.size > 0 else None
                values = data['values'].copy()
                crs_str = str(data['crs'])
                crs = (CRS.from_user_input(crs_str)
                       if crs_str else None)

                # Close NpzFile handle before deleting
                data.close()
                del data
                meshdata_hfun = MeshData(
                    coords=coords, tria=tria,
                    quad=quad, values=values, crs=crs
                )
                # Clean up intermediate .npz — data is in memory now
                try:
                    os.remove(npz_path)
                except OSError:
                    _logger.debug(
                        f"Could not remove temp file {npz_path}"
                    )
            else:
                # Worker failed for this index — skip
                continue

            # Clip against all previously-accumulated bounding boxes
            _logger.info("Removing bounds from hfun mesh...")
            for ibox in bbox_list:
                meshdata_hfun = utils.clip_mesh_by_shape(
                    meshdata_hfun,
                    ibox,
                    use_box_only=True,
                    fit_inside=True,
                    inverse=True)

            if len(meshdata_hfun.coords) == 0:
                _logger.debug("Hfun ignored due to overlap")
                continue

            # Check meshdata_hfun.value against hmin & hmax
            hmin = self._size_info['hmin']
            hmax = self._size_info['hmax']
            if hmin:
                meshdata_hfun.values[
                    meshdata_hfun.values < hmin] = hmin
            if hmax:
                meshdata_hfun.values[
                    meshdata_hfun.values > hmax] = hmax

            mesh = Mesh(meshdata_hfun)
            bbox_list.append(mesh.get_bbox(crs="EPSG:4326"))
            file_counter = file_counter + 1
            _logger.info(f'write mesh {file_counter} to file...')
            file_path = out_dir / f'hfun_{pid}_{file_counter}.2dm'
            mesh.write(file_path, format='2dm')
            path_list.append(file_path)
            _logger.info('Done writing 2dm file.')
            del mesh
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
