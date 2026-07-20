import pathlib
from importlib import util
import tempfile
import os
import sys
import platform

# When running under MPI, pin numerical
# library threads to 1 BEFORE numpy is imported. NumPy delegates
# matrix operations to compiled libraries (OpenBLAS, MKL) that
# spawn threads internally. Without pinning, each MPI rank spawns
# N threads on an N-core node → N ranks × N threads = massive
# oversubscription. Must happen here (before any ocsmesh submodule
# imports numpy) to take effect.
#
# Detection uses environment variables that MPI launchers set
# automatically — no mpi4py import needed at this stage.
# TODO : review if all of these are necessary.
_MPI_ENV_HINTS = (
    'OMPI_COMM_WORLD_SIZE',     # Open MPI
    'PMI_SIZE',                  # MPICH / Cray
    'MPI_LOCALNRANKS',           # Intel MPI
    'SLURM_NTASKS',              # SLURM (srun)
)
_MPI_THREAD_PIN_VARS = (
    'OMP_NUM_THREADS',           # OpenMP (generic)
    'MKL_NUM_THREADS',           # Intel MKL (conda numpy)
    'OPENBLAS_NUM_THREADS',      # OpenBLAS (pip numpy)
)

# TODO: Consider another approach like a context manager.
if any(var in os.environ for var in _MPI_ENV_HINTS):
    # Pin numerical library threads to 1 to prevent oversubscription
    for _var in _MPI_THREAD_PIN_VARS:
        os.environ.setdefault(_var, '1')

    # Force 'spawn' start method for multiprocessing. Under MPI,
    # 'fork' copies the MPI communicator state into Pool children,
    # causing deadlocks or silent corruption. Must be set here
    # before any import triggers multiprocessing initialization.
    import multiprocessing as _mp
    try:
        _mp.set_start_method('spawn', force=False)
    except RuntimeError:
        # Already set — check if it's safe
        if _mp.get_start_method() != 'spawn':
            import warnings
            warnings.warn(
                f"multiprocessing start method is "
                f"'{_mp.get_start_method()}', but MPI requires "
                f"'spawn' to avoid deadlocks. Call "
                f"multiprocessing.set_start_method('spawn') before "
                f"importing ocsmesh.",
                UserWarning
            )

from .internal import MeshData
from .raster import Raster
from .mesh import Mesh
from .geom import Geom
from .hfun import Hfun
from .driver import MeshDriver

if util.find_spec("colored_traceback") is not None:
    import colored_traceback
    colored_traceback.add_hook(always=True)

tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/ocsmesh'))+'/'
os.makedirs(tmpdir, exist_ok=True)

__all__ = [
    "Geom",
    "Hfun",
    "Raster",
    "Mesh",
    "MeshDriver",
    "MeshData",
]

# mpl.rcParams['agg.path.chunksize'] = 10000
