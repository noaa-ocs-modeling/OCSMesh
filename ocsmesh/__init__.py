import os
import pathlib
import tempfile
from importlib import util

# Check if process was launched under an MPI environment (e.g., mpiexec, srun).
# If detected, pins numerical library thread pools (OpenMP/MKL/OpenBLAS) to 1 to
# prevent CPU thrashing and sets multiprocessing start method to 'spawn'.
# NOTE: Must execute before any submodule imports NumPy because BLAS/MKL thread
# pools are initialized at C-library import time.
# NOTE: If NOT running under MPI launcher, _configure_mpi_environment is a no-op.
from .mpi import _configure_mpi_environment
_configure_mpi_environment()

# pylint: disable=wrong-import-position
from .internal import MeshData
from .raster import Raster
from .mesh import Mesh
from .geom import Geom
from .hfun import Hfun
from .driver import MeshDriver

if util.find_spec("colored_traceback") is not None:
    import colored_traceback
    colored_traceback.add_hook(always=True)

tmpdir = str(pathlib.Path(tempfile.gettempdir() + '/ocsmesh')) + '/'
os.makedirs(tmpdir, exist_ok=True)

__all__ = [
    "Geom",
    "Hfun",
    "Raster",
    "Mesh",
    "MeshDriver",
    "MeshData",
]
