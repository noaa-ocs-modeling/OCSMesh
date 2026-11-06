import os
import pathlib
import tempfile
from importlib import util

# Under an MPI launcher with more than one task (mpiexec -n N, srun with
# N > 1): pin OpenMP/MKL/OpenBLAS threads to 1 unless already set, and set
# the multiprocessing start method to 'spawn'. Nothing changes otherwise.
# NOTE: Must execute before any submodule imports NumPy because BLAS/MKL thread
# pools are initialized at C-library import time.
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

tmpdir = str(pathlib.Path(os.environ.get('TMPDIR', tempfile.gettempdir()) + '/ocsmesh')) + '/'
os.makedirs(tmpdir, exist_ok=True)

__all__ = [
    "Geom",
    "Hfun",
    "Raster",
    "Mesh",
    "MeshDriver",
    "MeshData",
]
