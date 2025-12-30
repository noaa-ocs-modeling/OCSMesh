import pathlib
from importlib import util
import tempfile
import os
import sys
import platform
import warnings

# Major API Change Warning
warnings.warn(
"\n\n"
"********************************************************************************\n"
"                  MAJOR UPDATE ALERT - Upcoming OCSMesh 2.0.0                   \n"
"********************************************************************************\n"
"OCSMesh is going through significant architectural changes.\n"
"The current architecture will be depracated soon, among the upcoming changes we highlight:\n"
"1. Mesh Engines (Jigsaw and Triangle) will become OPTIONAL dependencies.\n"
"   After version 2.0.0 you will have to install them manually.\n"
"2. A new default mesh engine will be adopted to replace Jigsaw.\n"
"3. 'jigsaw_msh_t' will be replaced by a new custom Python class called 'MeshData'.\n"
"4. Functions operating on msh_t objects will be updated to support MeshData.\n"
"This v1.7.0 is the last release version before OCSMesh 2.0.0.\n"
"********************************************************************************\n",
    FutureWarning,
    stacklevel=2
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
