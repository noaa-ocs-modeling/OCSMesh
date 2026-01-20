import pathlib
from importlib import util
import tempfile
import os
import sys
import platform

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
