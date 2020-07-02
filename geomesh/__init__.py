from importlib import util
import os
import pathlib
import tempfile
from geomesh import logger
from geomesh.geom import Geom
from geomesh.raster import Raster
from geomesh.driver import JigsawDriver
from geomesh.collections.rasters import RasterCollection

if util.find_spec("colored_traceback") is not None:
    import colored_traceback
    colored_traceback.add_hook(always=True)

tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/geomesh'))+'/'
os.makedirs(tmpdir, exist_ok=True)

__all__ = [
    "Geom",
    "Raster",
    "RasterCollection",
    "JigsawDriver",
    "logger"
]

# mpl.rcParams['agg.path.chunksize'] = 10000
