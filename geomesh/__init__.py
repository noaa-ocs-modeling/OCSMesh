from importlib import util
import os
import tempfile
import pathlib


if util.find_spec("colored_traceback") is not None:
    import colored_traceback
    colored_traceback.add_hook(always=True)

tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/geomesh'))+'/'
os.makedirs(tmpdir, exist_ok=True)

from geomesh import geom
from geomesh import driver


__all__ = [
    "geom",
    "driver"
    ]


# import matplotlib as mpl
# from geomesh.mesh import Mesh
# from geomesh.pslg import PlanarStraightLineGraph
# from geomesh.size_function import SizeFunction
# from geomesh.driver import Jigsaw
# from geomesh.raster import Raster
# from geomesh.raster_collection import RasterCollection
# from geomesh.refiner import MeshRefiner
# __all__ = ["Mesh",
#            "PlanarStraightLineGraph",
#            "Jigsaw",
#            "SizeFunction",
#            "Raster",
#            "RasterCollection",
#            "MeshRefiner"]
# mpl.rcParams['agg.path.chunksize'] = 10000
