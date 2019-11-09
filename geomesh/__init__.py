import matplotlib as mpl
import os
import tempfile
import pathlib
from geomesh.mesh import TriangularMesh, Mesh
from geomesh.pslg import PlanarStraightLineGraph
from geomesh.size_function import SizeFunction
from geomesh.driver import Jigsaw
from geomesh.raster import Raster
from geomesh.raster_collection import RasterCollection
__all__ = ["Mesh",  # just an alias for TriangularMesh
           "TriangularMesh",
           "PlanarStraightLineGraph",
           "Jigsaw",
           "SizeFunction",
           "Raster",
           "RasterCollection"]
mpl.rcParams['agg.path.chunksize'] = 10000
tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/geomesh'))+'/'
os.makedirs(tmpdir, exist_ok=True)
