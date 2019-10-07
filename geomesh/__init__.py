import matplotlib as mpl
from geomesh.mesh import TriMesh
from geomesh.pslg import PlanarStraightLineGraph
from geomesh.size_function import SizeFunction
from geomesh.driver import Jigsaw
from geomesh.raster import Raster
from geomesh.raster_collection import RasterCollection
__all__ = ["TriMesh",
           "PlanarStraightLineGraph",
           "Jigsaw",
           "SizeFunction",
           "Raster",
           "RasterCollection"]
mpl.rcParams['agg.path.chunksize'] = 10000
