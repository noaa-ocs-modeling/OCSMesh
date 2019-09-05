import numpy as np
import matplotlib.pyplot as plt
from osgeo import ogr, osr
from jigsawpy import jigsaw_msh_t
from geomesh.gdal_dataset_collection import GdalDatasetCollection
from geomesh.gdal_tools import GdalTools


class PlanarStraightLineGraph(GdalTools):

    def __init__(self, zmin, zmax, SpatialReference=3395):
        super(PlanarStraightLineGraph, self).__init__()
        self._zmin = zmin
        self._zmax = zmax
        self._GdalDatasetCollection = GdalDatasetCollection()
        self._SpatialReference = SpatialReference

    def __iter__(self):
        for gdal_dataset in self._GdalDatasetCollection:
            yield gdal_dataset

    def make_plot(self, show=False):
        for _LinearRing in self.Polygon:
            array = np.asarray(_LinearRing.GetPoints())
            plt.plot(array[:, 0], array[:, 1])
        plt.gca().axis('scaled')
        plt.show()

    def add_dataset(self, path):
        self._GdalDatasetCollection.add_dataset(path, pslg=True, hfun=False)

    @property
    def SpatialReference(self):
        return self.__SpatialReference

    @property
    def Polygon(self):
        areas = [_Polygon.GetArea() for _Polygon in self.MultiPolygon]
        idx = int(np.where(areas == np.max(areas))[0][0])
        _MultiPolygon = self.MultiPolygon
        _Polygon = _MultiPolygon.GetGeometryRef(idx)
        return _Polygon.Clone()

    @property
    def MultiPolygon(self):
        _MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        _MultiPolygon.AssignSpatialReference(self.SpatialReference)
        for __MultiPolygon in list(self.MultiPolygons):
            __MultiPolygon.TransformTo(self.SpatialReference)
            for __Polygon in __MultiPolygon:
                _MultiPolygon.AddGeometry(__Polygon)
        return _MultiPolygon.Buffer(0)

    @property
    def MultiPolygons(self):
        for gdal_dataset in self._GdalDatasetCollection:
            gdal_dataset.zmin = self.zmin
            gdal_dataset.zmax = self.zmax
            gdal_dataset.SpatialReference = self.SpatialReference
            _MultiPolygon = gdal_dataset.MultiPolygon
            gdal_dataset.reset()
            yield _MultiPolygon

    @property
    def zmin(self):
        return self.__zmin

    @property
    def zmax(self):
        return self.__zmax

    @property
    def vert2(self):
        _Polygon = self.Polygon
        vert2 = list()
        for _LinearRing in _Polygon:
            vert2 = [*vert2, *_LinearRing.GetPoints()[:-1]]
        return vert2

    @property
    def edge2(self):
        _Polygon = self.Polygon
        edge2 = list()
        for _LinearRing in _Polygon:
            _edge2 = list()
            for i in range(_LinearRing.GetPointCount()-2):
                _edge2.append((i, i+1))
            _edge2.append((_edge2[-1][1], _edge2[0][0]))
            _edge2 = np.asarray(_edge2) + len(edge2)
            edge2 = [*edge2, *_edge2.tolist()]
        return edge2

    @SpatialReference.setter
    def SpatialReference(self, SpatialReference):
        self.MultiPolygon.TransformTo(SpatialReference)

    @property
    def _zmin(self):
        return self.__zmin

    @property
    def _zmax(self):
        return self.__zmax

    @property
    def _ndim(self):
        return 2

    @property
    def _mshID(self):
        return "euclidean-mesh"

    @property
    def _geom(self):
        geom = jigsaw_msh_t()
        geom.mshID = self._mshID
        vert2 = list()
        for i, (x, y) in enumerate(self.vert2):
            vert2.append(((x, y), 0))   # why 0?
        geom.vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)
        edge2 = list()
        for i, (e0, e1) in enumerate(self.edge2):
            edge2.append(((e0, e1), 0))   # why 0?
        geom.edge2 = np.asarray(edge2, dtype=jigsaw_msh_t.EDGE2_t)
        return geom

    @property
    def _SpatialReference(self):
        return self.MultiPolygon.GetSpatialReference()

    @property
    def _GdalDatasetCollection(self):
        return self.__GdalDatasetCollection

    @_SpatialReference.setter
    def _SpatialReference(self, SpatialReference):
        SpatialReference = self.sanitize_SpatialReference(SpatialReference)
        self.__SpatialReference = SpatialReference

    @_GdalDatasetCollection.setter
    def _GdalDatasetCollection(self, GdalDatasetCollection):
        self.__GdalDatasetCollection = GdalDatasetCollection

    @_zmin.setter
    def _zmin(self, zmin):
        self.__zmin = float(zmin)

    @_zmax.setter
    def _zmax(self, zmax):
        self.__zmax = float(zmax)

# from netCDF4 import Dataset
# pyenv_prefix = "/".join(sys.executable.split('/')[:-2])
# if os.getenv('SRTM15_PATH') is not None:
#     nc = pathlib.Path(os.getenv('SRTM15_PATH'))
# else:
#     nc = pathlib.Path(pyenv_prefix + '/lib/SRTM15+V2.nc')
