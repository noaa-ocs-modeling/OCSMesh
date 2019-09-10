import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
from scipy.spatial import cKDTree
from osgeo import ogr
from jigsawpy import jigsaw_msh_t
from geomesh.pslg import PlanarStraightLineGraph as _PSLG
from geomesh.dataset_collection import DatasetCollection
from geomesh.gdal_tools import GdalTools


class SizeFunction(GdalTools):

    def __init__(
        self,
        PlanarStraightLineGraph,
        expansion_rate=0.2,
        SpatialReference=3395
    ):
        super(SizeFunction, self).__init__()
        self._PlanarStraightLineGraph = PlanarStraightLineGraph
        self._dfdx = expansion_rate
        self._SpatialReference = SpatialReference

    def make_plot(self, axes=None, show=False):
        if axes is None:
            axes = plt.figure().add_subplot(111)
        plt.tricontourf(self.mpl_tri, self.values)
        if show:
            plt.show()

        # for _LineString in self._PlanarStraightLineGraph.Polygon:
        #     xyz = np.asarray(_LineString.GetPoints())
        #     plt.plot(xyz[:, 0], xyz[:, 1])

        # for i, dataset in enumerate(self._DatasetCollection):
        #     dataset.SpatialReference = self.SpatialReference
        #     plt.contour(dataset.x, dataset.y, dataset.values, levels=[0.])
        #     dataset.reset()
        # axes.axis('scaled')
        # plt.show()

    def __compute_hfun(self):
        _MutiPolygon = self._PlanarStraightLineGraph.MultiPolygon
        points = list()
        for _Polygon in _MutiPolygon:
            for _LinearRing in _Polygon:
                points = [*points, *_LinearRing.GetPoints()]
        print(np.asarray(points))
        BREAKME

    @property
    def shoreline(self):
        raise NotImplementedError

    @property
    def x(self):
        try:
            return self.__x
        except AttributeError:
            self.__compute_hfun()
            return self.__x

    @property
    def y(self):
        try:
            return self.__y
        except AttributeError:
            self.__compute_hfun()
            return self.__y

    @property
    def values(self):
        try:
            return self.__values
        except AttributeError:
            self.__compute_hfun()
            return self.__values

    @property
    def vertices(self):
        raise NotImplementedError
        try:
            return self.__vertices
        except AttributeError:
            self.__compute_hfun()
            return self.__vertices

    @property
    def expansion_rate(self):
        return self.__dfdx

    @property
    def hfun(self):
        hfun = jigsaw_msh_t()
        for dataset in self._DatasetCollection._hfun:
            dataset.SpatialReference = self.SpatialReference
        return hfun

    @property
    def mpl_tri(self):
        try:
            return self.__mpl_tri
        except AttributeError:
            pass
        mpl_tri = Triangulation(self.x, self.y)
        self.__mpl_tri = mpl_tri
        return self.__mpl_tri

    @property
    def SpatialReference(self):
        return self.__SpatialReference

    @property
    def _PlanarStraightLineGraph(self):
        return self.__PlanarStraightLineGraph

    @property
    def _SpatialReference(self):
        return self.__SpatialReference

    @property
    def _mshID(self):
        return "euclidean-mesh"

    @property
    def _hfun_scal(self):
        return 'absolute'

    @property
    def _dfdx(self):
        return self.__dfdx

    @_SpatialReference.setter
    def _SpatialReference(self, SpatialReference):
        self.__SpatialReference = self.sanitize_SpatialReference(
            SpatialReference)
        self._PlanarStraightLineGraph.SpatialReference = SpatialReference

    @_PlanarStraightLineGraph.setter
    def _PlanarStraightLineGraph(self, PlanarStraightLineGraph):
        assert isinstance(PlanarStraightLineGraph, _PSLG)
        self.__PlanarStraightLineGraph = PlanarStraightLineGraph

    @_dfdx.setter
    def _dfdx(self, dfdx):
        self.__dfdx = float(dfdx)
