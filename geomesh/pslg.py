import numpy as np
import matplotlib.pyplot as plt
import tempfile
import fiona
from shapely.geometry import shape, mapping, MultiPolygon
from geomesh.raster_collection import RasterCollection


class PlanarStraightLineGraph:

    def __init__(self, raster_collection, zmin, zmax, dst_crs="EPSG:3395"):
        self._raster_collection = raster_collection
        self._zmin = zmin
        self._zmax = zmax
        self._dst_crs = dst_crs

    def __iter__(self):
        for raster in self.raster_collection:
            yield raster

    def make_plot(self, view='pslg', **kwargs):
        assert view in ['topobathy', 'pslg']
        if view == 'topobathy':
            self.plot_topobathy(**kwargs)
        elif view == 'pslg':
            self.plot_pslg(**kwargs)

    def plot_topobathy(self, show=False):
        raise NotImplementedError

    def plot_pslg(self, show=False):
        for feature in self.collection:
            multipolygon = shape(feature["geometry"])
            for polygon in multipolygon:
                xy = np.asarray(polygon.exterior.coords)
                plt.plot(xy[:, 0], xy[:, 1], color='k')
                for inner_ring in polygon.interiors:
                    xy = np.asarray(inner_ring.coords)
                    plt.plot(xy[:, 0], xy[:, 1], color='r')
        if show:
            plt.gca().axis('scaled')
            plt.show()

    @property
    def raster_collection(self):
        return self._raster_collection

    @property
    def crs(self):
        return self.collection.crs

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    @property
    def collection(self):
        return self._collection

    @property
    def shp(self):
        return self._shp

    @property
    def dst_crs(self):
        return self._dst_crs

    @property
    def schema(self):
        return {
            'geometry': 'Polygon',
            'properties': {
                'zmin': 'float',
                'zmax': 'float'}}

    @property
    def _raster_collection(self):
        return self.__raster_collection

    @property
    def _collection(self):
        try:
            return self.__collection
        except AttributeError:
            polygon_collection = list()
            for raster in self.raster_collection:
                raster.zmin = self.zmin
                raster.zmax = self.zmax
                for feature in raster.collection:
                    multipolygon = shape(feature["geometry"])
                    for polygon in multipolygon:
                        polygon_collection.append(polygon)
            polygon = MultiPolygon(polygon_collection).buffer(0)
            collection = fiona.open(
                self.shp.name,
                'w',
                driver='ESRI Shapefile',
                crs=self.dst_crs,
                schema=self.schema)
            collection.write({
                "geometry": mapping(polygon),
                "properties": {
                    "zmin": self.zmin,
                    "zmax": self.zmax}})
            collection.close()
            self.__collection = fiona.open(self.shp.name)
            return self.__collection

    @property
    def _dst_crs(self):
        return self.__dst_crs

    @property
    def _zmin(self):
        return self.__zmin

    @property
    def _zmax(self):
        return self.__zmax

    @property
    def _shp(self):
        try:
            return self.__shp
        except AttributeError:
            self.__shp = tempfile.TemporaryDirectory()
            return self.__shp

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        self._dst_crs = dst_crs

    @_raster_collection.setter
    def _raster_collection(self, raster_collection):
        assert isinstance(raster_collection, RasterCollection)
        self.__raster_collection = raster_collection

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        self.raster_collection.dst_crs = dst_crs
        del(self._collection)
        self.__dst_crs = dst_crs

    @_zmin.setter
    def _zmin(self, zmin):
        if zmin is None:
            del(self._zmin)
        else:
            zmin = float(zmin)
            try:
                if zmin != self.__zmin:
                    del(self._collection)
            except AttributeError:
                pass
            self.__zmin = float(zmin)

    @_zmax.setter
    def _zmax(self, zmax):
        if zmax is None:
            del(self._zmax)
        else:
            zmax = float(zmax)
            try:
                if zmax != self.__zmax:
                    del(self._collection)
            except AttributeError:
                pass
            self.__zmax = float(zmax)

    @_collection.deleter
    def _collection(self):
        try:
            del(self.__collection)
        except AttributeError:
            pass

    @_zmin.deleter
    def _zmin(self):
        try:
            del(self.__zmin)
            del(self._collection)
        except AttributeError:
            pass

    @_zmax.deleter
    def _zmax(self):
        try:
            del(self.__zmax)
            del(self._collection)
        except AttributeError:
            pass





    # @property
    # def MultiPolygon(self):
    #     return self._MultiPolygon

    # @property
    # def MultiPolygons(self):
    #     return self._MultiPolygons

    # @property
    # def zmin(self):
    #     return self._zmin

    # @property
    # def zmax(self):
    #     return self._zmax

    # @property
    # def x(self):
    #     return self.points[:, 0]

    # @property
    # def y(self):
    #     return self.points[:, 1]

    # @property
    # def xy(self):
    #     return self.points[:, :2]

    # @property
    # def elevation(self):
    #     return self.points[:, 2]

    # @property
    # def values(self):
    #     return self.points[:, 2]

    # @property
    # def points(self):
    #     return self._points

    # @property
    # def container(self):
    #     return tuple(self._container)

    # @property
    # def ocean_boundary(self):
    #     idx = np.where(self.values < 0)[0]
    #     return self.values[idx]

    # @property
    # def land_boundary(self):
    #     raise NotImplementedError

    # @property
    # def ocean_boundaries(self):
    #     return tuple(self._ocean_boundaries)

    # @property
    # def land_boundaries(self):
    #     return tuple(self._land_boundaries)

    # @property
    # def inner_boundaries(self):
    #     return tuple(self._inner_boundaries)

    # @property
    # def geometry_type(self):
    #     return self._geometry_type

    # @property
    # def ndim(self):
    #     return self._ndim

    # @property
    # def mpl_tri(self):
    #     return self._mpl_tri

    # @property
    # def mpl_tri_mask(self):
    #     return self._mpl_tri_mask

    # @property
    # def vert2(self):
    #     return self._vert2

    # @property
    # def edge2(self):
    #     return self._edge2

    # @property
    # def geom(self):
    #     geom = jigsaw_msh_t()
    #     geom.vert2 = self.vert2
    #     geom.edge2 = self.edge2
    #     geom.ndim = self.ndim
    #     geom.mshID = self.geometry_type
    #     return geom

    # @property
    # def SpatialReference(self):
    #     return self._SpatialReference

    # @property
    # def _MultiPolygon(self):
    #     try:
    #         return self.__MultiPolygon
    #     except AttributeError:
    #         MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    #         MultiPolygon.AssignSpatialReference(self.SpatialReference)
    #         for _MultiPolygon in list(self.MultiPolygons):
    #             for Polygon in _MultiPolygon:
    #                 MultiPolygon.AddGeometry(Polygon)
    #         MultiPolygon = MultiPolygon.Buffer(0)
    #         self._MultiPolygon = MultiPolygon
    #         return self.__MultiPolygon

    # @property
    # def _MultiPolygons(self):
    #     try:
    #         return self.__MultiPolygons
    #     except AttributeError:
    #         self._MultiPolygons = list()
    #         return self.__MultiPolygons

    # @property
    # def _zmin(self):
    #     return self.__zmin

    # @property
    # def _zmax(self):
    #     return self.__zmax

    # @property
    # def _mpl_tri(self):
    #     try:
    #         return self.__mpl_tri
    #     except AttributeError:
    #         self._mpl_tri = self.points
    #         return self.__mpl_tri

    # @property
    # def _points(self):
    #     try:
    #         return self.__points
    #     except AttributeError:
    #         self._points = list()
    #         return self.__points

    # @property
    # def _ndim(self):
    #     return 2

    # @property
    # def _vert2(self):
    #     try:
    #         return self.__vert2
    #     except AttributeError:
    #         self._vert2 = list()
    #         return self.__vert2

    # @property
    # def _edge2(self):
    #     try:
    #         return self.__edge2
    #     except AttributeError:
    #         self._edge2 = list()
    #         return self.__edge2

    # @property
    # def _geometry_type(self):
    #     try:
    #         return self.__geometry_type
    #     except AttributeError:
    #         self._geometry_type = 'euclidean-mesh'
    #         return self.__geometry_type

    # @property
    # def _ocean_boundaries(self):
    #     # mask = np.full(self.elevation.size, True)
    #     try:
    #         return self.__ocean_boundaries
    #     except AttributeError:
    #         self._ocean_boundaries = list()
    #         return self.__ocean_boundaries

    # @property
    # def _land_boundaries(self):
    #     try:
    #         return self.__land_boundaries
    #     except AttributeError:
    #         self._land_boundaries = list()
    #         return self.__land_boundaries

    # @property
    # def _inner_boundaries(self):
    #     try:
    #         return self.__inner_boundaries
    #     except AttributeError:
    #         self._inner_boundaries = list()
    #         return self.__inner_boundaries

    # @property
    # def _SpatialReference(self):
    #     return self.__SpatialReference

    # @property
    # def _container(self):
    #     try:
    #         return self.__container
    #     except AttributeError:
    #         self.__container = list()
    #         return self.__container
    #     return self.__container

    # @property
    # def _DatasetCollection(self):
    #     for gdal_dataset in self.__DatasetCollection:
    #         gdal_dataset.zmin = self.zmin
    #         gdal_dataset.zmax = self.zmax
    #     return self.__DatasetCollection

    # def make_plot(self, show=False):
    #     for _Polygon in self.MultiPolygon:
    #         for _LinearRing in _Polygon:
    #             _LinearRing = _LinearRing.Clone()
    #             array = np.asarray(_LinearRing.GetPoints())
    #             plt.plot(array[:, 0], array[:, 1])
    #     plt.gca().axis('scaled')
    #     if show:
    #         plt.show()
    #     return plt.gca()

    # @_DatasetCollection.setter
    # def _DatasetCollection(self, DatasetCollection):
    #     assert isinstance(DatasetCollection,  geomesh.DatasetCollection)
    #     self.__DatasetCollection = DatasetCollection

    # @_zmin.setter
    # def _zmin(self, zmin):
    #     del(self._MultiPolygons)
    #     self.__zmin = float(zmin)

    # @_zmax.setter
    # def _zmax(self, zmax):
    #     del(self._MultiPolygons)
    #     self.__zmax = float(zmax)

    # @_SpatialReference.setter
    # def _SpatialReference(self, SpatialReference):
    #     SpatialReference = gdal_tools.sanitize_SpatialReference(
    #             SpatialReference)
    #     self.__SpatialReference = SpatialReference

    # @_MultiPolygon.setter
    # def _MultiPolygon(self, MultiPolygon):
    #     del(self._MultiPolygon)
    #     self.__MultiPolygon = MultiPolygon

    # @_points.setter
    # def _points(self, points):
    #     for _Polygon in self.MultiPolygon:
    #         for _LinearRing in _Polygon:
    #             points = [*points, *_LinearRing.GetPoints()]
    #     self.__points = np.asarray(points)

    # @_mpl_tri.setter
    # def _mpl_tri(self, points):
    #     self.__mpl_tri = Triangulation(points[:, 0], points[:, 1])

    # @_ocean_boundaries.setter
    # def _ocean_boundaries(self, ocean_boundaries):
    #     dry_mask = np.full(self.elevation.shape, True)
    #     dry_mask[np.where(self.elevation < 0.)] = False
    #     idxs = np.where(~dry_mask)[0]
    #     _idxs = [idxs[0]]
    #     for i in range(1, len(idxs)):
    #         if idxs[i-1] == idxs[i]-1:
    #             _idxs.append(int(idxs[i]))
    #         else:
    #             ocean_boundaries.append(_idxs)
    #             _idxs = list()
    #     ocean_boundaries.append(_idxs)
    #     self.__ocean_boundaries = ocean_boundaries

    # @_land_boundaries.setter
    # def _land_boundaries(self, land_boundaries):
    #     wet_mask = np.full(self.elevation.shape, True)
    #     wet_mask[np.where(self.elevation > 0.)] = False
    #     idxs = np.where(~wet_mask)[0]
    #     _idxs = [idxs[0]]
    #     for i in range(1, len(idxs)):
    #         if idxs[i-1] == idxs[i]-1:
    #             _idxs.append(int(idxs[i]))
    #         else:
    #             land_boundaries.append(_idxs)
    #             _idxs = list()
    #     land_boundaries.append(_idxs)
    #     self.__land_boundaries = land_boundaries

    # @_inner_boundaries.setter
    # def _inner_boundaries(self, inner_boundaries):
    #     initial_idx = 0
    #     for _Polygon in self.MultiPolygon:
    #         _inner_boundaries = list()
    #         for i, _LinearRing in enumerate(_Polygon):
    #             if i == 0:
    #                 initial_idx += _LinearRing.GetPointCount()
    #             else:
    #                 final_idx = initial_idx + _LinearRing.GetPointCount()
    #                 _inner_boundaries.append(np.arange(initial_idx, final_idx))
    #                 initial_idx = final_idx
    #         inner_boundaries.append(_inner_boundaries)
    #     self.__inner_boundaries = inner_boundaries

    # @_MultiPolygons.setter
    # def _MultiPolygons(self, MultiPolygons):
    #     for ds in self:
    #         _MultiPolygon = ds.get_MultiPolygon(self.SpatialReference)
    #         del(ds._MultiPolygon)  # free up memory
    #         MultiPolygons.append(_MultiPolygon)
    #     self.__MultiPolygons = MultiPolygons
    #     return self.__MultiPolygons

    # @_vert2.setter
    # def _vert2(self, vert2):
    #     for _Polygon in self.MultiPolygon:
    #         for _LinearRing in _Polygon:
    #             _vert2 = list()
    #             for x, y in np.asarray(_LinearRing.GetPoints())[:-1, :2]:
    #                 _vert2.append(((x, y), 0))  # always 0?
    #             vert2 = [*vert2, *_vert2]
    #     self.__vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)

    # @_edge2.setter
    # def _edge2(self, edge2):
    #     for _Polygon in self.MultiPolygon:
    #         for _LinearRing in _Polygon:
    #             _edge2 = list()
    #             for i in range(_LinearRing.GetPointCount()-2):
    #                 _edge2.append((i, i+1))
    #             _edge2.append((_edge2[-1][1], _edge2[0][0]))
    #             _edge2 = np.asarray(_edge2) + len(edge2)
    #             edge2 = [*edge2, *_edge2.tolist()]
    #     edge2 = [((x, y), 0) for x, y in edge2]
    #     self.__edge2 = np.asarray(edge2, dtype=jigsaw_msh_t.EDGE2_t)

    # @_geometry_type.setter
    # def _geometry_type(self, geometry_type):
    #     assert geometry_type in ["euclidean-mesh"]
    #     self.__geometry_type = geometry_type

    # @_MultiPolygon.deleter
    # def _MultiPolygon(self):
    #     try:
    #         del(self.__MultiPolygon)
    #         del(self._points)
    #         del(self._vert2)
    #     except AttributeError:
    #         pass

    # @_vert2.deleter
    # def _vert2(self):
    #     try:
    #         del(self.__vert2)
    #         del(self._edge2)
    #     except AttributeError:
    #         pass

    # @_points.deleter
    # def _points(self):
    #     try:
    #         del(self.__points)
    #         del(self._mpl_tri)
    #     except AttributeError:
    #         pass

    # @_edge2.deleter
    # def _edge2(self):
    #     try:
    #         del(self.__edge2)
    #         del(self._vert2)
    #     except AttributeError:
    #         pass

    # @_mpl_tri.deleter
    # def _mpl_tri(self):
    #     try:
    #         del(self.__mpl_tri)
    #     except AttributeError:
    #         pass

    # @_MultiPolygons.deleter
    # def _MultiPolygons(self):
    #     try:
    #         del(self.__MultiPolygons)
    #     except AttributeError:
    #         pass
