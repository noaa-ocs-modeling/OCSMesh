import numpy as np
from osgeo import ogr, osr
import matplotlib.pyplot as plt
from matplotlib.path import Path
from geomesh import gdal_tools


class GdalDataset:

    def __init__(self, path):
        self.__path = path

    def downsample(self, xRes, yRes):
        dx, dy = self.get_resolution(self.Dataset)
        if xRes >= dx or yRes >= dy:
            raise Exception('Cannot upsample DEM.')
        self.__Dataset = self.Warp(self.__Dataset, xRes=xRes, yRes=yRes)

    def reset(self):
        self.reopen()
        self.zmin = None
        self.zmax = None

    def close(self):
        del self.__Dataset

    def reopen(self):
        self.__Dataset = self.__original

    def get_arrays(self, SpatialReference=None):
        return gdal_tools.get_arrays(self.Dataset, SpatialReference)

    def get_xyz(self, SpatialReference=None):
        return gdal_tools.get_arrays(self.Dataset, SpatialReference)

    def get_xy(self, SpatialReference=None):
        return gdal_tools.get_xy(self.Dataset, SpatialReference)

    def get_bbox(self, SpatialReference=None, Path=False):
        return gdal_tools.get_bbox(self.Dataset, SpatialReference, Path)

    def get_GeoTransform(self, SpatialReference=None):
        return gdal_tools.get_GeoTransform(self.Dataset, SpatialReference)

    def get_resolution(self, SpatialReference=None):
        return gdal_tools.get_resolution(self.Dataset, SpatialReference)

    @property
    def xyz(self):
        return self.get_xyz(self.SpatialReference)

    @property
    def x(self):
        return self.get_arrays(self.SpatialReference)[0]

    @property
    def y(self):
        return self.get_arrays(self.SpatialReference)[1]

    @property
    def values(self):
        values = self.get_arrays(self.SpatialReference)[2]
        return np.ma.masked_equal(values, 255)

    @property
    def zmin(self):
        try:
            return self.__zmin
        except AttributeError:
            raise AttributeError('Must set attribute zmin.')

    @property
    def zmax(self):
        try:
            return self.__zmax
        except AttributeError:
            raise AttributeError('Must set attribute zmax.')

    @property
    def Dataset(self):
        try:
            return self.__Dataset
        except AttributeError:
            # keep copy of original to avoid calling Open since it can take
            # a long time for internet residing DEMs, but it takes + memory
            try:
                self.__Dataset = self.__original
            except AttributeError:
                self.__Dataset = gdal_tools.Open(self.path)
                self.__original = self.__Dataset
            return self.__Dataset

    @property
    def MultiLineString(self):
        _QuadContourSet = plt.contour(
            self.x, self.y, self.values, levels=[self.zmin, self.zmax])
        plt.close(plt.gcf())
        for _PathCollection in _QuadContourSet.collections:
            _MultiLineString = ogr.Geometry(ogr.wkbMultiLineString)
            for _Path in _PathCollection.get_paths():
                _LineString = ogr.Geometry(ogr.wkbLineString)
                for x, y in _Path.vertices:
                    _LineString.AddPoint_2D(x, y)
                _LineString.CloseRings()
                _MultiLineString.AddGeometry(_LineString)
            yield _MultiLineString

    @property
    def MultiPolygon(self):
        _QuadContourSet = plt.contourf(
            self.x, self.y, self.values, levels=[self.zmin, self.zmax])
        plt.close(plt.gcf())
        paths = list()
        for _PathCollection in _QuadContourSet.collections:
            for _Path in _PathCollection.get_paths():
                polygons = _Path.to_polygons(closed_only=True)
                for polygon in polygons:
                    paths.append(Path(polygon, closed=True))
        # now we need to sort into separate polygons/holes.
        rings = list()
        for path in paths:
            _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
            _LinearRing.AssignSpatialReference(self.SpatialReference)
            for x, y in path.vertices:
                _LinearRing.AddPoint_2D(x, y)
            _LinearRing.CloseRings()
            rings.append(_LinearRing)
        # with the areas we can take the largest and find the inner ones
        areas = [_.GetArea() for _ in rings]
        multipolygon = list()
        while len(rings) > 0:
            _idx = np.where(np.max(areas) == areas)[0]
            path = paths[_idx[0]]
            _idxs = np.where([
                path.contains_point(_.vertices[0, :]) for _ in paths])[0]
            _idxs = np.hstack([_idx, _idxs])
            polygon = list()
            for _idx in np.flip(_idxs):
                polygon.insert(0, rings.pop(_idx))
                paths.pop(_idx)
                areas.pop(_idx)
            multipolygon.append(polygon)

        _MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        _MultiPolygon.AssignSpatialReference(self.SpatialReference)
        for _LinearRings in multipolygon:
            _Polygon = ogr.Geometry(ogr.wkbPolygon)
            _Polygon.AssignSpatialReference(self.SpatialReference)
            for _LinearRing in _LinearRings:
                _Polygon.AddGeometry(_LinearRing)
            _MultiPolygon.AddGeometry(_Polygon)
        return _MultiPolygon

    @property
    def path(self):
        return self.__path

    @property
    def bbox(self):
        return self.get_bbox()

    @property
    def SpatialReference(self):
        return gdal_tools.get_SpatialReference(self.Dataset)

    @zmin.setter
    def zmin(self, zmin):
        if zmin is None:
            del(self.zmin)
        else:
            self.__zmin = float(zmin)

    @zmax.setter
    def zmax(self, zmax):
        if zmax is None:
            del(self.zmax)
        else:
            self.__zmax = float(zmax)

    @SpatialReference.setter
    def SpatialReference(self, SpatialReference):
        SpatialReference = gdal_tools.sanitize_SpatialReference(
            SpatialReference)
        self.__Dataset = gdal_tools.Warp(self.Dataset, dstSRS=SpatialReference)

    @zmin.deleter
    def zmin(self):
        try:
            del(self.__zmin)
        except AttributeError:
            pass

    @zmax.deleter
    def zmax(self):
        try:
            del(self.__zmax)
        except AttributeError:
            pass

    @property
    def _pslg(self):
        try:
            return self.__pslg
        except AttributeError:
            return False

    @property
    def _hfun(self):
        try:
            return self.__hfun
        except AttributeError:
            return False

    @_pslg.setter
    def _pslg(self, pslg):
        assert isinstance(pslg, bool)
        self.__pslg = pslg

    @_hfun.setter
    def _hfun(self, hfun):
        assert isinstance(hfun, bool)
        self.__hfun = hfun

# return empty polygon if tile is fully internal/external
# if np.all(np.logical_and(z >= self.zmin, z <= self.zmax)) \
#         or (np.all(z >= self.zmax) or np.all(z <= self.zmin)):
#     return _Polygon
# compute polygon using matplotlib


    # def __get_polygon_bbox(self):
    #     bbox = self.get_bbox()
    #     _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
    #     _LinearRing.AssignSpatialReference(self.SpatialReference)
    #     _LinearRing.AddPoint_2D(bbox.xmin, bbox.ymin)
    #     _LinearRing.AddPoint_2D(bbox.xmax, bbox.ymin)
    #     _LinearRing.AddPoint_2D(bbox.xmax, bbox.ymax)
    #     _LinearRing.AddPoint_2D(bbox.xmin, bbox.ymax)
    #     _LinearRing.AddPoint_2D(bbox.xmin, bbox.ymin)
    #     _Polygon = ogr.Geometry(ogr.wkbPolygon)
    #     _Polygon.AssignSpatialReference(self.SpatialReference)
    #     _Polygon.AddGeometry(_LinearRing)
    #     _Polygon.CloseRings()
    #     return _Polygon

    # def check_overlap(self, other):
    #     # using 32-bit precision (~7 decimal digits).
    #     self_polygon = self.__get_polygon_bbox()
    #     other_polygon = other.__get_polygon_bbox()
    #     # print(self_polygon.Overlaps(other_polygon))
    #     intersection = self_polygon.Intersection(other_polygon)
    #     # print(self_polygon)
    #     # print()
    #     # mercator = osr.SpatialReference()
    #     # mercator.ImportFromEPSG(3395)
    #     # intersection.TransformTo(mercator)

    #     # plt.plot(self_polygon.GetGeometryRef(0).GetPoints())
    #     xy = np.asarray(self_polygon.GetGeometryRef(0).GetPoints())
    #     plt.plot(xy[:, 0], xy[:, 1])
    #     xy = np.asarray(other_polygon.GetGeometryRef(0).GetPoints())
    #     plt.plot(xy[:, 0], xy[:, 1])
    #     xy = np.asarray(intersection.GetGeometryRef(0).GetPoints())
    #     plt.plot(xy[:, 0], xy[:, 1])
    #     # plt.plot(intersection.GetGeometryRef(0).GetPoints())
    #     plt.show()
    #     print(self_polygon)
    #     print(other_polygon)
    #     print(intersection)
    #     print(intersection.GetArea())

    #     BREAKEM