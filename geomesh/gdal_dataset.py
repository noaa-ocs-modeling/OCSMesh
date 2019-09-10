import numpy as np
from osgeo import ogr
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib.path import Path as mpl_Path
from geomesh import gdal_tools


class GdalDataset:

    def __init__(
        self,
        path,
        SpatialReference=3395,
        feature_size=None,
    ):
        self._path = path
        self._SpatialReference = SpatialReference
        self._feature_size = feature_size

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

    def get_dx(self, SpatialReference=None):
        return gdal_tools.get_dx(self.Dataset, SpatialReference)

    def get_dy(self, SpatialReference=None):
        return gdal_tools.get_dy(self.Dataset, SpatialReference)

    def get_SpatialReference(self):
        return gdal_tools.get_SpatialReference(self.Dataset)

    def get_value(self, x, y):
        return self._RectBivariateSpline.ev(x, y)

    def __get_empty_Polygon(self):
        _Polygon = ogr.Geometry(ogr.wkbPolygon)
        _Polygon.AssignSpatialReference(self.SpatialReference)
        return _Polygon

    def __get_empty_LinearRing(self):
        _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
        _LinearRing.AssignSpatialReference(self.SpatialReference)
        return _LinearRing

    def __get_empty_MultiPolygon(self):
        _MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        _MultiPolygon.AssignSpatialReference(self.SpatialReference)
        return _MultiPolygon

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
            raise AttributeError('Must set attribute zmax.')

    @property
    def zmax(self):
        try:
            return self.__zmax
        except AttributeError:
            raise AttributeError('Must set attribute zmax.')

    @property
    def hmin(self):
        try:
            return self.__hmin
        except AttributeError:
            pass

    @property
    def hmax(self):
        try:
            return self.__hmax
        except AttributeError:
            pass

    @property
    def Dataset(self):
        try:
            return self.__Dataset
        except AttributeError:
            pass
        srcSRS = gdal_tools.get_SpatialReference(self._original)
        dstSRS = self._SpatialReference
        xRes, yRes = gdal_tools.get_resolution(self._original, dstSRS)
        if srcSRS.IsSame(dstSRS):
            dstSRS = None
        h0 = self._h0
        if xRes < h0:
            xRes = h0
        else:
            xRes = None
        if yRes < h0:
            yRes = h0
        else:
            yRes = None
        if all(_ is None for _ in [dstSRS, xRes, yRes]):
            Dataset = self._original
        else:
            Dataset = gdal_tools.Warp(
                self._original, xRes=xRes, yRes=yRes, dstSRS=dstSRS)
        self.__Dataset = Dataset
        return self.__Dataset

    @property
    def MultiPolygon(self):
        try:
            return self.__MultiPolygon
        except AttributeError:
            pass

        # fully external tile.
        if np.all(self.values > self.zmax) or np.all(self.values < self.zmin):
            self.__MultiPolygon = self.__get_empty_MultiPolygon()
            return self.__MultiPolygon

        # fully internal tile
        elif np.min(self.values) > self.zmin \
                and np.max(self.values) < self.zmax:
            _LinearRing = self.__get_empty_LinearRing()
            bbox = self.bbox.get_points()
            x0, y0 = float(bbox[0][0]), float(bbox[0][1])
            x1, y1 = float(bbox[1][0]), float(bbox[1][1])
            _LinearRing.AddPoint(x0, y0, float(self.get_value(x0, y0)))
            _LinearRing.AddPoint(x1, y0, float(self.get_value(x1, y0)))
            _LinearRing.AddPoint(x1, y1, float(self.get_value(x1, y1)))
            _LinearRing.AddPoint(x0, y1, float(self.get_value(x0, y1)))
            _LinearRing.AddPoint(*_LinearRing.GetPoint(0))
            _Polygon = self.__get_empty_Polygon()
            _Polygon.AddGeometry(_LinearRing)
            _MultiPolygon = self.__get_empty_MultiPolygon()
            _MultiPolygon.AddGeometry(_Polygon)
            self.__MultiPolygon = _MultiPolygon
            return self.__MultiPolygon

        # tile containing boundary
        _QuadContourSet = plt.contourf(
            self.x, self.y, self.values, levels=[self.zmin, self.zmax])
        plt.close(plt.gcf())
        for _PathCollection in _QuadContourSet.collections:
            _LinearRings = list()
            for _Path in _PathCollection.get_paths():
                linear_rings = _Path.to_polygons(closed_only=True)
                for linear_ring in linear_rings:
                    _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
                    _LinearRing.AssignSpatialReference(self.SpatialReference)
                    for x, y in linear_ring:
                        _LinearRing.AddPoint(
                            float(x), float(y), float(self.get_value(x, y)))
                    _LinearRing.CloseRings()
                    _LinearRings.append(_LinearRing)
        # create output object
        _MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        _MultiPolygon.AssignSpatialReference(self.SpatialReference)
        # sort list of linear rings into polygons
        areas = [_LinearRing.GetArea() for _LinearRing in _LinearRings]
        idx = np.where(areas == np.max(areas))[0][0]
        _Polygon = ogr.Geometry(ogr.wkbPolygon)
        _Polygon.AssignSpatialReference(self.SpatialReference)
        _Polygon.AddGeometry(_LinearRings.pop(idx))
        while len(_LinearRings) > 0:
            _Path = mpl_Path(np.asarray(
                _Polygon.GetGeometryRef(0).GetPoints())[:, :2], closed=True)
            for i, _LinearRing in reversed(list(enumerate(_LinearRings))):
                x = _LinearRing.GetX(0)
                y = _LinearRing.GetY(0)
                if _Path.contains_point((x, y)):
                    _Polygon.AddGeometry(_LinearRings.pop(i))
            _Polygon.CloseRings()
            _MultiPolygon.AddGeometry(_Polygon)
            if len(_LinearRings) > 0:
                areas = [_LinearRing.GetArea() for _LinearRing in _LinearRings]
                idx = np.where(areas == np.max(areas))[0][0]
                _Polygon = ogr.Geometry(ogr.wkbPolygon)
                _Polygon.AssignSpatialReference(self.SpatialReference)
                _Polygon.AddGeometry(_LinearRings.pop(idx))
        self.__MultiPolygon = _MultiPolygon
        return self.__MultiPolygon

    @property
    def path(self):
        return self.__path

    @property
    def bbox(self):
        return self.get_bbox()

    @property
    def SpatialReference(self):
        return self.get_SpatialReference()

    @property
    def feature_size(self):
        return self.__h0

    @property
    def _RectBivariateSpline(self):
        try:
            return self.__RectBivariateSpline
        except AttributeError:
            pass
        _RectBivariateSpline = RectBivariateSpline(
            self.x, self.y, self.values.T)
        self.__RectBivariateSpline = _RectBivariateSpline
        return self.__RectBivariateSpline

    @property
    def _SpatialReference(self):
        return self.__SpatialReference

    @property
    def _path(self):
        return self.__path

    @property
    def _feature_size(self):
        return self.__feature_size

    @property
    def _h0(self):
        return 0.5*(self.feature_size / np.sqrt(2))

    @property
    def _Dataset(self):
        return self.__Dataset

    @property
    def _MultiPolygon(self):
        return self.__MultiPolygon

    @property
    def _original(self):
        try:
            return self.__original
        except AttributeError:
            self.__original = gdal_tools.Open(self.path)
            return self.__original

    @SpatialReference.setter
    def SpatialReference(self, SpatialReference):
        curSRS = self.__SpatialReference
        dstSRS = gdal_tools.sanitize_SpatialReference(SpatialReference)
        if not curSRS.IsSame(dstSRS):
            del(self._Dataset)
        self.__SpatialReference = dstSRS

    @zmin.setter
    def zmin(self, zmin):
        zmin = float(zmin)
        try:
            if not self.__zmin == zmin:
                del(self._MultiPolygon)
        except AttributeError:
            pass
        self.__zmin = zmin

    @zmax.setter
    def zmax(self, zmax):
        zmax = float(zmax)
        try:
            if not self.__zmax == zmax:
                del(self._MultiPolygon)
        except AttributeError:
            pass
        self.__zmax = zmax

    @feature_size.setter
    def feature_size(self, feature_size):
        self._feature_size = feature_size

    @_path.setter
    def _path(self, path):
        self.__path = str(path)

    @_SpatialReference.setter
    def _SpatialReference(self, SpatialReference):
        srcSRS = gdal_tools.get_SpatialReference(self._original)
        dstSRS = gdal_tools.sanitize_SpatialReference(SpatialReference)
        if not srcSRS.IsSame(dstSRS):
            del(self._Dataset)
        self.__SpatialReference = dstSRS

    @_feature_size.setter
    def _feature_size(self, feature_size):
        assert isinstance(feature_size, (float, type(None)))
        try:
            if self.__h0 != feature_size:
                del(self._Dataset)
        except AttributeError:
            pass
        self.__h0 = feature_size

    @_Dataset.deleter
    def _Dataset(self):
        try:
            del(self.__Dataset)
        except AttributeError:
            pass
        del(self._MultiPolygon)

    @_MultiPolygon.deleter
    def _MultiPolygon(self):
        try:
            del(self.__MultiPolygon)
        except AttributeError:
            pass
