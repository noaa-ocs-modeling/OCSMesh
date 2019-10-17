import numpy as np
import pathlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Bbox
import rasterio
from rasterio import warp
from rasterio.mask import mask
from rasterio.enums import Resampling
import multiprocessing
from shapely.geometry import Polygon, LinearRing, MultiPolygon, mapping, shape
import fiona
import tempfile
from pyproj import Proj


class Raster:

    def __init__(
        self,
        path,
        zmin=None,
        zmax=None,
        dst_crs=None,
        xres=None,
        yres=None
    ):
        self._path = path
        self._zmin = zmin
        self._zmax = zmax
        self._dst_crs = dst_crs
        self._xres = xres
        self._yres = yres

    def __call__(self, zmin, zmax):
        if np.all(self.values > zmax) or np.all(self.values < zmin):
            # fully external tile.
            return MultiPolygon()

        elif (np.min(self.values) > zmin
              and np.max(self.values) < zmax):
            # fully internal tile
            raise NotImplementedError
            # _LinearRing = self.__get_empty_LinearRing()
            # bbox = self.bbox.get_points()
            # x0, y0 = float(bbox[0][0]), float(bbox[0][1])
            # x1, y1 = float(bbox[1][0]), float(bbox[1][1])
            # _LinearRing.AddPoint(x0, y0, float(self.get_value(x0, y0)))
            # _LinearRing.AddPoint(x1, y0, float(self.get_value(x1, y0)))
            # _LinearRing.AddPoint(x1, y1, float(self.get_value(x1, y1)))
            # _LinearRing.AddPoint(x0, y1, float(self.get_value(x0, y1)))
            # _LinearRing.AddPoint(*_LinearRing.GetPoint(0))
            # _Polygon = self.__get_empty_Polygon()
            # _Polygon.AddGeometry(_LinearRing)
            # _MultiPolygon = self.__get_empty_MultiPolygon()
            # _MultiPolygon.AddGeometry(_Polygon)
            # self.__MultiPolygon = _MultiPolygon
            # return self.__MultiPolygon
        else:
            # tile containing boundary
            ax = plt.contourf(
                self.x, self.y, self.values, levels=[zmin, zmax])
            plt.close(plt.gcf())
            # extract linear_rings from plot
            linear_ring_collection = list()
            for path_collection in ax.collections:
                for path in path_collection.get_paths():
                    polygons = path.to_polygons(closed_only=True)
                    for linear_ring in polygons:
                        linear_ring_collection.append(LinearRing(linear_ring))
            # reorder linear rings from above
            areas = [Polygon(linear_ring).area
                     for linear_ring in linear_ring_collection]
            idx = np.where(areas == np.max(areas))[0][0]
            polygon_collection = list()
            outer_ring = linear_ring_collection.pop(idx)
            path = Path(np.asarray(outer_ring.coords), closed=True)
            while len(linear_ring_collection) > 0:
                inner_rings = list()
                for i, linear_ring in reversed(
                        list(enumerate(linear_ring_collection))):
                    xy = np.asarray(linear_ring.coords)[0, :]
                    if path.contains_point(xy):
                        inner_rings.append(linear_ring_collection.pop(i))
                polygon_collection.append(Polygon(outer_ring, inner_rings))
                if len(linear_ring_collection) > 0:
                    areas = [Polygon(linear_ring).area
                             for linear_ring in linear_ring_collection]
                    idx = np.where(areas == np.max(areas))[0][0]
                    outer_ring = linear_ring_collection.pop(idx)
                    path = Path(np.asarray(outer_ring.coords), closed=True)
            return MultiPolygon(polygon_collection)

    def tags(self, i=None):
        if i is None:
            return self.src.tags()
        else:
            return self.src.tags(i)

    def read(self, i, **kwargs):
        return self.src.read(i, **kwargs)

    def dtype(self, i):
        return self.src.dtypes[i-1]

    def nodataval(self, i):
        return self.src.nodatavals[i-1]

    def sample(self, xy, i):
        return self.src.sample(xy, i)

    # def close(self):
    #     del(self._src)

    def add_band(self,  band_type, values):
        kwargs = self.src.meta.copy()
        band_id = kwargs["count"]+1
        kwargs.update(count=band_id)
        tmpfile = tempfile.NamedTemporaryFile()
        fname = tmpfile.name
        with rasterio.open(fname, 'w', **kwargs) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
                dst.update_tags(i, **self.src.tags(i))
            dst.write_band(band_id, values.astype(self.src.dtypes[i-1]))
            dst.update_tags(band_id, BAND_TYPE=band_type)
        self._tmpfile = tmpfile

    def mask(self, features, i=None):
        kwargs = self.src.meta.copy()
        # kwargs.update({'nodata': nodata})
        out_images, out_transform = mask(self.src, features)
        tmpfile = tempfile.NamedTemporaryFile()
        fname = tmpfile.name
        with rasterio.open(fname, 'w', **kwargs) as dst:
            if i is None:
                for j in range(1, self.src.count + 1):
                    dst.write_band(j, out_images[j-1])
                    dst.update_tags(j, **self.src.tags(j))
            else:
                for j in range(1, self.src.count + 1):
                    if i == j:
                        dst.write_band(j, out_images[j-1])
                        dst.update_tags(j, **self.src.tags(j))
                    else:
                        dst.write_band(j, self.src.read(j))
                        dst.update_tags(j, **self.src.tags(j))
        self._tmpfile = tmpfile

    def warp(self, dst_crs):
        src = rasterio.open(self.path)
        transform, width, height = warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height,
            *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        tmpfile = tempfile.NamedTemporaryFile()
        with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    # resampling=<Resampling.nearest: 0>,
                    num_threads=multiprocessing.cpu_count(),
                    )
                dst.update_tags(i, **self.src.tags(i))
        self._tmpfile = tmpfile

    def resample(self, xres, yres, method='bilinear'):
        transform, width, heigth = warp.aligned_target(
            self.transform, self.width, self.height, (xres, yres))
        kwargs = self.src.meta.copy()
        kwargs.update({
            "transform": transform,
            "width": width,
            "heigth": heigth})
        tmpfile = tempfile.NamedTemporaryFile()
        with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
            for i in self.count:
                x, y, band = self.get_resampled(i, xres, yres, method)
                dst.write_band(i, band)
                dst.update_tags(i, band)
        self._tmpfile = tempfile

    def save(self, path):
        with rasterio.open(pathlib.Path(path), 'w', **self.src.meta) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
                dst.update_tags(i, **self.src.tags(i))

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

    def get_resampled(self, i, xres, yres, method='bilinear'):
        method_dict = {
            'bilinear': Resampling.bilinear,
            'nearest': Resampling.nearest,
            'cubic': Resampling.cubic,
            'average': Resampling.average
        }
        if method in method_dict.keys():
            method = method_dict[method]
        else:
            msg = 'Method must be one of {} '.format(method_dict.keys())
            msg += 'or instance of type rasterio.enums.Resampling'
            assert isinstance(method, Resampling), msg
        transform, width, height = warp.aligned_target(
            self.transform, self.width, self.height, (xres, yres))
        band = self.read(
            i,
            out_shape=(height, width),
            resampling=method)
        x0, y0, x1, y1 = rasterio.transform.array_bounds(
            height, width, transform)
        x = np.linspace(x0, x1, width)
        y = np.linspace(y1, y0, height)
        return x, y, band

    @property
    def path(self):
        return self._path

    @property
    def src(self):
        return self._src

    @property
    def count(self):
        return self.src.count

    @property
    def shape(self):
        return self.src.shape

    @property
    def height(self):
        return self.src.height

    @property
    def bbox(self):
        x0, y0, x1, y1 = rasterio.transform.array_bounds(
            self.height, self.width, self.transform)
        return Bbox([[x0, y0], [x1, y1]])

    @property
    def width(self):
        return self.src.width

    @property
    def dx(self):
        return self.src.transform[0]

    @property
    def dy(self):
        return -self.src.transform[4]

    @property
    def crs(self):
        return self.src.crs

    @property
    def dst_crs(self):
        return self._dst_crs

    @property
    def srs(self):
        return self.proj.srs

    @property
    def proj(self):
        return Proj(init=str(self.crs))

    @property
    def nodatavals(self):
        return self.src.nodatavals

    @property
    def transform(self):
        return self.src.transform

    @property
    def dtypes(self):
        return self.src.dtypes

    @property
    def x(self):
        return np.linspace(
            self.src.bounds.left,
            self.src.bounds.right,
            self.src.width)

    @property
    def y(self):
        return np.linspace(
            self.src.bounds.top,
            self.src.bounds.bottom,
            self.src.height)

    @property
    def values(self):
        return self.src.read(1)

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    @property
    def tmpdir_shp(self):
        return self._tmpdir_shp

    @property
    def collection(self):
        return self._collection

    @property
    def schema(self):
        return {
            'geometry': 'MultiPolygon',
            'properties': {
                'zmin': 'float',
                'zmax': 'float'}}

    @property
    def _src(self):
        try:
            return self.__src
        except AttributeError:
            tmpfile = tempfile.NamedTemporaryFile()
            with rasterio.open(self.path) as src:
                if src.count > 1 or src.count == 0:
                    msg = 'Input raster must have only a single band and it '
                    msg += 'must correspond to terrain elevation.'
                    raise TypeError(msg)
                kwargs = src.meta.copy()
                if self.dst_crs != src.crs:
                    transform, width, height = \
                        warp.calculate_default_transform(
                            src.crs, self.dst_crs, src.width, src.height,
                            *src.bounds)
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': self.dst_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })
                    with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
                        rasterio.warp.reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=self.dst_crs,
                            # resampling=<Resampling.nearest: 0>,
                            num_threads=multiprocessing.cpu_count(),
                            )
                        dst.update_tags(1, BAND_TYPE='ELEVATION')
                else:
                    with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
                        dst.write_band(1, src.read(1))
                        dst.update_tags(1, BAND_TYPE='ELEVATION')
            self._tmpfile = tmpfile
            return self.__src

    @property
    def _path(self):
        return self.__path

    @property
    def _zmin(self):
        try:
            return self.__zmin
        except AttributeError:
            raise AttributeError('Must set zmin attribute.')

    @property
    def _zmax(self):
        try:
            return self.__zmax
        except AttributeError:
            raise AttributeError('Must set zmax attribute.')

    @property
    def _tmpdir_shp(self):
        try:
            return self.__tmpdir_shp
        except AttributeError:
            self.__tmpdir_shp = tempfile.TemporaryDirectory()
            return self.__tmpdir_shp

    @property
    def _tmpfile(self):
        return self.__tmpfile

    @property
    def _dst_crs(self):
        return self.__dst_crs

    @property
    def _collection(self):
        try:
            return self.__collection
        except AttributeError:
            collection = fiona.open(
                self.tmpdir_shp.name,
                'w',
                driver='ESRI Shapefile',
                crs=self.dst_crs,
                schema=self.schema)
            collection.write({
                "geometry": mapping(self(self.zmin, self.zmax)),
                "properties": {
                    "zmin": self.zmin,
                    "zmax": self.zmax}})
            collection.close()
            self.__collection = fiona.open(self.tmpdir_shp.name)
            return self.__collection

    @zmin.setter
    def zmin(self, zmin):
        self._zmin = zmin

    @zmax.setter
    def zmax(self, zmax):
        self._zmax = zmax

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        self._dst_crs = dst_crs

    @_path.setter
    def _path(self, path):
        self.__path = path

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        if dst_crs is None:
            dst_crs = rasterio.open(self.path)
        self.__dst_crs = dst_crs

    @_tmpfile.setter
    def _tmpfile(self, tmpfile):
        self._src = rasterio.open(tmpfile.name)
        del(self._tmpfile)
        self.__tmpfile = tmpfile

    @_src.setter
    def _src(self, src):
        del(self._src)
        self.__src = src

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

    @_src.deleter
    def _src(self):
        try:
            self.__src.close()
            del(self.__src)
            del(self._tmpfile)
        except AttributeError:
            pass

    @_tmpfile.deleter
    def _tmpfile(self):
        try:
            del(self.__tmpfile)
            del(self._collection)
        except AttributeError:
            pass

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

    #     self.__dataset = str(dataset)

    # @_target_crs.setter
    # def _target_crs(self, target_crs):

    # def get_arrays(self, SpatialReference=None):
    #     return gdal_tools.get_arrays(self.Dataset, SpatialReference)

    # def get_xyz(self, SpatialReference=None):
    #     return gdal_tools.get_arrays(self.Dataset, SpatialReference)

    # def get_xy(self, SpatialReference=None):
    #     return gdal_tools.get_xy(self.Dataset, SpatialReference)

    # def get_x(self, SpatialReference=None):
    #     return gdal_tools.get_x(self.Dataset, SpatialReference)

    # def get_y(self, SpatialReference=None):
    #     return gdal_tools.get_y(self.Dataset, SpatialReference)

    # def get_bbox(self, SpatialReference=None, Path=False):
    #     return gdal_tools.get_bbox(self.Dataset, SpatialReference, Path)

    # def get_GeoTransform(self, SpatialReference=None):
    #     return gdal_tools.get_GeoTransform(self.Dataset, SpatialReference)

    # def get_resolution(self, SpatialReference=None):
    #     return gdal_tools.get_resolution(self.Dataset, SpatialReference)

    # def get_dx(self, SpatialReference=None):
    #     return gdal_tools.get_dx(self.Dataset, SpatialReference)

    # def get_dy(self, SpatialReference=None):
    #     return gdal_tools.get_dy(self.Dataset, SpatialReference)

    # def get_SpatialReference(self):
    #     return gdal_tools.get_SpatialReference(self.Dataset)

    # def get_value(self, x, y):
    #     return float(self._RectBivariateSpline.ev(x, y))

    # def get_MultiPolygon(self, SpatialReference=None):
    #     _MultiPolygon = self.MultiPolygon
    #     if SpatialReference is None:
    #         return _MultiPolygon
    #     else:
    #         SpatialReference = gdal_tools.sanitize_SpatialReference(
    #             SpatialReference)
    #         _MultiPolygon.TransformTo(SpatialReference)
    #         return _MultiPolygon

    # def __get_empty_Polygon(self):
    #     _Polygon = ogr.Geometry(ogr.wkbPolygon)
    #     _Polygon.AssignSpatialReference(self.SpatialReference)
    #     return _Polygon

    # def __get_empty_LinearRing(self):
    #     _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
    #     _LinearRing.AssignSpatialReference(self.SpatialReference)
    #     return _LinearRing

    # def __get_empty_MultiPolygon(self):
    #     _MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    #     _MultiPolygon.AssignSpatialReference(self.SpatialReference)
    #     return _MultiPolygon

    # @property
    # def xyz(self):
    #     return self.get_xyz(self.SpatialReference)

    # @property
    # def x(self):
    #     return self.get_arrays(self.SpatialReference)[0]

    # @property
    # def y(self):
    #     return self.get_arrays(self.SpatialReference)[1]

    # @property
    # def elevation(self):
    #     return self.values

    # @property
    # def values(self):
    #     values = self.get_arrays(self.SpatialReference)[2]
    #     return np.ma.masked_equal(values, 255)

    # @property
    # def Dataset(self):
    #     return self._Dataset

    # @property
    # def MultiPolygon(self):
    #     # not used here but might be relevant:
    #     # https://stackoverflow.com/questions/22100453/gdal-python-creating-contourlines
    #     try:
    #         return self.__MultiPolygon
    #     except AttributeError:
    #         pass

    #     # fully external tile.
    #     if np.all(self.values > self.zmax) or np.all(self.values < self.zmin):
    #         self.__MultiPolygon = self.__get_empty_MultiPolygon()
    #         return self.__MultiPolygon

    #     # fully internal tile
    #     elif np.min(self.values) > self.zmin \
    #             and np.max(self.values) < self.zmax:
    #         _LinearRing = self.__get_empty_LinearRing()
    #         bbox = self.bbox.get_points()
    #         x0, y0 = float(bbox[0][0]), float(bbox[0][1])
    #         x1, y1 = float(bbox[1][0]), float(bbox[1][1])
    #         _LinearRing.AddPoint(x0, y0, float(self.get_value(x0, y0)))
    #         _LinearRing.AddPoint(x1, y0, float(self.get_value(x1, y0)))
    #         _LinearRing.AddPoint(x1, y1, float(self.get_value(x1, y1)))
    #         _LinearRing.AddPoint(x0, y1, float(self.get_value(x0, y1)))
    #         _LinearRing.AddPoint(*_LinearRing.GetPoint(0))
    #         _Polygon = self.__get_empty_Polygon()
    #         _Polygon.AddGeometry(_LinearRing)
    #         _MultiPolygon = self.__get_empty_MultiPolygon()
    #         _MultiPolygon.AddGeometry(_Polygon)
    #         self.__MultiPolygon = _MultiPolygon
    #         return self.__MultiPolygon

    #     # tile containing boundary
    #     _QuadContourSet = plt.contourf(
    #         self.x, self.y, self.values, levels=[self.zmin, self.zmax])
    #     plt.close(plt.gcf())
    #     for _PathCollection in _QuadContourSet.collections:
    #         _LinearRings = list()
    #         for _Path in _PathCollection.get_paths():
    #             linear_rings = _Path.to_polygons(closed_only=True)
    #             for linear_ring in linear_rings:
    #                 _LinearRing = ogr.Geometry(ogr.wkbLinearRing)
    #                 _LinearRing.AssignSpatialReference(self.SpatialReference)
    #                 for x, y in linear_ring:
    #                     _LinearRing.AddPoint(
    #                         float(x), float(y), float(self.get_value(x, y)))
    #                 _LinearRing.CloseRings()
    #                 _LinearRings.append(_LinearRing)
    #     # create output object
    #     _MultiPolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    #     _MultiPolygon.AssignSpatialReference(self.SpatialReference)
    #     # sort list of linear rings into polygons
    #     areas = [_LinearRing.GetArea() for _LinearRing in _LinearRings]
    #     idx = np.where(areas == np.max(areas))[0][0]
    #     _Polygon = ogr.Geometry(ogr.wkbPolygon)
    #     _Polygon.AssignSpatialReference(self.SpatialReference)
    #     _Polygon.AddGeometry(_LinearRings.pop(idx))
    #     while len(_LinearRings) > 0:
    #         _Path = mpl_Path(np.asarray(
    #             _Polygon.GetGeometryRef(0).GetPoints())[:, :2], closed=True)
    #         for i, _LinearRing in reversed(list(enumerate(_LinearRings))):
    #             x = _LinearRing.GetX(0)
    #             y = _LinearRing.GetY(0)
    #             if _Path.contains_point((x, y)):
    #                 _Polygon.AddGeometry(_LinearRings.pop(i))
    #         _Polygon.CloseRings()
    #         _MultiPolygon.AddGeometry(_Polygon)
    #         if len(_LinearRings) > 0:
    #             areas = [_LinearRing.GetArea() for _LinearRing in _LinearRings]
    #             idx = np.where(areas == np.max(areas))[0][0]
    #             _Polygon = ogr.Geometry(ogr.wkbPolygon)
    #             _Polygon.AssignSpatialReference(self.SpatialReference)
    #             _Polygon.AddGeometry(_LinearRings.pop(idx))
    #     self.__MultiPolygon = _MultiPolygon
    #     return self.__MultiPolygon

    # @property
    # def bbox(self):
    #     return self.get_bbox()

    # @property
    # def SpatialReference(self):
    #     return self.get_SpatialReference()

    # @property
    # def _Dataset(self):
    #     try:
    #         return self.__Dataset
    #     except AttributeError:
    #         self._Dataset = gdal_tools.Open(self.path)
    #         return self.__Dataset

    # @property
    # def _xRes(self):
    #     return self.__xRes

    # @property
    # def _yRes(self):
    #     return self.__yRes

    # @property
    # def _RectBivariateSpline(self):
    #     try:
    #         return self.__RectBivariateSpline
    #     except AttributeError:
    #         pass
    #     _RectBivariateSpline = RectBivariateSpline(
    #         self.x, self.y, self.values.T)
    #     self.__RectBivariateSpline = _RectBivariateSpline
    #     return self.__RectBivariateSpline

    # @property
    # def _path(self):
    #     return self.__path

    # @property
    # def _MultiPolygon(self):
    #     return self.__MultiPolygon

    # @zmin.setter
    # def zmin(self, zmin):
    #     zmin = float(zmin)
    #     try:
    #         if not self.__zmin == zmin:
    #             del(self._MultiPolygon)
    #     except AttributeError:
    #         pass
    #     self.__zmin = zmin

    # @zmax.setter
    # def zmax(self, zmax):
    #     zmax = float(zmax)
    #     try:
    #         if not self.__zmax == zmax:
    #             del(self._MultiPolygon)
    #     except AttributeError:
    #         pass
    #     self.__zmax = zmax

    # @_Dataset.setter
    # def _Dataset(self, Dataset):
    #     if self.xRes is not None or self.yRes is not None:
    #         Dataset = gdal_tools.Warp(Dataset, xRes=self.xRes, yRes=self.yRes)
    #     self.__Dataset = Dataset


    # @_path.setter
    # def _path(self, path):
    #     self.__path = str(path)

    # @_MultiPolygon.deleter
    # def _MultiPolygon(self):
    #     try:
    #         del(self.__MultiPolygon)
    #     except AttributeError:
    #         pass
