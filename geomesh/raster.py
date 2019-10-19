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
from shapely.geometry import Polygon, LinearRing, MultiPolygon, mapping
import tempfile
from pyproj import Proj


class Raster:

    def __init__(
        self,
        path,
        dst_crs=None,
        xres=None,
        yres=None
    ):
        self._path = path
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
        with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
                dst.update_tags(i, **self.src.tags(i))
            dst.write_band(band_id, values.astype(self.src.dtypes[i-1]))
            dst.update_tags(band_id, BAND_TYPE=band_type)
        self._tmpfile = tmpfile

    def mask(self, shapes, i=None, **kwargs):
        _kwargs = self.src.meta.copy()
        _kwargs.update(kwargs)
        out_images, out_transform = mask(self.src, shapes)
        tmpfile = tempfile.NamedTemporaryFile()
        with rasterio.open(tmpfile.name, 'w', **_kwargs) as dst:
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

    def get_resampled(self, i, xres, yres, method='bilinear', masked=False):
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
            resampling=method,
            masked=masked)
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
    def _tmpfile(self):
        return self.__tmpfile

    @property
    def _dst_crs(self):
        return self.__dst_crs

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
        except AttributeError:
            pass
