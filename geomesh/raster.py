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
# import fiona
# from shapely.geometry import mapping
import tempfile
from pyproj import Proj
import geomesh


class Raster:

    def __init__(
        self,
        path,
        dst_crs=None,
    ):
        self._path = path
        self._dst_crs = dst_crs

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

    def close(self):
        del(self._src)

    def add_band(self, values,  **tags):
        kwargs = self.src.meta.copy()
        band_id = kwargs["count"]+1
        kwargs.update(count=band_id)
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=geomesh.tmpdir)
        with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
                dst.update_tags(i, **self.src.tags(i))
            dst.write_band(band_id, values.astype(self.src.dtypes[i-1]))
            dst.update_tags(band_id, **tags)
        self._tmpfile = tmpfile
        return band_id

    def mask(self, shapes, i=None, **kwargs):
        _kwargs = self.src.meta.copy()
        _kwargs.update(kwargs)
        out_images, out_transform = mask(self.src, shapes)
        tmpfile = tempfile.NamedTemporaryFile(prefix=geomesh.tmpdir)
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

    def read_masks(self, i=None):
        i = 1 if i is None else int(i)
        assert i in range(1, self.count + 1)
        return self.src.read_masks(i)

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
        tmpfile = tempfile.NamedTemporaryFile(prefix=geomesh.tmpdir)
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

    def save(self, path):
        with rasterio.open(pathlib.Path(path), 'w', **self.src.meta) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
                dst.update_tags(i, **self.src.tags(i))

    # def __add_feature(self, multipolygon, zmin, zmax):
    #     self.collection.close()
    #     with fiona.open(self.collection_tmpdir.name, 'a') as dst:
    #         dst.write({"geometry": mapping(multipolygon),
    #                    "properties": {"zmin": zmin, "zmax": zmax}})
    #     self.__collection = fiona.open(self.collection_tmpdir.name)

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
    def xres(self):
        return self.transform[0]

    @property
    def yres(self):
        return self.transform[4]

    @property
    def resampling_method(self):
        try:
            return self.__resampling_method
        except AttributeError:
            return 'bilinear'

    @property
    def _src(self):
        try:
            return self.__src
        except AttributeError:
            tmpfile = tempfile.NamedTemporaryFile(
                prefix=geomesh.tmpdir)
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
                        'height': height,
                        'driver': 'GTiff'
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

    @resampling_method.setter
    def resampling_method(self, resampling_method):
        method_dict = {
            'bilinear': Resampling.bilinear,
            'nearest': Resampling.nearest,
            'cubic': Resampling.cubic,
            'average': Resampling.average
        }
        assert resampling_method in method_dict.keys()
        self.__resampling_method = method_dict[resampling_method]

    @_path.setter
    def _path(self, path):
        path = pathlib.Path(path)
        self.__path = path

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        if dst_crs is None:
            dst_crs = rasterio.open(self.path).crs
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
