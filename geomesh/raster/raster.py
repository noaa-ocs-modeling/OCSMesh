import numpy as np
import pathlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio import warp
from rasterio.mask import mask
from rasterio.enums import Resampling
import multiprocessing
import tempfile
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Proj, CRS, Transformer
from scipy.ndimage import gaussian_filter
from shapely.ops import transform
import geomesh
from geomesh.figures import FixPointNormalize


class Raster:

    def __init__(
        self,
        path,
        crs=None,
        dst_crs=None,
        clip=None
    ):
        self._path = path
        self._crs = crs
        self._dst_crs = dst_crs
        self._clip = clip

    def make_plot(
        self,
        axes=None,
        vmin=None,
        vmax=None,
        cmap='topobathy',
        levels=None,
        show=False,
        title=None,
        figsize=None,
        colors=256,
        extent=None,
        cbar_label=None,
        norm=None,
        **kwargs
    ):
        if axes is None:
            fig = plt.figure(figsize=figsize)
            axes = fig.add_subplot(111)
        vmin = np.min(self.values) if vmin is None else float(vmin)
        vmax = np.max(self.values) if vmax is None else float(vmax)
        cmap, norm, levels, col_val = self._get_cmap(
            vmin, vmax, cmap, levels, colors, norm)
        axes.contourf(
            self.x,
            self.y,
            self.values,
            levels=levels,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            **kwargs
            )
        axes.axis('scaled')
        if extent is not None:
            axes.axis(extent)
        if title is not None:
            axes.set_title(title)
        mappable = ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(vmin, vmax)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("bottom", size="2%", pad=0.5)
        cbar = plt.colorbar(mappable, cax=cax,  # extend=cmap_extend,
                            orientation='horizontal')
        if col_val != 0:
            cbar.set_ticks([vmin, vmin + col_val * (vmax-vmin), vmax])
            cbar.set_ticklabels([np.around(vmin, 2), 0.0, np.around(vmax, 2)])
        else:
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([np.around(vmin, 2), np.around(vmax, 2)])
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        if show is True:
            plt.show()
        return axes

    def tags(self, i=None):
        if i is None:
            return self.src.tags()
        else:
            return self.src.tags(i)

    def read(self, i, masked=True, **kwargs):
        return self.src.read(i, masked=masked, **kwargs)

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

    def gaussian_filter(self, **kwargs):
        meta = self.src.meta.copy()
        band_id = meta["count"]+1
        meta.update(count=band_id)
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=geomesh.tmpdir)
        with rasterio.open(tmpfile.name, 'w', **meta) as dst:
            for i in range(1, self.src.count + 1):
                outband = gaussian_filter(self.src.read(i), **kwargs)
                dst.write_band(i, outband)
        self._tmpfile = tmpfile

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
        if i is None:
            return np.dstack(
                [self.src.read_masks(i) for i in range(1, self.count + 1)])
        else:
            return self.src.read_masks(i)

    def warp(self, dst_crs):
        dst_crs = CRS.from_user_input(dst_crs)
        src = self.src
        transform, width, height = warp.calculate_default_transform(
            src.crs,
            dst_crs.srs,
            src.width,
            src.height,
            *src.bounds,
            dst_width=src.width,
            dst_height=src.height
            )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs.srs,
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
                    dst_crs=dst_crs.srs,
                    resampling=self.resampling_method,
                    num_threads=multiprocessing.cpu_count(),
                    )
        self._tmpfile = tmpfile

    def save(self, path):
        with rasterio.open(pathlib.Path(path), 'w', **self.src.meta) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
                dst.update_tags(i, **self.src.tags(i))

    def _get_cmap(
        self,
        vmin,
        vmax,
        cmap=None,
        levels=None,
        colors=256,
        norm=None
    ):
        colors = int(colors)
        if cmap is None:
            cmap = plt.cm.get_cmap('jet')
            if levels is None:
                levels = np.linspace(vmin, vmax, colors)
            col_val = 0.
        elif cmap == 'topobathy':
            if vmax <= 0.:
                cmap = plt.cm.seismic
                col_val = 0.
                levels = np.linspace(vmin, vmax, colors)
            else:
                wet_count = int(
                    np.floor(colors*(
                        float((self.values < 0.).sum())
                        / float(self.values.size))))
                col_val = float(wet_count)/colors
                dry_count = int(
                    np.floor(colors*(
                        float((self.values > 0.).sum())
                        / float(self.values.size))))
                colors_undersea = plt.cm.bwr(np.linspace(1., 0., wet_count))
                colors_land = plt.cm.terrain(np.linspace(0.25, 1., dry_count))
                colors = np.vstack((colors_undersea, colors_land))
                cmap = LinearSegmentedColormap.from_list('cut_terrain', colors)
                _vals = self.values.astype(np.float64)
                _vals[np.where(_vals >= 0)] = -float('inf')
                largest_negative = np.max(_vals)
                _vals = self.values.astype(np.float64)
                _vals[np.where(_vals <= 0)] = float('inf')
                # smallest_positive = np.min(_vals)
                wlevels = np.linspace(vmin, largest_negative, wet_count)
                dlevels = np.linspace(0., vmax, dry_count)
                levels = np.hstack((wlevels, dlevels))
        else:
            cmap = plt.cm.get_cmap(cmap)
            levels = np.linspace(vmin, vmax, colors)
            col_val = 0.
        if vmax > 0:
            if norm is None:
                norm = FixPointNormalize(sealevel=0.0, vmax=vmax, vmin=vmin,
                                         col_val=col_val)
        return cmap, norm, levels, col_val

    @staticmethod
    def _transform_multipolygon(multipolygon, src_crs, dst_crs):
        if dst_crs.srs != src_crs.srs:
            transformer = Transformer.from_crs(
                src_crs, dst_crs, always_xy=True)
            polygon_collection = list()
            for polygon in multipolygon:
                polygon_collection.append(
                    transform(transformer.transform, polygon))
            outer = polygon_collection.pop(0)
            multipolygon = MultiPolygon([outer, *polygon_collection])
        return multipolygon

    @property
    def clip(self):
        return self._clip

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
    def is_masked(self):
        return self.read_masks().any()

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
        return Proj(self.crs)

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
        return self.read(1)

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
            return Resampling.nearest

    @property
    def _path(self):
        return self.__path

    @property
    def _clip(self):
        return self.__clip

    @property
    def _tmpfile(self):

        try:
            return self.__tmpfile
        except AttributeError:
            pass
        # copy original file into a tmpfile
        tmpfile = tempfile.NamedTemporaryFile(prefix=geomesh.tmpdir)
        with rasterio.open(self.path) as src:
            with rasterio.open(tmpfile.name, 'w', **src.meta.copy()) as dst:
                for i in range(1, src.count + 1):
                    dst.write_band(i, src.read(i))
        self._tmpfile = tmpfile
        return self.__tmpfile

    @property
    def _crs(self):
        return self.__crs

    @property
    def _dst_crs(self):
        return self.__dst_crs

    @property
    def _src(self):
        try:
            return self.__src
        except AttributeError:
            self.__src = rasterio.open(self._tmpfile.name)
            return self.__src

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        self._dst_crs = dst_crs

    @resampling_method.setter
    def resampling_method(self, resampling_method):
        self.__resampling_method = {
            'bilinear': Resampling.bilinear,
            'nearest': Resampling.nearest,
            'cubic': Resampling.cubic,
            'average': Resampling.average
            }[resampling_method]

    @_path.setter
    def _path(self, path):
        path = pathlib.Path(path)
        self.__path = path

    @_crs.setter
    def _crs(self, crs):
        """
        if CRS is present in path then this is ignored.
        """
        if crs is None:
            with rasterio.open(self.path) as src:
                if src.crs is None:
                    msg = 'CRS not found in raster file. Must specify CRS.'
                    raise IOError(msg)
                else:
                    crs = CRS.from_user_input(src.crs)
        else:
            crs = CRS.from_user_input(crs)
            tmpfile = tempfile.NamedTemporaryFile(
                    prefix=geomesh.tmpdir)
            with rasterio.open(self.path) as src:
                kwargs = src.meta.copy()
                kwargs.update({'crs': crs.srs})
                kwargs.update({'driver': 'GTiff'})
                with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        dst.write_band(i, src.read(i))
                self._tmpfile = tmpfile
        self.__crs = crs

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        if dst_crs is None:
            dst_crs = self._crs
        else:
            dst_crs = CRS.from_user_input(dst_crs)

        if dst_crs.srs != self._crs.srs:
            self.warp(dst_crs)

        self.__dst_crs = dst_crs

    @_clip.setter
    def _clip(self, clip):
        if clip is not None:
            assert isinstance(clip, (Polygon, MultiPolygon, Bbox))
            if isinstance(clip, Bbox):
                x0 = clip.xmin
                x1 = clip.xmax
                y0 = clip.ymin
                y1 = clip.ymax
                clip = Polygon([
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                        [x0, y0]])
            if isinstance(clip, Polygon):
                clip = MultiPolygon([clip])

            if self._crs.srs != self.dst_crs.srs:
                clip = self._transform_multipolygon(
                    clip, self._crs, self.dst_crs)

            with rasterio.open(self._tmpfile.name) as src:
                out_image, out_transform = rasterio.mask.mask(
                    src, clip, crop=True)
                kwargs = src.meta.copy()
                kwargs.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
                if kwargs['driver'] == 'netCDF':
                    kwargs.update({'driver': 'GTIff'})
                tmpfile = tempfile.NamedTemporaryFile(prefix=geomesh.tmpdir)
                with rasterio.open(tmpfile.name, "w", **kwargs) as dst:
                    dst.write(out_image)
                self._tmpfile = tmpfile

        self.__clip = clip

    @_tmpfile.setter
    def _tmpfile(self, tmpfile):
        del(self._tmpfile)
        self.__tmpfile = tmpfile

    @_tmpfile.deleter
    def _tmpfile(self):
        try:
            del(self._src)
            self.__tmpfile.close()
            del(self.__tmpfile)
        except AttributeError:
            pass

    @_src.deleter
    def _src(self):
        try:
            del(self.__src)
        except AttributeError:
            pass
