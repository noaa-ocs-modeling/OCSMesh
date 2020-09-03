import numpy as np
import pathlib
import matplotlib.pyplot as plt
import rasterio
import os
import multiprocessing
import tempfile
import hashlib
from matplotlib.transforms import Bbox
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from rasterio import warp
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.transform import array_bounds
from rasterio import windows
from shapely.geometry import Polygon, MultiPolygon, LinearRing, shape
from pyproj import Proj, CRS, Transformer
from scipy.ndimage import gaussian_filter
from shapely import ops
from functools import lru_cache
# from geomesh.geom import Geom
# from geomesh.hfun import Hfun
from geomesh import figures


tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/geomesh'))+'/'
os.makedirs(tmpdir, exist_ok=True)


class Raster:

    def __init__(
            self,
            path,
            crs=None,
            chunk_size=None,
            overlap=None
    ):
        self._path = path
        self._crs = crs
        self.chunk_size = chunk_size
        self.overlap = overlap

    # def __call__(self, **kwargs):
    #     return self.get_geom(**kwargs)

    def __iter__(self, chunk_size=None, overlap=None):
        for window in self.iter_windows(chunk_size, overlap):
            bounds = array_bounds(
                window.height,
                window.width,
                self.get_window_transform(window)
            )
            yield window, bounds

    def get_x(self, window=None):
        window = windows.Window(0, 0, self._src.shape[1], self._src.shape[0]) \
            if window is None else window
        if window is not None:
            assert isinstance(window, windows.Window)
            width = window.width
        else:
            width = self.shape[0]
        x0, y0, x1, y1 = self.get_array_bounds(window)
        return np.linspace(x0, x1, width)

    def get_y(self, window=None):
        window = windows.Window(0, 0, self._src.shape[1], self._src.shape[0]) \
            if window is None else window
        if window is not None:
            assert isinstance(window, windows.Window)
            height = window.height
        else:
            height = self.shape[1]
        x0, y0, x1, y1 = self.get_array_bounds(window)
        return np.linspace(y1, y0, height)

    def get_xy(self, window=None):
        x, y = np.meshgrid(self.get_x(window), self.get_y(window))
        return np.vstack([x.flatten(), y.flatten()]).T

    def get_values(self, window=None, band=None, **kwargs):
        i = 1 if band is None else band
        window = windows.Window(0, 0, self._src.shape[1], self._src.shape[0]) \
            if window is None else window
        if window is not None:
            assert isinstance(window, windows.Window)
        return self._src.read(i, window=window, **kwargs)

    def get_xyz(self, window=None, band=None):
        xy = self.get_xy(window)
        values = self.get_values(window=window, band=band).reshape(
            (xy.shape[0], 1))
        return np.hstack([xy, values])

    # def get_geom(self, **kwargs):
    #     return Geom(self.get_multipolygon(**kwargs), self.crs)

    # def get_hfun(self, **kwargs):
    #     return Hfun(self, **kwargs)

    def get_multipolygon(
            self,
            zmin=None,
            zmax=None,
            window=None,
            join_method=None,
            driver=None,
            overlap=None,
            dst_crs=None,
            nprocs=None,
    ):
        nprocs = multiprocessing.cpu_count() if nprocs == -1 else nprocs
        nprocs = 1 if nprocs is None else nprocs

        # certify driver
        driver = 'matplotlib' if driver is None \
            else driver
        assert driver in ['rasterio', 'matplotlib']

        # certify join_method
        join_method = 'unary_union' if join_method is None else join_method
        assert join_method in ['buffer', 'unary_union']

        polygon_collection = []

        if window is None:
            iter_windows = self.iter_windows(overlap=2)
        else:
            iter_windows = [window]
        # case: fast
        if driver == 'rasterio':
            tmpfile = tempfile.NamedTemporaryFile(
                prefix=tmpdir
                )
            with rasterio.open(tmpfile.name, 'w', **self._src.meta) as dst:
                for window in iter_windows:
                    data = self.get_values(window=window, masked=True)
                    if zmin is not None:
                        data[np.where(data < zmin)] = self._src.nodata
                    if zmax is not None:
                        data[np.where(data > zmax)] = self._src.nodata
                    dst.write(data, window=window)

            with rasterio.open(tmpfile.name) as src:
                req = rasterio.features.dataset_features(
                    src,
                    # bidx=None,
                    # sampling=1,
                    band=False,
                    # as_mask=False,
                    with_nodata=False,
                    # geographic=False,
                    # precision=-1
                    )
                for res in req:
                    multipolygon = shape(res['geometry'])
                    if isinstance(multipolygon, Polygon):
                        multipolygon = MultiPolygon([multipolygon])
                    for polygon in multipolygon:
                        polygon_collection.append(polygon)

                #     if isinstance(geom, geometry.Polygon):
                #         geom = geometry.MultiPolygon([geom])
                #     for polygon in geom:
                #         plt.plot(*polygon.exterior.xy, color='k')
                #         for interior in polygon.interiors:
                #             plt.plot(*interior.xy, color='r')
                # plt.show()
                # exit()

        # case: accurate
        if driver == 'matplotlib':

            for window in iter_windows:
                x, y, z = self.get_window_data(window)
                new_mask = np.full(z.mask.shape, 0)
                new_mask[np.where(z.mask)] = -1
                new_mask[np.where(~z.mask)] = 1

                if zmin is not None:
                    new_mask[np.where(z < zmin)] = -1

                if zmax is not None:
                    new_mask[np.where(z > zmax)] = -1

                if np.all(new_mask == -1):  # or not new_mask.any():
                    continue

                else:
                    ax = plt.contourf(
                        x, y, new_mask[0, :, :], levels=[0, 1])
                    plt.close(plt.gcf())
                    for polygon in self._get_multipolygon_from_axes(ax):
                        polygon_collection.append(polygon)

        if join_method == 'buffer':
            multipolygon = MultiPolygon(polygon_collection).buffer(0)

        if join_method == 'unary_union':
            multipolygon = ops.unary_union(polygon_collection)

        if isinstance(multipolygon, Polygon):
            multipolygon = MultiPolygon([multipolygon])

        if dst_crs is not None:
            dst_crs = CRS.from_user_input(dst_crs)
            if not dst_crs.equal(self.crs):
                transformer = Transformer.from_crs(
                    self.crs, dst_crs, always_xy=True)
                polygon_collection = list()
                for polygon in multipolygon:
                    polygon_collection.append(
                        ops.transform(transformer.transform, polygon))
                outer = polygon_collection.pop(0)
                multipolygon = MultiPolygon([outer, *polygon_collection])

        return multipolygon

    def contourf(
            self,
            band=1,
            window=None,
            axes=None,
            vmin=None,
            vmax=None,
            cmap='topobathy',
            levels=None,
            show=False,
            title=None,
            figsize=None,
            colors=256,
            cbar_label=None,
            norm=None,
            **kwargs
    ):
        if axes is None:
            fig = plt.figure(figsize=figsize)
            axes = fig.add_subplot(111)
        values = self.get_values(band=band, masked=True, window=window)
        vmin = np.min(values) if vmin is None else float(vmin)
        vmax = np.max(values) if vmax is None else float(vmax)

        # def get_cmap(
        #     values,
        #     vmin,
        #     vmax,
        #     cmap=None,
        #     levels=None,
        #     colors=256,
        #     norm=None,
        # ):
        #     colors = int(colors)
        #     if cmap is None:
        #         cmap = plt.cm.get_cmap('jet')
        #         if levels is None:
        #             levels = np.linspace(vmin, vmax, colors)
        #         col_val = 0.
        #     elif cmap == 'topobathy':
        #         if vmax <= 0.:
        #             cmap = plt.cm.seismic
        #             col_val = 0.
        #             levels = np.linspace(vmin, vmax, colors)
        #         else:
        #             wet_count = int(np.floor(colors*(float((values < 0.).sum())
        #                                              / float(values.size))))
        #             col_val = float(wet_count)/colors
        #             dry_count = colors - wet_count
        #             colors_undersea = plt.cm.bwr(np.linspace(1., 0., wet_count))
        #             colors_land = plt.cm.terrain(np.linspace(0.25, 1., dry_count))
        #             colors = np.vstack((colors_undersea, colors_land))
        #             cmap = LinearSegmentedColormap.from_list('cut_terrain', colors)
        #             wlevels = np.linspace(vmin, 0.0, wet_count, endpoint=False)
        #             dlevels = np.linspace(0.0, vmax, dry_count)
        #             levels = np.hstack((wlevels, dlevels))
        #     else:
        #         cmap = plt.cm.get_cmap(cmap)
        #         levels = np.linspace(vmin, vmax, colors)
        #         col_val = 0.
        #     if vmax > 0:
        #         if norm is None:
        #             norm = FixPointNormalize(
        #                 sealevel=0.0,
        #                 vmax=vmax,
        #                 vmin=vmin,
        #                 col_val=col_val
        #                 )
        #     return cmap, norm, levels, col_val
        # cmap, norm, levels, col_val = get_cmap(
        #     values, vmin, vmax, cmap, levels, colors, norm)
        cmap, norm, levels, col_val = figures.get_topobathy_kwargs()
        axes.contourf(
            self.get_x(window),
            self.get_y(window),
            values,
            levels=levels,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            **kwargs
            )
        axes.axis('scaled')
        # if extent is not None:
        #     axes.axis(extent)
        if title is not None:
            axes.set_title(title)
        mappable = ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(vmin, vmax)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("bottom", size="2%", pad=0.5)
        cbar = plt.colorbar(
            mappable,
            cax=cax,
            # extend=cmap_extend,
            orientation='horizontal'
            )
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
            return self._src.tags()
        else:
            return self._src.tags(i)

    def read(self, i, masked=True, **kwargs):
        return self._src.read(i, masked=masked, **kwargs)

    def dtype(self, i):
        return self._src.dtypes[i-1]

    def nodataval(self, i):
        return self._src.nodatavals[i-1]

    def sample(self, xy, i):
        return self._src.sample(xy, i)

    def close(self):
        del(self._src)

    def add_band(self, values,  **tags):
        kwargs = self._src.meta.copy()
        band_id = kwargs["count"]+1
        kwargs.update(count=band_id)
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=tmpdir)
        with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
            for i in range(1, self._src.count + 1):
                dst.write_band(i, self._src.read(i))
            dst.write_band(band_id, values.astype(self._src.dtypes[i-1]))
        self._tmpfile = tmpfile
        return band_id

    def fill_nodata(self):
        """
        A parallelized version is presented here:
        https://github.com/basaks/rasterio/blob/master/examples/fill_large_raster.py
        """
        tmpfile = tempfile.NamedTemporaryFile(prefix=tmpdir)
        with rasterio.open(tmpfile.name, 'w', **self._src.meta.copy()) as dst:
            for window in self.iter_windows():
                dst.write(
                    fillnodata(self._src.read(window=window, masked=True)),
                    window=window
                    )
        self._tmpfile = tmpfile

    def gaussian_filter(self, **kwargs):
        meta = self._src.meta.copy()
        band_id = meta["count"]+1
        meta.update(count=band_id)
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=tmpdir)
        with rasterio.open(tmpfile.name, 'w', **meta) as dst:
            for i in range(1, self._src.count + 1):
                outband = gaussian_filter(self._src.read(i), **kwargs)
                dst.write_band(i, outband)
        self._tmpfile = tmpfile

    def mask(self, shapes, i=None, **kwargs):
        _kwargs = self._src.meta.copy()
        _kwargs.update(kwargs)
        out_images, out_transform = mask(self._src, shapes)
        tmpfile = tempfile.NamedTemporaryFile(prefix=tmpdir)
        with rasterio.open(tmpfile.name, 'w', **_kwargs) as dst:
            if i is None:
                for j in range(1, self._src.count + 1):
                    dst.write_band(j, out_images[j-1])
                    dst.update_tags(j, **self._src.tags(j))
            else:
                for j in range(1, self._src.count + 1):
                    if i == j:
                        dst.write_band(j, out_images[j-1])
                        dst.update_tags(j, **self._src.tags(j))
                    else:
                        dst.write_band(j, self._src.read(j))
                        dst.update_tags(j, **self._src.tags(j))
        self._tmpfile = tmpfile

    def read_masks(self, i=None):
        if i is None:
            return np.dstack(
                [self._src.read_masks(i) for i in range(1, self.count + 1)])
        else:
            return self._src.read_masks(i)

    def warp(self, dst_crs):
        dst_crs = CRS.from_user_input(dst_crs)
        transform, width, height = warp.calculate_default_transform(
            self._src.crs,
            dst_crs.srs,
            self._src.width,
            self._src.height,
            *self._src.bounds,
            dst_width=self._src.width,
            dst_height=self._src.height
            )
        kwargs = self._src.meta.copy()
        kwargs.update({
            'crs': dst_crs.srs,
            'transform': transform,
            'width': width,
            'height': height
        })
        tmpfile = tempfile.NamedTemporaryFile(prefix=tmpdir)

        with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
            for i in range(1, self._src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(self._src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=self._src.transform,
                    crs=self._src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs.srs,
                    resampling=self.resampling_method,
                    num_threads=multiprocessing.cpu_count(),
                    )

        self._tmpfile = tmpfile

    def resample(self, scaling_factor, resampling_method=None):
        if resampling_method is None:
            resampling_method = self.resampling_method
        else:
            msg = "resampling_method must be None or one of "
            msg += f"{self._resampling_methods.keys()}"
            assert resampling_method in self._resampling_methods.keys(), msg
            resampling_method = self._resampling_methods[resampling_method]

        tmpfile = tempfile.NamedTemporaryFile(prefix=tmpdir)
        # resample data to target shape
        width = int(self._src.width * scaling_factor)
        height = int(self._src.height * scaling_factor)
        data = self._src.read(
            out_shape=(
                self._src.count,
                height,
                width
            ),
            resampling=resampling_method
        )
        kwargs = self._src.meta.copy()
        transform = self._src.transform * self._src.transform.scale(
            (self._src.width / data.shape[-1]),
            (self._src.height / data.shape[-2])
        )
        kwargs.update({
            'transform': transform,
            'width': width,
            'height': height
            })
        with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
            dst.write(data)
        self._tmpfile = tmpfile

    def save(self, path):
        with rasterio.open(pathlib.Path(path), 'w', **self._src.meta) as dst:
            for i in range(1, self._src.count + 1):
                dst.write_band(i, self._src.read(i))
                dst.update_tags(i, **self._src.tags(i))

    def clip(self, multipolygon):
        out_image, out_transform = rasterio.mask.mask(
            self._src, multipolygon, crop=True)
        out_meta = self._src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform}
            )
        tmpfile = tempfile.NamedTemporaryFile(prefix=tmpdir)
        with rasterio.open(tmpfile.name, "w", **out_meta) as dest:
            dest.write(out_image)
        self._tmpfile = tmpfile

    def iter_windows(self, chunk_size=None, overlap=None):
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        overlap = self.overlap if overlap is None else overlap
        if chunk_size in [0, None, False]:
            yield rasterio.windows.Window(0, 0, self.width, self.height)
            return

        def get_iter_windows(
            width,
            height,
            chunk_size=0,
            overlap=0,
            row_off=0,
            col_off=0
        ):
            h = chunk_size
            for i in range(
                int(row_off),
                int(row_off + height + chunk_size),
                chunk_size
                    ):
                if i + h > row_off + height:
                    h = height - i
                    if h <= 0:
                        break
                w = chunk_size
                for j in range(
                    int(col_off),
                    int(col_off + width + chunk_size),
                    chunk_size
                        ):
                    if j + w > col_off + width:
                        w = width - j
                        if w <= 0:
                            break
                    o = overlap
                    while j + w + o > width:
                        o -= 1
                    w += o
                    o = overlap
                    while i + h + o > height:
                        o -= 1
                    h += o
                    yield windows.Window(j, i, w, h)

        for window in get_iter_windows(
                self.width, self.height, chunk_size, overlap):
            yield window

    @property
    def x(self):
        return self.get_x()

    @property
    def y(self):
        return self.get_y()

    @property
    def values(self):
        return self.get_values()

    @property
    def path(self):
        return self._path

    @property
    def md5(self):
        hash_md5 = hashlib.md5()
        with open(self._tmpfile.resolve(), "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @property
    def count(self):
        return self._src.count

    @property
    def is_masked(self):
        for window in self.iter_windows(self.chunk_size):
            if self._src.nodata in self._src.read(window=window):
                return True
        return False

    @property
    def shape(self):
        return self._src.shape

    @property
    def height(self):
        return self._src.height

    @property
    def bbox(self):
        x0, y0, x1, y1 = rasterio.transform.array_bounds(
            self.height, self.width, self.transform)
        return Bbox([[x0, y0], [x1, y1]])

    @property
    def width(self):
        return self._src.width

    @property
    def dx(self):
        return self._src.transform[0]

    @property
    def dy(self):
        return -self._src.transform[4]

    @property
    def crs(self):
        return self._src.crs

    @property
    def srs(self):
        return self.proj.srs

    @property
    def proj(self):
        return Proj(self.crs)

    @property
    def nodatavals(self):
        return self._src.nodatavals

    @property
    def transform(self):
        return self._src.transform

    @property
    def dtypes(self):
        return self._src.dtypes

    @property
    def nodata(self):
        return self._src.nodata

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
    def chunk_size(self):
        return self.__chunk_size

    @property
    def overlap(self):
        return self.__overlap

    @chunk_size.setter
    def chunk_size(self, chunk_size):
        chunk_size = 0 if chunk_size is None else int(chunk_size)
        assert chunk_size >= 0, "chunk_size must be >= 0."
        self.__chunk_size = chunk_size

    @overlap.setter
    def overlap(self, overlap):
        overlap = 0 if overlap is None else overlap
        self.__overlap = overlap

    @resampling_method.setter
    def resampling_method(self, resampling_method):
        self.__resampling_method = self._resampling_methods[resampling_method]

    def _get_multipolygon_from_axes(self, ax):
        # extract linear_rings from plot
        linear_ring_collection = list()
        for path_collection in ax.collections:
            for path in path_collection.get_paths():
                polygons = path.to_polygons(closed_only=True)
                for linear_ring in polygons:
                    if linear_ring.shape[0] > 3:
                        linear_ring_collection.append(
                            LinearRing(linear_ring))
        if len(linear_ring_collection) > 1:
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
            multipolygon = MultiPolygon(polygon_collection)
        else:
            multipolygon = MultiPolygon(
                [Polygon(linear_ring_collection.pop())])
        return multipolygon

    def get_window_data(self, window, masked=True):
        x0, y0, x1, y1 = self.get_array_bounds(window)
        x = np.linspace(x0, x1, window.width)
        y = np.linspace(y1, y0, window.height)
        data = self._src.read(masked=masked, window=window)
        return x, y, data

    def get_array_bounds(self, window):
        return array_bounds(
            window.height,
            window.width,
            self.get_window_transform(window))

    def get_window_transform(self, window):
        if window is None:
            return
        return windows.transform(window, self.transform)

    @property
    def _path(self):
        return self.__path

    @property
    def _crs(self):
        return self.__crs

    @property
    def _tmpfile(self):
        try:
            return pathlib.Path(self.__tmpfile.name)
        except AttributeError:
            return self._path.resolve()
        # copy original file into a tmpfile
        # tmpfile = tempfile.NamedTemporaryFile(prefix=tmpdir)
        # with rasterio.open(self.path) as src:
        #     with rasterio.open(tmpfile.name, 'w', **src.meta.copy()) as dst:
        #         for window in self._get_iter_windows(
        #                 src.width, src.height, self.chunk_size):
        #             dst.write(src.read(window=window), window=window)
        # self.__tmpfile = 
        # return self.__tmpfile

    @property
    def _src(self):
        try:
            return self.__src
        except AttributeError:
            return rasterio.open(self._path)

    @property
    @lru_cache(maxsize=None)
    def _resampling_methods(self):
        self.__resampling_method = {
            'bilinear': Resampling.bilinear,
            'nearest': Resampling.nearest,
            'cubic': Resampling.cubic,
            'average': Resampling.average
        }

    @_path.setter
    def _path(self, path):
        self.__path = pathlib.Path(path)

    @_crs.setter
    def _crs(self, crs):
        if crs is None:  # return if CRS in file || raise if no CRS
            # check if CRS is in file
            with rasterio.open(self.path) as src:
                # If CRS not in file, raise. All Rasters objects must have a
                # defined CRS. Cannot work with undefined CRS.
                if src.crs is None:
                    msg = 'CRS not found in raster file. Must specify CRS.'
                    raise IOError(msg)
                else:
                    # return because crs will be read directly from file.
                    return
        else:
            raise NotImplementedError('Raster._crs.setter')
            # CRS is specified by user
        #     with rasterio.open(self.path) as src:
        #         kwargs = src.meta.copy()
        #         if crs is not None:
        #             kwargs.update({'crs': crs.srs})
        #         kwargs.update({'driver': 'GTiff'})
        #         with rasterio.open(out_h, 'w', **kwargs,) as dst:
        #             if read_mode == "array":
        #                 for i in range(1, in_h.count + 1):
        #                     dst.write_band(i, in_h.read(i))
        #             elif read_mode == 'block_windows':
        #                 for _, window in in_h.block_windows():
        #                     dst.write(
        #                         in_h.read(window=window),
        #                         window=window
        #                         )
        #             else:
        #                 raise NotImplementedError('Duck-type invalid')
        # self._tmpfile = tmpfile

    @_tmpfile.setter
    def _tmpfile(self, tmpfile):
        del(self._tmpfile)
        self.__src = rasterio.open(tmpfile.name)
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
