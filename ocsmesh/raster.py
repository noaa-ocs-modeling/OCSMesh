"""Wrapper for raster image functionalities commonly used for meshing

This module implements wrapper for basic functionalities of handling
raster files such as CRS transformations, resampling, clipping, 
extracting & overriding data, plotting, etc.
"""

import math
import hashlib
import logging
import multiprocessing
import os
import pathlib
import tempfile
from time import time
from typing import Union
from contextlib import contextmanager, ExitStack
import warnings

# from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pyproj import CRS, Transformer
import rasterio
import rasterio.mask
from rasterio import warp
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.transform import array_bounds
from rasterio import windows
from scipy.ndimage import gaussian_filter
from shapely import ops
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, box)

# from ocsmesh.geom import Geom
# from ocsmesh.hfun import Hfun
from ocsmesh import figures
from ocsmesh import utils

_logger = logging.getLogger(__name__)


tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/ocsmesh'))+'/'
os.makedirs(tmpdir, exist_ok=True)


class RasterPath:
    """Descriptor class for storing the path of original input raster
    """

    def __set__(self, obj, val: Union[str, os.PathLike]):
        obj.__dict__['path'] = pathlib.Path(val)

    def __get__(self, obj, val):
        return obj.__dict__['path']


class Crs:
    """Descriptor class for CRS of the input raster.

    If user specifies this value, the existing CRS of the input raster
    is ignored (if any). If the user doesn't provide this value and
    the raster itself doesn't have any CRS either it's an error.
    """

    def __set__(self, obj, val: Union[str, CRS, None]):

        # check if CRS is in file
        if val is None:
            with rasterio.open(obj.path) as src:
                # Raise if CRS not in file and the user did not provide a CRS.
                # All Rasters objects must have a defined CRS.
                # Program cannot operate with an undefined CRS.
                val = src.crs
                if val is None:
                    raise IOError(
                        'CRS not found in raster file. Must specify CRS.')
        # CRS is specified by user rewrite raster but add CRS to meta
        else:
            if isinstance(val, str):
                val = CRS.from_user_input(val)

            if not isinstance(val, CRS):
                raise TypeError(f'Argument crs must be of type {str} or {CRS},'
                                f' not type {type(val)}.')

            # create a temporary copy of the original file and update meta.
            with ExitStack() as stack:

                src = stack.enter_context(rasterio.open(obj.path))
                if obj.chunk_size is not None:
                    wins = get_iter_windows(
                        src.width, src.height, chunk_size=obj.chunk_size)
                else:
                    wins = [windows.Window(
                        0, 0, src.width, src.height)]

                dst = stack.enter_context(
                        obj.modifying_raster(crs=val, driver='GTiff'))
                for window in wins:
                    dst.write(src.read(window=window), window=window)


class TemporaryFile:
    """Descriptor class for storing tempfile object of working raster

    All the raster operations done using `Raster` object is written
    to disk. In order to avoid modifying the original input raster
    a the input data is first writton to a temporary file. This 
    temporary file is created using `TemporaryFile` to have auto
    cleanup capabities on object destruction.
    """

    def __set__(self, obj, val):
        obj.__dict__['tmpfile'] = val
        obj._src = rasterio.open(val.name)

    def __get__(self, obj, val) -> pathlib.Path:
        tmpfile = obj.__dict__.get('tmpfile')
        if tmpfile is None:
            return obj.path
        return pathlib.Path(tmpfile.name)


class SourceRaster:
    """Descriptor class for storing the raster dataset handler.

    `Raster` objects are manipulated using `rasterio` package. All
    the access to the raster data using this package happens through
    `rasterio.DatasetReader`. This descriptor holds onto the opened
    dataset to use as a shorthand of checking file existance and
    opening it everytime need arises.
    """

    def __get__(self, obj, val) -> rasterio.DatasetReader:
        source = obj.__dict__.get('source')
        if source is None:
            source = rasterio.open(obj.path)
            obj.__dict__['source'] = source
        return source

    def __set__(self, obj, val: rasterio.DatasetReader):
        obj.__dict__['source'] = val


class ChunkSize:
    """Descriptor class for storing the size for windowed operations
    """

    def __set__(self, obj, val):
        chunk_size = 0 if val is None else int(val)
        if not chunk_size >= 0:
            raise ValueError("Argument chunk_size must be >= 0.")
        obj.__dict__['chunk_size'] = val

    def __get__(self, obj, val):
        return obj.__dict__['chunk_size']


class Overlap:
    """Descriptor class for storing the overlap for windowed operations
    """

    def __set__(self, obj, val):
        obj.__dict__['overlap'] = 0 if val is None else val

    def __get__(self, obj, val):
        return obj.__dict__['overlap']


class Raster:
    """Wrapper class for basic raster handling

    This 

    Attributes
    ----------
    x
    y
    values
    path
    tmpfile
    md5
    count
    is_masked
    shape
    height
    bbox
    src
    width
    dx
    dy
    crs
    nodatavals
    transform
    dtypes
    nodata
    xres
    yres
    resampling_method
    chunk_size
    overlap
    
    Methods
    -------
    modifying_raster(use_src_meta=True, **kwargs)
        Context for modifying raster and saving to disk after.
    get_x(window=None)
        Get X values from the raster.
    get_y(window=None)
        Get Y values from the raster.
    get_xy(window=None)
        Get raster position tuples.
    get_values(window=None, band=None, **kwargs)
        Get values at all the points in the raster.
    get_xyz(window=None, band=None)
        Get raster position tuples and values horizontally stacked.
    get_multipolygon(zmin=None, zmax=None, window=None,
                     overlap=None, band=1)
        Extract multipolygon from raster data.
    get_bbox(crs=None, output_type=None)
        Get the raster bounding box.
    contourf(band=1, window=None, axes=None, vmin=None, vmax=None,
             cmap='topobathy', levels=None, show=False, title=None,
             figsize=None, colors=256, cbar_label=None, norm=None,
             **kwargs)
        Plot a filled contour from the raster data.
    tags(i=None)
        Get tags set on raster dataset.
    read(i, masked=True, **kwargs)
        Call underlying dataset `read` method.
    dtype(i)
        Return data type from the source raster.
    nodataval(i)
        Return the no-data value of the source raster.
    sample(xy, i)
        Call underlying dataset `sample` method.
    close()
        Release source raster object.
    add_band(values,  **tags)
        Add a new band of data with specified tags to the raster.
    fill_nodata()
        Fill no-data points in the raster dataset.
    gaussian_filter(**kwargs)
        Apply Gaussian filter on the raster data.
    mask(shapes, i=None, **kwargs)
        Mask raster data by shape.
    read_masks(i=None)
        Read source raster masks.
    warp(dst_crs, nprocs=-1)
        Reproject raster data.
    resample(scaling_factor, resampling_method=None)
        Resample raster data.
    save(path)
        Save as raster data to the provided path.
    clip(geom)
        Clip raster data by provided shape.
    adjust(geom, inside_min=0.5, outside_max=-0.5)
        Modify raster values based on constraints and shape.
    get_contour(level, window)
        Calculate contour of specified level on raster data.
    get_channels(level=0, width=1000, tolerance=None)
        Calculate narrow areas based on input level and width.
    iter_windows(chunk_size=None, overlap=None)
        Return raster view windows based on chunk size and overlap.
    get_window_data(window, masked=True, band=None)
        Return raster values based for the input window.
    get_window_bounds(window)
        Return window bounds.
    get_window_transform(window)
        Return window transform for the input raster

    Notes
    -----
    This class makes use of property and descriptor type attributes.
    Some of the properties even point to the value set by the 
    descriptors. It's important to not that the properties are
    set lazily. It's also noteworthy to mention that this class is
    currently **not** picklable due temporary file and file-like
    object attributes.
    """

    _path = RasterPath()
    _crs = Crs()
    _chunk_size = ChunkSize()
    _overlap = Overlap()
    _tmpfile = TemporaryFile()
    _src = SourceRaster()

    def __init__(
            self,
            path: Union[str, os.PathLike],
            crs: Union[str, CRS] = None,
            chunk_size=None,
            overlap=None
    ):
        """Raster data manipulator.

        Parameters
        ----------
        path : str or os.PathLike
            Path to a raster image to work on (.tiff or .nc).
        crs : str or CRS, default=None
            CRS to use and override input raster data with.
            Note that no transformation takes place.
        chunk_size : int , default=None
            Square window size to be used for data chunking
        overlap : int , default=None
            Overlap size for calculating chunking windows on the raster
        """

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._path = path
        self._crs = crs

    def __iter__(self, chunk_size=None, overlap=None):
        for window in self.iter_windows(chunk_size, overlap):
            yield window, self.get_window_bounds(window)

    @contextmanager
    def modifying_raster(self, use_src_meta=True, **kwargs):
        r"""Context manager for modifying and storing raster data

        This is a helper context manager method that handles creating
        new temporary file and replacing the old one with it when
        raster data is successfully modified.

        Parameters
        ----------
        use_src_meta : bool, default=True
            Whether or not to copy the metadata of the source raster 
            when creating the new empty raster file
        **kwargs : dict
            Options to be passed as metadata to raster database. These
            options override values taken from source raster in case
            `use_src_meta` is `True`

        Yields
        ------
        rasterio.DatasetReader
            Handle to the opened dataset on temporary file which
            will override the old values if no exception occurs
            during the context
        """

        no_except = False
        try:
            # pylint: disable=R1732
            tmpfile = tempfile.NamedTemporaryFile(prefix=tmpdir)

            new_meta = kwargs
            # Flag to workaround cases where "src" is NOT set yet
            if use_src_meta:
                new_meta = self.src.meta.copy()
                new_meta.update(**kwargs)
            with rasterio.open(tmpfile.name, 'w', **new_meta) as dst:
                yield dst

            no_except = True

        finally:
            if no_except:
                # So that tmpfile is NOT destroyed when it locally
                # goes out of scope
                self._tmpfile = tmpfile



    def get_x(self, window=None):
        """Get X positions of the raster grid.

        Parameters
        ----------
        window : windows.Window, default=None
            The window over which X positions are to be returned

        Returns
        -------
        np.ndarray
            A vector X positions from minimum to the maximum
        """

        window = windows.Window(0, 0, self.src.shape[1], self.src.shape[0]) \
            if window is None else window
        if window is not None:
            assert isinstance(window, windows.Window)
            width = window.width
        else:
            width = self.shape[0]
        x0, y0, x1, y1 = self.get_window_bounds(window)
        return np.linspace(x0, x1, width)

    def get_y(self, window=None):
        """Get Y positions of the raster grid.

        Parameters
        ----------
        window : windows.Window, default=None
            The window over which Y positions are to be returned

        Returns
        -------
        np.ndarray
            A vector Y positions from minimum to the maximum
        """

        window = windows.Window(0, 0, self.src.shape[1], self.src.shape[0]) \
            if window is None else window
        if window is not None:
            assert isinstance(window, windows.Window)
            height = window.height
        else:
            height = self.shape[1]
        x0, y0, x1, y1 = self.get_window_bounds(window)
        return np.linspace(y1, y0, height)

    def get_xy(self, window=None):
        """Get raster positions tuple array

        Parameters
        ----------
        window : windows.Window, default=None
            The window over which positions are to be returned

        Returns
        -------
        np.ndarray
            A `n` :math:`\times 2` matrix of positions
        """

        x, y = np.meshgrid(self.get_x(window), self.get_y(window))
        return np.vstack([x.flatten(), y.flatten()]).T

    def get_values(self, window=None, band=None, **kwargs):
        r"""Return the data stored at each point in the raster grid

        Parameters
        ----------
        window : windows.Window, default=None
            The window over which values are to be returned.
        band : int, default=None
            The band from which the values should be read. If `None`
            return data from band `1`.
        **kwargs : dict
            Additional arguments to pass to `rasterio.DatasetReader.read`.

        Returns
        -------
        np.ndarray
            A 2D matrix of values on the raster grid.
        """

        i = 1 if band is None else band
        window = windows.Window(0, 0, self.src.shape[1], self.src.shape[0]) \
            if window is None else window
        if window is not None:
            assert isinstance(window, windows.Window)
        return self.src.read(i, window=window, **kwargs)

    def get_xyz(self, window=None, band=None):
        """Return the data stored at each point in the raster grid

        Parameters
        ----------
        window : windows.Window, default=None
            The window over which positions and values are to be
            returned.
        band : int, default=None
            The band from which the values should be read. If `None`
            return data from band `1`.

        Returns
        -------
        np.ndarray
            A `n` :math:`\times 3` matrix of x, y and values
        """

        xy = self.get_xy(window)
        values = self.get_values(window=window, band=band).reshape(
            (xy.shape[0], 1))
        return np.hstack([xy, values])

    def get_multipolygon(
            self,
            zmin=None,
            zmax=None,
            window=None,
            overlap=None,
            band=1,
    ):
        """Calculate and return a multipolygon based on the raster data

        Calculates filled contour from raster data between specified
        limits on the specified band and creates a multipolygon from
        the filled contour to return.

        Parameters
        ----------
        zmin : float or None, default=None
            Lower bound of raster data for filled contour calculation
        zmax : float or None, default=None
            Upper bound of raster data for filled contour calculation
        window : windows.Window or None, default=None
            Window over whose data the multipolygon is calculated
        overlap : int or None, default=None
            Overlap used for generating windows if `window` is not provided
        band : int or None, default=1
            Raster band over whose data multipolygon is calculated

        Returns
        -------
        MultiPolygon
            The calculated multipolygon from raster data
        """

        polygon_collection = []
        if window is None:
            iter_windows = self.iter_windows(overlap=overlap)
        else:
            iter_windows = [window]

        for win in iter_windows:
            x, y, z = self.get_window_data(win, band=band)
            new_mask = np.full(z.mask.shape, 0)
            new_mask[np.where(z.mask)] = -1
            new_mask[np.where(~z.mask)] = 1

            if zmin is not None:
                new_mask[np.where(z < zmin)] = -1

            if zmax is not None:
                new_mask[np.where(z > zmax)] = -1

            if np.all(new_mask == -1):  # or not new_mask.any():
                continue

            fig, ax = plt.subplots()
            ax.contourf(x, y, new_mask, levels=[0, 1])
            plt.close(fig)
            polygon_collection.extend(
                    list(utils.get_multipolygon_from_pathplot(ax)))

        union_result = ops.unary_union(polygon_collection)
        if not isinstance(union_result, MultiPolygon):
            union_result = MultiPolygon([union_result])
        return union_result

    def get_bbox(
            self,
            crs: Union[str, CRS] = None,
            output_type: str = None
    ) -> Union[Polygon, Bbox]:
        output_type = 'polygon' if output_type is None else output_type
        xmin, xmax = np.min(self.x), np.max(self.x)
        ymin, ymax = np.min(self.y), np.max(self.y)
        crs = self.crs if crs is None else crs
        if crs is not None:
            if not self.crs.equals(crs):
                transformer = Transformer.from_crs(
                    self.crs, crs, always_xy=True)
                # pylint: disable=E0633
                (xmin, xmax), (ymin, ymax) = transformer.transform(
                    (xmin, xmax), (ymin, ymax))
        if output_type == 'polygon': # pylint: disable=R1705
            return box(xmin, ymin, xmax, ymax)
        elif output_type == 'bbox':
            return Bbox([[xmin, ymin], [xmax, ymax]])

        raise TypeError(
            'Argument output_type must a string literal \'polygon\' or '
            '\'bbox\'')

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
        cmap, norm, levels, col_val = figures.get_topobathy_kwargs(
            values, vmin, vmax)
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
            return self.src.tags()
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
        del self._src

    def add_band(self, values,  **tags):
        kwargs = self.src.meta.copy()
        band_id = kwargs["count"] + 1
        with self.modifying_raster(count=band_id) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
            dst.write_band(band_id, values.astype(self.src.dtypes[i-1]))
        return band_id

    def fill_nodata(self):
        """
        A parallelized version is presented here:
        https://github.com/basaks/rasterio/blob/master/examples/fill_large_raster.py
        """

        with self.modifying_raster() as dst:
            for window in self.iter_windows():
                dst.write(
                    fillnodata(self.src.read(window=window, masked=True)),
                    window=window
                    )

    def gaussian_filter(self, **kwargs):

        # TODO: Don't overwrite; add additoinal bands for filtered values

        # NOTE: Adding new bands in this function can result in issues
        # in other parts of the code. Thorough testing is needed for
        # modifying the raster (e.g. hfun add_contour is affected)
        with self.modifying_raster() as dst:
            for i in range(1, self.src.count + 1):
                outband = self.src.read(i)
#                # Write orignal band
#                dst.write_band(i + n_bands_new // 2, outband)
                # Write filtered band
                outband = gaussian_filter(outband, **kwargs)
                dst.write_band(i, outband)

    def mask(self, shapes, i=None, **kwargs):
        out_images, out_transform = rasterio.mask.mask(self._src, shapes)
        with self.modifying_raster(**kwargs) as dst:
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

    def read_masks(self, i=None):
        if i is None:
            return np.dstack(
                [self.src.read_masks(i) for i in range(1, self.count + 1)])

        return self.src.read_masks(i)

    def warp(self, dst_crs, nprocs=-1):
        nprocs = -1 if nprocs is None else nprocs
        nprocs = multiprocessing.cpu_count() if nprocs == -1 else nprocs
        dst_crs = CRS.from_user_input(dst_crs)
        transform, width, height = warp.calculate_default_transform(
            self.src.crs,
            dst_crs.srs,
            self.src.width,
            self.src.height,
            *self.src.bounds,
            dst_width=self.src.width,
            dst_height=self.src.height
            )

        meta_update = {
            'crs': dst_crs.srs,
            'transform': transform,
            'width': width,
            'height': height
        }
        with self.modifying_raster(**meta_update) as dst:
            for i in range(1, self.src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(self._src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=self.src.transform,
                    crs=self.src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs.srs,
                    resampling=self.resampling_method,
                    num_threads=nprocs,
                    )


    def resample(self, scaling_factor, resampling_method=None):
        if resampling_method is None:
            resampling_method = self.resampling_method
        else:
            msg = "resampling_method must be a valid name or None"
            raise ValueError(msg)

        # resample data to target shape
        width = int(self.src.width * scaling_factor)
        height = int(self.src.height * scaling_factor)
        data = self.src.read(
            out_shape=(
                self.src.count,
                height,
                width
            ),
            resampling=resampling_method
        )
        transform = self.src.transform * self.src.transform.scale(
            (self.src.width / data.shape[-1]),
            (self.src.height / data.shape[-2])
        )
        meta_update = {
            'transform': transform,
            'width': width,
            'height': height
            }
        with self.modifying_raster(**meta_update) as dst:
            dst.write(data)

    def save(self, path):
        with rasterio.open(pathlib.Path(path), 'w', **self.src.meta) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
                dst.update_tags(i, **self.src.tags(i))

    def clip(self, geom: Union[Polygon, MultiPolygon]):
        if isinstance(geom, Polygon):
            geom = MultiPolygon([geom])
        out_image, out_transform = rasterio.mask.mask(
            self.src, geom, crop=True)
        meta_update = {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
            }

        with self.modifying_raster(**meta_update) as dest:
            dest.write(out_image)


    def adjust(
            self,
            geom: Union[Polygon, MultiPolygon],
            inside_min=0.5,
            outside_max=-0.5):

        if isinstance(geom, Polygon):
            geom = MultiPolygon([geom])

        with self.modifying_raster(driver='GTiff') as dst:
            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)
            for i, window in enumerate(iter_windows):
                _logger.debug(f'Processing window {i+1}/{tot}.')

                # NOTE: We should NOT transform polygon, user just
                # needs to make sure input polygon has the same CRS
                # as the hfun (we don't calculate distances in this
                # method)

                _logger.info('Creating mask from shape ...')
                start = time()
                values = self.get_values(window=window).copy()
                mask = np.zeros_like(values)
                try:
                    mask, _, _ = rasterio.mask.raster_geometry_mask(
                        self.src, geom,
                        all_touched=True, invert=True)
                    mask = mask[rasterio.windows.window_index(window)]

                except ValueError:
                    # If there's no overlap between the raster and
                    # shapes then it throws ValueError, instead of
                    # checking for intersection, if there's a value
                    # error we assume there's no overlap
                    _logger.debug(
                        'Polygons don\'t intersect with the raster')

                _logger.info(
                    f'Creating mask from shape took {time()-start}.')

                if mask.any():
                    values[np.where(np.logical_and(
                            values < inside_min, mask)
                            )] = inside_min

                    values[np.where(np.logical_and(
                            values > outside_max, np.logical_not(mask))
                            )] = outside_max
                else:
                    values[values > outside_max] = outside_max

                _logger.info('Write array to file...')
                start = time()
                dst.write_band(1, values, window=window)
                _logger.info(f'Write array to file took {time()-start}.')


    def get_contour(
            self,
            level: float,
            window: rasterio.windows.Window = None
    ):
        _logger.debug(
            f'RasterHfun.get_raster_contours(level={level}, window={window})')
        if window is None:
            iter_windows = list(self.iter_windows())
        else:
            iter_windows = [window]
        if len(iter_windows) > 1:
            return self._get_raster_contour_feathered(level, iter_windows)

        return self._get_raster_contour_windowed(level, window)

    def get_channels(
            self,
            level: float = 0,
            width: float = 1000, # in meters
            tolerance: Union[None, float] = None
    ):

        multipoly = self.get_multipolygon(zmax=level)

        utm_crs = utils.estimate_bounds_utm(
            self.get_bbox().bounds, self.crs)

        if utm_crs is not None:
            transformer = Transformer.from_crs(
                self.src.crs, utm_crs, always_xy=True)
            multipoly = ops.transform(transformer.transform, multipoly)
        channels = utils.get_polygon_channels(
                multipoly, width, simplify=tolerance)
        if channels is None:
            return None

        if utm_crs is not None:
            transformer = Transformer.from_crs(
                utm_crs, self.src.crs, always_xy=True)
            channels = ops.transform(transformer.transform, channels)

        return channels

    def _get_raster_contour_windowed(self, level, window):
        x, y = self.get_x(), self.get_y()
        features = []
        values = self.get_values(band=1, window=window)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            _logger.debug('Computing contours...')
            start = time()
            fig, ax = plt.subplots()
            ax.contour(x, y, values, levels=[level])
            _logger.debug(f'Took {time()-start}...')
            plt.close(fig)
        for path_collection in ax.collections:
            for path in path_collection.get_paths():
                try:
                    features.append(LineString(path.vertices))
                except ValueError:
                    # LineStrings must have at least 2 coordinate tuples
                    pass
        return ops.linemerge(features)

    def _get_raster_contour_feathered(self, level, iter_windows):

        with tempfile.TemporaryDirectory(dir=tmpdir) as feather_dir:
            results = self._get_raster_contour_feathered_internal(
                    level, iter_windows, feather_dir)
        return results

    def _get_raster_contour_feathered_internal(self, level, iter_windows, temp_dir):
        feathers = []
        total_windows = len(iter_windows)
        _logger.debug(f'Total windows to process: {total_windows}.')
        for i, window in enumerate(iter_windows):
            x, y = self.get_x(window), self.get_y(window)
            _logger.debug(f'Processing window {i+1}/{total_windows}.')
            features = []
            values = self.get_values(band=1, window=window)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                _logger.debug('Computing contours...')
                start = time()
                fig, ax = plt.subplots()
                ax.contour(x, y, values, levels=[level])
                _logger.debug(f'Took {time()-start}...')
                plt.close(fig)
            for path_collection in ax.collections:
                for path in path_collection.get_paths():
                    try:
                        features.append(LineString(path.vertices))
                    except ValueError:
                        # LineStrings must have at least 2 coordinate tuples
                        pass
            if len(features) > 0:
                tmpfile = os.path.join(temp_dir, f'file_{i}.feather')
                _logger.debug('Saving feather.')
                features = ops.linemerge(features)
                gpd.GeoDataFrame(
                    [{'geometry': features}]
                    ).to_feather(tmpfile)
                feathers.append(tmpfile)
        _logger.debug('Concatenating feathers.')
        features = []
        out = gpd.GeoDataFrame()
        for feather in feathers:
            out = out.append(gpd.read_feather(feather), ignore_index=True)
            feather.unlink()
            geometry = []
            for geom in out.geometry:
                if isinstance(geom, LineString):
                    geometry = MultiLineString([geom])
                    break
            for linestring in geometry:
                features.append(linestring)
        _logger.debug('Merging features.')
        return ops.linemerge(features)

    def iter_windows(self, chunk_size=None, overlap=None):
        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        overlap = self.overlap if overlap is None else overlap
        if chunk_size in [0, None, False]:
            yield rasterio.windows.Window(0, 0, self.width, self.height)
            return

        for window in get_iter_windows(
                self.width, self.height, chunk_size, overlap):
            yield window

    def get_window_data(self, window, masked=True, band=None):
        x0, y0, x1, y1 = self.get_window_bounds(window)
        x = np.linspace(x0, x1, window.width)
        y = np.linspace(y1, y0, window.height)
        if band is not None:
            data = self.src.read(band, masked=masked, window=window)
        else:
            data = self.src.read(masked=masked, window=window)
        return x, y, data

    def get_window_bounds(self, window):
        return array_bounds(
            window.height,
            window.width,
            self.get_window_transform(window))

    def get_window_transform(self, window):
        if window is None:
            return None
        return windows.transform(window, self.transform)

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
    def tmpfile(self):
        return self._tmpfile

    @property
    def md5(self):
        hash_md5 = hashlib.md5()
        with open(self._tmpfile.resolve(), "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @property
    def count(self):
        return self.src.count

    @property
    def is_masked(self):
        for window in self.iter_windows(self.chunk_size):
            if self.src.nodata in self.src.read(window=window):
                return True
        return False

    @property
    def shape(self):
        return self.src.shape

    @property
    def height(self):
        return self.src.height

    @property
    def bbox(self):
        return self.get_bbox()

    @property
    def src(self):
        return self._src

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
    def crs(self) -> CRS:
        # cast rasterio.CRS to pyproj.CRS for API consistency
        return CRS.from_user_input(self.src.crs)

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
    def nodata(self):
        return self.src.nodata

    @property
    def xres(self):
        return self.transform[0]

    @property
    def yres(self):
        return self.transform[4]

    @property
    def resampling_method(self):
        if not hasattr(self, '_resampling_method'):
            self._resampling_method = Resampling.nearest
        return self._resampling_method

    @resampling_method.setter
    def resampling_method(self, resampling_method):
        if not isinstance(resampling_method, Resampling):
            TypeError(
                f'Argument resampling_method must be of type  {Resampling}, '
                f'not type {type(resampling_method)}.')
        self._resampling_method = resampling_method

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size):
        self._chunk_size = chunk_size

    @property
    def overlap(self):
        return self._overlap

    @overlap.setter
    def overlap(self, overlap):
        self._overlap = overlap


def get_iter_windows(
        width,
        height,
        chunk_size=0,
        overlap=0,
        row_off=0,
        col_off=0
):
    win_h = chunk_size + overlap
    win_w = chunk_size + overlap
    n_win_h = math.ceil(height / chunk_size)
    n_win_w = math.ceil(width / chunk_size)
    for i in range(n_win_h):
        for j in range(n_win_w):
            off_h = i * chunk_size
            off_w = j * chunk_size
            h = chunk_size + overlap
            h = h - (off_h + h) % height if off_h + h > height else h
            w = chunk_size + overlap
            w = w - (off_w + w) % width if off_w + w > width else w
            yield windows.Window(off_w, off_h, w, h)


def redistribute_vertices(geom, distance):
    if geom.geom_type == 'LineString': # pylint: disable=R1705
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])

    raise ValueError(f'unhandled geometry {geom.geom_type}')
