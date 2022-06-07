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
import warnings
from time import time
from contextlib import contextmanager, ExitStack
from typing import (
        Union, Generator, Any, Optional, Dict, List, Tuple, Iterable,
        Callable)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.typing as npt
from numpy import ma
from pyproj import CRS, Transformer
import rasterio
import rasterio.mask
from rasterio import warp, Affine
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.transform import array_bounds
from rasterio import windows
from scipy.ndimage import gaussian_filter, generic_filter
from scipy import LowLevelCallable
from shapely import ops
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, box)
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer

from ocsmesh import figures
from ocsmesh import utils

_logger = logging.getLogger(__name__)


tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/ocsmesh'))+'/'
os.makedirs(tmpdir, exist_ok=True)


# From https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/
@cfunc(intc(CPointer(float64), intp,
            CPointer(float64), voidptr))
def nbmean(values_ptr, len_values, result, data):
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = 0
    n_vals = 0
    for v in values:
        if not np.isnan(v):
            result[0] = result[0] + v
            n_vals = n_vals + 1
    # NOTE: Divide by number of used values, NOT the whole foot area
    if n_vals > 1:
        result[0] = result[0] / n_vals

    return 1


class RasterPath:
    """Descriptor class for storing the path of original input raster
    """

    def __set__(self, obj, val: Union[str, os.PathLike]):
        obj.__dict__['path'] = pathlib.Path(val)

    def __get__(self, obj, objtype=None):
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

    def __set__(self, obj, val: tempfile.NamedTemporaryFile):
        obj.__dict__['tmpfile'] = val
        obj._src = rasterio.open(val.name)

    def __get__(self, obj, objtype=None) -> pathlib.Path:
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

    def __set__(self, obj, val: rasterio.DatasetReader):
        obj.__dict__['source'] = val

    def __get__(self, obj, objtype=None) -> rasterio.DatasetReader:
        source = obj.__dict__.get('source')
        if source is None:
            source = rasterio.open(obj.path)
            obj.__dict__['source'] = source
        return source


class ChunkSize:
    """Descriptor class for storing the size for windowed operations
    """

    def __set__(self, obj, val: int):
        chunk_size = 0 if val is None else int(val)
        if not chunk_size >= 0:
            raise ValueError("Argument chunk_size must be >= 0.")
        obj.__dict__['chunk_size'] = val

    def __get__(self, obj, objtype=None) -> int:
        return obj.__dict__['chunk_size']


class Overlap:
    """Descriptor class for storing the overlap for windowed operations
    """

    def __set__(self, obj, val: int):
        obj.__dict__['overlap'] = 0 if val is None else val

    def __get__(self, obj, objtype=None) -> int:
        return obj.__dict__['overlap']


class Raster:
    """Wrapper class for basic raster handling

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
    get_multipolygon(zmin=None, zmax=None, window=None, overlap=None, band=1)
        Extract multipolygon from raster data.
    get_bbox(crs=None, output_type='polygon')
        Get the raster bounding box.
    contourf(...)
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
    average_filter(size, drop_above, drop_below)
        Apply average filter on the raster data.
    generic_filter(function, **kwargs)
        Apply generic filter on the raster data.
    mask(shapes, i=None, **kwargs)
        Mask raster data by shape.
    read_masks(i=None)
        Read source raster masks.
    warp(dst_crs, nprocs=-1)
        Reproject raster data.
    resample(scaling_factor, resampling_method=None)
        Resample raster data.
    save(path)
        Save-as raster data to the provided path.
    clip(geom)
        Clip raster data by provided shape.
    adjust(geom=None, inside_min=-np.inf, outside_max=np.inf, cond=None)
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
            crs: Union[str, CRS, None] = None,
            chunk_size: Optional[int] = None,
            overlap: Optional[int] = None
    ):
        """Raster data manipulator.

        Parameters
        ----------
        path : str or os.PathLike
            Path to a raster image to work on (.tiff or .nc).
        crs : str or CRS, default=None
            CRS to use and override input raster data with.
            Note that no transformation takes place.
        chunk_size : int or None, default=None
            Square window size to be used for data chunking
        overlap : int or None , default=None
            Overlap size for calculating chunking windows on the raster
        """

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._path = path
        self._crs = crs

    def __iter__(self, chunk_size: int = None, overlap: int = None):
        for window in self.iter_windows(chunk_size, overlap):
            yield window, self.get_window_bounds(window)

    @contextmanager
    def modifying_raster(
            self,
            use_src_meta: bool = True,
            **kwargs: Any
            ) -> Generator[rasterio.DatasetReader, None, None]:
        r"""Context manager for modifying and storing raster data

        This is a helper context manager method that handles creating
        new temporary file and replacing the old one with it when
        raster data is successfully modified.

        Parameters
        ----------
        use_src_meta : bool, default=True
            Whether or not to copy the metadata of the source raster
            when creating the new empty raster file
        **kwargs : dict, optional
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
                if use_src_meta:
                    for i, desc in enumerate(self.src.descriptions):
                        dst.set_band_description(i+1, desc)
                yield dst

            no_except = True

        finally:
            if no_except:
                # So that tmpfile is NOT destroyed when it locally
                # goes out of scope
                self._tmpfile = tmpfile



    def get_x(
            self,
            window: Optional[windows.Window] = None
            ) -> npt.NDArray[float]:
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

    def get_y(
            self,
            window: Optional[windows.Window] = None
            ) -> npt.NDArray[float]:
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

    def get_xy(
            self,
            window: Optional[windows.Window] = None
            ) -> np.ndarray:
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

    def get_values(
            self,
            window: Optional[windows.Window] = None,
            band: Optional[int] = None,
            **kwargs: Any
            ) -> npt.NDArray[float]:
        r"""Return the data stored at each point in the raster grid

        Parameters
        ----------
        window : windows.Window, default=None
            The window over which values are to be returned.
        band : int, default=None
            The band from which the values should be read. If `None`
            return data from band `1`.
        **kwargs : dict, optional
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

    def get_xyz(
            self,
            window: Optional[windows.Window] = None,
            band: Optional[int] = None
            ) -> np.ndarray:
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
            zmin: Optional[float] = None,
            zmax: Optional[float] = None,
            window: Optional[windows.Window] = None,
            overlap: Optional[int] = None,
            band: int = 1,
    ) -> MultiPolygon:
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
        band : int, default=1
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
                    utils.get_multipolygon_from_pathplot(ax).geoms)

        union_result = ops.unary_union(polygon_collection)
        if not isinstance(union_result, MultiPolygon):
            union_result = MultiPolygon([union_result])
        return union_result

    def get_bbox(
            self,
            crs: Union[str, CRS, None] = None,
            output_type: Literal['polygon', 'bbox'] = 'polygon'
        ) -> Union[Polygon, Bbox]:
        """Calculate the bounding box of the raster.

        Parameters
        ----------
        crs : str or CRS or None, default=None
            The CRS in which the bounding box is requested.
        output_type : {'polygon', 'bbox'}
            The label of the return type for bounding box, either a
            `shapely` 'polygon' or `matplotlib` 'bbox'.

        Returns
        -------
        box or Bbox
            The bounding box of the raster.

        Raises
        ------
        TypeError
            If the label of return type is not valid.
        """

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
            band: int = 1,
            window: Optional[windows.Window] = None,
            axes: Optional[Axes] = None,
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            cmap: str = 'topobathy',
            levels: Optional[List[float]] = None,
            show: bool = False,
            title: Optional[str] = None,
            figsize: Optional[Tuple[float, float]] = None,
            colors: int = 256,
            cbar_label: Optional[str] = None,
            norm=None,
            **kwargs : Any
            ) -> Axes:
        """Plot filled contour for raster data.

        Parameters
        ----------
            band : int, default=1
                Raster band from which data is used.
            window : windows.Window or None, default=None
                Raster window from which data is used.
            axes : Axes or None, default=None
                Matplotlib axes to draw contour on>
            vmin : float or None, default=None
                Minimum value of the filled contour.
            vmax : float or None, default=None
                Maximum value of the filled contour.
            cmap : str, default='topobathy'
                Colormap to use for filled contour.
            levels : list of float or None, default=None
                Prespecified list of contour levels.
            show : bool, default=False
                Whether to show the contour on creation or not.
            title : str or None, default=None
                Title used on the axes of the contour
            figsize : tuple of float or None, default=None
                Figure size used for the contour figure
            colors : int, default=256
                Contour colors associated with levels
            cbar_label : str or None, default=None
                Label of the colorbar
            norm : Normalize or None, default=None
                Normalizer object
            **kwargs : dict, optional
                Keyword arguments passed to the matplotlib contourf()
                function

        Returns
        -------
        Axes
            Axes object from matplotlib library that holds onto the
            contour plot object
        """

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

    def tags(self, i: Optional[int] = None) -> Dict[str, str]:
        """Return a dictionary of dataset or band's tags

        Parameters
        ----------
        i : int or None, default=None
            The band from which the tags are read.

        Returns
        -------
        dict
            Dictionary of tags
        """

        if i is None:
            return self.src.tags()
        return self.src.tags(i)

    def read(self, i: int, masked: bool = True, **kwargs: Any) -> npt.NDArray[float]:
        """Read the data from raster opened file

        Parameters
        ----------
        i : int
            The index of the band to read the data from
        masked : bool, default=True
            Whether or not to return a masked array
        **kwargs : dict, optional
            Additional keyword arguments passed to rasterio read()

        Returns
        -------
        ndarray
            Array of raster data
        """

        return self.src.read(i, masked=masked, **kwargs)

    def dtype(self, i: int) -> npt.NDArray[float]:
        """Raster data type

        Parameters
        ----------
        i : int
            The index of the band to read the data from

        Returns
        -------
        Any
            Data type of raster values
        """

        return self.src.dtypes[i-1]

    def nodataval(self, i: int) -> float:
        """Value used for filling no-data points

        Parameters
        ----------
        i : int
            The index of the band to read the data from

        Returns
        -------
        float
            The value to be used for points that have missing data
        """

        return self.src.nodatavals[i-1]

    def sample(self, xy: Iterable, i: int) -> npt.NDArray[float]:
        """Get value of the data in specified positions

        Parameters
        ----------
        xy : iterable
            Pairs of xy coordinates for which data is retrieved
        i : int
            The index of the band to read the data from

        Returns
        -------
        ndarray
            Array of values for specified input positions
        """

        return self.src.sample(xy, i)

    def close(self) -> None:
        """Delete source object"""

        del self._src

    def add_band(self, values: npt.NDArray[float], **tags: Any) -> int:
        """Add a new band for `values` with tags `tags` to the raster

        Parameters
        ----------
            values : array-like
                The values to be added to the raster, it must have the
                correct shape as the raster.
            **tags : dict, optional
                The tags to be added for the new band of data.

        Returns
        -------
        int
            ID of the new band added to the raster.
        """

        kwargs = self.src.meta.copy()
        band_id = kwargs["count"] + 1
        with self.modifying_raster(count=band_id) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
            dst.write_band(band_id, values.astype(self.src.dtypes[i-1]))
        return band_id

    def fill_nodata(self) -> None:
        """Fill missing values in the raster in-place"""

        # A parallelized version is presented here:
        # https://github.com/basaks/rasterio/blob/master/examples/fill_large_raster.py

        with self.modifying_raster() as dst:
            for window in self.iter_windows():
                dst.write(
                    fillnodata(self.src.read(window=window, masked=True)),
                    window=window
                    )

    def gaussian_filter(self, **kwargs: Any) -> None:
        """Apply Gaussian filter to the raster data in-place

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments passed to SciPy `gaussian_filter` function

        Returns
        -------
        None
        """

        # TODO: Don't overwrite; add additoinal bands for filtered values

        # NOTE: Adding new bands in this function can result in issues
        # in other parts of the code. Thorough testing is needed for
        # modifying the raster (e.g. hfun add_contour is affected)
        with self.modifying_raster() as dst:
            for i in range(1, self.src.count + 1):
                outband = self.src.read(i, masked=True)
#                # Write orignal band
#                dst.write_band(i + n_bands_new // 2, outband)
                # Write filtered band
                outband = gaussian_filter(outband, **kwargs)
                dst.write_band(i, outband)

    def average_filter(
            self,
            size: Union[int, npt.NDArray[int]],
            drop_above: Optional[float] = None,
            drop_below: Optional[float] = None,
            apply_on_bands: Optional[List[int]] = None
            ) -> None:
        """Apply average(mean) filter on the raster

        Parameters
        ----------
        size: int, npt.NDArray[int]
            size of the footprint
        drop_above: float or None
            elevation above which the cells are ignored for averaging
        drop_below: float or None
            elevation below which the cells are ignored for averaging

        Returns
        -------
        None
        """

        # TODO: Don't overwrite; add additoinal bands for filtered values

        # NOTE: Adding new bands in this function can result in issues
        # in other parts of the code. Thorough testing is needed for
        # modifying the raster (e.g. hfun add_contour is affected)

        bands = apply_on_bands
        if bands is None:
            bands = range(1, self.src.count + 1)
        with self.modifying_raster() as dst:
            for i in bands:
                bnd_idx = i - 1
                outband = self.src.read(i, masked=True)
                mask = outband.mask.copy()

#                # Write orignal band
#                dst.write_band(i + n_bands_new // 2, outband)

                # Make the values out of range of interest to be nan
                # so that they are ignored in filtering
                if drop_above is not None:
                    mask_above = ma.getdata(outband) > drop_above
                    outband[mask_above] = np.nan
                if drop_below is not None:
                    mask_below = ma.getdata(outband) < drop_below
                    outband[mask_below] = np.nan

                outband_new = generic_filter(
                    outband, LowLevelCallable(nbmean.ctypes), size=size)

                # Mask raster based on result of filter?
                if drop_above is not None:
                    mask_above = ma.getdata(outband) > drop_above
                    mask = np.logical_or(mask, mask_above)
                if drop_below is not None:
                    mask_below = ma.getdata(outband) < drop_below
                    mask = np.logical_or(mask, mask_below)

                outband_new[mask] = dst.nodatavals[bnd_idx]
                outband_ma = ma.masked_array(outband_new, mask=mask)
                dst.write_band(i, outband_ma)


    def generic_filter(self, function, **kwargs: Any) -> None:
        """Apply Gaussian filter to the raster data in-place

        Parameters
        ----------
        function: callable, LowLevelCallable
            Function to be used on the footprint array
        **kwargs : dict, optional
            Keyword arguments passed to SciPy `generic_filter` function

        Returns
        -------
        None
        """

        # TODO: Don't overwrite; add additoinal bands for filtered values

        # NOTE: Adding new bands in this function can result in issues
        # in other parts of the code. Thorough testing is needed for
        # modifying the raster (e.g. hfun add_contour is affected)
        with self.modifying_raster() as dst:
            for i in range(1, self.src.count + 1):
                outband = self.src.read(i, masked=True)
#                # Write orignal band
#                dst.write_band(i + n_bands_new // 2, outband)
                # Write filtered band
                outband = generic_filter(outband, function, **kwargs)
                dst.write_band(i, outband)


    def mask(self,
             shapes: Iterable,
             i: Optional[int] = None,
             **kwargs: Any
             ) -> None:
        """Mask data based on input shapes in-place

        Parameters
        ----------
        shapes : iterable
            List of GeoJSON like dict or objects that implement Python
            geo interface protocol (passed to `rasterio.mask.mask`).
        i : int or None, default=None
            The index of the band to read the data from.
        **kwargs : dict, optional
            Keyword arguments used to create new raster Dataset.

        Returns
        -------
        None
        """

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

    def read_masks(self, i: Optional[int] = None) -> npt.NDArray[bool]:
        """Read existing masks on the raster data

        Parameters
        ----------
        i : int or None, default=None

        Returns
        -------
        np.ndarray or view
            Raster band mask from the dataset
        """

        if i is None:
            return np.dstack(
                [self.src.read_masks(i) for i in range(1, self.count + 1)])

        return self.src.read_masks(i)

    def warp(self,
             dst_crs: Union[CRS, str],
             nprocs: int = -1
             ) -> None:
        """Reproject the raster data to specified `dst_crs` in-place

        Parameters
        ----------
        dst_crs : CRS or str
            Destination CRS to which raster must be transformed
        nprocs : int, default=-1
            Number of processors to use for the operation

        Returns
        -------
        None
        """

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


    def resample(self,
                 scaling_factor: float,
                 resampling_method: Optional[str] = None
                 ) -> None:
        """Resample raster data in-place based on a scaling factor

        Parameters
        ----------
        scaling_factor : float
            The scaling factor to use for resampling data
        resampling_method : str or None, default=None
            Name of the resampling method passed to
            `rasterio.DatasetReader.read`

        Returns
        -------
        None
        """

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

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save-as raster dataset to a new location

        Parameters
        ----------
        path : str or path-like
            The path to which raster must be saved.

        Returns
        -------
        None
        """

        with rasterio.open(pathlib.Path(path), 'w', **self.src.meta) as dst:
            for i in range(1, self.src.count + 1):
                dst.write_band(i, self.src.read(i))
                dst.update_tags(i, **self.src.tags(i))

    def clip(self, geom: Union[Polygon, MultiPolygon]) -> None:
        """Clip raster data in-place, outside the specified shape.

        Parameters
        ----------
        geom : Polygon or MultiPolygon
            Shape used to clip the raster data

        Returns
        -------
        None
        """

        if isinstance(geom, Polygon):
            geom = MultiPolygon([geom])
        out_image, out_transform = rasterio.mask.mask(
            self.src, geom.geoms, crop=True)
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
            geom: Union[None, Polygon, MultiPolygon] = None,
            inside_min: float = -np.inf,
            outside_max: float = np.inf,
            cond: Optional[Callable[[npt.NDArray[float]], npt.NDArray[bool]]] = None
            ) -> None:
        """Adjust raster data in-place based on specified shape.

        This method can be used to adjust e.g. raster elevation values
        based on a more accurate land-mass polygon.

        Parameters
        ----------
        geom : None or Polygon or MultiPolygon
            Filled shape to determine which points are considered
            inside or outside (usually land-mass polygon)
        inside_min : float
            The minimum value to truncate raster data that falls
            inside the specified `geom` shape
        outside_max : float
            The maximum value to truncate raster data that falls
            outside the specified `geom` shape

        Returns
        -------
        None
        """

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

                start = time()
                values = self.get_values(window=window, masked=True).copy()
                mask = np.zeros_like(values)
                if geom is not None:
                    _logger.info('Creating mask from shape ...')
                    try:
                        mask, _, _ = rasterio.mask.raster_geometry_mask(
                            self.src, geom.geoms,
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

                if cond:
                    if geom is None:
                        mask = cond(values)
                    else:
                        mask = np.logical_and(mask, cond(values))

                if cond is None and geom is None:
                    raise ValueError(
                        "Neither shape nor condition are provided for adjustment!")

                if mask.any():
                    in_mask = values.mask.copy()
                    values[np.where(np.logical_and(
                            values < inside_min, mask)
                            )] = inside_min
                    values[np.where(np.logical_and(
                            values > outside_max, np.logical_not(mask))
                            )] = outside_max
                    values[in_mask] = dst.nodata
                else:
                    values[values > outside_max] = outside_max

                _logger.info('Write array to file...')
                start = time()
                dst.write_band(1, values, window=window)

                _logger.info(f'Write array to file took {time()-start}.')


    def get_contour(
            self,
            level: float,
            window: Optional[windows.Window] = None
            ) -> Union[LineString, MultiLineString]:
        """Calculate contour lines for specified data level.

        This method can be used e.g. to calculated coastline based on
        raster data.

        Parameters
        ----------
        level : float
            The level for which contour lines must be calculated
        window : windows.Window or None
            The raster window for which contour lines must be calculated

        Returns
        -------
        LineString or MultiLineString
            The contour lines calculated for the specified level
        """

        _logger.debug(
            f'RasterHfun.get_raster_contours(level={level}, window={window})')
        if window is None:
            iter_windows = list(self.iter_windows())
        else:
            iter_windows = [window]
        if len(iter_windows) > 1:
            return self._get_raster_contour_feathered(level, iter_windows)

        return self._get_raster_contour_single_window(level, window)

    def get_channels(
            self,
            level: float = 0,
            width: float = 1000, # in meters
            tolerance: Optional[float] = None
            ) -> Union[Polygon, MultiPolygon]:
        """Calculate narrow width polygons based on specified input

        By using `buffer` functionality this method finds narrow
        regions of the domain. The `level` specifies at which data
        level domain polygon should be calculated and `width`
        describes the narrow region cut-off. `tolerance` is used
        for simplifying the polygon before buffering it to reduce
        computational cost.

        Parameters
        ----------
        level : float, default=0
            Reference level to calculate domain polygon for narrow
            region calculation.
        width : float, default=1000
            Cut-off used for designating narrow regions.
        tolerance : float or None, default=None
            Tolerance used for simplifying domain polygon.

        Returns
        -------
        Polygon or MultiPolygon
            The calculated narrow regions based on raster data
        """

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

    def _get_raster_contour_single_window(
            self,
            level: float,
            window: windows.Window
            ) -> Union[LineString, MultiLineString]:
        """Calculate contour on raster data for a single window

        Parameters
        ----------
        level : float
            The level for which contour lines must be calculated
        window : windows.Window or None
            The raster window for which contour lines must be calculated

        Returns
        -------
        LineString or MultiLineString
            The contour lines calculated for the specified level
        """

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

    def _get_raster_contour_feathered(
            self,
            level : float,
            iter_windows : Iterable[windows.Window]
            ) -> Union[LineString, MultiLineString]:
        """Wrapper to calculate contour on raster data for a list of windows.

        Parameters
        ----------
        level : float
            The level for which contour lines must be calculated
        iter_windows : iterable
            Sequence of raster windows to calculated the contour lines
            on

        Returns
        -------
        LineString or MultiLineString
            The contour lines calculated for the specified level

        Notes
        -----
        This method calculates contour for each window and then merges
        the results. This private method is a wrapper to the
        method that actually computes the contours.
        """

        with tempfile.TemporaryDirectory(dir=tmpdir) as feather_dir:
            results = self._get_raster_contour_feathered_internal(
                    level, iter_windows, feather_dir)
        return results

    def _get_raster_contour_feathered_internal(
            self,
            level : float,
            iter_windows : Iterable[windows.Window],
            temp_dir : str
            ) -> Union[LineString, MultiLineString]:
        """Calculate contour on raster data for a list of windows.

        Parameters
        ----------
        level : float
            The level for which contour lines must be calculated
        iter_windows : iterable
            Sequence of raster windows to calculated the contour lines
            on
        temp_dir : str
            Path to the temporary directory used for storing feather
            files for each raster window before combining the results

        Returns
        -------
        LineString or MultiLineString
            The contour lines calculated for the specified level

        Notes
        -----
        This method calculates contour for each window and offloads it
        to disk to conserve memory. When all windows are processed,
        `geopandas` out of core method is used to merge the results
        into a `LineString` or `MultiLineString` on the memory
        """

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

    def iter_windows(
            self,
            chunk_size: Optional[int] = None,
            overlap: Optional[int] = None
            ) -> Generator[windows.Window, None, None]:
        """Calculates sequence of windows for the raster

        This method calculates the sequence of square windows for
        the raster based on the provided `chunk_size` and `overlap`.

        Parameters
        ----------
        chunk_size : int or None , default=None
            Square window size to be used for data chunking
        overlap : int or None , default=None
            Overlap size for calculating chunking windows on the raster

        Yields
        ------
        windows.Window
            Calculated square window on raster based on the window
            size and windows overlap values.
        """

        chunk_size = self.chunk_size if chunk_size is None else chunk_size
        overlap = self.overlap if overlap is None else overlap
        if chunk_size in [0, None, False]:
            yield rasterio.windows.Window(0, 0, self.width, self.height)
            return

        for window in get_iter_windows(
                self.width, self.height, chunk_size, overlap):
            yield window

    def get_window_data(
            self,
            window : windows.Window,
            masked : bool = True,
            band : Optional[int] = None
            ) -> Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
        """Return the positions and values of raster data for the window

        Paramaters
        ----------
        window : windows.Window
            The window for which data is to be returned.
        masked : bool, default=True
            Whether to return a masked array in case of missing data.
        band : int or None, default=None
            The ID of the band to read the data from.

        Returns
        -------
        tuple of np.ndarray
            The triple of x-ticks, y-ticks and data on the raster grid
        """

        x0, y0, x1, y1 = self.get_window_bounds(window)
        x = np.linspace(x0, x1, window.width)
        y = np.linspace(y1, y0, window.height)
        if band is not None:
            data = self.src.read(band, masked=masked, window=window)
        else:
            data = self.src.read(masked=masked, window=window)
        return x, y, data

    def get_window_bounds(
            self,
            window : windows.Window
            ) -> Tuple[int, int, int, int]:
        """Returns west, south, east, north bounds of the window

        Paramaters
        ----------
        window : windows.Window
            The window for which bounds are calculated

        Returns
        -------
        tuple of int
            West, south, east, north bounds of the window
        """

        return array_bounds(
            window.height,
            window.width,
            self.get_window_transform(window))

    def get_window_transform(
            self,
            window : windows.Window
            ) -> Union[None, Affine]:
        """Returns raster's affine transform calculated for the window

        Paramaters
        ----------
        window : windows.Window
            The window for which the affine transform is calculated

        Returns
        -------
        Affine
            Affine transform matrix for specified window
        """

        if window is None:
            return None
        return windows.transform(window, self.transform)

    @property
    def x(self) -> npt.NDArray[float]:
        """Read-only attribute for the x-ticks of raster grid

        This is a read-only property that returns the same results as
        `get_x` method
        """

        return self.get_x()

    @property
    def y(self) -> npt.NDArray[float]:
        """Read-only attribute for the y-ticks of raster grid

        This is a read-only property that returns the same results as
        `get_y` method
        """

        return self.get_y()

    @property
    def values(self) -> npt.NDArray[float]:
        """Read-only attribute for the raster grid data

        This is a read-only property that returns the same results as
        `get_values` method
        """

        return self.get_values()

    @property
    def path(self) -> pathlib.Path:
        """Read-only attribute for the path to original raster file"""

        return self._path

    @property
    def tmpfile(self) -> pathlib.Path:
        """Read-only attribute for the path to working raster file"""

        return self._tmpfile

    @property
    def md5(self) -> str:
        """Read-only attribute for the hash of working raster file content

        This is a read-only property that is recalculated every time
        from the content of the temporary working raster file.
        """

        hash_md5 = hashlib.md5()
        with open(self._tmpfile.resolve(), "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @property
    def count(self) -> int:
        """Read-only attribute for the number of bands of raster dataset"""

        return self.src.count

    @property
    def is_masked(self) -> bool:
        """Read-only attribute indicating whether raster has missing data.

        This is a read-only property that indicates whether or not
        the raster has missing data points. The value of this
        property is recalculated every time the property is retrieved.
        """

        for window in self.iter_windows(self.chunk_size):
            if self.src.nodata in self.src.read(window=window):
                return True
        return False

    @property
    def shape(self) -> Tuple[int, int]:
        """Read-only attribute indicating the shape of raster grid"""

        return self.src.shape

    @property
    def height(self) -> int:
        """Read-only attribute for the number of rows of raster grid"""

        return self.src.height

    @property
    def bbox(self) -> Union[Polygon, Bbox]:
        """Read-only attribute for the bounding box of the raster grid

        This is a read-only property that returns the same results as
        `get_bbox` method
        """

        return self.get_bbox()

    @property
    def src(self) -> rasterio.DatasetReader:
        """Read-only attribute for access to opened dataset handle"""

        return self._src

    @property
    def width(self) -> int:
        """Read-only attribute for the number of columns of raster grid"""

        return self.src.width

    @property
    def dx(self) -> float:
        """Read-only attribute for grid distance in x direction"""
        return self.src.transform[0]

    @property
    def dy(self) -> float:
        """Read-only attribute for grid distance in y direction"""
        return -self.src.transform[4]

    @property
    def crs(self) -> CRS:
        """Read-only attribute for raster CRS

        This read-only property returns CRS as a pyproj.CRS type
        **not** rasterio.CRS.
        """

        # cast rasterio.CRS to pyproj.CRS for API consistency
        return CRS.from_user_input(self.src.crs)

    @property
    def nodatavals(self) -> float:
        return self.src.nodatavals

    @property
    def transform(self) -> Affine:
        """Read-only attribute for raster's transform"""
        return self.src.transform

    @property
    def dtypes(self) -> Any:
        """Read-only attribute for raster's data type"""
        return self.src.dtypes

    @property
    def nodata(self):
        return self.src.nodata

    @property
    def xres(self) -> float:
        """Read-only attribute for grid resolution in x direction

        This read-only property returns the same value as `dx`
        """
        return self.transform[0]

    @property
    def yres(self) -> float:
        """Read-only attribute for grid resolution in y direction

        This read-only property returns the same value as `-dy`
        """
        return self.transform[4]

    @property
    def resampling_method(self) -> Resampling:
        """Modifiable attribute for stored raster resampling method"""

        if not hasattr(self, '_resampling_method'):
            self._resampling_method = Resampling.nearest
        return self._resampling_method

    @resampling_method.setter
    def resampling_method(self, resampling_method: Resampling) -> None:
        """Set `resampling_method`"""

        if not isinstance(resampling_method, Resampling):
            TypeError(
                f'Argument resampling_method must be of type  {Resampling}, '
                f'not type {type(resampling_method)}.')
        self._resampling_method = resampling_method

    @property
    def chunk_size(self) -> int:
        """Modfiable attribute for stored square raster window size"""
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, chunk_size: int) -> None:
        """Set `chunk_size`"""

        self._chunk_size = chunk_size

    @property
    def overlap(self) -> int:
        """Modfiable attribute for stored raster windows overlap amount"""
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: int) -> None:
        """Set `overlap`"""

        self._overlap = overlap


def get_iter_windows(
        width: int,
        height: int,
        chunk_size: int = 0,
        overlap: int = 0,
        row_off: int = 0,
        col_off: int = 0
        ) -> Generator[windows.Window, None, None]:
    """Calculates sequence of raster windows based on basic inputs

    This stand-alone function calculates the sequence of square windows
    based on inputs that are not necessarily tied to a specific raster
    dataset.

    Parameters
    ----------
    width : int
        The total width of the all unioned windows combined
    height : int
        The total height of the all unioned windows combined
    chunk_size : int, default=0
        Square window size to be used for data chunking
    overlap: int, default=0
        Overlap size for calculating chunking windows on the raster
    row_off: int, default=0
        Unused!
    col_off: int, default=0
        Unused!

    Yields
    ------
    windows.Window
        Calculated square window based on the total size as well as
        chunk size and windows overlap values.
    """

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


def redistribute_vertices(
        geom: Union[LineString, MultiLineString],
        distance: float
        ) -> Union[LineString, MultiLineString]:
    """Redistribute the vertices of input line strings

    Parameters
    ----------
    geom : LineString or MultiLineString
        Input line strings whose vertices is to be redistributed.
    distance : float
        The distance to be used for redistribution.

    Returns
    -------
    LineString or MultiLineString
        The resulting line strings with redistributed vertices.

    Raises
    ------
    ValueError
        If input geometry is not LineString or MultiLineString.
    """

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
