"""This module define class for raster based size function
"""

import functools
import gc
import logging
from multiprocessing import cpu_count, Pool
import operator
import tempfile
from time import time
from typing import Union, List, Callable, Any, Optional, Iterable, Tuple
from contextlib import ExitStack
import warnings
try:
    # pylint: disable=C0412
    from typing import Literal
except ImportError:
    # pylint: disable=C0412
    from typing_extensions import Literal

from jigsawpy import jigsaw_msh_t, jigsaw_jig_t
from jigsawpy import libsaw
import numpy as np
import numpy.typing as npt
from pyproj import CRS, Transformer
import rasterio
from scipy.spatial import cKDTree
from shapely import ops
from shapely.geometry import (
    LineString, MultiLineString, box, GeometryCollection,
    Polygon, MultiPolygon)

from ocsmesh.hfun.base import BaseHfun
from ocsmesh.raster import Raster, get_iter_windows
from ocsmesh.geom.shapely import PolygonGeom
from ocsmesh.features.constraint import (
    Constraint,
    TopoConstConstraint,
    TopoFuncConstraint,
    CourantNumConstraint
)
from ocsmesh import utils

# supress feather warning
warnings.filterwarnings(
    'ignore', message='.*initial implementation of Parquet.*')

_logger = logging.getLogger(__name__)


def _apply_constraints(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for (re)applying constraint after updating hfun spec

    This function is a deocrator that takes in a callable and assumes
    the callable is a method whose first argument is an object. In the
    wrapper function, after the wrapped method is called the
    `apply_added_constraints` method is called on the object and
    then the return value of the wrapped method is returned to the
    caller.

    Parameters
    ----------
    method : callable
        Method to be wrapped so that the constraints are automatically
        re-applied after method is executed.

    Returns
    -------
    callable
        The wrapped method
    """

    def wrapped(obj, *args, **kwargs):
        rv = method(obj, *args, **kwargs)
        obj.apply_added_constraints()
        return rv
    return wrapped


class HfunInputRaster:
    """Descriptor class for holding reference to the input raster"""

    def __set__(self, obj, raster: Raster) -> None:
        if not isinstance(raster, Raster):
            raise TypeError(f'Argument raster must be of type {Raster}, not '
                            f'type {type(raster)}.')
        # init output raster file
        with ExitStack() as stack:

            src = stack.enter_context(rasterio.open(raster.tmpfile))
            if raster.chunk_size is not None:
                windows = get_iter_windows(
                    src.width, src.height, chunk_size=raster.chunk_size)
            else:
                windows = [rasterio.windows.Window(
                    0, 0, src.width, src.height)]

            meta = src.meta.copy()
            meta.update({'driver': 'GTiff', 'dtype': np.float32})
            dst = stack.enter_context(
                    obj.modifying_raster(use_src_meta=False, **meta))
            for window in windows:
                values = src.read(window=window).astype(np.float32)
                values[:] = np.finfo(np.float32).max
                dst.write(values, window=window)

        obj.__dict__['raster'] = raster
        obj._chunk_size = raster.chunk_size
        obj._overlap = raster.overlap

    def __get__(self, obj, objtype=None) -> Raster:
        return obj.__dict__['raster']


class HfunRaster(BaseHfun, Raster):
    """Raster based size function.

    Creates a raster based size function. The mesh size is specified
    at each point of the grid of the input raster, based on the
    specified criteria.

    Attributes
    ----------
    raster
    hmin
    hmax
    verbosity

    Methods
    -------
    msh_t()
        Return mesh sizes interpolated on an size-optimized
        unstructured mesh
    apply_added_constraints()
        Re-apply the existing constraint. Mostly used internally.
    apply_constraints(constraint_list)
        Apply constraint objects in the `constraint_list`.
    add_topo_bound_constraint(...)
        Add size fixed-per-point value constraint to the area
        bounded by specified bounds with expansion/contraction
        rate `rate` specified.
    add_topo_func_constraint(...)
        Add size value constraint based on function of depth/elevation
        to the area bounded by specified bounds with the expansion or
        contraction rate `rate` specified.
    add_patch(...)
        Add a region of fixed size refinement with optional expansion
        rate for points outside the region to achieve smooth size
        transition.
    add_contour(level expansion_rate target_size=None, nprocs=None)
        Add refinement based on contour lines auto-extrcted from the
        underlying raster data. The size is calculated based on the
        specified `rate`, `target_size` and the distance from the
        extracted feature line.
    add_channel(...)
        Add refinement for auto-detected narrow domain regions.
        Optionally use an expansion rate for points outside detected
        narrow regions for smooth size transition.
    add_feature(...)
        Decorated method to add size refinement based on the specified
        `expansion_rate`, `target_size`, and distance from the input
        feature lines `feature`.
    get_xy_memcache(window, dst_crs)
        Get XY grid cached onto disk. Useful for when XY needs to be
        projected to UTM so as to avoid reprojecting on every call.
    add_subtidal_flow_limiter(...)
        Add mesh size refinement based on the value as well as
        gradient of the topography within the region between
        specified by lower and upper bound on topography.
    add_constant_value(value, lower_bound=None, upper_bound=None)
        Add fixed size mesh refinement in the region specified by
        upper and lower bounds on topography.

    Notes
    -----
    Currently the implementation of this size function is such that
    when an object is created, the "size" values of this object is
    set to be maximum `np.float32`, and maximum and minimum constraints
    are not applied. After application of any refinement or constrant
    these constraints are actually applied. As a result if a raster
    based size function is created and no `add_*` or `apply_*` method
    is called, the values of `hmin` and `hmax` attributes as well as
    any value on the size function grid is meaningless.

    An important distinction that must be made is between
    **refinements** and **constraints**. These two concepts are applied
    differently when it comes to calculating the sizes. Refinements
    specification guarantees that the sizes in the specified region
    is at most equal to the specified value, however it does **not**
    have any guarantee on the minimum of sizes. In other words if
    multiple refinements are specified, then the size at everypoint
    is equal to the minimum calculated from all of those refinements.
    Contraints are applied a bit differently. Constraints ensure that
    a given condition is met at a given point. A list of specified
    constraints is created and after application of any refinements
    or other constraints, all the constraint are re-applied.
    As a result constraint can be used to ensure the size at a given
    point is not smaller that a specified value. Constraint can also
    be used similar to refinements to ensure the size is smaller
    than a specified value.

    Another important different between contraints and refinements is
    that for refinements the final value is the minimum of all, but
    constraints are applied one at a time and depending on their type
    and condition, it might result on a value between all specified
    maximums and minimums.

    Note that constraints can be conflicting, currently there's no
    automatic conflict resolutions and the constrains are applied in
    the order specified, so if applicable the last one overrides all
    else.
    """

    _raster = HfunInputRaster()

    def __init__(self,
                 raster: Raster,
                 hmin: Optional[float] = None,
                 hmax: Optional[float] = None,
                 verbosity: int = 0
                 ) -> None:
        """Initialize a raster based size function object

        Parameters
        ----------
        raster : Raster
            The input raster file. The sizes are calculated on the
            grid points of the raster's grid. The input raster is
            not modified so that its elevation data can be used for
            topo-based refinement calculations.
        hmin : float or None, default=None
            Global minimum size of mesh for the size function, if not
            specified, the calculated values during refinement and
            constraint applications are not capped off.
        hmax : float or None, default=None
            Global maximum size of mesh for the size function, if not
            specified, the calculated values during refinement and
            constraint applications are not capped off.
        verbosity : int, default=0
            The verbosity of the outputs.

        Notes
        -----
        All the points in the raster grid are set to have maximum
        `np.float32` value. The `hmin` and `hmax` arguments,
        even if provided, are not applied in during initialization.
        """

        self._xy_cache = {}
        # NOTE: unlike Raster, HfunRaster has no "path" set
        self._raster = raster
        # TODO: Store max and min as two separate constraints instead
        # of private attributes
        self._hmin = float(hmin) if hmin is not None else hmin
        self._hmax = float(hmax) if hmax is not None else hmax
        self._verbosity = int(verbosity)
        self._constraints = []


    def msh_t(
            self,
            window: Optional[rasterio.windows.Window] = None,
            marche: bool = False,
            verbosity : Optional[bool] = None
            ) -> jigsaw_msh_t:
        """Interpolates mesh size function on an unstructred mesh

        Interpolate the calculated mesh sizes from the raster grid
        onto an unstructured mesh. This mesh is generated by meshing
        the input raster using the size function values. The return
        value is in a projected CRS. If the input raster CRS is
        geographic, then a local UTM CRS is calculated and used
        for the output of this method.

        Parameters
        ----------
        window : rasterio.windows.Window or None, default=None
            If provided, a single window on raster for which the
            mesh size is to be returned.
        marche : bool, default=False
            Whether to run `marche` algorithm on the complete
            size function before calculating the unstructured mesh
            and interpolate values on it.
        verbosity : bool or None, default=None
            The verbosity of the output.

        Returns
        -------
        jigsaw_msh_t
            Size function calculated and interpolated on an
            unstructured mesh.

        Notes
        -----
        In case the underlying raster is created in windowed
        calculation mode, this method calculated the mesh for each
        window separately and then combines (no remeshing) the
        elements of all the windows.

        The output of this method needs to have length unit for
        distances (i.e. not degrees) since mesh size is specified
        in length units and the domain and size function are the
        passed to the mesh engine for cartesian meshing.

        The reason the full high-resolution size function is
        interpolated on a generated mesh it to save memory and
        have the ability to process and combine many DEMs. By doing
        more sizes are specified at points where the size needs to
        be smaller in the final mesh.

        To generate the mesh for size function interpolation, the
        raster size function (called ``hmat``) is passed to the mesh
        engine along with the bounding box of the size function as
        the meshing domain.
        """


        if window is None:
            iter_windows = list(self.iter_windows())
        else:
            iter_windows = [window]


        output_mesh = jigsaw_msh_t()
        output_mesh.ndims = +2
        output_mesh.mshID = "euclidean-mesh"
        output_mesh.crs = self.crs
        for win in iter_windows:

            hfun = jigsaw_msh_t()
            hfun.ndims = +2

            x0, y0, x1, y1 = self.get_window_bounds(win)

            utm_crs = utils.estimate_bounds_utm(
                    (x0, y0, x1, y1), self.crs)

            if utm_crs is not None:
                hfun.mshID = 'euclidean-mesh'
                # If these 3 objects (vert2, tria3, value) don't fit into
                # memroy, then the raster needs to be chunked. We need to
                # implement auto-chunking.
                start = time()
                # get bbox data
                xgrid = self.get_x(window=win)
                ygrid = np.flip(self.get_y(window=win))
                xgrid, ygrid = np.meshgrid(xgrid, ygrid)
                bottom = xgrid[0, :]
                top = xgrid[1, :]
                del xgrid
                left = ygrid[:, 0]
                right = ygrid[:, 1]
                del ygrid

                _logger.info('Building hfun.tria3...')

                dim1 = win.width
                dim2 = win.height

                tria3 = np.empty(
                    ((dim1 - 1), (dim2  - 1)),
                    dtype=jigsaw_msh_t.TRIA3_t)
                index = tria3["index"]
                helper_ary = np.ones(
                        ((dim1 - 1), (dim2  - 1)),
                        dtype=jigsaw_msh_t.INDEX_t).cumsum(1) - 1
                index[:, :, 0] = np.arange(
                        0, dim1 - 1,
                        dtype=jigsaw_msh_t.INDEX_t).reshape(dim1 - 1, 1)
                index[:, :, 0] += (helper_ary + 0) * dim1

                index[:, :, 1] = np.arange(
                        1, dim1 - 0,
                        dtype=jigsaw_msh_t.INDEX_t).reshape(dim1 - 1, 1)
                index[:, :, 1] += (helper_ary + 0) * dim1

                index[:, :, 2] = np.arange(
                        1, dim1 - 0,
                        dtype=jigsaw_msh_t.INDEX_t).reshape(dim1 - 1, 1)
                index[:, :, 2] += (helper_ary + 1) * dim1

                hfun.tria3 = tria3.ravel()
                del tria3, helper_ary
                gc.collect()
                _logger.info('Done building hfun.tria3...')

                # BUILD VERT2_t. this one comes from the memcache array
                _logger.info('Building hfun.vert2...')
                hfun.vert2 = np.empty(
                    win.width*win.height,
                    dtype=jigsaw_msh_t.VERT2_t)
                hfun.vert2['coord'] = np.array(
                    self.get_xy_memcache(win, utm_crs))
                _logger.info('Done building hfun.vert2...')

                # Build REALS_t: this one comes from hfun raster
                _logger.info('Building hfun.value...')
                hfun.value = np.array(
                    self.get_values(window=win, band=1).flatten().reshape(
                        (win.width*win.height, 1)),
                    dtype=jigsaw_msh_t.REALS_t)
                _logger.info('Done building hfun.value...')

                # Build Geom
                _logger.info('Building initial geom...')
                transformer = Transformer.from_crs(
                    self.crs, utm_crs, always_xy=True)
                bbox = [
                    *[(x, left[0]) for x in bottom],
                    *[(bottom[-1], y) for y in reversed(right)],
                    *[(x, right[-1]) for x in reversed(top)],
                    *[(bottom[0], y) for y in reversed(left)]]
                geom = PolygonGeom(
                    ops.transform(transformer.transform, Polygon(bbox)),
                    utm_crs
                ).msh_t()
                _logger.info('Building initial geom done.')
                kwargs = {'method': 'nearest'}

            else:
                _logger.info('Forming initial hmat (euclidean-grid).')
                start = time()
                hfun.mshID = 'euclidean-grid'
                hfun.xgrid = np.array(
                    np.array(self.get_x(window=win)),
                    dtype=jigsaw_msh_t.REALS_t)
                hfun.ygrid = np.array(
                    np.flip(self.get_y(window=win)),
                    dtype=jigsaw_msh_t.REALS_t)
                hfun.value = np.array(
                    np.flipud(self.get_values(window=win, band=1)),
                    dtype=jigsaw_msh_t.REALS_t)
                kwargs = {'kx': 1, 'ky': 1}  # type: ignore[dict-item]
                geom = PolygonGeom(box(x0, y1, x1, y0), self.crs).msh_t()

            _logger.info(f'Initial hfun generation took {time()-start}.')

            _logger.info('Configuring jigsaw...')

            opts = jigsaw_jig_t()

            # additional configuration options
            opts.mesh_dims = +2
            opts.hfun_scal = 'absolute'
            # no need to optimize for size function generation
            opts.optm_tria = False

            opts.hfun_hmin = np.min(hfun.value) if self.hmin is None else \
                self.hmin
            opts.hfun_hmax = np.max(hfun.value) if self.hmax is None else \
                self.hmax
            opts.verbosity = self.verbosity if verbosity is None else \
                verbosity

            # mesh of hfun window
            window_mesh = jigsaw_msh_t()
            window_mesh.mshID = 'euclidean-mesh'
            window_mesh.ndims = +2

            if marche is True:
                libsaw.marche(opts, hfun)

            libsaw.jigsaw(opts, geom, window_mesh, hfun=hfun)

            del geom
            # do post processing
            hfun.crs = utm_crs
            utils.interpolate(hfun, window_mesh, **kwargs)

            # reproject to combine with other windows
            if utm_crs is not None:
                window_mesh.crs = utm_crs
                utils.reproject(window_mesh, self.crs)


            # combine with results from previous windows
            output_mesh.tria3 = np.append(
                output_mesh.tria3,
                np.array([((idx + len(output_mesh.vert2)), tag)
                          for idx, tag in window_mesh.tria3],
                         dtype=jigsaw_msh_t.TRIA3_t),
                axis=0)
            output_mesh.vert2 = np.append(
                output_mesh.vert2,
                np.array(list(window_mesh.vert2),
                         dtype=jigsaw_msh_t.VERT2_t),
                axis=0)
            if output_mesh.value.size:
                output_mesh.value = np.append(
                    output_mesh.value,
                    np.array(list(window_mesh.value),
                             dtype=jigsaw_msh_t.REALS_t),
                    axis=0)
            else:
                output_mesh.value = np.array(
                        list(window_mesh.value),
                        dtype=jigsaw_msh_t.REALS_t)

        # NOTE: In the end we need to return in a CRS that
        # uses meters as units. UTM based on the center of
        # the bounding box of the hfun is used
        utm_crs = utils.estimate_bounds_utm(
                self.get_bbox().bounds, self.crs)
        if utm_crs is not None:
            transformer = Transformer.from_crs(
                self.crs, utm_crs, always_xy=True)
            output_mesh.vert2['coord'] = np.vstack(
                transformer.transform(
                    output_mesh.vert2['coord'][:, 0],
                    output_mesh.vert2['coord'][:, 1]
                    )).T
            output_mesh.crs = utm_crs

        return output_mesh


    def apply_added_constraints(self) -> None:
        """Apply all the added constraints

        This method is implemented for internal use. It's public
        because it needs to be called from outside the class through
        a decorator.

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.apply_constraints(self._constraints)


    def apply_constraints(
            self,
            constraint_list: Iterable[Constraint]
            ) -> None:
        """Applies constraints specified by the list of contraint objects.

        Applies constraints from the provided list `constraint_list`,
        but doesn't not store them in the internal size function
        constraint list. This is mostly for internal use.

        Parameters
        ----------
        constraint_list : iterable of Constraint
            List of constraint objects to be applied to (not stored in)
            the size function.

        Returns
        -------
        None
        """

        # TODO: Validate conflicting constraints

        # Apply constraints
        with self.modifying_raster() as dst:
            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)

            for i, window in enumerate(iter_windows):
                hfun_values = self.get_values(band=1, window=window)
                rast_values = self.raster.get_values(band=1, window=window)


                # Get locations
                utm_crs = utils.estimate_bounds_utm(
                        self.get_window_bounds(window), self.crs)

                if utm_crs is not None:
                    xy = self.get_xy_memcache(window, utm_crs)
                else:
                    xy = self.get_xy(window)


                # Apply custom constraints
                _logger.debug(f'Processing window {i+1}/{tot}.')
                for constraint in constraint_list:
                    hfun_values = constraint.apply(
                            rast_values, hfun_values, locations=xy)

                # Apply global constraints
                if self.hmin is not None:
                    hfun_values[hfun_values < self.hmin] = self.hmin
                if self.hmax is not None:
                    hfun_values[hfun_values > self.hmax] = self.hmax

                dst.write_band(1, hfun_values, window=window)
                del rast_values
                gc.collect()


    @_apply_constraints
    def add_topo_bound_constraint(
            self,
            value: Union[float, npt.NDArray[np.float32]],
            upper_bound: float = np.inf,
            lower_bound: float = -np.inf,
            value_type: Literal['min', 'max'] = 'min',
            rate: float = 0.01
            ) -> None:
        """Add a fixed-value or fixed-matrix constraint.

        Add a fixed-value or fixed-matrix constraint to the region
        of the size function specified by lower and upper elevation
        of the underlying DEM. Optionally a `rate` can be specified
        to relax the constraint gradually outside the bounds.

        Parameters
        ----------
        value : float or array-like
            A single fixed value or array of values to be used for
            mesh size if condition is not met based on `value_type`.
            In case of an array the dimensions must match or be
            broadcastable to the raster grid.
        upper_bound : float, default=np.inf
            Maximum elevation to cut off the region where the
            constraint needs to be applied
        lower_bound : float, default=-np.inf
            Minimum elevation to cut off the region where the
            constraint needs to be applied
        value_type : {'min', 'max'}, default='min'
            Type of contraint. If 'min', it means the mesh size
            should not be smaller than the specified `value` at each
            point.
        rate : float, default=0.01
            Rate of relaxation of constraint outside the region defined
            by `lower_bound` and `upper_bound`.

        Returns
        -------
        None

        See Also
        --------
        add_topo_func_constraint :
            Addint constraint based on function of topography
        add_courant_num_constraint :
            Add constraint based on approximated Courant number
        """

        # TODO: Validate conflicting constraints, right now last one wins
        self._constraints.append(TopoConstConstraint(
            value, upper_bound, lower_bound, value_type, rate))


    @_apply_constraints
    def add_topo_func_constraint(
            self,
            func: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
                = lambda i: i / 2.0,
            upper_bound: float = np.inf,
            lower_bound: float = -np.inf,
            value_type: Literal['min', 'max'] = 'min',
            rate: float = 0.01
            ) -> None:
        """Add constraint based on a function of the topography

        Add a constraint based on the provided function `func` of
        the topography and apply to the region of the size function
        specified by lower and upper elevation of the underlying DEM.
        Optionally a `rate` can be specified to relax the constraint
        gradually outside the bounds.

        Parameters
        ----------
        func : callable
            A function to be applied on the topography to acquire the
            values to be used for mesh size if condition is not met
            based on `value_type`.
        upper_bound : float, default=np.inf
            Maximum elevation to cut off the region where the
            constraint needs to be applied
        lower_bound : float, default=-np.inf
            Minimum elevation to cut off the region where the
            constraint needs to be applied
        value_type : {'min', 'max'}, default='min'
            Type of contraint. If 'min', it means the mesh size
            should not be smaller than the value calculated from the
            specified `func` at each point.
        rate : float, default=0.01
            Rate of relaxation of constraint outside the region defined
            by `lower_bound` and `upper_bound`.

        Returns
        -------
        None

        See Also
        --------
        add_topo_bound_constraint :
            Add fixed-value or fixed-matrix constraint.
        add_courant_num_constraint :
            Add constraint based on approximated Courant number
        """

        # TODO: Validate conflicting constraints, right now last one wins
        self._constraints.append(TopoFuncConstraint(
            func, upper_bound, lower_bound, value_type, rate))

    @_apply_constraints
    def add_courant_num_constraint(
            self,
            upper_bound: float = 0.9,
            lower_bound: Optional[float] = None,
            timestep: float = 150,
            wave_amplitude: float = 2
            ) -> None:
        """Add constraint based on approximated Courant number bounds


        Parameters
        ----------
        upper_bound : float, default=0.9
            Maximum Courant number to allow on this mesh size function
        lower_bound : float or None, default=None
            Minimum Courant number to allow on this mesh size function
        timestep : float
            Timestep size (:math:`seconds`) to
        wave_amplitude : float, default=2
            Free surface elevation (:math:`meters`) from the reference
            (i.e. wave height)

        Returns
        -------
        None

        See Also
        --------
        add_topo_bound_constraint :
            Add fixed-value or fixed-matrix constraint.
        add_topo_func_constraint :
            Add constraint based on a function of the topography.
        """

        # TODO: Validate conflicting constraints, right now last one wins
        if upper_bound is None and lower_bound is None:
            raise ValueError("Both upper and lower Courant bounds can NOT be None!")

        if upper_bound is not None:
            self._constraints.append(
                CourantNumConstraint(
                    value=upper_bound,
                    timestep=timestep,
                    wave_amplitude=wave_amplitude,
                    value_type='max'
                )
            )
        if lower_bound is not None:
            self._constraints.append(
                CourantNumConstraint(
                    value=lower_bound,
                    timestep=timestep,
                    wave_amplitude=wave_amplitude,
                    value_type='min'
                )
            )


    @_apply_constraints
    def add_patch(
            self,
            multipolygon: Union[MultiPolygon, Polygon],
            expansion_rate: Optional[float] = None,
            target_size: Optional[float] = None,
            nprocs: Optional[int] = None
            ) -> None:
        """Add refinement as a region of fixed size with an optional rate

        Add a refinement based on a region specified by `multipolygon`.
        The fixed `target_size` refinement can be expanded outside the
        region specified by the shape if `expansion_rate` is provided.

        Parameters
        ----------
        multipolygon : MultiPolygon or Polygon
            Shape of the region to use specified `target_size` for
            refinement.
        expansion_rate : float or None, default=None
            Optional rate to use for expanding refinement outside
            the specified shape in `multipolygon`.
        target_size : float or None, default=None
            Fixed target size of mesh to use for refinement in
            `multipolygon`
        nprocs : int or None, default=None
            Number of processors to use in parallel sections of the
            algorithm

        Returns
        -------
        None

        See Also
        --------
        add_feature :
            Add refinement for specified line string
        add_contour :
            Add refinement for auto-extracted contours
        add_channel :
            Add refinement for auto-extracted narrow regions
        add_subtidal_flow_limiter :
            Add refinement based on topograph
        add_constant_value :
            Add refinement with fixed value
        """

        # TODO: Add pool input support like add_feature for performance

        # TODO: Support other shapes - call buffer(1) on non polygons(?)
        if not isinstance(multipolygon, (Polygon, MultiPolygon)):
            raise TypeError(
                    f"Wrong type \"{type(multipolygon)}\""
                    f" for multipolygon input.")

        if isinstance(multipolygon, Polygon):
            multipolygon = MultiPolygon([multipolygon])

        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs
        _logger.debug(f'Using nprocs={nprocs}')


        # check target size
        target_size = self.hmin if target_size is None else target_size
        if target_size is None:
            # pylint: disable=W0101
            raise ValueError('Argument target_size must be specified if no '
                             'global hmin has been set.')
        if target_size <= 0:
            raise ValueError("Argument target_size must be greater than zero.")

        # For expansion_rate
        if expansion_rate is not None:
            exteriors = [ply.exterior for ply in multipolygon]
            interiors = [
                inter for ply in multipolygon for inter in ply.interiors]

            features = MultiLineString([*exteriors, *interiors])
            # pylint: disable=E1123, E1125
            self.add_feature(
                feature=features,
                expansion_rate=expansion_rate,
                target_size=target_size,
                nprocs=nprocs)

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
                try:
                    mask, _, _ = rasterio.mask.raster_geometry_mask(
                        self.src, multipolygon,
                        all_touched=True, invert=True)
                    mask = mask[rasterio.windows.window_index(window)]

                except ValueError:
                    # If there's no overlap between the raster and
                    # shapes then it throws ValueError, instead of
                    # checking for intersection, if there's a value
                    # error we assume there's no overlap
                    _logger.debug(
                        'Polygons don\'t intersect with the raster')
                    continue
                _logger.info(
                    f'Creating mask from shape took {time()-start}.')

                values = self.get_values(window=window).copy()
                if mask.any():
                    # NOTE: Don't continue, otherwise the final
                    # destination file might end up being empty!
                    values[mask] = target_size
                if self.hmin is not None:
                    values[np.where(values < self.hmin)] = self.hmin
                if self.hmax is not None:
                    values[np.where(values > self.hmax)] = self.hmax
                values = np.minimum(self.get_values(window=window), values)

                _logger.info('Write array to file...')
                start = time()
                dst.write_band(1, values, window=window)
                _logger.info(f'Write array to file took {time()-start}.')


    @_apply_constraints
    def add_contour(
            self,
            level: Union[List[float], float],
            expansion_rate: float,
            target_size: Optional[float] = None,
            nprocs: Optional[int] = None
            ) -> None:
        """Add refinement for auto extracted contours

        Add refinement for the contour lines extracted based on
        level or levels specified by `level`. The refinement
        is relaxed with `expansion_rate` and distance from the
        extracted contour lines.

        Parameters
        ----------
        level : float or list of floats
            Level(s) at which contour lines should be extracted.
        expansion_rate : float
            Rate to use for expanding refinement with distance away
            from the extracted contours.
        target_size : float or None, default=None
            Target size to use on the extracted contours and expand
            from with distance.
        nprocs : int or None, default=None
            Number of processors to use in parallel sections of the
            algorithm

        Returns
        -------
        None

        See Also
        --------
        add_feature :
            Add refinement for specified line string
        add_patch :
            Add refinement for region specified polygon
        add_channel :
            Add refinement for auto-extracted narrow regions
        add_subtidal_flow_limiter :
            Add refinement based on topograph
        add_constant_value :
            Add refinement with fixed value

        Notes
        -----
        This method extracts contours at specified levels and
        the calls `add_feature` by passing those contour lines.
        """

        if not isinstance(level, list):
            level = [level]

        contours = []
        for _level in level:
            # pylint: disable=R1724

            _contours = self.raster.get_contour(_level)
            if isinstance(_contours, GeometryCollection):
                continue
            elif isinstance(_contours, LineString):
                contours.append(_contours)
            elif isinstance(_contours, MultiLineString):
                for _cont in _contours.geoms:
                    contours.append(_cont)

        if len(contours) == 0:
            _logger.info('No contours found!')
            return

        contours = MultiLineString(contours)

        _logger.info('Adding contours as features...')
        # pylint: disable=E1123, E1125
        self.add_feature(
                contours, expansion_rate, target_size,
                nprocs=nprocs)

    @_apply_constraints
    def add_channel(
            self,
            level: float = 0,
            width: float = 1000,
            target_size: float = 200,
            expansion_rate: Optional[float] = None,
            nprocs: Optional[int] = None,
            tolerance: Optional[float] = None
            ) -> None:
        """Add refinement for auto detected channels

        Automatically detects narrow regions in the domain and apply
        refinement size with an expanion rate (if provided) outside
        the detected area.

        Parameters
        ----------
        level : float, default=0
            High water mark at which domain is extracted for narrow
            region or channel calculations.
        width : float, default=1000
            The cut-off width for channel detection.
        target_size : float, default=200
            Target size to use on the detected channels and expand
            from with distance.
        expansion_rate : float or None, default=None
            Rate to use for expanding refinement with distance away
            from the detected channels.
        nprocs : int or None, default=None
            Number of processes to use for parallel sections of the
            workflow.
        tolerance: float or None, default=None
            Tolerance to use for simplifying the polygon extracted
            from DEM data. If `None` don't simplify.

        Returns
        -------
        None

        See Also
        --------
        add_contour :
            Add refinement for auto-extracted contours
        add_patch :
            Add refinement for region specified polygon
        add_feature :
            Add refinement for specified line string
        add_subtidal_flow_limiter :
            Add refinement based on topograph
        add_constant_value :
            Add refinement with fixed value

        Notes
        -----
        """

        channels = self.raster.get_channels(
                level=level, width=width, tolerance=tolerance)

        if channels is None:
            return

        self.add_patch(
            channels, expansion_rate, target_size, nprocs)



    @_apply_constraints
    @utils.add_pool_args
    def add_feature(
            self,
            feature: Union[LineString, MultiLineString],
            expansion_rate: float,
            target_size: Optional[float] = None,
            max_verts: int = 200,
            *, # kwarg-only comes after this
            pool: Pool,
            ) -> None:
        """Add refinement for specified linestring `feature`

        Add refinement for the specified linestrings `feature`.
        The refinement is relaxed with `expansion_rate` and distance
        from the extracted contour lines.

        Parameters
        ----------
        feature : LineString or MultiLineString
            The user-specified line strings for applying refinement on.
        expansion_rate : float
            Rate to use for expanding refinement with distance away
            from the extracted contours.
        target_size : float or None, default=None
            Target size to use on the extracted contours and expand
            from with distance.
        max_verts : int, default=200
            Number of maximum vertices in a feature line that is
            passed to a separate process in parallel section of
            the algorithm.
        pool : Pool
            Pre-created and initialized process pool to be used for
            parallel sections of the algorithm.

        Returns
        -------
        None

        See Also
        --------
        add_contour :
            Add refinement for auto-extracted contours
        add_patch :
            Add refinement for region specified polygon
        add_channel :
            Add refinement for auto-extracted narrow regions
        add_subtidal_flow_limiter :
            Add refinement based on topograph
        add_constant_value :
            Add refinement with fixed value

        Notes
        -----
        See https://outline.com/YU7nSM for an explanation
        about tree algorithms.

        Creating a local projection allows having similar area/length
        calculations as if great circle calculations was being used.

        Another useful refererence:
        https://gis.stackexchange.com/questions/214261/should-we-always-calculate-length-and-area-in-lat-lng-to-get-accurate-sizes-leng
        """

        # TODO: Consider using BallTree with haversine or Vincenty
        # metrics instead of a locally projected window.

        # TODO: Partition features if they are too "long" which results in an
        # improvement for parallel pool. E.g. if a feature is too long, 1
        # processor will be busy and the rest will be idle.

        if not isinstance(feature, (LineString, MultiLineString)):
            raise TypeError(
                f'Argument feature must be of type {LineString} or '
                f'{MultiLineString}, not type {type(feature)}.')

        if isinstance(feature, LineString):
            feature = [feature]

        elif isinstance(feature, MultiLineString):
            feature = list(feature.geoms)

        # check target size
        target_size = self.hmin if target_size is None else target_size
        if target_size is None:
            raise ValueError('Argument target_size must be specified if no '
                             'global hmin has been set.')
        if target_size <= 0:
            raise ValueError("Argument target_size must be greater than zero.")
        with self.modifying_raster(driver='GTiff') as dst:
            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)
            for i, window in enumerate(iter_windows):
                _logger.debug(f'Processing window {i+1}/{tot}.')
                utm_crs = utils.estimate_bounds_utm(
                        self.get_window_bounds(window), self.crs)

                _logger.info('Repartitioning features...')
                start = time()
                res = pool.starmap(
                    utils.repartition_features,
                    [(linestring, max_verts) for linestring in feature]
                    )
                win_feature = functools.reduce(operator.iconcat, res, [])
                _logger.info(f'Repartitioning features took {time()-start}.')

                _logger.info('Resampling features on ...')
                start = time()

                # We don't want to recreate the same transformation
                # many times (it takes time) and we can't pass
                # transformation object to subtask (cinit issue)
                transformer = None
                if utm_crs is not None:
                    start2 = time()
                    transformer = Transformer.from_crs(
                        self.src.crs, utm_crs, always_xy=True)
                    _logger.info(
                            f"Transform creation took {time() - start2:f}")
                    start2 = time()
                    win_feature = [
                        ops.transform(transformer.transform, linestring)
                        for linestring in win_feature]
                    _logger.info(
                            f"Transform apply took {time() - start2:f}")

                transformed_features = pool.starmap(
                    utils.transform_linestring,
                    [(linestring, target_size) for linestring in win_feature]
                )
                _logger.info(f'Resampling features took {time()-start}.')
                _logger.info('Concatenating points...')
                start = time()
                points = []
                for geom in transformed_features:
                    if isinstance(geom, LineString):
                        points.extend(geom.coords)
                    elif isinstance(geom, MultiLineString):
                        for linestring in geom:
                            points.extend(linestring.coords)
                _logger.info(f'Point concatenation took {time()-start}.')

                _logger.info('Generating KDTree...')
                start = time()
                tree = cKDTree(np.array(points))
                _logger.info(f'Generating KDTree took {time()-start}.')
                if utm_crs is not None:
                    xy = self.get_xy_memcache(window, utm_crs)
                else:
                    xy = self.get_xy(window)

                _logger.info(f'Transforming points took {time()-start}.')
                _logger.info('Querying KDTree...')
                start = time()
                if self.hmax:
                    r = (self.hmax - target_size) / (expansion_rate * target_size)
                    near_dists, neighbors = tree.query(
                        xy, workers=pool._processes, distance_upper_bound=r)
                    distances = r * np.ones(len(xy))
                    mask = np.logical_not(np.isinf(near_dists))
                    distances[mask] = near_dists[mask]
                else:
                    distances, _ = tree.query(xy, workers=pool._processes)
                _logger.info(f'Querying KDTree took {time()-start}.')
                values = expansion_rate*target_size*distances + target_size
                values = values.reshape(window.height, window.width).astype(
                    self.dtype(1))
                if self.hmin is not None:
                    values[np.where(values < self.hmin)] = self.hmin
                if self.hmax is not None:
                    values[np.where(values > self.hmax)] = self.hmax
                values = np.minimum(self.get_values(window=window), values)
                _logger.info('Write array to file...')
                start = time()
                dst.write_band(1, values, window=window)
                _logger.info(f'Write array to file took {time()-start}.')

    def get_xy_memcache(
            self,
            window : rasterio.windows.Window,
            dst_crs: Union[CRS, str]
            ) -> npt.NDArray[float]:
        """Get the transformed locations of raster points.

        Get the locations of raster points in the `dst_crs` CRS.
        This method caches these transformed values for fast retrieval
        upon multiple calls.

        Parameters
        ----------
        window : rasterio.windows.Window
            The raster window for querying location data.
        dst_crs : CRS or str
            The destination CRS for the raster points locations.

        Returns
        -------
        np.ndarray
            Locations of raster points after projecting to `dst_crs`

        See Also
        --------
        get_xy :
            Get the locations of raster points from the raster file.
        """

        tmpfile = self._xy_cache.get(f'{window}{dst_crs}')
        if tmpfile is None:
            _logger.info('Transform points to local CRS...')
            transformer = Transformer.from_crs(
                self.src.crs, dst_crs, always_xy=True)
            # pylint: disable=R1732
            tmpfile = tempfile.NamedTemporaryFile()
            xy = self.get_xy(window)
            fp = np.memmap(tmpfile, dtype='float32', mode='w+', shape=xy.shape)
            fp[:] = np.vstack(
                transformer.transform(xy[:, 0], xy[:, 1])).T
            _logger.info('Saving values to memcache...')
            fp.flush()
            _logger.info('Done!')
            self._xy_cache[f'{window}{dst_crs}'] = tmpfile
            return fp[:]

        _logger.info('Loading values from memcache...')
        return np.memmap(tmpfile, dtype='float32', mode='r',
                         shape=((window.width*window.height), 2))[:]

    @_apply_constraints
    def add_subtidal_flow_limiter(
            self,
            hmin: Optional[float] = None,
            hmax: Optional[float] = None,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None
            ) -> None:
        """Add mesh refinement based on topography.

        Calculates a pre-defined function of topography to use
        as values for refinement. The function values are cut off
        using `hmin` and `hmax` and is applied to the region bounded
        by `lower_bound` and `upper_bound`.

        Parameters
        ----------
        hmin : float or None, default=None
            Minimum mesh size in the refinement
        hmax : float or None, default=None
            Maximum mesh size in the refinement
        lower_bound : float or None, default=None
            Lower limit of the cut-off elevation for region to apply
            the fixed `value`.
        upper_bound : float or None, default=None
            Higher limit of the cut-off elevation for region to apply
            the fixed `value`.

        Returns
        -------
        None

        See Also
        --------
        add_feature :
            Add refinement for specified line string
        add_contour :
            Add refinement for auto-extracted contours
        add_patch :
            Add refinement for region specified polygon
        add_channel :
            Add refinement for auto-extracted narrow regions
        add_constant_value :
            Add refinement with fixed value

        Notes
        -----
        The size refinement value is calculated from the topography
        using:

        .. math::
            h = {1 \\over 3} \\times {{|z|} \\over {\\left\\| \\grad z \\right\\|}}

        where :math:`z` is the elevation and :math:`h` is the value of
        the mesh size. This refinement is not applied wherever the
        magnitude of the gradient of topography is equal to zero.
        """

        hmin = float(hmin) if hmin is not None else hmin
        hmax = float(hmax) if hmax is not None else hmax

        with self.modifying_raster() as dst:

            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)

            for i, window in enumerate(iter_windows):

                _logger.debug(f'Processing window {i+1}/{tot}.')

                x0, y0, x1, y1 = self.get_window_bounds(window)
                utm_crs = utils.estimate_bounds_utm(
                        (x0, y0, x1, y1), self.crs)
                if utm_crs is not None:
                    transformer = Transformer.from_crs(
                            self.crs, utm_crs, always_xy=True)
                    (x0, x1), (y0, y1) = transformer.transform(
                            [x0, x1], [y0, y1])
                    dx = np.diff(np.linspace(x0, x1, window.width))[0]
                    dy = np.diff(np.linspace(y0, y1, window.height))[0]
                else:
                    dx = self.dx
                    dy = self.dy
                topobathy = self.raster.get_values(band=1, window=window)
                dx, dy = np.gradient(topobathy, dx, dy)
                with warnings.catch_warnings():
                    # in case self._src.values is a masked array
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dh = np.sqrt(dx**2 + dy**2)
                dh = np.ma.masked_equal(dh, 0.)
                hfun_values = np.abs((1./3.)*(topobathy/dh))
                # values = values.filled(np.max(values))

                if upper_bound is not None:
                    idxs = np.where(topobathy > upper_bound)
                    hfun_values[idxs] = self.get_values(
                        band=1, window=window)[idxs]
                if lower_bound is not None:
                    idxs = np.where(topobathy < lower_bound)
                    hfun_values[idxs] = self.get_values(
                        band=1, window=window)[idxs]

                if hmin is not None:
                    hfun_values[np.where(hfun_values < hmin)] = hmin

                if hmax is not None:
                    hfun_values[np.where(hfun_values > hmax)] = hmax

                if self._hmin is not None:
                    hfun_values[np.where(hfun_values < self._hmin)] = self._hmin
                if self._hmax is not None:
                    hfun_values[np.where(hfun_values > self._hmax)] = self._hmax

                hfun_values = np.minimum(
                    self.get_values(band=1, window=window),
                    hfun_values).astype(
                    self.dtype(1))
                dst.write_band(1, hfun_values, window=window)

    @_apply_constraints
    def add_constant_value(
            self,
            value: float,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None
            ) -> None:
        """Add refinement of fixed value in the region specified by bounds.

        Apply fixed value mesh size refinement to the region specified
        by bounds (if provided) or the whole domain.

        Parameters
        ----------
        value : float
            Fixed value to use for refinement size
        lower_bound : float or None, default=None
            Lower limit of the cut-off elevation for region to apply
            the fixed `value`.
        upper_bound : float or None, default=None
            Higher limit of the cut-off elevation for region to apply
            the fixed `value`.

        Returns
        -------
        None

        See Also
        --------
        add_feature :
            Add refinement for specified line string
        add_contour :
            Add refinement for auto-extracted contours
        add_patch :
            Add refinement for region specified polygon
        add_channel :
            Add refinement for auto-extracted narrow regions
        add_subtidal_flow_limiter :
            Add refinement based on topograph
        """

        lower_bound = -float('inf') if lower_bound is None \
            else float(lower_bound)
        upper_bound = float('inf') if upper_bound is None \
            else float(upper_bound)

        with self.modifying_raster() as dst:

            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)

            for i, window in enumerate(iter_windows):

                _logger.debug(f'Processing window {i+1}/{tot}.')
                hfun_values = self.get_values(band=1, window=window)
                rast_values = self.raster.get_values(band=1, window=window)
                hfun_values[np.where(np.logical_and(
                    rast_values > lower_bound,
                    rast_values < upper_bound))] = value
                hfun_values = np.minimum(
                    self.get_values(band=1, window=window),
                    hfun_values.astype(self.dtype(1)))
                dst.write_band(1, hfun_values, window=window)
                del rast_values
                gc.collect()

    @property
    def raster(self):
        """Read-only attribute to reference to the input raster"""

        return self._raster

    @property
    def hmin(self):
        """Read-only attribute for the minimum mesh size constraint"""

        return self._hmin

    @property
    def hmax(self):
        """Read-only attribute for the maximum mesh size constraint"""

        return self._hmax

    @property
    def verbosity(self):
        """Modifiable attribute for the verbosity of the output"""

        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity: int):
        self._verbosity = verbosity


def transform_point(
        x: npt.NDArray[float],
        y: npt.NDArray[float],
        src_crs: Union[CRS, str],
        utm_crs: Union[CRS, str],
        ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Transform input locations to the destination CRS `utm_crs`

    Parameters
    ----------
    x : npt.NDArray[float]
        Vector of x locations.
    y : npt.NDArray[float]
        Vector of y locations.
    src_crs : Union[CRS, str]
        Source CRS for location transformation.
    utm_crs : Union[CRS, str]
        Destination CRS for location transformation.

    Returns
    -------
    Tuple[npt.NDArray[float], npt.NDArray[float]]
        Tuplpe of transformed x and y arrays.
    """
    transformer = Transformer.from_crs(src_crs, utm_crs, always_xy=True)
    return transformer.transform(x, y)


def transform_polygon(
    polygon: Polygon,
    src_crs: CRS = None,
    utm_crs: CRS = None
    ) -> Polygon:
    """Transform the input polygon to the destination CRS `utm_crs`

    Parameters
    ----------
    polygon : Polygon
        Input polygon to be transformed from `src_crs` to `utm_crs`.
    src_crs : Union[CRS, str]
        Source CRS for location transformation.
    utm_crs : Union[CRS, str]
        Destination CRS for location transformation.

    Returns
    -------
    Polygon
        Transformed polygon in the destination `utm_crs`.
    """

    if utm_crs is not None:
        transformer = Transformer.from_crs(
            src_crs, utm_crs, always_xy=True)

        polygon = ops.transform(
                transformer.transform, polygon)
    return polygon
