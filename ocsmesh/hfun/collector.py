"""This module defines size function collector.
`
Size function collector objects accepts a list of valid basic `Hfun`
inputs and creates an object that merges the results of all the
other types of size functions, e.g. mesh-based and raster-based.

Notes
-----
This enables the user to define size on multiple DEM and mesh
without having to worry about the details of merging the output size
functions defined on each DEM or mesh.
"""

import os
import gc
import logging
import warnings
import tempfile
from pathlib import Path
from time import time
from multiprocessing import Pool, cpu_count
from copy import copy, deepcopy
from typing import (
    Union, Sequence, List, Tuple, Iterable, Any, Optional, Callable)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import numpy.typing as npt
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection, box
from shapely import ops
from jigsawpy import jigsaw_msh_t
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import rasterio

from ocsmesh import utils
from ocsmesh.hfun.base import BaseHfun
from ocsmesh.hfun.raster import HfunRaster
from ocsmesh.hfun.mesh import HfunMesh
from ocsmesh.mesh.mesh import Mesh, EuclideanMesh2D
from ocsmesh.raster import Raster, get_iter_windows
from ocsmesh.features.contour import Contour
from ocsmesh.features.patch import Patch
from ocsmesh.features.channel import Channel
from ocsmesh.features.constraint import (
    TopoConstConstraint, TopoFuncConstraint, CourantNumConstraint
)

CanCreateSingleHfun = Union[Raster, EuclideanMesh2D]
CanCreateMultipleHfun = Iterable[Union[CanCreateSingleHfun, str]]
CanCreateHfun = Union[CanCreateSingleHfun, CanCreateMultipleHfun]

SizeFuncList = Sequence[Union[HfunRaster, HfunMesh]]

_logger = logging.getLogger(__name__)

class _RefinementContourInfoCollector:
    """Collection for contour refinement specification

    Accumulates information about the specified contour refinements
    to be applied later when computing the return size function.
    Provides interator interface for looping over the collection items.
    """

    def __init__(self) -> None:
        self._contours_info = {}

    def add(self, contour_defn: Contour, **size_info: Any) -> None:
        """Add contour specification to the collection

        Parameters
        ----------
        contour_defn : Contour
            The level at which contour lines need to be extracted.
        size_info : dict
            Information related to contour application such as
            target size, rate, etc.

        Returns
        -------
        None
        """

        self._contours_info[contour_defn] = size_info

    def __iter__(self) -> Tuple[Contour, dict]:
        """Iterator method for this collection object

        Yields
        ------
        defn : Contour
            Contour definition added to the collection.
        info : dict
            Dictionary of contour refinement specifications.
        """

        for defn, info in self._contours_info.items():
            yield defn, info




class _RefinementContourCollector:
    """Collection for extracted refinement contours

    Extracts and stores on the disk the contours specified by the user.
    Provides interator interface for looping over the collection items.
    """

    def __init__(
            self,
            contours_info: _RefinementContourInfoCollector
            ) -> None:
        """Initialize the collection object with empty output list.

        Parameters
        ----------
        contours_info : _RefinementPatchInfoCollector
            Handle to the collection of user specified contours
            specification.
        """

        self._contours_info = contours_info
        self._container: List[Union[Tuple, None]] = []

    def calculate(
            self,
            source_list: Iterable[HfunRaster],
            out_path: Union[Path, str]
            ) -> None:
        """Extract specified contours and store on disk in `out_path`.

        Parameters
        ----------
        source_list : list of HfunRaster
            List of raster size functions from which the contours
            must be calculated.
        out_path : path-like
            Path for storing calculated contours and their crs data.

        Returns
        -------
        None
        """

        out_dir = Path(out_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        file_counter = 0
        pid = os.getpid()
        self._container.clear()
        for contour_defn, size_info in self._contours_info:
            if not contour_defn.has_source:
                # Copy so that in case of a 2nd run the no-source
                # contour still gets all current sources
                contour_defn = copy(contour_defn)
                for source in source_list:
                    contour_defn.add_source(source)

            for contour, crs in contour_defn.iter_contours():
                file_counter = file_counter + 1
                feather_path = out_dir / f"contour_{pid}_{file_counter}.feather"
                crs_path = out_dir / f"crs_{pid}_{file_counter}.json"
                gpd.GeoDataFrame(
                    { 'geometry': [contour],
                      'expansion_rate': size_info['expansion_rate'],
                      'target_size': size_info['target_size'],
                    },
                    crs=crs).to_feather(feather_path)
                gc.collect()
                with open(crs_path, 'w') as fp:
                    fp.write(crs.to_json())
                self._container.append((feather_path, crs_path))


    def __iter__(self) -> gpd.GeoDataFrame:
        """Iterator method for this collection object

        Yields
        ------
        gpd.GeoDataFrame
            Data from containing the extracted contour line and the
            CRS information.
        """

        for raster_data in self._container:
            feather_path, crs_path = raster_data
            gdf = gpd.read_feather(feather_path)
            with open(crs_path) as fp:
                gdf.set_crs(CRS.from_json(fp.read()))
            yield gdf




class _ConstantValueContourInfoCollector:
    """Collection for constant value refinement specification"""

    def __init__(self) -> None:
        self._contours_info = {}

    def add(self,
            src_idx: Optional[Sequence[int]],
            contour_defn0: Contour,
            contour_defn1: Contour,
            value: float
            ) -> None:
        """Add a new fixed-value refinement specification to the spec.

        Parameters
        ----------
        src_idx : tuple of int or None
            Indices of sources (indexed based on all `HfunCollector`
            **not** just rasters) on which constant value refinement
            must be applied.
        contour_defn0 : Contour
            Lower bound of region to apply constant value refinement.
        contour_defn1 : Contour
            Upper bound of region to apply constant value refinement.
        value : float
            Fixed-value to be applied as refinement.

        Returns
        -------
        None
        """

        srcs = tuple(src_idx) if src_idx is not None else None
        self._contours_info[
                (srcs, contour_defn0, contour_defn1)] = value

    def __iter__(self) -> Tuple[Tuple[Sequence[int], Contour, Contour], dict]:
        """Iterator method for this collection object

        Yields
        ------
        defn : Tuple[Sequence[int], Contour, Contour]
            The lower and upper bound contours definitions provided
            by the user as well as the list of source.
        info : dict
            Dictionary of specifications for constant value refinement.
        """

        for defn, info in self._contours_info.items():
            yield defn, info



class _RefinementPatchInfoCollector:
    """Collection for patch refinement specifications"""

    def __init__(self) -> None:
        self._patch_info = {}

    def add(self, patch_defn: Patch, **size_info: Any) -> None:
        """Add patch refinement specifications to the collection

        Parameters
        ----------
        patch_defn : Patch
            Shape of the region to apply the refinement during
            application.
        size_info : dict
            Information related to patch application such as
            target size, rate, etc.

        Returns
        -------
        None
        """

        self._patch_info[patch_defn] = size_info

    def __iter__(self) -> Tuple[Patch, dict]:
        """Iterator method for this collection object

        Yields
        ------
        defn : Patch
            Patch object representing the shape of the patch area.
        info : dict
            Dictionary of specifications for patch refinement.
        """

        for defn, info in self._patch_info.items():
            yield defn, info



class _FlowLimiterInfoCollector:
    """Collection for subtidal flow limiter refinement spec."""

    def __init__(self) -> None:
        self._flow_lim_info = []

    def add(self,
            src_idx: Optional[Sequence[int]],
            hmin: float,
            hmax: float,
            upper_bound: float,
            lower_bound: float
            ) -> None:
        """Add subtidal flow limiter refinement spec to the collection.

        Parameters
        ----------
        src_idx : tuple of int or None
            Indices of sources (indexed based on all `HfunCollector`
            **not** just rasters) on which subtidal flow limiter
            refinement must be applied.
        hmin : float
            Minimum mesh size to be applied based on limiter
            calculations.
        hmax : float
            Maximum mesh size to be applied based on limiter
            calculations.
        upper_bound : float
            Elevation upper bound of the area that the limiter
            refinement is applied.
        lower_bound : float
            Elevation lower bound of the area that the limiter
            refinement is applied.

        Returns
        -------
        None
        """

        srcs = tuple(src_idx) if src_idx is not None else None
        self._flow_lim_info.append(
                (src_idx, hmin, hmax, upper_bound, lower_bound))

    def __iter__(self) -> Tuple[Sequence[int], float, float, float, float]:
        """Iterator method for this collection object

        Yields
        ------
        src_idx : tuple of int
            Similar to `add`
        hmin : float
            Similar to `add`
        hmax : float
            Similar to `add`
        ub : float
            Similar to `add`
        lb : float
            Similar to `add`
        """

        for src_idx, hmin, hmax, ub, lb in self._flow_lim_info:
            yield src_idx, hmin, hmax, ub, lb



class _ChannelRefineInfoCollector:
    """Collection for channel refinement specifications"""

    def __init__(self) -> None:
        self._ch_info_dict = {}

    def add(self,
            channel_defn: Channel,
            **size_info: Any
            ) -> None:
        """Add channel refinement spec to the collection.

        Parameters
        ----------
        channel_defn : Channel
            Definition of channel detection specification.
        size_info : dict
            Information related to channel refinement application
            such as target size, rate, etc.

        Returns
        -------
        None
        """

        self._ch_info_dict[channel_defn] = size_info

    def __iter__(self) -> Tuple[Channel, dict]:
        """Iterator method for this collection object

        Yields
        ------
        defn : Channel
            Similar to `channel_defn` in `add`
        info : dict
            Similar to `size_info` in `add`
        """

        for defn, info in self._ch_info_dict.items():
            yield defn, info


class _ChannelRefineCollector:
    """Collection for extracted refinement channels"""

    def __init__(self, channels_info) -> None:
        """Initialize the collection object with empty output list.

        Parameters
        ----------
        channels_info : _ChannelRefineInfoCollector
            Handle to the collection of user specified channel
            refinement specification.
        """

        self._channels_info = channels_info
        self._container: List[Union[Tuple, None]] = []

    def calculate(
            self,
            source_list,
            out_path
            ) -> None:
        """Extract specified channels and store on disk in `out_path`.

        Parameters
        ----------
        source_list : list of HfunRaster
            List of raster size functions from which the channels
            must be calculated.
        out_path : path-like
            Path for storing calculated channels and their crs data.

        Returns
        -------
        None
        """

        out_dir = Path(out_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        file_counter = 0
        pid = os.getpid()
        self._container.clear()
        for channel_defn, size_info in self._channels_info:
            if not channel_defn.has_source:
                # Copy so that in case of a 2nd run the no-source
                # channel still gets all current sources
                channel_defn = copy(channel_defn)
                for source in source_list:
                    channel_defn.add_source(source)

            for channels, crs in channel_defn.iter_channels():
                file_counter = file_counter + 1
                feather_path = out_dir / f"channels_{pid}_{file_counter}.feather"
                crs_path = out_dir / f"crs_{pid}_{file_counter}.json"
                gpd.GeoDataFrame(
                    { 'geometry': [channels],
                      'expansion_rate': size_info['expansion_rate'],
                      'target_size': size_info['target_size'],
                    },
                    crs=crs).to_feather(feather_path)
                gc.collect()
                with open(crs_path, 'w') as fp:
                    fp.write(crs.to_json())
                self._container.append((feather_path, crs_path))

    def __iter__(self):
        """Iterator method for this collection object

        Yields
        ------
        gpd.GeoDataFrame
            Data from containing the extracted channel polygons and
            the CRS information.
        """

        for raster_data in self._container:
            feather_path, crs_path = raster_data
            gdf = gpd.read_feather(feather_path)
            with open(crs_path) as fp:
                gdf.set_crs(CRS.from_json(fp.read()))
            yield gdf


class _ConstraintInfoCollector:
    """Collection for the applied constraints"""

    def __init__(self) -> None:
        self._constraints_info = []

    def add(self,
            src_idx: Optional[Sequence[int]],
            constraint: Union[TopoConstConstraint, TopoFuncConstraint]
            ) -> None:
        """Add size function constraint spec to the collection.

        Parameters
        ----------
        src_idx : tuple of int or None
            Indices of sources (indexed based on all `HfunCollector`
            **not** just rasters) on which constant value refinement
            must be applied.
        constraint : TopoConstConstraint or TopoFuncConstraint
            The constraint definition object.

        Returns
        -------
        None
        """

        srcs = tuple(src_idx) if src_idx is not None else None
        self._constraints_info.append((srcs, constraint))

    def __iter__(self) -> Union[TopoConstConstraint, TopoFuncConstraint]:
        """Iterator method for this collection object

        Yields
        ------
        TopoConstConstraint or TopoFuncConstraint
            The constraint object provided in `add`
        """

        for defn in self._constraints_info:
            yield defn



class HfunCollector(BaseHfun):
    """Define size function based on multiple inputs of different types

    Attributes
    ----------

    Methods
    -------
    msh_t()
        Return mesh sizes interpolated on an size-optimized
        unstructured mesh
    add_topo_bound_constraint(...)
        Add size fixed-per-point value constraint to the area
        bounded by specified bounds with expansion/contraction
        rate `rate` specified. This refinement is only applied on
        rasters with specified indices. The index is w.r.t the
        full input list for collector object creation.
    add_topo_func_constraint(upper_bound=np.inf, lower_bound=-np.inf,
                             value_type='min', rate=0.01)
        Add size value constraint based on function of depth/elevation
        to the area bounded by specified bounds with the expansion or
        contraction rate `rate` specified. This constraint is only
        applied on rasters with specified indices. The index is w.r.t
        the full input list for collector object creation.
    add_patch(...)
        Add a region of fixed size refinement with optional expansion
        rate for points outside the region to achieve smooth size
        transition.
    add_contour(...)
        Add refinement based on contour lines auto-extrcted from the
        underlying raster data. The size is calculated based on the
        specified `rate`, `target_size` and the distance from the
        extracted feature line. For refinement contours are extracted
        only from raster inputs, but are applied on all input size
        function bases.
    add_channel(...)
        Add refinement for auto-detected narrow domain regions.
        Optionally use an expansion rate for points outside detected
        narrow regions for smooth size transition.
    add_subtidal_flow_limiter(...)
        Add mesh size refinement based on the value as well as
        gradient of the topography within the region between
        specified by lower and upper bound on topography.
        This refinement is only applied on rasters with specified
        indices. The index is w.r.t the full input list for collector
        object creation.
    add_constant_value(...)
        Add fixed size mesh refinement in the region specified by
        upper and lower bounds on topography.  This refinement is
        only applied on rasters with specified indices. The index is
        w.r.t the full input list for collector object creation.

    Notes
    -----
    All the refinements and constraints of this collector size
    function are applied lazily. That means the size values are **not**
    evaluated at the time they are called. Instead the effect of
    all these refinements and constraints on the size is calculated
    when `msh_t()` method is called.

    Two distinct algorithms are implemented for storing the size
    function values during evaluation and before creating the
    "background mesh" on which sizes are specified. Currently
    the difference between algorithms are only due to how raster
    inputs to the collector are processed. The **exact** algorithm
    is more computationally expensive; it processes all refinements
    on the original rasters and applies the indivitually on all
    of those individual rasters; this results in exponential time
    calculation of contours or raster features as the extracted
    features on all rasters must be applied on all rasters. Application
    of features usually involve distance calculation using a tree
    and can be very expensive when many rasters-features are involved.
    The **fast** approach is less exact and can use more memory, but
    it is much faster. The approach it takes is to still extract
    the raster features individually but then apply it to a lower
    resolution large raster that covers all the input rasters.
    """

    def __init__(
            self,
            in_list: CanCreateMultipleHfun,
            base_mesh: Optional[Mesh] = None,
            hmin: Optional[float] = None,
            hmax: Optional[float] = None,
            nprocs: Optional[int] = None,
            verbosity: int = 0,
            method: Literal['exact', 'fast'] = 'exact',
            base_as_hfun: bool = True,
            base_shape: Optional[Union[Polygon, MultiPolygon]] = None,
            base_shape_crs: Union[str, CRS] = 'EPSG:4326'
            ) -> None:
        """Initialize a collector size function object

        Parameters
        ----------
        in_list : CanCreateMultipleHfun
        base_mesh : Mesh or None, default=None
        hmin : float or None, default=None
        hmax : float or None, default=None
        nprocs : int or None, default=None
        verbosity : int, default=0
        method : {'exact', 'fast'}, default='exact
        base_as_hfun : bool, default=True
        base_shape: Polygon or MultiPolygon or None, default=None
        base_shape_crs: str or CRS, default='EPSG:4326'
        """

        # NOTE: Input Hfuns and their Rasters can get modified

        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs

        self._applied = False
        self._size_info = dict(hmin=hmin, hmax=hmax)
        self._nprocs = nprocs
        self._hfun_list = []
        self._method = method

        self._base_shape = base_shape
        self._base_shape_crs = CRS.from_user_input(base_shape_crs)
        self._base_as_hfun = base_as_hfun

        # NOTE: Base mesh has to have a crs otherwise HfunMesh throws
        # exception
        self._base_mesh = None
        if base_mesh:
            self._base_mesh = HfunMesh(base_mesh)
            if self._base_as_hfun:
                self._base_mesh.size_from_mesh()

        self._contour_info_coll = _RefinementContourInfoCollector()
        self._contour_coll = _RefinementContourCollector(
                self._contour_info_coll)

        self._const_val_contour_coll = _ConstantValueContourInfoCollector()

        self._refine_patch_info_coll = _RefinementPatchInfoCollector()

        self._flow_lim_coll = _FlowLimiterInfoCollector()

        self._ch_info_coll = _ChannelRefineInfoCollector()
        self._channels_coll = _ChannelRefineCollector(
                self._ch_info_coll)

        self._constraint_info_coll = _ConstraintInfoCollector()

        self._type_chk(in_list)

        # TODO: Interpolate max size on base mesh basemesh?
        #
        # TODO: CRS considerations

        for in_item in in_list:
            # Add supports(ext) to each hfun type?

            if isinstance(in_item, BaseHfun):
                hfun = in_item

            elif isinstance(in_item, Raster):
                if self._base_shape:
                    clip_shape = self._base_shape
                    if not self._base_shape_crs.equals(in_item.crs):
                        transformer = Transformer.from_crs(
                            self._base_shape_crs, in_item.crs, always_xy=True)
                        clip_shape = ops.transform(
                                transformer.transform, clip_shape)
                    try:
                        in_item.clip(clip_shape)
                    except ValueError as err:
                        # This raster does not intersect shape
                        _logger.debug(err)
                        continue

                elif self._base_mesh:
                    try:
                        in_item.clip(self._base_mesh.mesh.get_bbox(crs=in_item.crs))
                    except ValueError as err:
                        # This raster does not intersect shape
                        _logger.debug(err)
                        continue

                hfun = HfunRaster(in_item, **self._size_info)

            elif isinstance(in_item, EuclideanMesh2D):
                hfun = HfunMesh(in_item)

            elif isinstance(in_item, str):
                if in_item.endswith('.tif'):
                    raster = Raster(in_item)
                    if self._base_shape:
                        clip_shape = self._base_shape
                        if not self._base_shape_crs.equals(raster.crs):
                            transformer = Transformer.from_crs(
                                self._base_shape_crs, raster.crs, always_xy=True)
                            clip_shape = ops.transform(
                                    transformer.transform, clip_shape)
                        try:
                            in_item.clip(clip_shape)
                        except ValueError as err:
                            # This raster does not intersect shape
                            _logger.debug(err)
                            continue

                    elif self._base_mesh:
                        try:
                            raster.clip(self._base_mesh.mesh.get_bbox(crs=raster.crs))
                        except ValueError as err:
                            # This raster does not intersect shape
                            _logger.debug(err)
                            continue

                    hfun = HfunRaster(raster, **self._size_info)

                elif in_item.endswith(
                        ('.14', '.grd', '.gr3', '.msh', '.2dm')):
                    mesh = Mesh.open(in_item)
                    hfun = HfunMesh(mesh)

                else:
                    raise TypeError("Input file extension not supported!")

            self._hfun_list.append(hfun)


    def msh_t(self) -> jigsaw_msh_t:
        """Interpolates mesh size functions on an unstructred mesh

        Calculates and the interpolate the mesh sizes from all inputs
        onto an unstructured mesh. This mesh is generated by meshing
        the inputs using the size function values. The return
        value is in a projected CRS. If the inputs CRS the same and
        geographic, then a local UTM CRS is calculated and used
        for the output of this method.

        Parameters
        ----------

        Returns
        -------
        jigsaw_msh_t
            Size function calculated and interpolated on an
            unstructured mesh.

        Notes
        -----
        The actual application of refinements and constrains for this
        collector size function happens after calling this method.
        This calculation is cached in case of 'exact' algorithm and
        not cached for 'fast' algorithm.
        """

        composite_hfun = jigsaw_msh_t()

        if self._method == 'exact':
            self._apply_features()

            with tempfile.TemporaryDirectory() as temp_dir:
                hfun_path_list = self._write_hfun_to_disk(temp_dir)
                composite_hfun = self._get_hfun_composite(hfun_path_list)


        elif self._method == 'fast':

            with tempfile.TemporaryDirectory() as temp_dir:
                rast = self._create_big_raster(temp_dir)
                hfun = self._apply_features_fast(rast)
                composite_hfun = self._get_hfun_composite_fast(hfun)

        else:
            raise ValueError(f"Invalid method specified: {self._method}")

        return composite_hfun


    def add_topo_bound_constraint(
            self,
            value: Union[float, npt.NDArray[np.float32]],
            upper_bound: float = np.inf,
            lower_bound: float = -np.inf,
            value_type: Literal['min', 'max'] = 'min',
            rate: float = 0.01,
            source_index: Union[List[int], int, None] = None
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
        source_index : int or list of ints or None, default=None
            The index of raster entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None
        """

        self._applied = False

        constraint_defn = TopoConstConstraint(
            value, upper_bound, lower_bound, value_type, rate)

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]
        self._constraint_info_coll.add(source_index, constraint_defn)


    def add_topo_func_constraint(
            self,
            func: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
                = lambda i: i / 2.0,
            upper_bound: float = np.inf,
            lower_bound: float = -np.inf,
            value_type: Literal['min', 'max'] = 'min',
            rate: float = 0.01,
            source_index: Union[List[int], int, None] = None
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
        source_index : int or list of ints or None, default=None
            The index of raster entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None
        """

        self._applied = False

        constraint_defn = TopoFuncConstraint(
            func, upper_bound, lower_bound, value_type, rate)

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]
        self._constraint_info_coll.add(source_index, constraint_defn)


    def add_courant_num_constraint(
            self,
            upper_bound: float = 0.9,
            lower_bound: Optional[float] = None,
            timestep: float = 150,
            wave_amplitude: float = 2,
            source_index: Union[List[int], int, None] = None
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
        source_index : int or list of ints or None, default=None
            The index of raster entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None
        """

        self._applied = False

        # TODO: Validate conflicting constraints, right now last one wins
        if upper_bound is None and lower_bound is None:
            raise ValueError("Both upper and lower Courant bounds can NOT be None!")

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]

        if upper_bound is not None:
            self._constraint_info_coll.add(
                source_index,
                CourantNumConstraint(
                    value=upper_bound,
                    timestep=timestep,
                    wave_amplitude=wave_amplitude,
                    value_type='min'
                )
            )
        if lower_bound is not None:
            self._constraint_info_coll.add(
                source_index,
                CourantNumConstraint(
                    value=lower_bound,
                    timestep=timestep,
                    wave_amplitude=wave_amplitude,
                    value_type='min'
                )
            )



    def add_contour(
            self,
            level: Union[List[float], float, None] = None,
            expansion_rate: float = 0.01,
            target_size: Optional[float] = None,
            contour_defn: Optional[Contour] = None
            ) -> None:
        """Add refinement for auto extracted contours

        Add refinement for the contour lines extracted based on
        level or levels specified by `level`. The refinement
        is relaxed with `expansion_rate` and distance from the
        extracted contour lines. Contours are extracted only from the
        raster inputs, but are applied on all inputs passed to the
        collector constructor.

        Parameters
        ----------
        level : float or list of floats or None, default=None
            Level(s) at which contour lines should be extracted.
        expansion_rate : float, default=0.01
            Rate to use for expanding refinement with distance away
            from the extracted contours.
        target_size : float or None, default=None
            Target size to use on the extracted contours and expand
            from with distance.
        contour_defn : Contour or None, default=None
            Contour definition objects which defines contour extraction
            specification from rasters.

        Returns
        -------
        None
        """

        # Always lazy
        self._applied = False

        levels = []
        if isinstance(level, (list, tuple)):
            levels.extend(level)
        else:
            levels.append(level)


        contour_defns = []
        if contour_defn is None:
            for lvl in levels:
                contour_defns.append(Contour(level=lvl))

        elif not isinstance(contour_defn, Contour):
            raise TypeError(
                f"Contour definition must be of type {Contour} not"
                f" {type(contour_defn)}!")

        elif level is not None:
            msg = "Level is ignored since a contour definition is provided!"
            warnings.warn(msg)
            _logger.info(msg)

        else:
            contour_defns.append(contour_defn)

        for ctr_dfn in contour_defns:
            self._contour_info_coll.add(
                ctr_dfn,
                expansion_rate=expansion_rate,
                target_size=target_size)

    def add_channel(
            self,
            level: float = 0,
            width: float = 1000,
            target_size: float = 200,
            expansion_rate: Optional[float] = None,
            tolerance: Optional[float] = None,
            channel_defn: Optional[Channel] = None
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
        tolerance : float or None, default=None
            Tolerance to use for simplifying the polygon extracted
            from DEM data. If `None` don't simplify.
        channel_defn : Contour or None
            Channel definition objects which defines channel
            extraction specification from rasters.

        Returns
        -------
        None
        """

        self._applied = False

        # Always lazy
        self._applied = False

        # Even a tolerance of 1 for simplifying polygon for channel
        # calculations is much faster than no simplification. 50
        # is much faster than 1. The reason is in simplify we don't
        # preserve topology
        if channel_defn is None:
            channel_defn = Channel(
                level=level, width=width, tolerance=tolerance)

        elif not isinstance(channel_defn, Channel):
            raise TypeError(
                f"Channel definition must be of type {Channel} not"
                f" {type(channel_defn)}!")

        self._ch_info_coll.add(
            channel_defn,
            expansion_rate=expansion_rate,
            target_size=target_size)


    def add_subtidal_flow_limiter(
            self,
            hmin: Optional[float] = None,
            hmax: Optional[float] = None,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None,
            source_index: Union[List[int], int, None] = None
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
        source_index : int or list of ints or None, default=None
            The index of raster entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None

        See Also
        --------
        hfun.raster.HfunRaster.add_subtidal_flow_limiter :
        """

        self._applied = False

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]

        # TODO: Checks on hmin/hmax, etc?

        self._flow_lim_coll.add(
            source_index,
            hmin=hmin,
            hmax=hmax,
            upper_bound=upper_bound,
            lower_bound=lower_bound)


    def add_constant_value(
            self,
            value: float,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None,
            source_index: Union[List[int], int, None] =None):
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
        source_index : int or list of ints or None, default=None
            The index of raster entries from the input list argument
            of the constructor of collector size function. If `None`
            all input rasters are used.

        Returns
        -------
        None
        """

        self._applied = False

        contour_defn0 = None
        contour_defn1 = None
        if lower_bound is not None and not np.isinf(lower_bound):
            contour_defn0 = Contour(level=lower_bound)
        if upper_bound is not None and not np.isinf(upper_bound):
            contour_defn1 = Contour(level=upper_bound)

        if source_index is not None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]
        self._const_val_contour_coll.add(
            source_index, contour_defn0, contour_defn1, value)


    def add_patch(
            self,
            shape: Union[MultiPolygon, Polygon, None] = None,
            patch_defn: Optional[Patch] = None,
            shapefile: Union[None, str, Path] = None,
            expansion_rate: Optional[float] = None,
            target_size: Optional[float] = None,
            ) -> None:
        """Add refinement as a region of fixed size with an optional rate

        Add a refinement based on a region specified by `shape`,
        `patch_defn` or `shapefile`.  The fixed `target_size`
        refinement can be expanded outside the region specified by the
        shape if `expansion_rate` is provided.

        Parameters
        ----------
        shape : MultiPolygon or Polygon or None, default=None
            Shape of the region to use specified `target_size` for
            refinement. Only one of `shape`, `patch_defn` or `shapefile`
            must be specified.
        patch_defn : Patch or None, default=None
            Shape of the region to use specified `target_size` for
            refinement. Only one of `shape`, `patch_defn` or `shapefile`
            must be specified.
        shapefile : None or str or Path, default=None
            Shape of the region to use specified `target_size` for
            refinement. Only one of `shape`, `patch_defn` or `shapefile`
            must be specified.
        expansion_rate : float or None, default=None
            Optional rate to use for expanding refinement outside
            the specified shape in `multipolygon`.
        target_size : float or None, default=None
            Fixed target size of mesh to use for refinement in
            `multipolygon`

        Returns
        -------
        None
        """

        # "shape" should be in 4326 CRS. For shapefile or patch_defn
        # CRS info is included

        self._applied = False

        if not patch_defn:
            if shape:
                patch_defn = Patch(shape=shape)

            elif shapefile:
                patch_defn = Patch(shapefile=shapefile)

        self._refine_patch_info_coll.add(
            patch_defn,
            expansion_rate=expansion_rate,
            target_size=target_size)


    @staticmethod
    def _type_chk(input_list: List[Any]) -> None:
        """Checks the if the input types are supported for size function

        Checks if size function collector supports handling
        size functions created from the input types.

        Parameters
        ----------
        input_list : List[Any]

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the any of the inputs are not supported
        """

        valid_types = (str, Raster, Mesh, HfunRaster, HfunMesh)
        if not all(isinstance(item, valid_types) for item in input_list):
            raise TypeError(
                f'Input list items must be of type'
                f' {", ".join(str(i) for i in valid_types)},'
                f' or a derived type.')

    def _apply_features(self) -> None:
        """Internal: apply all specified refinements and constraints

        Apply all specified refinements and constrains for the exact
        algorithm.

        Parameters
        ----------

        Returns
        -------
        None

        See Also
        --------
        _apply_features_fast :
        """

        if not self._applied:
            self._apply_contours()
            self._apply_flow_limiters()
            self._apply_const_val()
            self._apply_patch()
            self._apply_channels()
            self._apply_constraints()

        self._applied = True


    def _apply_constraints(self) -> None:
        """Internal: apply specified constraints.

        Apply specified constraints for the exact algorithm.

        Parameters
        ----------

        Returns
        -------
        None

        See Also
        --------
        _apply_constraints_fast :
        """

        if self._method == 'fast':
            raise NotImplementedError(
                "This function does not suuport fast hfun method")

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        for in_idx, hfun in enumerate(raster_hfun_list):
            constraint_list = []
            for src_idx, constraint_defn in self._constraint_info_coll:
                if src_idx is not None and in_idx not in src_idx:
                    continue

                constraint_list.append(constraint_defn)

            if constraint_list:
                hfun.apply_constraints(constraint_list)


    def _apply_contours(self, apply_to: Optional[SizeFuncList] = None) -> None:
        """Internal: apply specified constraints.

        Parameters
        ----------
        apply_to : SizeFuncList or None, default=None
            Size functions on which contours must be applied. If `None`
            all inputs are used to apply the calculated contours.

        Returns
        -------
        None
        """

        # TODO: Consider CRS before applying to different hfuns
        #
        # NOTE: for parallelization make sure a single hfun is NOT
        # passed to multiple processes

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        if apply_to is None:
            mesh_hfun_list = [
                i for i in self._hfun_list if isinstance(i, HfunMesh)]
            if self._base_mesh and self._base_as_hfun:
                mesh_hfun_list.insert(0, self._base_mesh)
            apply_to = [*mesh_hfun_list, *raster_hfun_list]

        with tempfile.TemporaryDirectory() as temp_path:
            with Pool(processes=self._nprocs) as p:
                # Contours are ONLY extracted from raster sources
                self._contour_coll.calculate(raster_hfun_list, temp_path)
                counter = 0
                for hfun in apply_to:
                    for gdf in self._contour_coll:
                        for row in gdf.itertuples():
                            _logger.debug(row)
                            shape = row.geometry
                            if isinstance(shape, GeometryCollection):
                                continue
                            # NOTE: CRS check is done AFTER
                            # GeometryCollection check because
                            # gdf.to_crs results in an error in case
                            # of empty GeometryCollection
                            if not gdf.crs.equals(hfun.crs):
                                _logger.info("Reprojecting feature...")
                                transformer = Transformer.from_crs(
                                    gdf.crs, hfun.crs, always_xy=True)
                                shape = ops.transform(
                                        transformer.transform, shape)
                            counter = counter + 1
                            hfun.add_feature(**{
                                'feature': shape,
                                'expansion_rate': row.expansion_rate,
                                'target_size': row.target_size,
                                'pool': p
                            })
            p.join()
            # hfun objects cause issue with pickling
            # -> cannot be passed to pool
#            with Pool(processes=self._nprocs) as p:
#                p.starmap(
#                    _apply_contours_worker,
#                    [(hfun, self._contour_coll, self._nprocs)
#                     for hfun in apply_to])

    def _apply_channels(self, apply_to: Optional[SizeFuncList] = None) -> None:
        """Internal: apply specified channel refinements.

        Parameters
        ----------
        apply_to : SizeFuncList or None, default=None
            Size functions on which channels must be applied. If `None`
            all inputs are used to apply the calculated channels.

        Returns
        -------
        None
        """

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        if apply_to is None:
            mesh_hfun_list = [
                i for i in self._hfun_list if isinstance(i, HfunMesh)]
            if self._base_mesh and self._base_as_hfun:
                mesh_hfun_list.insert(0, self._base_mesh)
            apply_to = [*mesh_hfun_list, *raster_hfun_list]

        with tempfile.TemporaryDirectory() as temp_path:
            # Channels are ONLY extracted from raster sources
            self._channels_coll.calculate(raster_hfun_list, temp_path)
            counter = 0
            for hfun in apply_to:
                for gdf in self._channels_coll:
                    for row in gdf.itertuples():
                        _logger.debug(row)
                        shape = row.geometry
                        if isinstance(shape, GeometryCollection):
                            continue
                        # NOTE: CRS check is done AFTER
                        # GeometryCollection check because
                        # gdf.to_crs results in an error in case
                        # of empty GeometryCollection
                        if not gdf.crs.equals(hfun.crs):
                            _logger.info("Reprojecting feature...")
                            transformer = Transformer.from_crs(
                                gdf.crs, hfun.crs, always_xy=True)
                            shape = ops.transform(
                                    transformer.transform, shape)
                        counter = counter + 1
                        hfun.add_patch(**{
                            'multipolygon': shape,
                            'expansion_rate': row.expansion_rate,
                            'target_size': row.target_size,
                            'nprocs': self._nprocs
                        })


    def _apply_flow_limiters(self) -> None:
        """Internal: apply specified sub tidal flow limiter refinements

        Applies specified subtidal flow limiter refinements for
        the exact algorithm.

        Parameters
        ----------

        Returns
        -------
        None

        See Also
        --------
        _apply_flow_limiters_fast :
        """

        if self._method == 'fast':
            raise NotImplementedError(
                "This function does not suuport fast hfun method")

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        for in_idx, hfun in enumerate(raster_hfun_list):
            for src_idx, hmin, hmax, zmax, zmin in self._flow_lim_coll:
                if src_idx is not None and in_idx not in src_idx:
                    continue
                if hmin is None:
                    hmin = self._size_info['hmin']
                if hmax is None:
                    hmax = self._size_info['hmax']
                hfun.add_subtidal_flow_limiter(hmin, hmax, zmax, zmin)


    def _apply_const_val(self):
        """Internal: apply specified constant value refinements.

        Applies constant value refinements for the exact algorithm.

        Returns
        -------
        None

        See Also
        --------
        _apply_const_val_fast :
        """

        if self._method == 'fast':
            raise NotImplementedError(
                "This function does not suuport fast hfun method")

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        for in_idx, hfun in enumerate(raster_hfun_list):
            for (src_idx, ctr0, ctr1), const_val in self._const_val_contour_coll:
                if src_idx is not None and in_idx not in src_idx:
                    continue
                level0 = None
                level1 =  None
                if ctr0 is not None:
                    level0 = ctr0.level
                if ctr1 is not None:
                    level1 = ctr1.level
                hfun.add_constant_value(const_val, level0, level1)


    def _apply_patch(self, apply_to: Optional[SizeFuncList] = None) -> None:
        """Internal: apply the specified patch refinements.

        Parameters
        ----------
        apply_to : SizeFuncList or None, default=None
            Size functions on which patches must be applied. If `None`
            all inputs are used to apply the patches.

        Returns
        -------
        None
        """

        raster_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        if apply_to is None:
            mesh_hfun_list = [
                i for i in self._hfun_list if isinstance(i, HfunMesh)]
            if self._base_mesh and self._base_as_hfun:
                mesh_hfun_list.insert(0, self._base_mesh)
            apply_to = [*mesh_hfun_list, *raster_hfun_list]

        # TODO: Parallelize
        for hfun in apply_to:
            for patch_defn, size_info in self._refine_patch_info_coll:
                shape, crs = patch_defn.get_multipolygon()
                if hfun.crs != crs:
                    transformer = Transformer.from_crs(
                        crs, hfun.crs, always_xy=True)
                    shape = ops.transform(
                            transformer.transform, shape)

                hfun.add_patch(
                        shape, nprocs=self._nprocs, **size_info)


    def _write_hfun_to_disk(
            self,
            out_path: Union[str, Path]
            ) -> List[Union[str, Path]]:
        """Internal: write individual size function output mesh to file

        Calculate the interpolated on-mesh size function from each
        individual input, clip overlaps based on priority, and
        write the results to disk for later combining.

        Parameters
        ----------
        out_path : path-like
            The path of the (temporary) directory to which mesh size
            functions must be written.

        Returns
        -------
        list of path-like
            List of individual file path for mesh size function of
            each input.
        """

        out_dir = Path(out_path)
        path_list = []
        file_counter = 0
        pid = os.getpid()
        bbox_list = []

        hfun_list = self._hfun_list[::-1]
        if self._base_mesh and self._base_as_hfun:
            hfun_list = [*self._hfun_list[::-1], self._base_mesh]

        # Last user input item has the highest priority (its trias
        # are not dropped) so process in reverse order
        for hfun in hfun_list:
            # TODO: Calling msh_t() on HfunMesh more than once causes
            # issue right now due to change in crs of internal Mesh

            # To avoid removing verts and trias from mesh hfuns
            hfun_mesh = deepcopy(hfun.msh_t())
            # If no CRS info, we assume EPSG:4326
            if hasattr(hfun_mesh, "crs"):
                dst_crs = CRS.from_user_input("EPSG:4326")
                if hfun_mesh.crs != dst_crs:
                    utils.reproject(hfun_mesh, dst_crs)

            # Get all previous bbox and clip to resolve overlaps
            # removing all tria that have NODE in bbox because it's
            # faster and so we can resolve all overlaps
            _logger.info("Removing bounds from hfun mesh...")
            for ibox in bbox_list:
                hfun_mesh = utils.clip_mesh_by_shape(
                    hfun_mesh,
                    ibox,
                    use_box_only=True,
                    fit_inside=True,
                    inverse=True)

            if len(hfun_mesh.vert2) == 0:
                _logger.debug("Hfun ignored due to overlap")
                continue

            # Check hfun_mesh.value against hmin & hmax
            hmin = self._size_info['hmin']
            hmax = self._size_info['hmax']
            if hmin:
                hfun_mesh.value[hfun_mesh.value < hmin] = hmin
            if hmax:
                hfun_mesh.value[hfun_mesh.value > hmax] = hmax

            mesh = Mesh(hfun_mesh)
            bbox_list.append(mesh.get_bbox(crs="EPSG:4326"))
            file_counter = file_counter + 1
            _logger.info(f'write mesh {file_counter} to file...')
            file_path = out_dir / f'hfun_{pid}_{file_counter}.2dm'
            mesh.write(file_path, format='2dm')
            path_list.append(file_path)
            _logger.info('Done writing 2dm file.')
            del mesh
            gc.collect()
        return path_list



    def _get_hfun_composite(
            self,
            hfun_path_list: List[Union[str, Path]]
            ) -> jigsaw_msh_t:
        """Internal: combine the size functions written to disk

        Combine the size functions written to disk from the list of
        input files `hfun_path_list`. This is used for `exact` method.

        Parameters
        ----------
        hfun_path_list : list of path-like

        Retruns
        -------
        jigsaw_msh_t
            The combined size function interpolated on an optimized
            mesh.

        See Also
        --------
        _get_hfun_composite_fast :
        """

        collection = []
        _logger.info('Reading 2dm hfun files...')
        start = time()
        for path in hfun_path_list:
            collection.append(Mesh.open(path, crs='EPSG:4326'))
        _logger.info(f'Reading 2dm hfun files took {time()-start}.')

        # NOTE: Overlaps are taken care of in the write stage

        coord = []
        index = []
        value = []
        offset = 0
        for hfun in collection:
            index.append(hfun.tria3['index'] + offset)
            coord.append(hfun.coord)
            value.append(hfun.value)
            offset += hfun.coord.shape[0]

        composite_hfun = jigsaw_msh_t()
        composite_hfun.mshID = 'euclidean-mesh'
        composite_hfun.ndims = 2

        composite_hfun.vert2 = np.array(
                [(coo, 0) for coo in np.vstack(coord)],
                dtype=jigsaw_msh_t.VERT2_t)
        composite_hfun.tria3 = np.array(
                [(idx, 0) for idx in np.vstack(index)],
                dtype=jigsaw_msh_t.TRIA3_t)
        composite_hfun.value = np.array(
                np.vstack(value),
                dtype=jigsaw_msh_t.REALS_t)

        composite_hfun.crs = CRS.from_user_input("EPSG:4326")

        # NOTE: In the end we need to return in a CRS that
        # uses meters as units. UTM based on the center of
        # the bounding box of the hfun is used
        # Up until now all calculation was in EPSG:4326
        utils.msh_t_to_utm(composite_hfun)

        return composite_hfun


    def _create_big_raster(self, out_path: Union[str, Path]) -> Raster:
        """Internal: create a large raster covering all input rasters.

        Parameters
        ----------
        out_path : path-like
            Path of the (tempoerary) directory to which the large
            raster needs to be written

        Returns
        -------
        Raster
            Lower resolution raster covering all input rasters.
        """

        out_dir = Path(out_path)
        out_rast = out_dir / 'big_raster.tif'

        rast_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        if len(rast_hfun_list) == 0:
            return None

        all_bounds = []
        n_cell_lim = 0
        for hfun_in in rast_hfun_list:
            n_cell_lim = max(
                hfun_in.raster.src.shape[0]
                    * hfun_in.raster.src.shape[1],
                n_cell_lim)
            all_bounds.append(
                    hfun_in.get_bbox(crs='EPSG:4326').bounds)
        # 3 is just a arbitray tolerance for memory limit calculations
        n_cell_lim = n_cell_lim * self._nprocs / 3
        all_bounds = np.array(all_bounds)

        x0, y0 = np.min(all_bounds[:, [0, 1]], axis=0)
        x1, y1 = np.max(all_bounds[:, [2, 3]], axis=0)

        utm_crs = utils.estimate_bounds_utm(
                (x0, y0, x1, y1), "EPSG:4326")
        assert utm_crs is not None
        transformer = Transformer.from_crs(
                'EPSG:4326', utm_crs, always_xy=True)

        box_epsg4326 = box(x0, y0, x1, y1)
        poly_utm = ops.transform(transformer.transform, Polygon(box_epsg4326))
        x0, y0, x1, y1 = poly_utm.bounds

        worst_res = 0
        for hfun_in in rast_hfun_list:
            bnd1 = hfun_in.get_bbox(crs=utm_crs).bounds
            dim1 = np.max([bnd1[2] - bnd1[0], bnd1[3] - bnd1[1]])
            bnd2 = hfun_in.get_bbox(crs='EPSG:4326').bounds
            dim2 = np.max([bnd2[2] - bnd2[0], bnd2[3] - bnd2[1]])
            ratio = dim1 / dim2
            pixel_size_x = hfun_in.raster.src.transform[0] * ratio
            pixel_size_y = -hfun_in.raster.src.transform[4] * ratio

            worst_res = np.max([worst_res, pixel_size_x, pixel_size_y])

        # TODO: What if no hmin? -> use smallest raster res!
        g_hmin = self._size_info['hmin']
        res = np.max([g_hmin / 2, worst_res])
        _logger.info(
                f"Spatial resolution"
                f" chosen: {res}, worst: {worst_res}")
        shape0 = int(np.ceil(abs(x1 - x0) / res))
        shape1 = int(np.ceil(abs(y1 - y0) / res))

        approx =  int(np.sqrt(n_cell_lim))
        window_size = None #default of OCSMesh.raster.Raster
        mem_lim = 0 # default of rasterio
        if approx < max(shape0, shape1):
            window_size = np.min([shape0, shape1, approx])
            # Memory limit in MB
            mem_lim = n_cell_lim * np.float32(1).itemsize / 10e6


        # NOTE: Upper-left vs lower-left origin
        # (this only works for upper-left)
        transform = from_origin(x0 - res / 2, y1 + res / 2, res, res)

        rast_profile = {
                'driver': 'GTiff',
                'dtype': np.float32,
                'width': shape0,
                'height': shape1,
                'crs': utm_crs,
                'transform': transform,
                'count': 1,
        }
        with rasterio.open(str(out_rast), 'w', **rast_profile) as dst:
            # For places where raster is DEM is not provided it's
            # assumed deep ocean for contouring purposes
            if window_size is not None:
                write_wins = get_iter_windows(
                    shape0, shape1, chunk_size=window_size)
                for win in write_wins:
                    z = np.full((win.width, win.height), -99999, dtype=np.float32)
                    dst.write(z, 1, window=win)

            else:
                z = np.full((shape0, shape1), -99999, dtype=np.float32)
                dst.write(z, 1)
                del z


            # Reproject if needed (for now only needed if constant
            # value levels or subtidal limiters are added)
            for in_idx, hfun in enumerate(rast_hfun_list):
                ignore = True
                for (src_idx, _, _), _ in self._const_val_contour_coll:
                    if src_idx is None or in_idx in src_idx:
                        ignore = False
                        break
                for src_idx, _, _, _, _ in self._flow_lim_coll:
                    if src_idx is None or in_idx in src_idx:
                        ignore = False
                        break
                for src_idx, _ in self._constraint_info_coll:
                    if src_idx is None or in_idx in src_idx:
                        ignore = False
                        break
                if ignore:
                    continue

                # NOTE: Last one implicitely has highest priority in
                # case of overlap
                reproject(
                    source=rasterio.band(hfun.raster.src, 1),
                    destination=rasterio.band(dst, 1),
                    resampling=Resampling.nearest,
                    init_dest_nodata=False, # To avoid overwrite
                    num_threads=self._nprocs,
                    warp_mem_limit=mem_lim)



        return Raster(out_rast, chunk_size=window_size)

    def _apply_features_fast(self, big_raster: HfunRaster):
        """Internal: apply all specified refinements and constraints

        Apply all specified refinements and constrains for the fast
        algorithm.

        Parameters
        ----------
        big_raster : HfunRaster
            The lower-resolution large raster that covers all the
            input rasters.

        Returns
        -------
        None

        See Also
        --------
        _apply_features :
        """

        # NOTE: Caching applied doesn't work here since we apply
        # everything on a temporary big raster
        rast_hfun_list = []
        hfun_rast = None
        if big_raster:
            hfun_rast = HfunRaster(big_raster, **self._size_info)
            rast_hfun_list.append(hfun_rast)



        mesh_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunMesh)]
        if self._base_mesh and self._base_as_hfun:
            mesh_hfun_list.insert(0, self._base_mesh)

        # Mesh hfun parts are still stateful
        self._apply_contours([*mesh_hfun_list, *rast_hfun_list])
        if hfun_rast:
            # In fast method we only have big raster if any
            self._apply_flow_limiters_fast(hfun_rast)
            self._apply_const_val_fast(hfun_rast)
        # Mesh hfun parts are still stateful
        self._apply_patch([*mesh_hfun_list, *rast_hfun_list])
        self._apply_channels([*mesh_hfun_list, *rast_hfun_list])

        if hfun_rast:
            self._apply_constraints_fast(hfun_rast)


        return hfun_rast

    def _apply_flow_limiters_fast(self, big_hfun: HfunRaster) -> None:
        """Internal: apply specified sub tidal flow limiter refinements

        Applies specified subtidal flow limiter refinements for
        the fast algorithm.

        Parameters
        ----------
        big_hfun : HfunRaster
            The lower-resolution large raster that covers all the
            input rasters.

        Returns
        -------
        None

        See Also
        --------
        _apply_flow_limiters :
        """

        for src_idx, hmin, hmax, zmax, zmin in self._flow_lim_coll:
            # TODO: Account for source index
            if hmin is None:
                hmin = self._size_info['hmin']
            if hmax is None:
                hmax = self._size_info['hmax']

            # To avoid sharp gradient where no raster is projected
            if zmin is None:
                zmin = -99990
            else:
                zmin = max(zmin, -99990)

            big_hfun.add_subtidal_flow_limiter(hmin, hmax, zmax, zmin)

    def _apply_const_val_fast(self, big_hfun):
        """Internal: apply specified constant value refinements.

        Applies constant value refinements for the fast algorithm.

        Returns
        -------
        None

        See Also
        --------
        _apply_const_val :
        """

        for (src_idx, ctr0, ctr1), const_val in self._const_val_contour_coll:
            # TODO: Account for source index
            level0 = None
            level1 =  None
            if ctr0 is not None:
                level0 = ctr0.level
            if ctr1 is not None:
                level1 = ctr1.level
            big_hfun.add_constant_value(const_val, level0, level1)


    def _apply_constraints_fast(self, big_hfun: HfunRaster) -> None:
        """Internal: apply specified constraints.

        Apply specified constraints for the fast algorithm.

        Parameters
        ----------

        Returns
        -------
        None

        See Also
        --------
        _apply_constraints
        """

        constraint_list = []
        for src_idx, constraint_defn in self._constraint_info_coll:
            # TODO: Account for source index
            constraint_list.append(constraint_defn)

        if constraint_list:
            big_hfun.apply_constraints(constraint_list)


    def _get_hfun_composite_fast(self, big_hfun):
        """Internal: combine the size function functions for fast method

        Combine the size functions of the large raster with non-raster
        inputs. This is used for `fast` method.

        Parameters
        ----------
        big_hfun : HfunRaster
            The single large raster based size function covering all
            input rasters.

        Retruns
        -------
        jigsaw_msh_t
            The combined size function interpolated on an optimized
            mesh.

        See Also
        --------
        _get_hfun_composite
        """

        # In fast method all DEM hfuns have more priority than all
        # other inputs
        dem_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        nondem_hfun_list = [
            i for i in self._hfun_list if not isinstance(i, HfunRaster)]

        epsg4326 = CRS.from_user_input("EPSG:4326")

        dem_box_list = []
        for hfun in dem_hfun_list:
            dem_box_list.append(hfun.get_bbox(crs=epsg4326))

        index = []
        coord = []
        value = []
        offset = 0

        # Calculate multipoly and clip big hfun
        big_cut_shape = None
        if big_hfun:
            dem_gdf = gpd.GeoDataFrame(
                    geometry=dem_box_list, crs=epsg4326)
            big_cut_shape = dem_gdf.unary_union
            big_msh_t = big_hfun.msh_t()
            if hasattr(big_msh_t, "crs"):
                if not epsg4326.equals(big_msh_t.crs):
                    utils.reproject(big_msh_t, epsg4326)

            big_msh_t = utils.clip_mesh_by_shape(
                big_msh_t,
                big_cut_shape,
                use_box_only=False,
                fit_inside=False)


            index.append(big_msh_t.tria3['index'] + offset)
            coord.append(big_msh_t.vert2['coord'])
            value.append(big_msh_t.value)
            offset = offset + coord[-1].shape[0]

        hfun_list = nondem_hfun_list[::-1]
        if self._base_mesh and self._base_as_hfun:
            hfun_list = [*nondem_hfun_list[::-1], self._base_mesh]

        nondem_shape_list = []
        for hfun in hfun_list:
            nondem_msh_t = deepcopy(hfun.msh_t())
            if hasattr(nondem_msh_t, "crs"):
                if not epsg4326.equals(nondem_msh_t.crs):
                    utils.reproject(nondem_msh_t, epsg4326)

            nondem_shape = utils.get_mesh_polygons(hfun.mesh.msh_t)
            if not epsg4326.equals(hfun.crs):
                transformer = Transformer.from_crs(
                    hfun.crs, epsg4326, always_xy=True)
                nondem_shape = ops.transform(
                        transformer.transform, nondem_shape)

            # In fast method all DEM hfuns have more priority than all
            # other inputs
            if big_cut_shape:
                nondem_msh_t = utils.clip_mesh_by_shape(
                    nondem_msh_t,
                    big_cut_shape,
                    use_box_only=False,
                    fit_inside=True,
                    inverse=True)

            for ishp in nondem_shape_list:
                nondem_msh_t = utils.clip_mesh_by_shape(
                    nondem_msh_t,
                    ishp,
                    use_box_only=False,
                    fit_inside=True,
                    inverse=True)

            nondem_shape_list.append(nondem_shape)

            index.append(nondem_msh_t.tria3['index'] + offset)
            coord.append(nondem_msh_t.vert2['coord'])
            value.append(nondem_msh_t.value)
            offset += coord[-1].shape[0]

        composite_hfun = jigsaw_msh_t()
        composite_hfun.mshID = 'euclidean-mesh'
        composite_hfun.ndims = 2

        composite_hfun.vert2 = np.array(
                [(coord, 0) for coord in np.vstack(coord)],
                dtype=jigsaw_msh_t.VERT2_t)
        composite_hfun.tria3 = np.array(
                [(index, 0) for index in np.vstack(index)],
                dtype=jigsaw_msh_t.TRIA3_t)
        composite_hfun.value = np.array(
                np.vstack(value),
                dtype=jigsaw_msh_t.REALS_t)

        # TODO: Get user input for wether to force hmin and hmax on
        # final hfun (which includes non-raster and basemesh sizes)
        hmin = self._size_info['hmin']
        hmax = self._size_info['hmax']
        if hmin:
            composite_hfun.value[composite_hfun.value < hmin] = hmin
        if hmax:
            composite_hfun.value[composite_hfun.value > hmax] = hmax

        composite_hfun.crs = epsg4326

        # NOTE: In the end we need to return in a CRS that
        # uses meters as units. UTM based on the center of
        # the bounding box of the hfun is used
        # Up until now all calculation was in EPSG:4326
        utils.msh_t_to_utm(composite_hfun)

        return composite_hfun
