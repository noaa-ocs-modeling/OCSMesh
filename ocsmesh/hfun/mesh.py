"""This module define class for mesh based size function
"""

import functools
import logging
import operator
from collections import defaultdict
from typing import Union, Optional
from multiprocessing import cpu_count, Pool
from time import time

from matplotlib.transforms import Bbox
from scipy.spatial import cKDTree
from jigsawpy import jigsaw_msh_t
import numpy as np
from pyproj import Transformer
from shapely import ops
from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon)

from ocsmesh.hfun.base import BaseHfun
from ocsmesh.crs import CRS as CRSDescriptor
from ocsmesh import Mesh
from ocsmesh import utils


_logger = logging.getLogger(__name__)

class HfunMesh(BaseHfun):
    """Mesh based size function.

    Creates a mesh based size function. The mesh size is specified
    at each point of the mesh based on the specified criteria.

    Attributes
    ----------
    hmin
    hmax
    mesh
    crs

    Methods
    -------
    msh_t()
        Return mesh sizes specified on the points of the  underlying
        mesh.
    size_from_mesh()
        Calculate values of the size function at each point on the
        underlying mesh, based on the length of edges connected to
        that point.
    add_patch(multipolygon, expansion_rate=None,
              target_size=None, nprocs=None)
        Add a region of fixed size refinement with optional expansion
        rate for points outside the region to achieve smooth size
        transition.
    add_feature(feature, expansion_rate, target_size=None,
                max_verts=200, *, nprocs=None, pool=None)
        Decorated method to add size refinement based on the specified
        `expansion_rate`, `target_size`, and distance from the input
        feature lines `feature`.
    get_bbox(**kwargs)
        Return  the bounding box of the underlying mesh.

    Notes
    -----
    Unlike raster size function, mesh based size function doesn't
    support constraint at this point.
    """

    _crs = CRSDescriptor()

    def __init__(self, mesh: Mesh) -> None:
        """Initialize a mesh based size function object

        Parameters
        ----------
        mesh : Mesh
            Input mesh object whose points are used for specifying
            sizes of the mesh to be generated. Note the underlying
            mesh is not copied, it is overriden by methods in the
            object of this type.

        Notes
        -----
        When a size function is created from a mesh, it takes
        the values associated with the underlying mesh. Unless
        `size_from_mesh` is called or refinements are applied, the
        values can be meaningless.

        Unlike raster size function where the user defines the
        minimum and maximum, for mesh based hfun, the minimum and
        maximum is based on the values stored on the mesh.

        Note that currently object of this type holds onto a crs
        variable separate from `mesh.crs`. Because of this, if the
        user is not careful, one might run into unexpected behavior!
        """

        self._mesh = mesh
        self._crs = mesh.crs

    def msh_t(self) -> jigsaw_msh_t:
        """Return the size function specified on the underlying mesh

        Return the size function values stored on the underlying mesh.
        The return value is in a projected CRS. If the input mesh
        CRS is geographic, then a local UTM CRS is calculated and used
        for the output of this method.

        Parameters
        ----------

        Returns
        -------
        jigsaw_msh_t
            The size function specified on the points of input mesh.

        Notes
        -----
        This method effectively overrides the CRS of the objects and
        modifies it if the CRS is initially geographic. Note that
        this also affects the underlying mesh object which is **not**
        copied in the contructor.
        """

        utm_crs = utils.estimate_mesh_utm(self.mesh.msh_t)
        if utm_crs is not None:
            transformer = Transformer.from_crs(
                self.crs, utm_crs, always_xy=True)
            # TODO: This modifies the underlying mesh, is this
            # intended?
            self.mesh.msh_t.vert2['coord'] = np.vstack(
                transformer.transform(
                    self.mesh.msh_t.vert2['coord'][:, 0],
                    self.mesh.msh_t.vert2['coord'][:, 1]
                    )).T
            self.mesh.msh_t.crs = utm_crs
            self._crs = utm_crs

        return self.mesh.msh_t

    def size_from_mesh(self) -> None:
        """Calculates sizes based on the underlying mesh edge lengths.

        Get size function values based on the underlying input mesh
        This method overwrites the values in underlying `msh_t`.

        Parameters
        ----------

        Returns
        -------
        None

        Notes
        -----
        The size calculations are done in a projected CRS or a local
        UTM CRS. However, this method does not modify the
        CRS of the size function, it only updates the values and
        discard the projected CRS it used for calculations.


        """

        # Make sure it's in utm so that sizes are in meters
        hfun_msh = self.mesh.msh_t
        coord = hfun_msh.vert2['coord']

        transformer = None
        utm_crs = utils.estimate_mesh_utm(hfun_msh)
        if utm_crs is not None:
            _logger.info('Projecting to utm...')

            transformer = Transformer.from_crs(
                self.crs, utm_crs, always_xy=True)

        # Calculate length of all edges based on acquired coords
        _logger.info('Getting length of edges...')
        len_dict = utils.calculate_edge_lengths(hfun_msh, transformer)

        # Calculate the mesh size by getting average of lengths
        # associated with each vertex (note there's not id vs index
        # distinction here). This is the most time consuming section
        # as of 04/21
        vert_to_lens = defaultdict(list)
        for verts_idx, edge_len in len_dict.items():
            for vidx in verts_idx:
                vert_to_lens[vidx].append(edge_len)

        _logger.info('Creating size value array for vertices...')
        vert_value = np.array(
            [np.average(vert_to_lens[i]) if i in vert_to_lens else 0
             for i in range(coord.shape[0])])

        # NOTE: Modifying values of underlying mesh
        hfun_msh.value = vert_value.reshape(len(vert_value), 1)


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
            # TODO: Is this relevant for mesh type?
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

        coords = self.mesh.msh_t.vert2['coord']
        values = self.mesh.msh_t.value

        verts_in = utils.get_verts_in_shape(
            self.mesh.msh_t, shape=multipolygon, from_box=False)

        if len(verts_in):
            # NOTE: Don't continue, otherwise the final
            # destination file might end up being empty!
            values[verts_in, :] = target_size

        # NOTE: unlike raster self.hmin is based on values of this
        # hfun before applying feature; it is ignored so that
        # the new self.hmin becomes equal to "target" specified
#        if self.hmin is not None:
#            values[np.where(values < self.hmin)] = self.hmin
        if self.hmax is not None:
            values[np.where(values > self.hmax)] = self.hmax
        values = np.minimum(self.mesh.msh_t.value, values)
        values = values.reshape(self.mesh.msh_t.value.shape)

        self.mesh.msh_t.value = values

    @utils.add_pool_args
    def add_feature(
            self,
            feature: Union[LineString, MultiLineString],
            expansion_rate: float,
            target_size: float = None,
            max_verts=200,
            *, # kwarg-only comes after this
            pool: Pool
    ):
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
        add_patch :
            Add refinement for region specified polygon

        Notes
        -----
        See https://outline.com/YU7nSM for an explanation
        about tree algorithms.

        Creating a local projection allows having similar area/length
        calculations as if great circle calculations was being used.

        Another useful refererence:
        https://gis.stackexchange.com/questions/214261/should-we-always-calculate-length-and-area-in-lat-lng-to-get-accurate-sizes-leng
        """
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
            feature = list(feature)

        # check target size
        target_size = self.hmin if target_size is None else target_size
        if target_size is None:
            raise ValueError('Argument target_size must be specified if no '
                             'global hmin has been set.')
        if target_size <= 0:
            raise ValueError("Argument target_size must be greater than zero.")

        utm_crs = utils.estimate_mesh_utm(self.mesh.msh_t)

        _logger.info('Repartitioning features...')
        start = time()
        res = pool.starmap(
            utils.repartition_features,
            [(linestring, max_verts) for linestring in feature]
            )
        feature = functools.reduce(operator.iconcat, res, [])
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
                self.crs, utm_crs, always_xy=True)
            _logger.info(
                    f"Transform creation took {time() - start2:f}")
            start2 = time()
            feature = [
                ops.transform(transformer.transform, linestring)
                for linestring in feature]
            _logger.info(
                    f"Transform apply took {time() - start2:f}")

        transformed_features = pool.starmap(
            utils.transform_linestring,
            [(linestring, target_size) for linestring in feature]
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

        # We call msh_t() so that it also takes care of utm
        # transformation
        xy = self.msh_t().vert2['coord']

        _logger.info(f'transforming points took {time()-start}.')
        _logger.info('querying kdtree...')
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
        _logger.info(f'querying kdtree took {time()-start}.')
        values = expansion_rate*target_size*distances + target_size
        # NOTE: unlike raster self.hmin is based on values of this
        # hfun before applying feature; it is ignored so that
        # the new self.hmin becomes equal to "target" specified
#        if self.hmin is not None:
#            values[np.where(values < self.hmin)] = self.hmin
        if self.hmax is not None:
            values[np.where(values > self.hmax)] = self.hmax
        values = np.minimum(self.mesh.msh_t.value.ravel(), values)
        values = values.reshape(self.mesh.msh_t.value.shape)

        self.mesh.msh_t.value = values

    @property
    def hmin(self):
        """Read-only attribute for the minimum mesh size constraint"""

        return np.min(self.mesh.msh_t.value)

    @property
    def hmax(self):
        """Read-only attribute for the maximum mesh size constraint"""

        return np.max(self.mesh.msh_t.value)

    @property
    def mesh(self):
        """Read-only attribute to reference to the input mesh"""

        return self._mesh

    @property
    def crs(self):
        """Read-only attribute holding onto hfun CRS"""

        return self._crs

    def get_bbox(self, **kwargs) -> Union[Polygon, Bbox]:
        """Returns the bounding box of the underlying mesh

        Parameters
        ----------
        kwargs : dict, optional
            Arguments passed to the underlying mesh `get_bbox`

        Returns
        -------
        Polygon or Bbox
            The bounding box of the underlying mesh
        """

        return self.mesh.get_bbox(**kwargs)
