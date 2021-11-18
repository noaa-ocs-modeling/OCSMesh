import functools
import logging
import operator
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from time import time
from typing import Union

import numpy as np
from jigsawpy import jigsaw_msh_t
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import ops
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from ocsmesh import utils
from ocsmesh.crs import CRS as CRSDescriptor
from ocsmesh.hfun.base import BaseHfun

_logger = logging.getLogger(__name__)

class HfunMesh(BaseHfun):

    _crs = CRSDescriptor()

    def __init__(self, mesh):
        self._mesh = mesh
        self._crs = mesh.crs

    def msh_t(self) -> jigsaw_msh_t:

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

    def size_from_mesh(self):

        '''
        Get size function values based on the mesh underlying
        this size function. This method overwrites the values
        in underlying msh_t.
        Also note that for calculation coordinates are projected
        to utm, but the projected coordinates are discarded
        '''

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
            expansion_rate: float = None,
            target_size: float = None,
            nprocs: int = None
    ):

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
        return np.min(self.mesh.msh_t.value)

    @property
    def hmax(self):
        return np.max(self.mesh.msh_t.value)

    @property
    def mesh(self):
        return self._mesh

    @property
    def crs(self):
        return self._crs

    def get_bbox(self, **kwargs):
        return self.mesh.get_bbox(**kwargs)
