import functools
import logging
import operator
from collections import defaultdict
from typing import Union
from multiprocessing import cpu_count, Pool
from time import time

from scipy.spatial import cKDTree
from jigsawpy import jigsaw_msh_t
import numpy as np
from pyproj import CRS, Transformer
import utm
from shapely import ops
from shapely.geometry import (
    LineString, MultiLineString, GeometryCollection,
    Polygon, MultiPolygon)

from ocsmesh.hfun.base import BaseHfun
from ocsmesh.crs import CRS as CRSDescriptor
from ocsmesh import utils


_logger = logging.getLogger(__name__)

class HfunMesh(BaseHfun):

    _crs = CRSDescriptor()

    def __init__(self, mesh):
        self._mesh = mesh
        self._crs = mesh.crs

    def msh_t(self) -> jigsaw_msh_t:
        if self.crs.is_geographic:
            x0, y0, x1, y1 = self.mesh.get_bbox().bounds
            _, _, number, letter = utm.from_latlon(
                    (y0 + y1)/2, (x0 + x1)/2)
            # PyProj 3.2.1 throws error if letter is provided
            utm_crs = CRS(
                    proj='utm',
                    zone=f'{number}',
                    south=(y0 + y1)/2 < 0,
                    ellps={
                        'GRS 1980': 'GRS80',
                        'WGS 84': 'WGS84'
                        }[self.crs.ellipsoid.name]
                )
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

        if self.crs.is_geographic:

            _logger.info('Projecting to utm...')

            x0, y0, x1, y1 = self.mesh.get_bbox().bounds
            _, _, number, letter = utm.from_latlon(
                    (y0 + y1)/2, (x0 + x1)/2)
            # PyProj 3.2.1 throws error if letter is provided
            utm_crs = CRS(
                    proj='utm',
                    zone=f'{number}',
                    south=(y0 + y1)/2 < 0,
                    ellps={
                        'GRS 1980': 'GRS80',
                        'WGS 84': 'WGS84'
                        }[self.crs.ellipsoid.name]
                )
            transformer = Transformer.from_crs(
                self.crs, utm_crs, always_xy=True)
            # Note self.mesh.msh_t is NOT overwritten as coord is
            # being reassigned, not modified
            coord = np.vstack(
                transformer.transform(coord[:, 0], coord[:, 1])).T

        # NOTE: For msh_t type vertex id and index are the same
        trias = hfun_msh.tria3['index']
        quads = hfun_msh.quad4['index']
        hexas = hfun_msh.hexa8['index']

        _logger.info('Getting edges...')
        # Get unique set of edges by rolling connectivity
        # and joining connectivities in 3rd dimension, then sorting
        # to get all edges with lower index first
        all_edges = np.empty(shape=(0, 2), dtype=trias.dtype)
        if trias.shape[0]:
            _logger.info('Getting tria edges...')
            edges = np.sort(
                    np.stack(
                        (trias, np.roll(trias, shift=1, axis=1)),
                        axis=2),
                    axis=2)
            edges = np.unique(
                    edges.reshape(np.product(edges.shape[0:2]), 2), axis=0)
            all_edges = np.vstack((all_edges, edges))
        if quads.shape[0]:
            _logger.info('Getting quad edges...')
            edges = np.sort(
                    np.stack(
                        (quads, np.roll(quads, shift=1, axis=1)),
                        axis=2),
                    axis=2)
            edges = np.unique(
                    edges.reshape(np.product(edges.shape[0:2]), 2), axis=0)
            all_edges = np.vstack((all_edges, edges))
        if hexas.shape[0]:
            _logger.info('Getting quad edges...')
            edges = np.sort(
                    np.stack(
                        (hexas, np.roll(hexas, shift=1, axis=1)),
                        axis=2),
                    axis=2)
            edges = np.unique(
                    edges.reshape(np.product(edges.shape[0:2]), 2), axis=0)
            all_edges = np.vstack((all_edges, edges))

        all_edges = np.unique(all_edges, axis=0)

        # ONLY TESTED FOR TRIA FOR NOW

        # This part of the function is generic for tria and quad

        # Get coordinates for all edge vertices
        _logger.info('Getting coordinate of edges...')
        edge_coords = coord[all_edges, :]

        # Calculate length of all edges based on acquired coords
        _logger.info('Getting length of edges...')
        edge_lens = np.sqrt(
                np.sum(
                    np.power(
                        np.abs(np.diff(edge_coords, axis=1)), 2)
                    ,axis=2)).squeeze()

        # Calculate the mesh size by getting average of lengths
        # associated with each vertex (note there's not id vs index
        # distinction here). This is the most time consuming section
        # as of 04/21
        _logger.info('Creating vertex to edge map...')
        vert_to_edge = defaultdict(list)
        for e, i in enumerate(all_edges.ravel()):
            vert_to_edge[i].append(e // 2)

        _logger.info('Creating size value array for vertices...')
        vert_value = np.array(
                [np.average(edge_lens[vert_to_edge[i]])
                    if i in vert_to_edge else 0
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

    def add_feature(
            self,
            feature: Union[LineString, MultiLineString],
            expansion_rate: float,
            target_size: float = None,
            nprocs=None,
            max_verts=200,
            proc_pool=None
    ):
        if proc_pool is not None:
            self._add_feature_internal(
                feature=feature,
                expansion_rate=expansion_rate,
                target_size=target_size,
                pool=proc_pool,
                max_verts=max_verts)

        else:
            # Check nprocs
            nprocs = -1 if nprocs is None else nprocs
            nprocs = cpu_count() if nprocs == -1 else nprocs
            _logger.debug(f'Using nprocs={nprocs}')

            with Pool(processes=nprocs) as pool:
                self._add_feature_internal(
                    feature=feature,
                    expansion_rate=expansion_rate,
                    target_size=target_size,
                    pool=pool,
                    max_verts=max_verts)
            pool.join()

    def _add_feature_internal(
            self,
            feature: Union[LineString, MultiLineString],
            expansion_rate: float,
            target_size: float = None,
            pool: Pool = None,
            max_verts=200
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

        utm_crs: Union[CRS, None] = None

        if self.crs.is_geographic:
            x0, y0, x1, y1 = self.mesh.get_bbox().bounds
            _, _, number, letter = utm.from_latlon(
                (y0 + y1)/2, (x0 + x1)/2)
            # PyProj 3.2.1 throws error if letter is provided
            utm_crs = CRS(
                proj='utm',
                zone=f'{number}',
                south=(y0 + y1)/2 < 0,
                ellps={
                    'GRS 1980': 'GRS80',
                    'WGS 84': 'WGS84'
                    }[self.crs.ellipsoid.name]
            )
        else:
            utm_crs = None

        _logger.info('Repartitioning features...')
        start = time()
        res = pool.starmap(
            repartition_features,
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
            transform_linestring,
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


def transform_linestring(
    linestring: LineString,
    target_size: float,
):
    distances = [0.]
    while distances[-1] + target_size < linestring.length:
        distances.append(distances[-1] + target_size)
    distances.append(linestring.length)
    linestring = LineString([
        linestring.interpolate(distance)
        for distance in distances
        ])
    return linestring


def repartition_features(linestring, max_verts):
    features = []
    if len(linestring.coords) > max_verts:
        new_feat = []
        for segment in list(map(LineString, zip(
                linestring.coords[:-1],
                linestring.coords[1:]))):
            new_feat.append(segment)
            if len(new_feat) == max_verts - 1:
                features.append(ops.linemerge(new_feat))
                new_feat = []
        if len(new_feat) != 0:
            features.append(ops.linemerge(new_feat))
    else:
        features.append(linestring)
    return features
