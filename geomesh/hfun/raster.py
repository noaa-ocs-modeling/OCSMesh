import functools
import gc
import logging
from multiprocessing import cpu_count, Pool
import operator
import tempfile
from time import time
from typing import Union, List
import warnings

from jigsawpy import jigsaw_msh_t, jigsaw_jig_t
from jigsawpy import libsaw
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from scipy.spatial import cKDTree
from shapely import ops
from shapely.geometry import (
    LineString, MultiLineString, box, GeometryCollection,
    Polygon, MultiPolygon)
import utm

from geomesh.hfun.base import BaseHfun
from geomesh.raster import Raster, get_iter_windows
from geomesh.geom.shapely import PolygonGeom
from geomesh import utils

# supress feather warning
warnings.filterwarnings(
    'ignore', message='.*initial implementation of Parquet.*')

_logger = logging.getLogger(__name__)


class HfunInputRaster:

    def __set__(self, obj, raster: Raster):
        if not isinstance(raster, Raster):
            raise TypeError(f'Argument raster must be of type {Raster}, not '
                            f'type {type(raster)}.')
        # init output raster file
        tmpfile = tempfile.NamedTemporaryFile()
        with rasterio.open(raster.tmpfile) as src:
            if raster.chunk_size is not None:
                windows = get_iter_windows(
                    src.width, src.height, chunk_size=raster.chunk_size)
            else:
                windows = [rasterio.windows.Window(
                    0, 0, src.width, src.height)]
            meta = src.meta.copy()
            meta.update({'driver': 'GTiff', 'dtype': np.float32})
            with rasterio.open(tmpfile, 'w', **meta,) as dst:
                for window in windows:
                    values = src.read(window=window).astype(np.float32)
                    values[:] = np.finfo(np.float32).max
                    dst.write(values, window=window)
        obj.__dict__['raster'] = raster
        obj._tmpfile = tmpfile
        obj._chunk_size = raster.chunk_size
        obj._overlap = raster.overlap

    def __get__(self, obj, val) -> Raster:
        return obj.__dict__['raster']


class FeatureCache:

    def __get__(self, obj, val):
        features = obj.__dict__.get('features')
        if features is None:
            features = {}


class HfunRaster(BaseHfun, Raster):

    _raster = HfunInputRaster()
    _feature_cache = FeatureCache()

    def __init__(self, raster: Raster, hmin: float = None, hmax: float = None,
                 verbosity=0):
        self._raster = raster
        self._hmin = float(hmin) if hmin is not None else hmin
        self._hmax = float(hmax) if hmax is not None else hmax
        self._verbosity = int(verbosity)

    def msh_t(self, window: rasterio.windows.Window = None,
              marche: bool = False, verbosity=None) -> jigsaw_msh_t:


        if window is None:
            iter_windows = list(self.iter_windows())
        else:
            iter_windows = [window]

        utm_crs = None

        output_mesh = jigsaw_msh_t()
        output_mesh.ndims = +2
        output_mesh.mshID = "euclidean-mesh"
        output_mesh.crs = self.crs
        for window in iter_windows:

            hfun = jigsaw_msh_t()
            hfun.ndims = +2

            x0, y0, x1, y1 = self.get_window_bounds(window)

            if self.crs.is_geographic:
                hfun.mshID = 'euclidean-mesh'
                # If these 3 objects (vert2, tria3, value) don't fit into
                # memroy, then the raster needs to be chunked. We need to
                # implement auto-chunking.
                start = time()
                _, _, number, letter = utm.from_latlon(
                    (y0 + y1)/2, (x0 + x1)/2)
                utm_crs = CRS(
                    proj='utm',
                    zone=f'{number}{letter}',
                    ellps={
                        'GRS 1980': 'GRS80',
                        'WGS 84': 'WGS84'
                        }[self.crs.ellipsoid.name]
                )
                # get bbox data
                xgrid = self.get_x(window=window)
                ygrid = np.flip(self.get_y(window=window))
                xgrid, ygrid = np.meshgrid(xgrid, ygrid)
                bottom = xgrid[0, :]
                top = xgrid[1, :]
                del xgrid
                left = ygrid[:, 0]
                right = ygrid[:, 1]
                del ygrid

                _logger.info('Building hfun.tria3...')

                dim1 = window.width
                dim2 = window.height

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
                    window.width*window.height,
                    dtype=jigsaw_msh_t.VERT2_t)
                hfun.vert2['coord'] = np.array(
                    self.get_xy_memcache(window, utm_crs))
                _logger.info('Done building hfun.vert2...')

                # Build REALS_t: this one comes from hfun raster
                _logger.info('Building hfun.value...')
                hfun.value = np.array(
                    self.get_values(window=window, band=1).flatten().reshape(
                        (window.width*window.height, 1)),
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
                    np.array(self.get_x(window=window)),
                    dtype=jigsaw_msh_t.REALS_t)
                hfun.ygrid = np.array(
                    np.flip(self.get_y(window=window)),
                    dtype=jigsaw_msh_t.REALS_t)
                hfun.value = np.array(
                    np.flipud(self.get_values(window=window, band=1)),
                    dtype=jigsaw_msh_t.REALS_t)
                kwargs = {'kx': 1, 'ky': 1}  # type: ignore[dict-item]
                geom = PolygonGeom(box(x0, y0, x1, y1), self.crs).msh_t()

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

            # reproject and combine with other windows
            # output_mesh is always in self.crs
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
                np.array([(coo, tag)
                          for coo, tag in window_mesh.vert2],
                         dtype=jigsaw_msh_t.VERT2_t),
                axis=0)
            if output_mesh.value.size:
                output_mesh.value = np.append(
                    output_mesh.value,
                    np.array([v for v in window_mesh.value],
                             dtype=jigsaw_msh_t.REALS_t),
                    axis=0)
            else:
                output_mesh.value = np.array(
                        [v for v in window_mesh.value],
                        dtype=jigsaw_msh_t.REALS_t)

        # NOTE: In the end we need to return in a CRS that
        # uses meters as units. UTM based on the center of
        # the bounding box of the hfun is used
        if self.crs.is_geographic:
            x0, y0, x1, y1 = self.get_bbox().bounds
            _, _, number, letter = utm.from_latlon(
                    (y0 + y1)/2, (x0 + x1)/2)
            utm_crs = CRS(
                    proj='utm',
                    zone=f'{number}{letter}',
                    ellps={
                        'GRS 1980': 'GRS80',
                        'WGS 84': 'WGS84'
                        }[self.crs.ellipsoid.name]
                )
            transformer = Transformer.from_crs(
                self.crs, utm_crs, always_xy=True)
            output_mesh.vert2['coord'] = np.vstack(
                transformer.transform(
                    output_mesh.vert2['coord'][:, 0],
                    output_mesh.vert2['coord'][:, 1]
                    )).T
            output_mesh.crs = utm_crs

        return output_mesh

    def add_patch(
            self,
            multipolygon: Union[MultiPolygon, Polygon],
            expansion_rate: float = None,
            target_size: float = None,
            nprocs: int = None
    ):

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
            raise ValueError('Argument target_size must be specified if no '
                             'global hmin has been set.')
        if target_size <= 0:
            raise ValueError("Argument target_size must be greater than zero.")

        # For expansion_rate
        if expansion_rate != None:
            exteriors = [ply.exterior for ply in multipolygon]
            interiors = [
                inter for ply in multipolygon for inter in ply.interiors]
            
            features = MultiLineString([*exteriors, *interiors])
            self.add_feature(
                feature=features,
                expansion_rate=expansion_rate,
                target_size=target_size,
                nprocs=nprocs)

        tmpfile = tempfile.NamedTemporaryFile()
        meta = self.src.meta.copy()
        meta.update({'driver': 'GTiff'})
        with rasterio.open(tmpfile, 'w', **meta,) as dst:
            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)
            for i, window in enumerate(iter_windows):
                _logger.debug(f'Processing window {i+1}/{tot}.')
                # NOTE: We should NOT transform polygon, user just
                # needs to make sure input polygon has the same CRS
                # as the hfun (we don't calculate distances in this
                # method)

                _logger.info(f'Creating mask from shape ...')
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
                
                _logger.info(f'Write array to file {tmpfile.name}...')
                start = time()
                dst.write_band(1, values, window=window)
                _logger.info(f'Write array to file took {time()-start}.')

        self._tmpfile = tmpfile


    def add_contour(
            self,
            level: Union[List[float], float],
            expansion_rate: float,
            target_size: float = None,
            nprocs: int = None,
    ):
        """ See https://outline.com/YU7nSM for an excellent explanation about
        tree algorithms.
        """
        if not isinstance(level, list):
            level = [level]

        contours = []
        for _level in level:
            _contours = self.raster.get_contour(_level)
            if isinstance(_contours, GeometryCollection):
                continue
            elif isinstance(_contours, LineString):
                contours.append(_contours)
            elif isinstance(_contours, MultiLineString):
                for _cont in _contours:
                    contours.append(_cont)

        if len(contours) == 0:
            _logger.info('No contours found!')
            return

        contours = MultiLineString(contours)

        _logger.info('Adding contours as features...')
        self.add_feature(contours, expansion_rate, target_size, nprocs)

    def add_feature(
            self,
            feature: Union[LineString, MultiLineString],
            expansion_rate: float,
            target_size: float = None,
            nprocs=None,
            max_verts=200
    ):
        '''Adds a linear distance size function constraint to the mesh.

        Arguments:
            feature: shapely.geometryLineString or MultiLineString

        https://gis.stackexchange.com/questions/214261/should-we-always-calculate-length-and-area-in-lat-lng-to-get-accurate-sizes-leng

        "Creating a local projection allowed us to have similar area/length
        calculations as if we was using great circle calculations."

        TODO: Consider using BallTree with haversine or Vincenty metrics
        instead of a locally projected window.
        '''

        # TODO: Partition features if they are too "long" which results in an
        # improvement for parallel pool. E.g. if a feature is too long, 1
        # processor will be busy and the rest will be idle.

        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs
        _logger.debug(f'Using nprocs={nprocs}')
        if not isinstance(feature, (LineString, MultiLineString)):
            raise TypeError(
                f'Argument feature must be of type {LineString} or '
                f'{MultiLineString}, not type {type(feature)}.')

        if isinstance(feature, LineString):
            feature = [feature]

        elif isinstance(feature, MultiLineString):
            feature = [linestring for linestring in feature]

        # check target size
        target_size = self.hmin if target_size is None else target_size
        if target_size is None:
            raise ValueError('Argument target_size must be specified if no '
                             'global hmin has been set.')
        if target_size <= 0:
            raise ValueError("Argument target_size must be greater than zero.")
        tmpfile = tempfile.NamedTemporaryFile()
        meta = self.src.meta.copy()
        meta.update({'driver': 'GTiff'})
        utm_crs: Union[CRS, None] = None
        with rasterio.open(tmpfile, 'w', **meta,) as dst:
            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)
            for i, window in enumerate(iter_windows):
                _logger.debug(f'Processing window {i+1}/{tot}.')
                if self.crs.is_geographic:
                    x0, y0, x1, y1 = self.get_window_bounds(window)
                    _, _, number, letter = utm.from_latlon(
                        (y0 + y1)/2, (x0 + x1)/2)
                    utm_crs = CRS(
                        proj='utm',
                        zone=f'{number}{letter}',
                        ellps={
                            'GRS 1980': 'GRS80',
                            'WGS 84': 'WGS84'
                            }[self.crs.ellipsoid.name]
                    )
                else:
                    utm_crs = None
                _logger.info('Repartitioning features...')
                start = time()
                with Pool(processes=nprocs) as pool:
                    res = pool.starmap(
                        repartition_features,
                        [(linestring, max_verts) for linestring in feature]
                        )
                pool.join()
                feature = functools.reduce(operator.iconcat, res, [])
                _logger.info(f'Repartitioning features took {time()-start}.')

                _logger.info(f'Resampling features on nprocs {nprocs}...')
                start = time()
                with Pool(processes=nprocs) as pool:

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
                        feature = [
                            ops.transform(transformer.transform, linestring)
                            for linestring in feature]
                        _logger.info(
                                f"Transform apply took {time() - start2:f}")

                    transformed_features = pool.starmap(
                        transform_linestring,
                        [(linestring, target_size) for linestring in feature]
                    )
                pool.join()
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
                        xy, workers=nprocs, distance_upper_bound=r)
                    distances = r * np.ones(len(xy))
                    mask = np.logical_not(np.isinf(near_dists))
                    distances[mask] = near_dists[mask]
                else:
                    distances, _ = tree.query(xy, workers=nprocs)
                _logger.info(f'Querying KDTree took {time()-start}.')
                values = expansion_rate*target_size*distances + target_size
                values = values.reshape(window.height, window.width).astype(
                    self.dtype(1))
                if self.hmin is not None:
                    values[np.where(values < self.hmin)] = self.hmin
                if self.hmax is not None:
                    values[np.where(values > self.hmax)] = self.hmax
                values = np.minimum(self.get_values(window=window), values)
                _logger.info(f'Write array to file {tmpfile.name}...')
                start = time()
                dst.write_band(1, values, window=window)
                _logger.info(f'Write array to file took {time()-start}.')
        self._tmpfile = tmpfile

    def get_xy_memcache(self, window, dst_crs):
        if not hasattr(self, '_xy_cache'):
            self._xy_cache = {}
        tmpfile = self._xy_cache.get(f'{window}{dst_crs}')
        if tmpfile is None:
            _logger.info('Transform points to local CRS...')
            transformer = Transformer.from_crs(
                self.src.crs, dst_crs, always_xy=True)
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
        else:
            _logger.info('Loading values from memcache...')
            return np.memmap(tmpfile, dtype='float32', mode='r',
                             shape=((window.width*window.height), 2))[:]

    def add_subtidal_flow_limiter(
            self,
            hmin=None,
            hmax=None,
            upper_bound=None,
            lower_bound=None
    ):

        hmin = float(hmin) if hmin is not None else hmin
        hmax = float(hmax) if hmax is not None else hmax

        tmpfile = tempfile.NamedTemporaryFile()
        utm_crs: Union[CRS, None] = None
        with rasterio.open(tmpfile.name, 'w', **self.src.meta) as dst:

            iter_windows = list(self.iter_windows())
            tot = len(iter_windows)

            for i, window in enumerate(iter_windows):

                _logger.debug(f'Processing window {i+1}/{tot}.')
                x0, y0, x1, y1 = self.get_window_bounds(window)

                if self.crs.is_geographic:
                    _, _, number, letter = utm.from_latlon(
                        (y0 + y1)/2, (x0 + x1)/2)
                    utm_crs = CRS(
                        proj='utm',
                        zone=f'{number}{letter}',
                        ellps={
                            'GRS 1980': 'GRS80',
                            'WGS 84': 'WGS84'
                            }[self.crs.ellipsoid.name]
                    )
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
        self._tmpfile = tmpfile

    def add_constant_value(self, value, lower_bound=None, upper_bound=None):
        lower_bound = -float('inf') if lower_bound is None \
            else float(lower_bound)
        upper_bound = float('inf') if upper_bound is None \
            else float(upper_bound)
        tmpfile = tempfile.NamedTemporaryFile()

        with rasterio.open(tmpfile.name, 'w', **self.src.meta) as dst:

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
        self._tmpfile = tmpfile

    @property
    def raster(self):
        return self._raster

    @property
    def output(self):
        return self

    @property
    def hmin(self):
        return self._hmin

    @property
    def hmax(self):
        return self._hmax

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity: int):
        self._verbosity = verbosity


def transform_point(x, y, src_crs, utm_crs):
    transformer = Transformer.from_crs(src_crs, utm_crs, always_xy=True)
    return transformer.transform(x, y)


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

def transform_polygon(
    polygon: Polygon,
    src_crs: CRS = None,
    utm_crs: CRS = None
):
    if utm_crs is not None:
        transformer = Transformer.from_crs(
            src_crs, utm_crs, always_xy=True)

        polygon = ops.transform(
                transformer.transform, polygon)
    return polygon


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
