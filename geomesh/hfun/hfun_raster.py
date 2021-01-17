from multiprocessing import Pool, cpu_count
import pathlib
import tempfile
import warnings

from jigsawpy import jigsaw_msh_t, jigsaw_jig_t
from jigsawpy import libsaw
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS
from pyproj.transformer import Transformer
import rasterio
from scipy.spatial import cKDTree
from shapely.ops import transform
from shapely.geometry import LineString, MultiLineString
import utm

from geomesh.figures import _figure
from geomesh.raster import Raster
from geomesh.geom import Geom
from geomesh.hfun.base import BaseHfun


tmpdir = pathlib.Path(tempfile.gettempdir()+'/geomesh') / 'hfun'
tmpdir.mkdir(parents=True, exist_ok=True)


def _contour_worker(path, window, level, tgt_size):
    raster = Raster(path)
    x = raster.get_x(window)
    y = raster.get_y(window)
    values = raster.get_values(band=1, window=window)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ax = plt.contour(x, y, values, levels=[level])
    plt.close(plt.gcf())
    features = []
    for path_collection in ax.collections:
        for path in path_collection.get_paths():
            _vertices = [(x, y) for (x, y), _ in path.iter_segments()]
            if len(_vertices) > 1:
                features.append(LineString(_vertices))
    return features


def _jigsaw_hmat_worker(path, window, hmin, hmax, geom):

    # TODO: Check for is_geographic on crs before passing to utm package
    raise NotImplementedError

    geom = None
    raster = Raster(path)

    x = raster.get_x(window)
    y = raster.get_y(window)
    _y = np.repeat(np.min(y), len(x))
    _x = np.repeat(np.min(x), len(y))
    _tx = utm.from_latlon(_y, x)[0]
    _ty = np.flip(utm.from_latlon(y, _x)[1])
    hmat = jigsaw_msh_t()
    hmat.mshID = "euclidean-grid"
    hmat.ndims = +2
    hmat.xgrid = _tx.astype(jigsaw_msh_t.REALS_t)
    hmat.ygrid = _ty.astype(jigsaw_msh_t.REALS_t)
    # TODO: We always get band = 1, so we should make sure the raster's
    # gaussian_filter will write the filtered band into band 1
    hmat.value = np.flipud(
            raster.get_values(band=1, window=window)
            ).astype(jigsaw_msh_t.REALS_t)

    # init opts
    opts = jigsaw_jig_t()

    # additional configuration options
    opts.verbosity = 1
    opts.mesh_dims = 2
    opts.hfun_scal = 'absolute'
    opts.optm_tria = True

    if hmin is not None:
        opts.hfun_hmin = hmin
    else:
        opts.hfun_hmin = np.min(hmat.value)

    if hmax is not None:
        opts.hfun_hmax = hmax
    else:
        opts.hfun_hmax = np.max(hmat.value)

    # output mesh
    mesh = jigsaw_msh_t()

    # call jigsaw to create local mesh
    libsaw.jigsaw(
        opts,
        geom,
        mesh,
        hfun=hmat
    )
    breakpoint()
    return mesh


class HfunRaster(BaseHfun):

    def __init__(self,
                 raster,
                 hmin=None,
                 hmax=None,
                 nprocs=None,
                 interface='cmdsaw'):

        self._raster = raster
        self._nprocs = nprocs
        self._hmin = hmin
        self._hmax = hmax

    def __iter__(self):
        for i, window in enumerate(self._src.iter_windows()):
            x = self._src.get_x(window)
            y = self._src.get_y(window)
            values = self._src.get_values(window=window)
            yield x, y, values

    @_figure
    def contourf(self, *args, **kwargs):
        plt.contourf(self._src.x, self._src.y, self._src.values)

    def add_contour(
            self,
            level: float,
            target_size: float,
            expansion_rate: float
    ):
        """ See https://outline.com/YU7nSM for an excellent explanation about
        tree algorithms.
        """
        expansion_rate = float(expansion_rate)
        target_size = float(target_size)
        assert target_size > 0.
        res = []
        if self._nprocs > 1:
            _old_backend = mpl.get_backend()
            mpl.use('agg')
            _job_args = []
            for window in self._src.iter_windows():
                _job_args.append(
                    (self._raster._tmpfile, window, level, target_size))
            with Pool(processes=self._nprocs) as pool:
                _res = pool.starmap(_contour_worker, _job_args)
            pool.join()
            mpl.use(_old_backend)
            for items in _res:
                res.extend(items)
        else:
            for window in self._src.iter_windows():
                res.extend(
                    _contour_worker(
                        self._raster._tmpfile, window, level, target_size))
        self.add_feature(MultiLineString(res), target_size, expansion_rate)

    def add_feature(self, feature, target_size, expansion_rate):
        expansion_rate = float(expansion_rate)
        target_size = float(target_size)
        assert target_size > 0.
        if not isinstance(feature, (LineString, MultiLineString)):
            raise TypeError(
                f"feature must be {LineString} or {MultiLineString}")
        if isinstance(feature, LineString):
            feature = MultiLineString([feature])
        tmpfile = tempfile.NamedTemporaryFile(prefix=str(tmpdir) + '/')
        meta = self._src._src.meta.copy()
        if self._src.crs.is_geographic:
            with rasterio.open(tmpfile.name, 'w', **meta) as dst:
                for window in self._src.iter_windows():
                    xy = self._src.get_xy(window)
                    _tx, _ty, zone, _ = utm.from_latlon(xy[:, 1], xy[:, 0])
                    dst_crs = CRS(proj='utm', zone=zone, ellps='WGS84')
                    transformer = Transformer.from_crs(
                        self._src.crs, dst_crs, always_xy=True)
                    res = []
                    for linestring in feature:
                        distances = [0]
                        linestring = transform(
                            transformer.transform, linestring)
                        while distances[-1] + target_size < linestring.length:
                            distances.append(distances[-1] + target_size)
                        distances.append(linestring.length)
                        linestring = LineString([
                            linestring.interpolate(distance)
                            for distance in distances
                            ])
                        res.extend(linestring.coords)
                    tree = cKDTree(np.vstack(res))
                    values = tree.query(
                        np.vstack([_tx, _ty]).T, n_jobs=self._nprocs)[0]
                    values = expansion_rate*target_size*values + target_size
                    values = values.reshape(
                        (1, window.height, window.width)).astype(meta['dtype'])
                    if self._hmin is not None:
                        values[np.where(values < self._hmin)] = self._hmin
                    if self._hmax is not None:
                        values[np.where(values > self._hmax)] = self._hmax
                    values = np.minimum(
                        self._src.get_values(window=window), values)
                    dst.write(values, window=window)

        else:  # is not geographic

            # NOTE: We are not iterating over windows here
            xy = self._src.get_xy(window=None)
            res = []
            for linestring in feature:
                distances = [0]
                while distances[-1] + target_size < linestring.length:
                    distances.append(distances[-1] + target_size)
                distances.append(linestring.length)
                linestring = LineString([
                    linestring.interpolate(distance)
                    for distance in distances
                    ])
                res.extend(linestring.coords)
            tree = cKDTree(np.vstack(res))
            with rasterio.open(tmpfile.name, 'w', **meta) as dst:
                for i, window in enumerate(self._src.iter_windows()):
                    values = tree.query(
                        self._src.get_xy(window),
                        n_jobs=self._nprocs)[0]
                    values = expansion_rate*target_size*values + target_size
                    dst.write(
                        np.minimum(
                            self._src.get_values(window=window),
                            values.reshape(
                                (1, window.height, window.width))
                            ).astype(meta['dtype']),
                        window=window)
        self._tmpfile = tmpfile

    def add_subtidal_flow_limiter(
            self, hmin=None, upper_bound=None, lower_bound=None):
        hmin = np.finfo(np.float32).eps if hmin is None else hmin
        if not self._src.crs.is_geographic:
            dx = np.abs(self._src.dx)
            dy = np.abs(self._src.dy)
        meta = self._src._src.meta.copy()
        tmpfile = tempfile.NamedTemporaryFile(prefix=str(tmpdir) + '/')
        with rasterio.open(tmpfile.name, 'w', **meta) as dst:
            for window, bounds in self._src:
                topobathy = self._raster.get_values(band=1, window=window)
                if self._src.crs.is_geographic:
                    west, south, east, north = bounds
                    _bounds = np.array([[east, south], [west, north]])
                    _x, _y = utm.from_latlon(_bounds[:, 1], _bounds[:, 0])[:2]
                    dx = np.diff(np.linspace(_x[0], _x[1], window.width))[0]
                    dy = np.diff(np.linspace(_y[0], _y[1], window.height))[0]
                _dx, _dy = np.gradient(topobathy, dx, dy)
                with warnings.catch_warnings():
                    # in case self._src.values is a masked array
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dh = np.sqrt(_dx**2 + _dy**2)
                dh = np.ma.masked_equal(dh, 0.)
                values = np.abs((1./3.)*(topobathy/dh))
                values = values.filled(np.max(values))

                if upper_bound is not None:
                    values[np.where(
                        topobathy > upper_bound)] = self._src.nodata
                if lower_bound is not None:
                    values[np.where(
                        topobathy < lower_bound)] = self._src.nodata
                values[np.where(values < hmin)] = hmin
                if self._hmin is not None:
                    values[np.where(values < self._hmin)] = self._hmin
                if self._hmax is not None:
                    values[np.where(values > self._hmax)] = self._hmax
                values = np.minimum(
                    self._src.get_values(band=1, window=window),
                    values).reshape((1, *values.shape)).astype(meta['dtype'])
                dst.write(values, window=window)
        self._tmpfile = tmpfile

    def get_mesh(self, geom=None):
        if geom is not None:
            if not isinstance(geom, Geom):
                raise TypeError(f"geom must be of type {Geom}")

        mesh = _jigsaw_hmat_worker(
            self._src._tmpfile,
            list(self._src.iter_windows())[0],
            self._hmin,
            self._hmax,
            geom.geom
            )
        exit()
        # vert2 = list()
        # tria3 = list()
        # value = list()

        # if self._nprocs > 1:
        #     _job_args = []
        #     for window in self._src.iter_windows():
        #         _args = []
        #         _args.append(self._src._tmpfile)
        #         _args.append(window)
        #         _args.append(self._hmin)
        #         _args.append(self._hmax)
        #         _args.append(geom)
        #         _job_args.append(_args)
        #     print(len(_job_args))
        #     with Pool(processes=self._nprocs) as pool:
        #         res = pool.starmap(_jigsaw_hmat_worker, _job_args)
        #     pool.join()
        #     for mesh in res:
        #         print(mesh)
        #     breakpoint()
        #     exit()

        # for i in range(len(self.raster_collection)):
        #     if self._interface == 'libsaw':
        #         # libsaw segfaults randomly when passing hmat.
        #         # cause is unknown. Set self._interface = 'cmdsaw'
        #         # to avoid this issue.
        #         mesh = self._generate_raster_hfun_libsaw(i)
        #     elif self._interface == 'cmdsaw':
        #         mesh = self._generate_raster_hfun_cmdsaw(i)
        #     for index, id_tag in mesh.tria3:
        #         tria3.append(((index + len(vert2)), id_tag))
        #     for coord, id_tag in mesh.vert2:
        #         vert2.append((coord, id_tag))
        #     for val in mesh.value:
        #         value.append(val)
        # hfun = jigsaw_msh_t()
        # hfun.ndims = +2
        # hfun.mshID = "euclidean-mesh"
        # hfun.vert2 = np.array(vert2, dtype=jigsaw_msh_t.VERT2_t)
        # hfun.tria3 = np.array(tria3, dtype=jigsaw_msh_t.TRIA3_t)
        # hfun.value = np.array(
        #     # np.array(value).reshape(len(value), 1),
        #     np.array(value),
        #     dtype=jigsaw_msh_t.REALS_t)
        # return hfun

    @property
    def _src(self):
        try:
            return self.__src
        except AttributeError:
            pass
        raster = Raster(
            self._raster._tmpfile,
            chunk_size=self._raster.chunk_size)
        raster.overlap = 0
        tmpfile = tempfile.NamedTemporaryFile(prefix=str(tmpdir) + '/')
        meta = raster._src.meta.copy()
        nodata = np.finfo(rasterio.float32).max
        meta.update({
            "dtype": rasterio.float32,
            "nodata": nodata
            })
        with rasterio.open(tmpfile.name, 'w', **meta) as dst:
            for i, window in enumerate(raster.iter_windows()):
                dst.write(
                    np.full((1, window.height, window.width), nodata),
                    window=window)
        self._tmpfile = tmpfile
        return self.__src

    @property
    def hfun(self):
        '''Return a jigsaw_msh_t object representing the mesh size'''
        raster = self._src
        x = raster.get_x()
        y = np.flip(raster.get_y())
        _tx = x
        _ty = y
        if self._src.crs.is_geographic:
            _y = np.repeat(np.min(y), len(x))
            _x = np.repeat(np.min(x), len(y))
            _tx = utm.from_latlon(_y, x)[0]
            _ty = utm.from_latlon(y, _x)[1]

        hmat = jigsaw_msh_t()
        hmat.mshID = "euclidean-grid"
        hmat.ndims = +2
        hmat.xgrid = _tx.astype(jigsaw_msh_t.REALS_t)
        hmat.ygrid = _ty.astype(jigsaw_msh_t.REALS_t)

        # TODO: Values of band=1 are only used. Make sure the
        # raster's gaussian_filter writes the relevant values into
        # band 1, or change the hardcoded band id here
        hmat.value = np.flipud(
                raster.get_values(band=1)
                ).astype(jigsaw_msh_t.REALS_t)

        return hmat

    @property
    def _raster(self):
        return self.__raster

    @_raster.setter
    def _raster(self, raster):
        assert isinstance(raster,  Raster)
        self.__raster = raster

    @property
    def _tmpfile(self):
        return self.__tmpfile

    @_tmpfile.setter
    def _tmpfile(self, tmpfile):
        try:
            del(self.__src)
        except AttributeError:
            pass
        self.__src = Raster(tmpfile.name, chunk_size=self._raster.chunk_size)
        self.__tmpfile = tmpfile
