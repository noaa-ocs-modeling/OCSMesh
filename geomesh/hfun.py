import tempfile
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import logging
import os
from functools import lru_cache
import warnings
from scipy.ndimage import gaussian_filter
from matplotlib.tri import Triangulation
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import cpu_count
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, MultiPolygon
from shapely.ops import transform
import jigsawpy
from jigsawpy import jigsaw_msh_t, jigsaw_jig_t, savemsh, loadmsh, certify
import geomesh
from geomesh import tmpdir
from geomesh import utils
from geomesh.raster import Raster, RasterCollection
from geomesh.geom import Geom


class SizeFunction:

    def __init__(
        self,
        hfun,
        hmin=None,
        hmax=None,
        zmin=None,
        zmax=None,
        crs=None,
        dst_crs="EPSG:3395",
        verbosity=0,
    ):
        self._hfun = hfun
        self._hmin = hmin
        self._hmax = hmax
        self._zmin = zmin
        self._zmax = zmax
        self._crs = crs
        self._dst_crs = dst_crs
        self.verbosity = verbosity
        self._interface = 'cmdsaw'

    def contourf(self, **kwargs):
        for i in range(len(self.raster_collection)):
            hmat = self._get_raster_hmat(i)
            plt.contourf(
                hmat.xgrid,
                hmat.ygrid,
                hmat.value,
                levels=kwargs.pop("levels", 256),
                cmap=kwargs.pop('cmap', 'jet'),
                vmin=kwargs.pop('vmin', np.min(hmat.value)),
                vmax=kwargs.pop('vmax', np.max(hmat.value)),
            )
        if kwargs.pop('show', False):
            plt.gca().axis('scaled')
            plt.show()

    def tricontourf(self, **kwargs):
        return utils.tricontourf(self.hfun, **kwargs)

    def triplot(self, **kwargs):
        return utils.triplot(self.hfun, **kwargs)

    def add_contour(
        self,
        level,
        expansion_rate,
        target_size=None,
        hmin=None,
        hmax=None,
        n_jobs=-1,
        idx=None
    ):
        """Adds a contour level to the size function.

        Args:
            level (float): The contour level to add wrt DEM vertical datum.
            expansion_rate (float): The rate at which elements expand.
            target_size (float)

        """
        # argument checks
        level = float(level)
        expansion_rate = float(expansion_rate)
        if target_size is None and self.hmin is not None:
            target_size = self.hmin
        else:
            target_size = float(target_size)
        idx = set(range(len(self.raster_collection))) if idx is None else idx
        assert target_size > 0.
        assert n_jobs == -1 or n_jobs in list(range(1, cpu_count()+1))
        if isinstance(
            self._hfun,
            (Raster,
             RasterCollection,
             Geom)
        ):
            if type(idx) is list:
                idx = set(idx)
            elif type(idx) is not set:
                idx = set([idx])
                for i in idx:
                    assert idx in list(range(len(self.raster_collection)))
            kwargs = {
                "level": level,
                "expansion_rate": expansion_rate,
                "target_size": target_size,
                "hmin": hmin,
                "hmax": hmax,
                "n_jobs": n_jobs}
            for _idx in idx:
                self._raster_contours[_idx].append(kwargs)
        else:
            msg = 'contour only available for '
            msg += f'{Raster}, {RasterCollection} and '
            msg += f'{Geom} inputs.'
            raise NotImplementedError(msg)

    def add_subtidal_flow_limiter(self, hmin=None, hmax=None, idx=None):
        """
        https://wiki.fvcom.pml.ac.uk/doku.php?id=configuration%3Agrid_scale_considerations
        """
        if isinstance(
            self._hfun,
            (
                Raster,
                RasterCollection,
                Geom)
        ):
            # argument check
            idx = set(
                range(len(self.raster_collection))) if idx is None else idx
            if type(idx) is list:
                i = set([idx])
            elif type(idx) is not set:
                idx = set([idx])
                for i in idx:
                    assert idx in list(range(len(self.raster_collection)))
            kwargs = {"hmin": hmin, "hmax": hmax}
            for _idx in idx:
                self._subtidal_flow_limiter[_idx] = kwargs
        else:
            msg = 'Subtidal flow limiter only available for '
            msg += f'{Raster} and {RasterCollection} and '
            msg += f'{Geom} geom inputs.'
            raise NotImplementedError(msg)

    def add_feature(self, geometry, target_size, expansion_rate, n_jobs=-1):
        msg = f"geometry must be and instance of type {LineString}"
        assert isinstance(geometry, (LineString, MultiLineString)), msg
        self._hfun_features.append({
            "geometry": geometry,
            "target_size": target_size,
            "expansion_rate": expansion_rate,
            "n_jobs": n_jobs
            })

    def add_gaussian_filter(self, sigma, indexes=None, **kwargs):
        if isinstance(
            self._hfun,
            (
                Raster,
                RasterCollection,
                Geom)
        ):
            if indexes is None:
                indexes = list(range(len(self.raster_collection)))
            indexes = set(list(indexes))
            for _idx in indexes:
                self._gaussian_filter[_idx] = kwargs
        else:
            msg = 'gaussian filter only available for '
            msg += f'{Raster}, {RasterCollection} and '
            msg += f'{Geom} geom inputs.'
            raise NotImplementedError(msg)

    def floodplain_size(self, size, indexes=None):
        if isinstance(
            self._hfun,
            (
                Raster,
                RasterCollection,
                Geom)
        ):
            if indexes is None:
                indexes = list(range(len(self.raster_collection)))
            indexes = set(list(indexes))
            for _idx in indexes:
                self._floodplain_size[_idx] = size
        else:
            msg = 'gaussian filter only available for '
            msg += f'{Raster}, {RasterCollection} and '
            msg += f'{Geom} geom inputs.'
            raise NotImplementedError(msg)

    def limgrad(self, expansion_rate, imax=100):
        self._value = utils.limgrad(
            self.hfun,
            expansion_rate,
            imax)

    @property
    def raster_level(self):
        try:
            return self.__raster_level
        except AttributeError:
            self.__raster_level = dict()
            return self.__raster_level

    @property
    def raster_collection(self):
        return tuple(self._raster_collection)

    @property
    def hfun_collection(self):
        return tuple(self._hfun_collection)

    @property
    def triangulation(self):
        return Triangulation(self.x, self.y, self.triangles)

    @property
    def coords(self):
        return self.hfun.vert2['coord']

    @property
    def triangles(self):
        return self.hfun.tria3['index']

    @property
    def values(self):
        return self.hfun.value.flatten()

    @property
    def multipolygon(self):
        return self.pslg.multipolygon

    @property
    def xy(self):
        return self.coords

    @property
    def x(self):
        return self.xy[:, 0]

    @property
    def y(self):
        return self.xy[:, 1]

    @property
    def scaling(self):
        try:
            return self.__scaling
        except AttributeError:
            self.__scaling = "absolute"
            return self.__scaling

    @property
    def crs(self):
        return self._dst_crs

    # @property
    # def dst_crs(self):
    #     return self._dst_crs

    @property
    def hmin(self):
        return self._hmin

    @property
    def hmax(self):
        return self._hmax

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    @property
    @lru_cache
    def geom(self):
        if isinstance(self._hfun, Geom):
            geom = self._hfun
            geom.dst_crs = self._dst_crs

        elif isinstance(self._hfun, (Raster, RasterCollection)):
            # possbile bug: assumes all rasters have same clip and same crs
            # self._hfun.dst_crs = self.dst_crs
            geom = Geom(
                geom=self._hfun,
                zmin=self.zmin,
                zmax=self.zmax,
                clip=self._clip,
                crs=self._hfun.dst_crs,  # does Raster have dst_crs?
                dst_crs=self._dst_crs)

        elif isinstance(self._hfun, jigsaw_msh_t):
            geom = Geom(
                geom=self._hfun,
                zmin=self.zmin,
                zmax=self.zmax,
                clip=self._clip,
                crs=self.crs,
                dst_crs=self._dst_crs)

        else:
            msg = "Must implement self._geom for hfun type: "
            msg += f"{type(self._hfun)}"
            raise NotImplementedError(msg)

        return geom

    @property
    def ndims(self):
        return self.geom.ndims

    @property
    def verbosity(self):
        return self.__verbosity

    @property
    def bbox(self):
        hfun = self.hfun
        x0 = np.min(hfun.vert2['coord'][:, 0])
        x1 = np.max(hfun.vert2['coord'][:, 0])
        y0 = np.min(hfun.vert2['coord'][:, 1])
        y1 = np.max(hfun.vert2['coord'][:, 1])
        return Bbox([[x0, y0], [x1, y1]])

    @property
    def hfun(self):
        try:
            hfun = jigsaw_msh_t()
            loadmsh(self.__hfun_tmpfile.name, hfun)
            return hfun
        except AttributeError:

            if isinstance(
                self._hfun,
                (Raster, RasterCollection, Geom)
            ):
                hfun = self._generate_raster_hfun()

            elif isinstance(self._hfun, jigsaw_msh_t):
                hfun = self._hfun

            else:
                msg = f"Must implement HFUN for hfun type {type(self._hfun)}"
                raise NotImplementedError(msg)

            tmpfile = tempfile.NamedTemporaryFile(prefix=tmpdir, suffix='.msh')

            savemsh(tmpfile.name, hfun)
            self.__hfun_tmpfile = tmpfile

            return hfun

    @property
    def vert2(self):
        return self.hfun.vert2

    @property
    def tria3(self):
        return self.hfun.tria3

    @property
    def value(self):
        return self.hfun.value

    @property
    def hmin_is_absolute_limit(self):
        try:
            return self.__hmin_is_absolute_limit
        except AttributeError:
            # Uses the data's hmin limit by default
            return False

    @property
    def hmax_is_absolute_limit(self):
        try:
            return self.__hmax_is_absolute_limit
        except AttributeError:
            # Uses the data's hmax limit by default
            return False

    @property
    def raster_contours(self):
        return tuple(self._raster_contours)

    @property
    def subtidal_flow_limiter(self):
        return tuple(self._subtidal_flow_limiter)

    @property
    def raster_gaussian_filter(self):
        return tuple(self._raster_gaussian_filter)

    @property
    def logger(self):
        try:
            return self.__logger
        except AttributeError:
            self.__logger = logging.getLogger(
                __name__ + '.' + self.__class__.__name__)
            return self.__logger

    @scaling.setter
    def scaling(self, scaling):
        assert scaling in ["absolute", "relative"]
        self.__scaling = scaling

    # @dst_crs.setter
    # def dst_crs(self, dst_crs):
    #     self._dst_crs = dst_crs

    @verbosity.setter
    def verbosity(self, verbosity):
        assert isinstance(verbosity, int)
        assert verbosity >= 0
        self.__verbosity = verbosity

    @hmin_is_absolute_limit.setter
    def hmin_is_absolute_limit(self, hmin_is_absolute_limit):
        assert isinstance(hmin_is_absolute_limit, bool)
        self.__hmin_is_absolute_limit = hmin_is_absolute_limit

    @hmax_is_absolute_limit.setter
    def hmax_is_absolute_limit(self, hmax_is_absolute_limit):
        assert isinstance(hmax_is_absolute_limit, bool)
        self.__hmax_is_absolute_limit = hmax_is_absolute_limit

    def _get_raster_hfun(self, idx):
        mesh = self.hfun_collection[idx]
        if mesh is None:
            mesh = self._generate_raster_hfun(idx)
            self._save_raster_hfun(idx, mesh)
        else:
            mesh = self._load_raster_hfun(idx)
        return mesh

# ---------- auxilliary functions

    def _generate_raster_hfun(self):
        vert2 = list()
        tria3 = list()
        value = list()
        for i in range(len(self.raster_collection)):
            if self._interface == 'libsaw':
                # libsaw segfaults randomly when passing hmat.
                # cause is unknown. Set self._interface = 'cmdsaw'
                # to avoid this issue.
                mesh = self._generate_raster_hfun_libsaw(i)
            elif self._interface == 'cmdsaw':
                mesh = self._generate_raster_hfun_cmdsaw(i)
            for index, id_tag in mesh.tria3:
                tria3.append(((index + len(vert2)), id_tag))
            for coord, id_tag in mesh.vert2:
                vert2.append((coord, id_tag))
            for val in mesh.value:
                value.append(val)
        hfun = jigsaw_msh_t()
        hfun.ndims = +2
        hfun.mshID = "euclidean-mesh"
        hfun.vert2 = np.array(vert2, dtype=jigsaw_msh_t.VERT2_t)
        hfun.tria3 = np.array(tria3, dtype=jigsaw_msh_t.TRIA3_t)
        hfun.value = np.array(
            np.array(value).reshape(len(value), 1),
            # np.array(value),
            dtype=jigsaw_msh_t.REALS_t)
        return hfun

    def _generate_raster_hfun_libsaw(self, idx):
        """
        sometimes jisawpy.libsaw.jigsaw() goes into segmentation fault
        when reading hmat on this function. The root cause for this
        segmentation fault is unknown.
        gdb output:
        Program terminated with signal SIGSEGV, Segmentation fault.
        #0  0x00007f02e9b567ec in containers::array<double, allocators::basic_alloc>::push_tail(double const&) ()  # noqa:E501
            from /geomesh/.geomesh_env/lib/libjigsaw.so
        """

        # get raster hmat
        self.logger.debug(f'_get_raster_hfun_libsaw({idx})')
        hmat = self._get_raster_hmat(idx)

        # get geom
        if isinstance(self.geom._geom, (Raster, RasterCollection)):
            self.logger.debug(
                f'_get_raster_libsaw_opts({idx}): get_raster_geom')
            geom = self.geom._get_raster_geom(idx)
        else:
            self.logger.debug(
                f'_get_raster_libsaw_opts({idx}): get_non_raster_geom')
            geom = self.geom.geom

        # init opts
        opts = jigsaw_jig_t()

        # additional configuration options
        opts.verbosity = self.verbosity
        opts.mesh_dims = self.ndims
        opts.hfun_scal = 'absolute'
        opts.optm_tria = False

        if self.hmin_is_absolute_limit:
            opts.hfun_hmin = self.hmin
        else:
            opts.hfun_hmin = np.min(hmat.value)

        if self.hmax_is_absolute_limit:
            opts.hfun_hmax = self.hmax
        else:
            opts.hfun_hmax = np.max(hmat.value)

        # output mesh
        mesh = jigsaw_msh_t()

        # call jigsaw to create local mesh
        jigsawpy.libsaw.jigsaw(
            opts,
            geom,
            mesh,
            hfun=hmat
        )

        # do post processing
        utils.cleanup_isolates(mesh)
        utils.interpolate_hmat(
            mesh,
            self._get_raster_hmat(idx),
            kx=1,
            ky=1
            )

        return mesh

    def _generate_raster_hfun_cmdsaw(self, idx):
        """
        default for calling jigsaw when handling hmat objects.
        """
        msg = f'_generate_raster_hfun_cmdsaw({idx})'
        self.logger.debug(msg)

        # get raster hmat
        self.logger.debug(f'_get_raster_cmdsaw_opts({idx})')
        hmat = self._get_raster_hmat(idx)

        # get geom
        if isinstance(self.geom._geom, RasterCollection):
            self.logger.debug(
                f'_get_raster_cmdsaw_opts({idx}): get_raster_geom')
            geom = self.geom._get_raster_geom(idx)
        else:
            self.logger.debug(
                f'_get_raster_cmdsaw_opts({idx}): get_pslg_geom')
            geom = self.geom.geom

        # init tmpfiles
        self.logger.debug(f'_get_raster_cmdsaw_opts({idx}): init tmpfiles')
        mesh_file = tempfile.NamedTemporaryFile(
            prefix=tmpdir, suffix='.msh')
        hmat_file = tempfile.NamedTemporaryFile(
            prefix=tmpdir, suffix='.msh')
        geom_file = tempfile.NamedTemporaryFile(
            prefix=tmpdir, suffix='.msh')
        jcfg_file = tempfile.NamedTemporaryFile(
            prefix=tmpdir, suffix='.jig')

        # dump data to tempfiles
        savemsh(hmat_file.name, hmat)
        savemsh(geom_file.name, geom)

        # init opts
        opts = jigsaw_jig_t()
        opts.mesh_file = mesh_file.name
        opts.hfun_file = hmat_file.name
        opts.geom_file = geom_file.name
        opts.jcfg_file = jcfg_file.name

        # additional configuration options
        opts.verbosity = self.verbosity
        opts.mesh_dims = self.ndims
        opts.hfun_scal = 'absolute'
        opts.optm_tria = False

        if self.hmin_is_absolute_limit:
            opts.hfun_hmin = self.hmin
        else:
            opts.hfun_hmin = np.min(hmat.value)

        if self.hmax_is_absolute_limit:
            opts.hfun_hmax = self.hmax
        else:
            opts.hfun_hmax = np.max(hmat.value)

        # init outputmesh
        mesh = jigsaw_msh_t()

        # call jigsaw
        self.logger.debug(f'_generate_raster_hfun_cmdsaw({idx}) call cmdsaw')
        jigsawpy.cmd.jigsaw(opts, mesh)

        # do post processing
        utils.cleanup_isolates(mesh)
        utils.interpolate_hmat(
                mesh,
                self._get_raster_hmat(idx),
                kx=1,
                ky=1
                )

        # cleanup log-file
        try:
            os.remove(opts.jcfg_file.strip('.jig')+'.log')
        # on some rare ocassions cmd.jigsaw fails to create the log file.
        except FileNotFoundError:
            pass

        # cleanup temporary files
        for tmpfile in (mesh_file, hmat_file, geom_file, jcfg_file):
            del(tmpfile)

        return mesh

    def _get_raster_hmat(self, idx):
        self.logger.debug(f'_get_raster_hmat({idx})')
        tmpfile = self._hmat_collection[idx]
        if tmpfile is None:
            hmat = self._generate_raster_hmat(idx)
            self._save_raster_hmat(idx, hmat)
        else:
            hmat = jigsaw_msh_t()
            loadmsh(tmpfile.name, hmat)
            # fix shape output in hmat.value from loadmsh()
            # TODO: this is not clear
            if self._interface == 'cmdsaw':
                hmat.value = hmat.value.reshape(
                    (hmat.xgrid.size, hmat.ygrid.size)).T
        return hmat

    def _generate_raster_hmat(self, idx):
        self.logger.debug(f'_generate_raster_hmat({idx})')
        raster = self.raster_collection[idx]
        # apply size function requests.
        outband = np.full(raster.shape, float("inf"))
        # contours
        for kwargs in self.raster_contours[idx]:
            outband = self._apply_raster_contour_level(
                raster, outband, **kwargs)
        # subtidal flow limiter
        kwargs = self.subtidal_flow_limiter[idx]
        if kwargs is not None:
            outband = self._apply_raster_subtidal_flow_limiter(
                raster, outband, **kwargs)

        # gaussian filter
        kwargs = self.raster_gaussian_filter[idx]
        if kwargs is not None:
            outband = self._apply_raster_gaussian_filter(outband, **kwargs)

        # floodplain size
        size = self._floodplain_size[idx]
        if size is not None:
            outband = self._apply_floodplain_size(raster, outband, size)

        for fid, _ in enumerate(self._hfun_features):
            outband = self._apply_hfun_features(outband, raster, fid)

        # global size limits
        if self.hmin is not None:
            outband[np.where(outband < self.hmin)] = self.hmin
        if self.hmax is not None:
            outband[np.where(outband > self.hmax)] = self.hmax

        # create hmat object
        hmat = jigsaw_msh_t()
        hmat.mshID = "euclidean-grid"
        hmat.ndims = +2
        hmat.xgrid = np.array(raster.x, dtype=jigsaw_msh_t.REALS_t)
        hmat.ygrid = np.array(np.flip(raster.y), dtype=jigsaw_msh_t.REALS_t)
        hmat.value = np.array(np.flipud(outband), dtype=jigsaw_msh_t.REALS_t)
        return hmat

    def _apply_raster_contour_level(
        self,
        raster,
        outband,
        level,
        expansion_rate,
        target_size,
        hmin,
        hmax,
        n_jobs
    ):
        # calculate distances between each pixel and nearest contour point
        msg = f'_apply_raster_contour_level({raster.path})'
        self.logger.debug(msg)
        tree = self._get_raster_level_kdtree(level)
        xt, yt = np.meshgrid(raster.x, raster.y)
        xt = xt.flatten()
        yt = yt.flatten()
        xy_target = np.vstack([xt, yt]).T
        values, _ = tree.query(xy_target, n_jobs=n_jobs)
        values = expansion_rate*target_size*values + target_size
        values = values.reshape(raster.values.shape)
        if hmin is not None:
            values[np.where(values < hmin)] = hmin
        if hmax is not None:
            values[np.where(values > hmax)] = hmax
        outband = np.minimum(outband, values)
        return outband

    def _apply_raster_subtidal_flow_limiter(self, raster, outband, hmin, hmax):
        msg = f'_apply_raster_subtidal_flow_limiter({raster.path})'
        self.logger.debug(msg)
        dx = np.abs(raster.dx)
        dy = np.abs(raster.dy)
        dx, dy = np.gradient(raster.values, dx, dy)
        with warnings.catch_warnings():
            # in case raster.values is a masked array
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dh = np.sqrt(dx**2 + dy**2)
        dh = np.ma.masked_equal(dh, 0.)
        values = np.abs((1./3.)*(raster.values/dh))
        values = values.filled(np.max(values))
        if hmin is not None:
            values[np.where(values < hmin)] = hmin
        if hmax is not None:
            values[np.where(values > hmax)] = hmax
        outband = np.minimum(outband, values)
        return outband

    def _apply_raster_gaussian_filter(self, outband, **kwargs):
        self.logger.debug('_apply_raster_gaussian_filter()')
        return gaussian_filter(outband, **kwargs)

    def _apply_floodplain_size(self, raster, outband, size):
        self.logger.debug('_apply_floodplain_size()')
        values = outband.copy()
        values[np.where(raster.values > 0)] = size
        return np.minimum(outband, values)

    def _apply_hfun_features(self, outband, raster, feature_id):
        self.logger.debug('_apply_hfun_features()')
        # magic starts here
        values = outband.copy()
        tree = self._hfun_features[feature_id].get('kdtree', None)
        n_jobs = self._hfun_features[feature_id]['n_jobs']
        expansion_rate = self._hfun_features[feature_id]['expansion_rate']
        target_size = self._hfun_features[feature_id]['target_size']
        if tree is None:
            line = self._hfun_features[feature_id]['geometry']
            distances = [0]
            while (distances[-1] +
                    self._hfun_features[feature_id]["target_size"]) \
                    < line.length:
                distances.append(
                    distances[-1] +
                    self._hfun_features[feature_id]["target_size"]
                    )
            distances.append(line.length)
            new_points = [
                line.interpolate(distance)
                for distance in distances
                ]
            new_points = [(p.x, p.y) for p in new_points]
            tree = cKDTree(np.array(new_points))
            self._hfun_features[feature_id].update({'kdtree': tree})
        xt, yt = np.meshgrid(raster.x, raster.y)
        xt = xt.flatten()
        yt = yt.flatten()
        xy_target = np.vstack([xt, yt]).T
        values, _ = tree.query(
            xy_target,
            n_jobs=n_jobs
            )
        values = expansion_rate*target_size*values + target_size
        values = values.reshape(raster.values.shape)
        return np.minimum(outband, values)

    def _get_raster_level(self, level):
        try:
            self.logger.debug(f'_get_raster_level({level})')
            return self.raster_level[level]["vertices"]
        except KeyError:
            self.logger.debug(f'_get_raster_level({level}):KeyError')
            vertices = list()
            for raster in self.raster_collection:
                with warnings.catch_warnings():
                    # suppress UserWarning: No contour levels were found within
                    # the data range.
                    warnings.simplefilter("ignore", category=UserWarning)
                    ax = plt.contour(
                        raster.x, raster.y, raster.values, levels=[level])
                plt.close(plt.gcf())
                for path_collection in ax.collections:
                    for path in path_collection.get_paths():
                        for (x, y), _ in path.iter_segments():
                            vertices.append((x, y))
            vertices = np.asarray(vertices)
            tmpfile = tempfile.NamedTemporaryFile(
                prefix=tmpdir, suffix='.raster_level')
            memmap_vertices = np.memmap(
                tmpfile.name, dtype=float, mode='w+', shape=vertices.shape)
            memmap_vertices[:] = vertices
            del memmap_vertices
            memmap_vertices = np.memmap(
                tmpfile.name, dtype=float, mode='r', shape=vertices.shape)
            self.raster_level[level] = {
                "tmpfile": tmpfile,
                "vertices": memmap_vertices
            }
            return self.raster_level[level]["vertices"]

    def _get_raster_level_kdtree(self, level):
        try:
            self.logger.debug(f'_get_raster_level_kdtree({level})')
            return self.raster_level[level]["kdtree"]
        except KeyError:
            self.logger.debug(f'_get_raster_level_kdtree({level}):KeyError')
            points = self._get_raster_level(level)
            self.raster_level[level]["kdtree"] = cKDTree(points)
            return self.raster_level[level]["kdtree"]

    def _save_raster_hfun(self, idx, mesh):
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=tmpdir, suffix='.msh')
        savemsh(tmpfile.name, mesh)
        self._hfun_collection[idx] = tmpfile

    def _load_raster_hfun(self, idx):
        mesh = jigsaw_msh_t()
        loadmsh(self.hfun_collection[idx].name, mesh)
        return mesh

    def _save_raster_hmat(self, idx, mesh):
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=tmpdir, suffix='.msh')
        savemsh(tmpfile.name, mesh)
        self._hmat_collection[idx] = tmpfile

    def _load_raster_hmat(self, idx):
        mesh = jigsaw_msh_t()
        loadmsh(self.hmat_collection[idx].name, mesh)
        return mesh

    def _transform_multipolygon(self, multipolygon):
        if self.dst_crs.srs != self._crs.srs:
            transformer = Transformer.from_crs(
                self._crs, self._dst_crs, always_xy=True)
            polygon_collection = list()
            for polygon in multipolygon:
                polygon_collection.append(
                    transform(transformer.transform, polygon))
            outer = polygon_collection.pop(0)
            multipolygon = MultiPolygon([outer, *polygon_collection])
        return multipolygon

    @property
    def _hfun_collection(self):
        try:
            return self.__hfun_collection
        except AttributeError:
            self.__hfun_collection = len(self.raster_collection)*[None]
            return self.__hfun_collection

    @property
    def _hmat_collection(self):
        try:
            return self.__hmat_collection
        except AttributeError:
            self.__hmat_collection = len(self.raster_collection)*[None]
            return self.__hmat_collection

    @property
    def _raster_collection(self):
        if isinstance(self._hfun, RasterCollection):
            return self._hfun

        if isinstance(self._hfun, Raster):
            return [self._hfun]

        if isinstance(self._hfun, Geom):
            if isinstance(self._hfun._geom, (Raster, RasterCollection)):
                return self._hfun.raster_collection

    @property
    def _hfun(self):
        return self.__hfun

    @property
    def _hmin(self):
        return self.__hmin

    @property
    def _hmax(self):
        return self.__hmax

    # @property
    # def _clip(self):
    #     return self.__clip

    @property
    def _value(self):
        return self.hfun.value

    @property
    def _crs(self):
        return self.__crs

    @property
    def _subtidal_flow_limiter(self):
        try:
            return self.__subtidal_flow_limiter
        except AttributeError:
            self.__subtidal_flow_limiter = len(self.raster_collection)*[None]
            return self.__subtidal_flow_limiter

    @property
    def _raster_contours(self):
        try:
            return self.__raster_contours
        except AttributeError:
            self.__raster_contours = len(self.raster_collection)*[list()]
            return self.__raster_contours

    @property
    def _raster_gaussian_filter(self):
        try:
            return self.__raster_gaussian_filter
        except AttributeError:
            self.__raster_gaussian_filter = len(self.raster_collection)*[None]
            return self.__raster_gaussian_filter

    @property
    def _timestep_limiter(self):
        try:
            return self.__timestep_limiter
        except AttributeError:
            self.__timestep_limiter = len(self.raster_collection)*[None]
            return self.__timestep_limiter

    @property
    def _zmin(self):
        return self.__zmin

    @property
    @lru_cache
    def _hfun_features(self):
        return []

    @property
    @lru_cache
    def _floodplain_size(self):
        return len(self.raster_collection)*[None]

    @property
    def _zmax(self):
        return self.__zmax

    @property
    def _dst_crs(self):
        return self.__dst_crs

    @property
    def _interface(self):
        try:
            return self.__interface
        except AttributeError:
            return 'cmdsaw'

    @_hfun.setter
    def _hfun(self, hfun):
        """
        input can be one of several object type instances.
        """
        assert isinstance(
            hfun,
            (Geom,
             geomesh.mesh.Mesh,
             jigsaw_msh_t,
             Raster,
             RasterCollection,
             float,
             int
             ))

        # sanitize inputs
        # case 1: jigsaw_msh_t
        if isinstance(hfun, jigsaw_msh_t):
            certify(hfun)

        # case 2: Geom
        elif isinstance(hfun, Geom):
            if len(hfun.raster_collection) == 0:
                msg = "A geomesh.Geom object used for hfun generation must "
                msg += "contain at least 1 raster associated to it."
                raise Exception(msg)

        self.__hfun = hfun

    @_hmin.setter
    def _hmin(self, hmin):
        if hmin is not None:
            assert isinstance(hmin, (float, int))
        self.__hmin = hmin

    @_hmax.setter
    def _hmax(self, hmax):
        if hmax is not None:
            assert isinstance(hmax, (float, int))
        self.__hmax = hmax

    @_zmin.setter
    def _zmin(self, zmin):
        if zmin is not None:
            assert isinstance(zmin, (int, float))
        self.__zmin = zmin

    @_zmax.setter
    def _zmax(self, zmax):
        if zmax is not None:
            assert isinstance(zmax, (int, float))
        self.__zmax = zmax

    @_value.setter
    def _value(self, value):
        hfun = self.hfun
        value = np.asarray(value).flatten()
        assert value.size == hfun.value.size
        hfun.value = np.array(
            value.reshape(value.size, 1),
            dtype=jigsaw_msh_t.REALS_t)
        tmpfile = tempfile.NamedTemporaryFile(
                prefix=tmpdir, suffix='.msh')
        savemsh(tmpfile.name, hfun)
        self.__hfun_tmpfile = tmpfile

    # @_clip.setter
    # def _clip(self, clip):
    #     if clip is not None:
    #         assert isinstance(clip, (Polygon, MultiPolygon))
    #         if isinstance(clip, Polygon):
    #             clip = MultiPolygon([clip])
    #         if self._crs.srs != self.dst_crs.srs:
    #             clip = self._transform_multipolygon(clip)
    #     self.__clip = clip

    @_crs.setter
    def _crs(self, crs):
        if crs is None:
            if isinstance(self._hfun, (Raster, RasterCollection)):
                crs = self._hfun.dst_crs

            elif isinstance(self._hfun, Geom):
                crs = self._hfun.crs

            else:
                msg = "Must specify CRS when hfun is of type "
                msg += f"{type(self._hfun)}."
                raise Exception(msg)
        crs = CRS.from_user_input(crs)
        self.__crs = crs

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        msg = "dst_crs cannot be None"
        assert dst_crs is not None, msg

        dst_crs = CRS.from_user_input(dst_crs)

        # transform hfun
        if isinstance(self._hfun, (Raster, RasterCollection)):
            self._hfun.dst_crs = dst_crs

        self.__dst_crs = dst_crs

    @_interface.setter
    def _interface(self, interface):
        assert interface in ['libsaw', 'cmdsaw']
        self.__interface = interface

    # def __add__(self, other):
    #     assert isinstance(other, SizeFunction)
    #     pslg = self.pslg + other.pslg
        # vert2 = list()
        # tria3 = list()
        # value = list()
        # bbox = other.bbox
        # vert2 = self.vert2.copy()
        # idxs = np.where(
        #     np.logical_and(
        #         np.logical_and(
        #             vert2['coord'][:, 0] >= bbox.xmin,
        #             vert2['coord'][:, 0] <= bbox.xmax),
        #         np.logical_and(
        #             vert2['coord'][:, 1] >= bbox.ymin,
        #             vert2['coord'][:, 1] <= bbox.ymax)))[0]
        # breakpoint()
        # print(np.where(np.isin(idxs == self.tria3['index'])))
        # exit()
        # vert2 = vert2.take(idxs)
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
        #     np.array(value).reshape(len(value), 1),
        #     dtype=jigsaw_msh_t.REALS_t)
        # return hfun

        # bbox = other.bbox
        # vert2 = self.vert2.copy()
        # idxs = np.where(
        #     ~np.logical_and(
        #         np.logical_and(
        #             vert2['coord'][:, 0] >= bbox.xmin,
        #             vert2['coord'][:, 0] <= bbox.xmax),
        #         np.logical_and(
        #             vert2['coord'][:, 1] >= bbox.ymin,
        #             vert2['coord'][:, 1] <= bbox.ymax)))[0]
        # vert2 = np.hstack([vert2.take(idxs), other.vert2])
        # value = self.value.copy()
        # value = np.hstack([value.take(idxs), other.value.flatten()])
        # tri = Triangulation(vert2['coord'][:, 0], vert2['coord'][:, 1])
        # mask = np.full((tri.triangles.shape[0],), True)
        # centroids = np.vstack(
        #     [np.sum(vert2['coord'][:, 0][tri.triangles], axis=1) / 3,
        #      np.sum(vert2['coord'][:, 1][tri.triangles], axis=1) / 3]).T
        # for polygon in self.multipolygon:
        #     path = Path(polygon.exterior.coords, closed=True)
        #     bbox = path.get_extents()
        #     idxs = np.where(np.logical_and(
        #                         np.logical_and(
        #                             bbox.xmin <= centroids[:, 0],
        #                             bbox.xmax >= centroids[:, 0]),
        #                         np.logical_and(
        #                             bbox.ymin <= centroids[:, 1],
        #                             bbox.ymax >= centroids[:, 1])))[0]
        #     mask[idxs] = np.logical_and(
        #                 mask[idxs], ~path.contains_points(centroids[idxs]))
        # for polygon in self.multipolygon:
        #     for interior in polygon.interiors:
        #         path = Path(interior.coords, closed=True)
        #         bbox = path.get_extents()
        #         idxs = np.where(np.logical_and(
        #                         np.logical_and(
        #                             bbox.xmin <= centroids[:, 0],
        #                             bbox.xmax >= centroids[:, 0]),
        #                         np.logical_and(
        #                             bbox.ymin <= centroids[:, 1],
        #                             bbox.ymax >= centroids[:, 1])))[0]
        #         mask[idxs] = np.logical_or(
        #                 mask[idxs], path.contains_points(centroids[idxs]))
        # tri = Triangulation(
        #     vert2['coord'][:, 0],
        #     vert2['coord'][:, 1],
        #     tri.triangles[~mask])
        # hfun = jigsaw_msh_t()
        # hfun.mshID = 'euclidean-mesh'
        # hfun.ndims = self.ndims
        # hfun.vert2 = vert2
        # hfun.tria3 = np.array(
        #     [(index, 0) for index in tri.triangles],
        #     dtype=jigsaw_msh_t.TRIA3_t)
        # hfun.value = np.array(
        #     value.reshape((value.size, 1)),
        #     dtype=jigsaw_msh_t.REALS_t)
        # return SizeFunction(hfun, crs=self.crs)
