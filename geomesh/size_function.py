import tempfile
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from matplotlib.tri import Triangulation
from matplotlib.path import Path
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import cpu_count
from jigsawpy.libsaw import jigsaw
from jigsawpy import jigsaw_msh_t, jigsaw_jig_t, savemsh, loadmsh
import geomesh
from geomesh.pslg import PlanarStraightLineGraph


class SizeFunction:

    def __init__(
        self,
        pslg,
        hmin=None,
        hmax=None,
        dst_crs="EPSG:3395",
        verbosity=0,
    ):
        self._pslg = pslg
        self._hmin = hmin
        self._hmax = hmax
        self._dst_crs = dst_crs
        self._verbosity = verbosity

    def tricontourf(self, show=False, **kwargs):
        plt.tricontourf(self.triangulation, self.values, **kwargs)
        plt.colorbar()
        if show:
            plt.gca().axis('scaled')
            plt.show()

    def tripcolor(self, i=0, show=False, **kwargs):
        if isinstance(i, int):
            assert i in list(range(len(self.raster_collection)))
            tri, values = self(i)

        plt.tripcolor(self.triangulation, self.values, **kwargs)
        plt.colorbar()
        if show:
            plt.gca().axis('scaled')
            plt.show()

    def triplot(
        self,
        i=None,
        show=False,
        linewidth=0.07,
        color='black',
        alpha=0.5,
        **kwargs
    ):
        if isinstance(i, int):
            assert i in list(range(len(self.raster_collection)))
            tri, values = self(i)

        elif i is None:
            plt.triplot(
                self.triangulation,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
                **kwargs
                )
            if show:
                plt.gca().axis('scaled')
                plt.show()

    def add_contour(
        self,
        level,
        expansion_rate=0.2,
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
        target_size = self.hmin if target_size is None else float(target_size)
        hmin = self.hmin if hmin is None else float(hmin)
        hmax = self.hmax if hmax is None else float(hmax)
        idx = set(range(len(self.raster_collection))) if idx is None else idx
        assert target_size > 0.
        assert hmin > 0.
        assert hmax > hmin
        assert n_jobs == -1 or n_jobs in list(range(1, cpu_count()+1))
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
            self.contours[_idx].append(kwargs)

    def add_subtidal_flow_limiter(self, hmin=None, hmax=None, idx=None):
        """
        https://wiki.fvcom.pml.ac.uk/doku.php?id=configuration%3Agrid_scale_considerations
        """
        # argument check
        hmin = self.hmin if hmin is None else float(hmin)
        hmax = self.hmax if hmax is None else float(hmax)
        idx = set(range(len(self.raster_collection))) if idx is None else idx
        assert hmin > 0.
        assert hmax > hmin
        if type(idx) is list:
            idx = set(idx)
        elif type(idx) is not set:
            idx = set([idx])
            for i in idx:
                assert idx in list(range(len(self.raster_collection)))
        kwargs = {"hmin": hmin, "hmax": hmax}
        for _idx in idx:
            self.subtidal_flow_limiter[_idx] = kwargs

    def add_gaussian_filter(self, sigma, idx=None, **kwargs):
        if isinstance(idx, int):
            assert idx in list(range(len(self.raster_collection)))
            idx = list(idx)
        elif idx is None:
            i = list(range(len(self.raster_collection)))
        for raster_id in i:
            assert raster_id in list(range(len(self.raster_collection)))
        kwargs.update({"sigma": sigma})
        for raster_id in i:
            self.gaussian_filter[raster_id] = kwargs

    def _get_hfun(self, idx):
        data = self.hfun_collection[idx]
        if data is None:
            mesh, values = self._generate_hfun(idx)
        else:
            mesh = self._load_hfun(idx)
            values = data['values']
        return mesh, values

    def _get_triangulation(self, idx):
        mesh, _ = self._get_hfun(idx)
        return Triangulation(
            mesh.vert2['coord'][:, 0],
            mesh.vert2['coord'][:, 1],
            mesh.tria3['index'])

    def _get_hmat(self, idx):
        raster = self.raster_collection[idx]
        # apply size function requests.
        outband = np.full(raster.shape, float("inf"))
        for kwargs in self.contours[idx]:
            outband = self._apply_contour_level(raster, outband, **kwargs)
        kwargs = self.subtidal_flow_limiter[idx]
        if kwargs is not None:
            outband = self._apply_subtidal_flow_limiter(
                raster, outband, **kwargs)
        kwargs = self.gaussian_filter[idx]
        if kwargs is not None:
            outband = self._apply_gaussian_filter(outband, **kwargs)
        outband[np.where(outband < self.hmin)] = self.hmin
        outband[np.where(outband > self.hmax)] = self.hmax
        # gcreate hmat object
        hmat = jigsaw_msh_t()
        hmat.mshID = "euclidean-grid"
        hmat.ndim = 2
        hmat.xgrid = np.array(raster.x, dtype=jigsaw_msh_t.REALS_t)
        hmat.ygrid = np.array(np.flip(raster.y), dtype=jigsaw_msh_t.REALS_t)
        hmat.value = np.array(np.flipud(outband), dtype=jigsaw_msh_t.REALS_t)
        return hmat

    def _save_hfun(self, idx, mesh, values):
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=geomesh.tmpdir, suffix='.msh')
        savemsh(tmpfile.name, mesh)
        self._hfun_collection[idx] = {'tmpfile': tmpfile, 'values': values}

    def _load_hfun(self, idx):
        mesh = jigsaw_msh_t()
        loadmsh(self.hfun_collection[idx]['tmpfile'].name, mesh)
        return mesh

    def _generate_hfun(self, idx):

        # generate raster size function
        values = self._get_hmat(idx)

        # jigsaw opts
        opts = jigsaw_jig_t()
        opts.verbosity = self.verbosity
        opts.mesh_dim = 2
        opts.hfun_hmin = np.min(values.value)
        opts.hfun_hmax = np.max(values.value)
        opts.hfun_scal = 'absolute'
        opts.optm_tria = False

        # pslg
        geom = self.pslg._get_geom(idx)

        # output mesh
        mesh = jigsaw_msh_t()

        # call jigsaw to optimize local mesh
        jigsaw(opts, geom, mesh, hfun=values)

        # do post processing
        mesh, values = self._jigsaw_post_process(mesh, values)

        # save results to hfun_collection
        self._save_hfun(idx, mesh, values)

        return mesh, values

    def _apply_contour_level(
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
        tree = self._get_raster_level_kdtree(level)
        xt, yt = np.meshgrid(raster.x, raster.y)
        xt = xt.flatten()
        yt = yt.flatten()
        xy_target = np.vstack([xt, yt]).T
        values, _ = tree.query(xy_target, n_jobs=n_jobs)
        values = expansion_rate*target_size*values + target_size
        values = values.reshape(raster.values.shape)
        values[np.where(values < hmin)] = hmin
        values[np.where(values > hmax)] = hmax
        outband = np.minimum(outband, values)
        return outband

    def _apply_subtidal_flow_limiter(self, raster, outband, hmin, hmax):
        dx = np.abs(raster.dx)
        dy = np.abs(raster.dy)
        dx, dy = np.gradient(raster.values, dx, dy)
        dh = np.sqrt(dx**2 + dy**2)
        dh = np.ma.masked_equal(dh, 0.)
        values = np.abs((1./3.)*(raster.values/dh))
        values = values.filled(np.max(values))
        values[np.where(values < hmin)] = hmin
        values[np.where(values > hmax)] = hmax
        outband = np.minimum(outband, values)
        return outband

    def _apply_gaussian_filter(self, outband, **kwargs):
        return gaussian_filter(outband, **kwargs)

    def _get_raster_level(self, level):
        try:
            return self.raster_level[level]["vertices"]
        except KeyError:
            vertices = list()
            for raster in self.raster_collection:
                ax = plt.contour(
                    raster.x, raster.y, raster.values, levels=[level])
                plt.close(plt.gcf())
                for path_collection in ax.collections:
                    for path in path_collection.get_paths():
                        for (x, y), _ in path.iter_segments():
                            vertices.append((x, y))
            vertices = np.asarray(vertices)
            tmpfile = tempfile.NamedTemporaryFile(
                prefix=geomesh.tmpdir, suffix='.raster_level')
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
            return self.raster_level[level]["kdtree"]
        except KeyError:
            points = self._get_raster_level(level)
            self.raster_level[level]["kdtree"] = cKDTree(points)
            return self.raster_level[level]["kdtree"]

    @classmethod
    def _jigsaw_post_process(cls, mesh, hmat):
        mesh = cls._cleanup_isolates(mesh)
        values = cls._get_interpolated_values(mesh, hmat)
        return mesh, values

    @staticmethod
    def _cleanup_isolates(mesh):
        # cleanup isolated nodes
        # TODO: Slow, needs optimization.
        node_indexes = np.arange(mesh.vert2['coord'].shape[0])
        used_indexes = np.unique(mesh.tria3['index'])
        vert2_idxs = np.where(
            np.isin(node_indexes, used_indexes, assume_unique=True))[0]
        tria3_idxs = np.where(
            ~np.isin(node_indexes, used_indexes, assume_unique=True))[0]
        tria3 = mesh.tria3['index'].flatten()
        for idx in reversed(tria3_idxs):
            _idx = np.where(tria3 >= idx)
            tria3[_idx] = tria3[_idx] - 1
        tria3 = tria3.reshape(mesh.tria3['index'].shape)
        _mesh = jigsaw_msh_t()
        _mesh.ndims = 2
        _mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)
        _mesh.tria3 = np.asarray(
            [(tuple(indices), mesh.tria3['IDtag'][i])
             for i, indices in enumerate(tria3)],
            dtype=jigsaw_msh_t.TRIA3_t)
        return _mesh

    @staticmethod
    def _get_interpolated_values(mesh, hmat, **kwargs):
        return RectBivariateSpline(
            hmat.xgrid,
            hmat.ygrid,
            hmat.value.T,
            **kwargs).ev(
            mesh.vert2['coord'][:, 0],
            mesh.vert2['coord'][:, 1])

    @property
    def pslg(self):
        return self._pslg

    @property
    def raster_collection(self):
        return self.pslg.raster_collection

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
        return self.hfun.value

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
    def dst_crs(self):
        return self._dst_crs

    @property
    def hmin(self):
        return self._hmin

    @property
    def hmax(self):
        return self._hmax

    @property
    def geom(self):
        return self.pslg.geom

    @property
    def ndim(self):
        return self.pslg.ndim

    @property
    def verbosity(self):
        return self.__verbosity

    @property
    def hfun(self):
        try:
            return self.__hfun
        except AttributeError:
            vert2 = list()
            tria3 = list()
            value = list()
            for i in range(len(self.raster_collection)):
                mesh, values = self._get_hfun(i)
                for index, id_tag in mesh.tria3:
                    tria3.append(((index + len(vert2)), id_tag))
                for coord, id_tag in mesh.vert2:
                    vert2.append((coord, id_tag))
                value.extend(values)
            hfun = jigsaw_msh_t()
            hfun.vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)
            hfun.tria3 = np.asarray(tria3, dtype=jigsaw_msh_t.TRIA3_t)
            hfun.value = np.asarray(value, dtype=jigsaw_msh_t.REALS_t)
            hfun.ndim = 2
            hfun.mshID = "euclidean-mesh"
            self.__hfun = hfun
            return self.__hfun

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
    def contours(self):
        try:
            return self.__contours
        except AttributeError:
            self.__contours = len(self.raster_collection)*[list()]
            return self.__contours

    @property
    def subtidal_flow_limiter(self):
        try:
            return self.__subtidal_flow_limiter
        except AttributeError:
            self.__subtidal_flow_limiter = len(self.raster_collection)*[None]
            return self.__subtidal_flow_limiter

    @property
    def gaussian_filter(self):
        try:
            return self.__gaussian_filter
        except AttributeError:
            self.__gaussian_filter = len(self.raster_collection)*[None]
            return self.__gaussian_filter

    @property
    def _hfun_collection(self):
        try:
            return self.__hfun_collection
        except AttributeError:
            self.__hfun_collection = len(
                self.raster_collection)*[None]
            return self.__hfun_collection

    @property
    def raster_level(self):
        try:
            return self.__raster_level
        except AttributeError:
            self.__raster_level = dict()
            return self.__raster_level

    @property
    def _hmin(self):
        return self.__hmin

    @property
    def _hmax(self):
        return self.__hmax

    @property
    def _pslg(self):
        return self.__pslg

    @property
    def _mesh(self):
        return self.__mesh

    @property
    def _verbosity(self):
        return self.__verbosity

    @scaling.setter
    def scaling(self, scaling):
        assert scaling in ["absolute", "relative"]
        self.__scaling = scaling

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        self.pslg.dst_crs = dst_crs
        self.__dst_crs = dst_crs

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = verbosity

    @hmin_is_absolute_limit.setter
    def hmin_is_absolute_limit(self, hmin_is_absolute_limit):
        assert isinstance(hmin_is_absolute_limit, bool)
        self.__hmin_is_absolute_limit = hmin_is_absolute_limit

    @hmax_is_absolute_limit.setter
    def hmax_is_absolute_limit(self, hmax_is_absolute_limit):
        assert isinstance(hmax_is_absolute_limit, bool)
        self.__hmax_is_absolute_limit = hmax_is_absolute_limit

    @_pslg.setter
    def _pslg(self, pslg):
        assert isinstance(pslg, PlanarStraightLineGraph)
        self.__pslg = pslg

    @_hmin.setter
    def _hmin(self, hmin):
        if hmin is None:
            # bound hmin to raster resolution.
            hmin = float("inf")
            for raster in self.raster_collection:
                hmin = np.min([np.abs(raster.dx), hmin])
                hmin = np.min([np.abs(raster.dy), hmin])
        self.__hmin = float(hmin)

    @_hmax.setter
    def _hmax(self, hmax):
        if hmax is None:
            # it's safe to keep hmax unbounded
            hmax = float("inf")
        self.__hmax = float(hmax)

    @_verbosity.setter
    def _verbosity(self, verbosity):
        assert isinstance(verbosity, int)
        assert verbosity >= 0
        self.__verbosity = verbosity
