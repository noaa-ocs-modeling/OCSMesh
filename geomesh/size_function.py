import tempfile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.tri import Triangulation
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import cpu_count
from jigsawpy.libsaw import jigsaw
from jigsawpy import jigsaw_msh_t, jigsaw_jig_t
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
        self.verbosity = verbosity

    def __call__(self, i):
        self.__fetch_triangulation(i)
        return tri, values

    # def __iter__(self):
    #     for i, data in enumerate(self.container):
    #        f data is None:
    #             x, y, z, elements, values = self(i)
    #         else:
    #             x, y, z, elements, values = *data
    #         yield data

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

    def __process_raster(self, idx):

        raster = self.raster_collection[idx]

        # generate outband
        outband = np.full(raster.shape, float("inf"))
        for kwargs in self.contours[idx]:
            outband = self.__apply_contour_level(raster, outband, **kwargs)
        kwargs = self.subtidal_flow_limiter[idx]
        if kwargs is not None:
            outband = self.__apply_subtidal_flow_limiter(
                raster, outband, **kwargs)
        kwargs = self.gaussian_filter[idx]
        if kwargs is not None:
            outband = self.__apply_gaussian_filter(outband, **kwargs)
        outband[np.where(outband < self.hmin)] = self.hmin
        outband[np.where(outband > self.hmax)] = self.hmax

        # hfun
        hmat = jigsaw_msh_t()
        hmat.mshID = "euclidean-grid"
        hmat.ndim = 2
        hmat.xgrid = np.array(raster.x, dtype=jigsaw_msh_t.REALS_t)
        hmat.ygrid = np.array(np.flip(raster.y), dtype=jigsaw_msh_t.REALS_t)
        hmat.value = np.array(np.flipud(outband), dtype=jigsaw_msh_t.REALS_t)

        # jigsaw opts
        opts = jigsaw_jig_t()
        opts.verbosity = self.verbosity
        opts.mesh_dim = 2
        opts.hfun_hmin = np.min(outband)
        opts.hfun_hmax = np.max(outband)
        opts.hfun_scal = 'absolute'
        # opts.mesh_top1 = True               # for sharp feat's
        # opts.geom_feat = True

        # pslg
        geom = self.pslg.geom(idx)

        # output mesh
        mesh = jigsaw_msh_t()

        jigsaw(opts, geom, mesh, hfun=hmat)

        print(f"Total nodes {len(mesh.vert2['coord'])}")
        density = len(mesh.vert2['coord']) / self.pslg.multipolygon(idx).area
        print(f"Local node density {density}")

        exit()

        from scipy.interpolate import RectBivariateSpline
        f = RectBivariateSpline(
            raster.x,
            np.flip(raster.y),
            np.flipud(outband).T,
            )

        values = f.ev(
            mesh.vert2['coord'][:, 0],
            mesh.vert2['coord'][:, 1])

        tri = Triangulation(
            mesh.vert2['coord'][:, 0],
            mesh.vert2['coord'][:, 1],
            mesh.tria3['index'])
        plt.triplot(tri, color='y', linewidth=0.1, alpha=0.5)
        plt.tricontourf(tri, values, cmap='jet')
        plt.gca().axis('scaled')
        # exit()
        plt.show()

    def __apply_contour_level(
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
        tree = self.__fetch_raster_level_tree(level)
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

    def __apply_subtidal_flow_limiter(self, raster, outband, hmin, hmax):
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

    def __apply_gaussian_filter(self, outband, **kwargs):
        return gaussian_filter(outband, **kwargs)

    def __fetch_raster_level(self, level):
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
            tmpfile = tempfile.NamedTemporaryFile(prefix=f"sf_rl_{level}_")
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

    def __fetch_raster_level_tree(self, level):
        try:
            return self.raster_level[level]["kdtree"]
        except KeyError:
            points = self.__fetch_raster_level(level)
            self.raster_level[level]["kdtree"] = cKDTree(points)
            return self.raster_level[level]["kdtree"]

    def __fetch_triangulation(self, i):
        data = self.triangulation_collection[i]
        if data is None:
            x, y, elements, values = self.__process_raster(i)
            raise NotImplementedError('continue...')

    @property
    def pslg(self):
        return self._pslg

    @property
    def raster_collection(self):
        return self.pslg.raster_collection

    @property
    def points(self):
        return self.memmap_points

    @property
    def elements(self):
        return self.memmap_elements

    @property
    def triangulation(self):
        return Triangulation(
            self.memmap_points[:, 0],
            self.memmap_points[:, 1],
            self.memmap_elements)

    @property
    def coords(self):
        return self.points[:, :2]

    @property
    def xy(self):
        return self.points[:, :2]

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 1]

    @property
    def values(self):
        return self.points[:, 2]

    @property
    def triangles(self):
        return self.memmap_elements

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
        return self.pslg

    @property
    def ndim(self):
        return 2

    @property
    def verbosity(self):
        return self.__verbosity

    @property
    def tempdir(self):
        return self._tempdir

    @property
    def tmpfile_points(self):
        return self._tmpfile_points

    @property
    def tmpfile_elements(self):
        return self._tmpfile_elements

    @property
    def hfun(self):
        hfun = jigsaw_msh_t()
        hfun.vert2 = self.vert2
        hfun.tria3 = self.tria3
        hfun.value = self.hfun_value
        hfun.ndim = self.ndim
        hfun.mshID = "euclidean-mesh"
        return hfun

    @property
    def vert2(self):
        return np.asarray(
            [([x, y], 0) for x, y in self.coords],
            dtype=jigsaw_msh_t.VERT2_t)

    @property
    def tria3(self):
        return np.asarray(
            [(tuple(indices), 0) for indices in self.elements],
            dtype=jigsaw_msh_t.TRIA3_t)

    @property
    def hfun_value(self):
        return np.asarray(self.values, dtype=jigsaw_msh_t.REALS_t)

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
    def triangulation_collection(self):
        try:
            return self.__triangulation_collection
        except AttributeError:
            self.__triangulation_collection = len(
                self.raster_collection)*[None]
            return self.__triangulation_collection

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

    # @property
    # def _memmap_points(self):
    #     try:
    #         return self.__memmap_points
    #     except AttributeError:
    #         for i, data in enumerate(self.memmap_container):
    #             if data is None:
    #                 data = self.__fill_memmap_container(i)
    #             x, y, z, elements, values = *data


                




    #             # raster.close()
    #         memmap_points = np.memmap(
    #             self.tmpfile_points.name, dtype=float, mode='w+',
    #             shape=points.shape)
    #         memmap_points[:] = points
    #         del memmap_points
    #         self.__memmap_points = np.memmap(
    #             self.tmpfile_points.name, dtype=float, mode='r',
    #             shape=points.shape)
    #         return self.__memmap_points

    # @property
    # def _memmap_elements(self):
    #     try:
    #         return self.__memmap_elements
    #     except AttributeError:
            
    #             raster.close()
    #             elements = tri.triangles[~mask]
    #             memmap_elements = np.memmap(
    #                         self.tmpfile_elements.name,
    #                         dtype=int, mode='r+', shape=elements.shape)
    #             memmap_elements[:] = elements
    #             del memmap_elements
    #             self.__memmap_elements = np.memmap(
    #                 self.tmpfile_elements.name, dtype=int, mode='r',
    #                 shape=elements.shape)
    #             return self.__memmap_elements


    @property
    def _tmpfile_points(self):
        try:
            return self.__tmpfile_points
        except AttributeError:
            self.__tmpfile_points = tempfile.NamedTemporaryFile()
            return self.__tmpfile_points

    @property
    def _tmpfile_elements(self):
        try:
            return self.__tmpfile_elements
        except AttributeError:
            self.__tmpfile_elements = tempfile.NamedTemporaryFile()
            return self.__tmpfile_elements

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
