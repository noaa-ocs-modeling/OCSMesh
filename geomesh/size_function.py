import tempfile
import gc
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.tri import Triangulation
import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
from jigsawpy import jigsaw_msh_t
from geomesh.pslg import PlanarStraightLineGraph
from geomesh import parallel_processing


class SizeFunction:

    def __init__(
        self,
        pslg,
        hmin=None,
        hmax=None,
        dst_crs="EPSG:3395",
        nproc=-1
    ):
        self._pslg = pslg
        self._hmin = hmin
        self._hmax = hmax
        self._dst_crs = dst_crs
        self._nproc = nproc

    def tricontourf(self, show=False, **kwargs):
        plt.tricontourf(self.triangulation, self.values, **kwargs)
        plt.colorbar()
        if show:
            plt.gca().axis('scaled')
            plt.show()

    def tripcolor(self, show=False, **kwargs):
        plt.tripcolor(self.triangulation, self.values, **kwargs)
        plt.colorbar()
        if show:
            plt.gca().axis('scaled')
            plt.show()

    def triplot(
        self,
        show=False,
        linewidth=0.07,
        color='black',
        alpha=0.5,
        **kwargs
    ):
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
        n_jobs=None
    ):

        # argument checks
        level = float(level)
        expansion_rate = float(expansion_rate)
        target_size = self.hmin if target_size is None else float(target_size)
        hmin = self.hmin if hmin is None else float(hmin)
        hmax = self.hmax if hmax is None else float(hmax)
        n_jobs = self.nproc if n_jobs is None else n_jobs
        assert target_size > 0.
        assert hmin > 0.
        assert hmax > hmin
        assert n_jobs == -1 or n_jobs in list(range(1, cpu_count()+1))

        # get vertices
        vertices = np.empty((0, 2), float)
        for i, raster in enumerate(self.raster_collection):
            ax = plt.contour(raster.x, raster.y, raster.values, levels=[level])
            plt.close(plt.gcf())
            for path_collection in ax.collections:
                for path in path_collection.get_paths():
                    vertices = np.vstack([vertices, path.vertices])

        # calculate distances between each pixel and nearest contour point
        tree = cKDTree(vertices)
        for i, raster in enumerate(self.raster_collection):
            xt, yt = np.meshgrid(raster.x, raster.y)
            xt = xt.flatten()
            yt = yt.flatten()
            xy_target = np.vstack([xt, yt]).T
            values, _ = tree.query(xy_target, n_jobs=n_jobs)
            values = expansion_rate*target_size*values + target_size
            values = values.reshape(raster.values.shape)
            values[np.where(values < hmin)] = hmin
            values[np.where(values > hmax)] = hmax
            raster.add_band("SIZE_FUNCTION", values)

    def add_subtidal_flow_limiter(self, hmin=None, hmax=None):
        """
        https://wiki.fvcom.pml.ac.uk/doku.php?id=configuration%3Agrid_scale_considerations
        """
        # argument check
        hmin = self.hmin if hmin is None else float(hmin)
        hmax = self.hmax if hmax is None else float(hmax)
        assert hmin > 0.
        assert hmax > hmin
        for raster in self.raster_collection:
            dx = np.abs(raster.src.transform[0])
            dy = np.abs(raster.src.transform[4])
            dx, dy = np.gradient(raster.values, dx, dy)
            dh = np.sqrt(dx**2 + dy**2)
            dh = np.ma.masked_equal(dh, 0.)
            values = np.abs((1./3.)*(raster.values/dh))
            values = values.filled(np.max(values))
            values[np.where(values < hmin)] = hmin
            values[np.where(values > hmax)] = hmax
            raster.add_band("SIZE_FUNCTION", values)

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
    def neighbors(self):
        return self.triangulation.neighbors

    @property
    def triangulation(self):
        return Triangulation(self.x, self.y, self.elements)

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
        return self.elements

    @property
    def scaling(self):
        return self._scaling

    @property
    def size_function_types(self):
        return self._size_function_types

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
    def nproc(self):
        return self._nproc

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
        hfun.ndim = 2
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
    def memmap_points(self):
        return self._memmap_points

    @property
    def memmap_elements(self):
        return self._memmap_elements

    @property
    def _nproc(self):
        return self.__nproc

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
    def _memmap_points(self):
        try:
            return self.__memmap_points
        except AttributeError:
            start_shape = (0, 3)
            for raster in self.raster_collection:
                band = np.full(raster.shape, float("inf"))
                for i in range(1, raster.count + 1):
                    if raster.tags(i)['BAND_TYPE'] == "SIZE_FUNCTION":
                        band = np.minimum(band, raster.read(i))
                raster.add_band("SIZE_FUNCTION_FINALIZED", band)
                raster.mask([self.pslg.polygon], raster.count)
                if raster.dx < self.hmin or raster.dy < self.hmin:
                    x, y, band = raster.get_resampled(
                        raster.count, self.hmin, self.hmin)
                else:
                    x, y = raster.x, raster.y
                band = np.ma.masked_equal(
                    band.astype(raster.dtype(raster.count)),
                    raster.nodataval(raster.count))
                x, y = np.meshgrid(x, y)
                x = x.flatten()
                y = y.flatten()
                band = band.flatten()
                x = x[~band.mask]
                y = y[~band.mask]
                band = band[~band.mask].data
                band[np.where(band < self.hmin)] = self.hmin
                band[np.where(band > self.hmax)] = self.hmax
                new_points = np.vstack([x, y, band]).T
                old_len = start_shape[0]
                new_len = old_len + new_points.shape[0]
                new_shape = (new_len, 3)
                points = np.memmap(
                    self.tmpfile_points.name,
                    dtype=float, mode='r+',
                    shape=new_shape)
                points[old_len:new_len] = new_points
                del new_points
                points.flush()
                start_shape = points.shape
                gc.collect()
            points.flush()
            points = np.memmap(
                self.tmpfile_points.name, dtype=float, mode='r',
                shape=points.shape)
            self.__memmap_points = points
            return self.__memmap_points

    @property
    def _memmap_elements(self):
        try:
            return self.__memmap_elements
        except AttributeError:
            tri = Triangulation(self.x, self.y)
            from shapely.geometry import LineString
            from rtree import index

            idx = index.Index()
            for i, (e0, e1) in enumerate(self.pslg.triangulation.edges):
                edge = LineString(
                    [(self.x[e0], self.y[e1]), (self.x[e0], self.y[e1])])
                idx.insert(i, edge.bounds)

            edges_mask = np.full((tri.edges.shape[0],), True)
            for i, (e0, e1) in enumerate(tri.edges):
                edge = LineString(
                    [(tri.x[e0], tri.y[e1]), (tri.x[e0], tri.y[e1])])
                edges_mask[list(idx.intersection(edge.bounds))] = False

            edges = tri.edges[~edges_mask]
            elements = np.where(np.any(tri.triangles
            plt.plot(tri.x[elements], tri.y[elements])
            plt.show()




            breakme

            tmpfile_elements = tempfile.NamedTemporaryFile()
            shape = elements.shape
            memmap_elements = np.memmap(
                        tmpfile_elements.name,
                        dtype=int, mode='r+', shape=shape)
            memmap_elements[:] = elements
            del elements
            del memmap_elements
            elements = np.memmap(
                        tmpfile_elements.name,
                        dtype=int, mode='r', shape=shape)

            # create mask to be populated
            tmpfile_mask = tempfile.NamedTemporaryFile()
            mask = np.memmap(
                        tmpfile_mask.name,
                        dtype=bool, mode='r+', shape=(shape[0],))
            mask[:] = True
            mask.flush()

            # create centroid list
            tmpfile_centroids = tempfile.NamedTemporaryFile()
            centroids = np.memmap(
                        tmpfile_centroids.name,
                        dtype=float, mode='r+', shape=(shape[0], 2))
            centroids[:] = np.vstack(
                [np.sum(self.x[elements], axis=1) / 3,
                 np.sum(self.y[elements], axis=1) / 3]).T
            del centroids
            centroids = np.memmap(
                        tmpfile_centroids.name,
                        dtype=float, mode='r', shape=(shape[0], 2))
            path = Path(self.pslg.polygon.exterior.coords, closed=True)
            bbox = path.get_extents()
            idxs = np.where(np.logical_and(
                                np.logical_and(
                                    bbox.xmin <= centroids[:, 0],
                                    bbox.xmax >= centroids[:, 0]),
                                np.logical_and(
                                    bbox.ymin <= centroids[:, 1],
                                    bbox.ymax >= centroids[:, 1])))[0]
            # ------ parallel inpoly test on polygon exterior
            if self.nproc > 1:
                pool = Pool(
                    self.nproc,
                    parallel_processing.inpoly_pool_initializer,
                    (path,))
                results = pool.map_async(
                            parallel_processing.inpoly_pool_worker,
                            (centroid for centroid in centroids[idxs]),
                            )
                results = results.get()
                pool.close()
                pool.join()
                results = np.asarray(results).flatten()
                mask[idxs] = np.logical_and(mask[idxs], ~results)
                del results
                gc.collect()
            # ------ serial inpoly test on polygon exterior
            else:
                mask[idxs] = np.logical_and(
                    mask[idxs], ~path.contains_points(centroids[idxs]))
            mask.flush()
            for interior in self.pslg.polygon.interiors:
                path = Path(interior.coords, closed=True)
                bbox = path.get_extents()
                idxs = np.where(np.logical_and(
                                np.logical_and(
                                    bbox.xmin <= centroids[:, 0],
                                    bbox.xmax >= centroids[:, 0]),
                                np.logical_and(
                                    bbox.ymin <= centroids[:, 1],
                                    bbox.ymax >= centroids[:, 1])))[0]
                # ------ parallel inpoly test on polygon interior
                if self.nproc > 1:
                    pool = Pool(
                        self.nproc,
                        parallel_processing.inpoly_pool_initializer,
                        (path,))
                    results = pool.map_async(
                                parallel_processing.inpoly_pool_worker,
                                (centroid for centroid in centroids[idxs]),
                                )
                    results = results.get()
                    pool.close()
                    pool.join()
                    results = np.asarray(results).flatten()
                    mask[idxs] = np.logical_or(mask[idxs], results)
                    del results
                    gc.collect()
                # ------ serial inpoly test on polygon interior
                else:
                    mask[idxs] = np.logical_or(
                        mask[idxs], path.contains_points(centroids[idxs]))
                mask.flush()
            del centroids
            shape = elements[~mask].shape
            memmap_elements = np.memmap(
                        self.tmpfile_elements.name,
                        dtype=int, mode='r+', shape=shape)
            memmap_elements[:] = elements[~mask]
            memmap_elements.flush()
            self.__memmap_elements = np.memmap(
                self.tmpfile_elements.name, dtype=int, mode='r', shape=shape)
            return self.__memmap_elements

    @property
    def _size_function_types(self):
        try:
            return self.__size_function_types
        except AttributeError:
            self.__size_function_types = {
                "contours": [],
                "subtidal_flow_limiter": False
            }
            return self.__size_function_types

    @property
    def _scaling(self):
        try:
            return self.__scaling
        except AttributeError:
            self._scaling = "absolute"
            return self.__scaling

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

    @property
    def _dst_crs(self):
        return self.__dst_crs

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = scaling

    @nproc.setter
    def nproc(self, nproc):
        self._nproc = nproc

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        self._dst_crs = dst_crs

    @_scaling.setter
    def _scaling(self, scaling):
        assert scaling in ["absolute", "relative"]
        self.__scaling = scaling

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        self.pslg.dst_crs = dst_crs
        self.__dst_crs = dst_crs

    @_pslg.setter
    def _pslg(self, pslg):
        assert isinstance(pslg, PlanarStraightLineGraph)
        self.__pslg = pslg

    @_hmin.setter
    def _hmin(self, hmin):
        if hmin is None:
            hmin = np.finfo(float).eps
        self.__hmin = float(hmin)

    @_hmax.setter
    def _hmax(self, hmax):
        if hmax is None:
            hmax = float("inf")
        self.__hmax = float(hmax)

    @_scaling.setter
    def _scaling(self, scaling):
        assert scaling in ["absolute", "relative"]
        self.__scaling = scaling

    @_nproc.setter
    def _nproc(self, nproc):
        if nproc == -1:
            nproc = cpu_count()
        assert isinstance(nproc, int)
        assert nproc > 0
        self.__nproc = nproc

# MASK GENERATION TESTS

# -------- serial mask generation
# path = Path(self.pslg.polygon.exterior.coords, closed=True)
# bbox = path.get_extents()
# idxs = np.where(np.logical_and(
#                     np.logical_and(
#                         bbox.xmin <= centroids[:, 0],
#                         bbox.xmax >= centroids[:, 0]),
#                     np.logical_and(
#                         bbox.ymin <= centroids[:, 1],
#                         bbox.ymax >= centroids[:, 1])))[0]
# mask[idxs] = np.logical_and(
#     mask[idxs], ~path.contains_points(centroids[idxs]))
# for interior in self.pslg.polygon.interiors:
#     path = Path(interior.coords, closed=True)
#     bbox = path.get_extents()
#     idxs = np.where(np.logical_and(
#                     np.logical_and(
#                         bbox.xmin <= centroids[:, 0],
#                         bbox.xmax >= centroids[:, 0]),
#                     np.logical_and(
#                         bbox.ymin <= centroids[:, 1],
#                         bbox.ymax >= centroids[:, 1])))[0]
#     mask[idxs] = np.logical_or(
#         mask[idxs],
#         path.contains_points(centroids[idxs]))
# -------- end serial mask generation


# -------------- parallel exterior/interior version
# global exterior
# global interior
# for polygon in self.pslg.multipolygon:
#     exterior = polygon.exterior
#     bbox = polygon.bounds
#     idxs = np.where(np.logical_and(
#                         np.logical_and(
#                             bbox[0] <= centroids[:, 0],
#                             bbox[2] >= centroids[:, 0]),
#                         np.logical_and(
#                             bbox[1] <= centroids[:, 1],
#                             bbox[3] >= centroids[:, 1])))[0]
#     p = Pool()
#     result = p.map(
#         parallel_exterior_contains,
#         [centroids[idx] for idx in idxs])
#     p.close()
#     p.join()
#     mask[idxs] = np.logical_and(
#         mask[idxs], ~np.asarray(result))
#     for interior in polygon.interiors:
#         bbox = interior.bounds
#         idxs = np.where(np.logical_and(
#                         np.logical_and(
#                             bbox[0] <= centroids[:, 0],
#                             bbox[2] >= centroids[:, 0]),
#                         np.logical_and(
#                             bbox[1] <= centroids[:, 1],
#                             bbox[3] >= centroids[:, 1])))[0]
#         p = Pool()
#         result = p.map(
#             parallel_interior_contains,
#             [centroids[idx] for idx in idxs])
#         p.close()
#         p.join()
#         mask[idxs] = np.logical_or(mask[idxs], np.asarray(result))
# del globals()['exterior']
# del globals()['interior']
# ---------- end parallel exterior/interior version

# -------------- parallel path version
# print('begin parallel path version')
# import time
# start = time.time()
# global path
# for polygon in self.pslg.multipolygon:
#     path = Path(polygon.exterior.coords, closed=True)
#     bbox = path.get_extents()
#     idxs = np.where(np.logical_and(
#                         np.logical_and(
#                             bbox.xmin <= centroids[:, 0],
#                             bbox.xmax >= centroids[:, 0]),
#                         np.logical_and(
#                             bbox.ymin <= centroids[:, 1],
#                             bbox.ymax >= centroids[:, 1])))[0]
#     p = Pool()
#     result = p.map_async(
#         parallel_path_contains_point,
#         [centroids[idx] for idx in idxs])
#     p.close()
#     p.join()
#     mask[idxs] = np.logical_and(
#         mask[idxs], ~np.asarray(result))
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
#         p = Pool()
#         result = p.map_async(
#             parallel_path_contains_point,
#             [centroids[idx] for idx in idxs])
#         p.close()
#         p.join()
#         mask[idxs] = np.logical_or(mask[idxs], np.asarray(result))
# print(f'parallel path version took {start-time.time()}')
# del globals()['path']
# ---------- end parallel path version

