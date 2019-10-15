import gc
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
from scipy.spatial import cKDTree
from jigsawpy import jigsaw_msh_t
from geomesh.pslg import PlanarStraightLineGraph

#  ----------- multiprocessing imports and functions
from multiprocessing import Pool, cpu_count


def parallel_raster_sample(coord):
    global raster
    try:
        sample = list(raster.sample(coord, raster.count))
        if sample[0][0] == 0.:
            return True
        else:
            return False
    except (TypeError, IndexError):
        return False


class SizeFunction:

    def __init__(
        self,
        pslg,
        hmin=None,
        hmax=None,
        dst_crs="EPSG:3395",
        nproc=1
    ):
        self._pslg = pslg
        self._hmin = hmin
        self._hmax = hmax
        self._dst_crs = dst_crs
        self._nproc = nproc

    def tricontourf(self, show=False, **kwargs):
        plt.tricontourf(self.triangulation, self.values, **kwargs)
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
        nproc=None
    ):
        if nproc is None:
            nproc = self.nproc
        if target_size is None:
            target_size = self.hmin
        level = float(level)
        target_size = float(target_size)
        expansion_rate = float(expansion_rate)
        vertices = np.empty((0, 2), float)
        for i, raster in enumerate(self.raster_collection):
            ax = plt.contour(raster.x, raster.y, raster.values, levels=[level])
            plt.close(plt.gcf())
            for path_collection in ax.collections:
                for path in path_collection.get_paths():
                    vertices = np.vstack([vertices, path.vertices])
        tree = cKDTree(vertices)
        for i, raster in enumerate(self.raster_collection):
            xt, yt = np.meshgrid(raster.x, raster.y)
            xt = xt.flatten()
            yt = yt.flatten()
            xy_target = np.vstack([xt, yt]).T
            values, _ = tree.query(xy_target, n_jobs=nproc)
            values = expansion_rate*target_size*values + target_size
            values = values.reshape(raster.values.shape)
            if hmin is not None:
                values[np.where(values < hmin)] = hmin
            if hmax is not None:
                values[np.where(values > hmax)] = hmax
            raster.add_band("SIZE_FUNCTION", values)

    def add_subtidal_flow_limiter(self, hmin=None, hmax=None):
        """
        https://wiki.fvcom.pml.ac.uk/doku.php?id=configuration%3Agrid_scale_considerations
        """
        for i, raster in enumerate(self.raster_collection):
            dx = np.abs(raster.src.transform[0])
            dy = np.abs(raster.src.transform[4])
            dx, dy = np.gradient(raster.values, dx, dy)
            dh = np.sqrt(dx**2 + dy**2)
            dh = np.ma.masked_equal(dh, 0.)
            values = np.abs((1./3.)*(raster.values/dh))
            values = values.filled(np.max(values))
            if hmin is not None:
                values[np.where(values < hmin)] = hmin
            if hmax is not None:
                values[np.where(values > hmax)] = hmax
            raster.add_band("SIZE_FUNCTION", values)

    def _set_triangulation(self):
        global raster
        points = np.empty((0, 3), float)
        for raster in self.raster_collection:
            band = np.full(raster.shape, float("inf"))
            for i in range(1, raster.count + 1):
                if raster.tags(i)['BAND_TYPE'] == "SIZE_FUNCTION":
                    band = np.minimum(band, raster.read(i))
            raster.add_band("SIZE_FUNCTION_FINALIZED", band)
            raster.mask([self.pslg.polygon], raster.count)
            if raster.dx < self.hmin or raster.dy < self.hmin:
                x, y, band = raster.resampled(
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
            points = np.vstack([points, np.vstack([x, y, band]).T])
        self._values = points[:, 2]
        points = points[:, :2]
        del(x)
        del(y)
        del(band)
        gc.collect()
        elements = Triangulation(points[:, 0], points[:, 1]).triangles.copy()
        # generate mask, this is the main bottleneck.
        centroids = np.vstack(
            [np.sum(points[:, 0][elements], axis=1) / 3,
             np.sum(points[:, 1][elements], axis=1) / 3]).T
        mask = np.full((1, elements.shape[0]), True).flatten()
        for raster in self.raster_collection:
            bbox = raster.bbox
            idxs = np.where(np.logical_and(
                            np.logical_and(
                                bbox.xmin <= centroids[:, 0],
                                bbox.xmax >= centroids[:, 0]),
                            np.logical_and(
                                bbox.ymin <= centroids[:, 1],
                                bbox.ymax >= centroids[:, 1])))[0]
            # ----serial version
            if self.nproc == 1:
                results = raster.sample(centroids[idxs], raster.count)
                mask[idxs] = np.ma.masked_equal(
                    np.asarray([value for value in results]),
                    raster.nodataval(raster.count)).mask.flatten()
            # ----parallel version
            else:
                pool = Pool(processes=self.nproc)
                results = pool.map_async(
                        parallel_raster_sample,
                        ([centroid] for centroid in centroids[idxs]))
                results = results.get()
                pool.close()
                pool.join()
                mask[idxs] = results
        self._triangulation = Triangulation(
            points[:, 0], points[:, 1], triangles=elements[~mask])

    @property
    def pslg(self):
        return self._pslg

    @property
    def raster_collection(self):
        return self.pslg._raster_collection

    @property
    def triangulation(self):
        return self._triangulation

    @property
    def coords(self):
        return np.vstack([self.triangulation.x, self.triangulation.y]).T

    @property
    def values(self):
        return self._values

    @property
    def scaling(self):
        return self._scaling

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
            [([x, y], 0) for x, y in self.coords[:, :2]],
            dtype=jigsaw_msh_t.VERT2_t)

    @property
    def tria3(self):
        return np.asarray(
            [(tuple(indices), 0) for indices in self.triangulation.triangles],
            dtype=jigsaw_msh_t.TRIA3_t)

    @property
    def hfun_value(self):
        return np.asarray(self.values, dtype=jigsaw_msh_t.REALS_t)

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
    def _scaling(self):
        try:
            return self.__scaling
        except AttributeError:
            self._scaling = "absolute"
            return self.__scaling

    @property
    def _triangulation(self):
        try:
            return self.__triangulation
        except AttributeError:
            self._set_triangulation()
            return self.__triangulation

    @property
    def _values(self):
        try:
            return self.__values
        except AttributeError:
            self._set_triangulation()
            return self.__values

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

    @_triangulation.setter
    def _triangulation(self, triangulation):
        self.__triangulation = triangulation

    @_values.setter
    def _values(self, values):
        values[np.where(values < self.hmin)] = self.hmin
        values[np.where(values > self.hmax)] = self.hmax
        self.__values = values

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