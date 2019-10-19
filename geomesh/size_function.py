import tempfile
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.tri import Triangulation
import numpy as np
from scipy.spatial import cKDTree
from jigsawpy import jigsaw_msh_t
from geomesh.pslg import PlanarStraightLineGraph


class SizeFunction:

    def __init__(
        self,
        pslg,
        hmin=None,
        hmax=None,
        dst_crs="EPSG:3395",
    ):
        self._pslg = pslg
        self._hmin = hmin
        self._hmax = hmax
        self._dst_crs = dst_crs

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
        n_jobs=1
    ):

        # argument checks
        level = float(level)
        expansion_rate = float(expansion_rate)
        target_size = self.hmin if target_size is None else float(target_size)
        hmin = self.hmin if hmin is None else float(hmin)
        hmax = self.hmax if hmax is None else float(hmax)
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
    def hmin_is_absolute_limit(self):
        return self._hmin_is_absolute_limit

    @property
    def hmax_is_absolute_limit(self):
        return self._hmax_is_absolute_limit

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
            points = np.empty((0, 3))
            for raster in self.raster_collection:
                band = np.full(raster.shape, float("inf"))
                for i in range(1, raster.count + 1):
                    if raster.tags(i)['BAND_TYPE'] == "SIZE_FUNCTION":
                        band = np.minimum(band, raster.read(i))
                # maybe apply filter to band here
                raster.add_band("SIZE_FUNCTION_FINALIZED", band)
                raster.mask(self.pslg.multipolygon, raster.count)
                if np.min(band) <= self.hmin:
                    x, y, band = raster.get_resampled(
                        raster.count, np.min(band), np.min(band))
                else:
                    x, y = raster.x, raster.y
                    band = raster.band(raster.count)
                if raster.nodataval(raster.count) is not None:
                    band = np.ma.masked_equal(
                        band.astype(raster.dtype(raster.count)),
                        raster.nodataval(raster.count))
                else:
                    # for rasters with no masked value defined 0 is returned
                    band = np.ma.masked_equal(
                        band.astype(raster.dtype(raster.count)), 0)
                x, y = np.meshgrid(x, y)
                x = x.flatten()
                y = y.flatten()
                band = band.flatten()
                x = x[~band.mask]
                y = y[~band.mask]
                band = band[~band.mask].data
                band[np.where(band < self.hmin)] = self.hmin
                band[np.where(band > self.hmax)] = self.hmax
                points = np.vstack([points, np.vstack([x, y, band]).T])
            memmap_points = np.memmap(
                self.tmpfile_points.name, dtype=float, mode='w+',
                shape=points.shape)
            memmap_points[:] = points
            del memmap_points
            self.__memmap_points = np.memmap(
                self.tmpfile_points.name, dtype=float, mode='r',
                shape=points.shape)
            return self.__memmap_points

    @property
    def _memmap_elements(self):
        try:
            return self.__memmap_elements
        except AttributeError:
            tri = Triangulation(
                self.memmap_points[:, 0],
                self.memmap_points[:, 1])
            mask = np.full((tri.triangles.shape[0],), True)
            centroids = np.vstack(
                [np.sum(tri.x[tri.triangles], axis=1) / 3,
                 np.sum(tri.y[tri.triangles], axis=1) / 3]).T
            for polygon in self.pslg.multipolygon:
                path = Path(polygon.exterior.coords, closed=True)
                bbox = path.get_extents()
                idxs = np.where(np.logical_and(
                                    np.logical_and(
                                        bbox.xmin <= centroids[:, 0],
                                        bbox.xmax >= centroids[:, 0]),
                                    np.logical_and(
                                        bbox.ymin <= centroids[:, 1],
                                        bbox.ymax >= centroids[:, 1])))[0]
                mask[idxs] = np.logical_and(
                    mask[idxs], ~path.contains_points(centroids[idxs]))
            for polygon in self.pslg.multipolygon:
                for interior in polygon.interiors:
                    path = Path(interior.coords, closed=True)
                    bbox = path.get_extents()
                    idxs = np.where(np.logical_and(
                                    np.logical_and(
                                        bbox.xmin <= centroids[:, 0],
                                        bbox.xmax >= centroids[:, 0]),
                                    np.logical_and(
                                        bbox.ymin <= centroids[:, 1],
                                        bbox.ymax >= centroids[:, 1])))[0]
                    mask[idxs] = np.logical_or(
                        mask[idxs], path.contains_points(centroids[idxs]))
                shape = tri.triangles[~mask].shape
                memmap_elements = np.memmap(
                            self.tmpfile_elements.name,
                            dtype=int, mode='r+', shape=shape)
                memmap_elements[:] = tri.triangles[~mask]
                del memmap_elements
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

    @property
    def _hmin_is_absolute_limit(self):
        try:
            return self.__hmin_is_absolute_limit
        except AttributeError:
            # Uses the data's limit by default to favor jigsaw stability
            return False

    @property
    def _hmax_is_absolute_limit(self):
        try:
            return self.__hmax_is_absolute_limit
        except AttributeError:
            # Uses the data's limit by default to favor jigsaw stability
            return False

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = scaling

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        self._dst_crs = dst_crs

    @hmin_is_absolute_limit.setter
    def hmin_is_absolute_limit(self, hmin_is_absolute_limit):
        self._hmin_is_absolute_limit = hmin_is_absolute_limit

    @hmax_is_absolute_limit.setter
    def hmax_is_absolute_limit(self, hmax_is_absolute_limit):
        self._hmax_is_absolute_limit = hmax_is_absolute_limit

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

    @_scaling.setter
    def _scaling(self, scaling):
        assert scaling in ["absolute", "relative"]
        self.__scaling = scaling

    @_hmin_is_absolute_limit.setter
    def _hmin_is_absolute_limit(self, hmin_is_absolute_limit):
        assert isinstance(hmin_is_absolute_limit, bool)
        self.__hmin_is_absolute_limit = hmin_is_absolute_limit

    @_hmax_is_absolute_limit.setter
    def _hmax_is_absolute_limit(self, hmax_is_absolute_limit):
        assert isinstance(hmax_is_absolute_limit, bool)
        self.__hmax_is_absolute_limit = hmax_is_absolute_limit
