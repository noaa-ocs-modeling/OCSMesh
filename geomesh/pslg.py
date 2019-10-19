import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.tri import Triangulation
import tempfile
import fiona
from shapely.geometry import shape, mapping, MultiPolygon
from jigsawpy import jigsaw_msh_t
from geomesh.raster import Raster
from geomesh.raster_collection import RasterCollection


class PlanarStraightLineGraph:

    def __init__(
        self,
        raster_collection,
        zmin,
        zmax,
        dst_crs="EPSG:3395",
    ):
        self._raster_collection = raster_collection
        self._zmin = zmin
        self._zmax = zmax
        self._dst_crs = dst_crs

    def __iter__(self):
        for raster in self.raster_collection:
            yield raster

    def plot(self, show=False):
        for polygon in self.multipolygon:
            plt.plot(*polygon.exterior.xy, color='k')
            for interior in polygon.interiors:
                plt.plot(*interior.xy, color='r')
        if show:
            plt.gca().axis('scaled')
            plt.show()
        return plt.gca()

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
            plt.axis('scaled')
            plt.show()

    @property
    def raster_collection(self):
        return self._raster_collection

    @property
    def crs(self):
        return self.raster_collection.dst_crs

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    @property
    def multipolygon(self):
        multipolygon = shape(self.collection[0]["geometry"])
        if self.simplify_multipolygon:
            areas = [polygon.area for polygon in multipolygon]
            idx = np.where(areas == np.max(areas))[0][0]
            return MultiPolygon([multipolygon[idx]])
        return multipolygon

    @property
    def simplify_multipolygon(self):
        return self._simplify_multipolygon

    @property
    def collection(self):
        return self._collection

    @property
    def points(self):
        return self.memmap_points

    @property
    def elements(self):
        return self.memmap_elements

    @property
    def coords(self):
        return self.memmap_points

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 0]

    @property
    def triangles(self):
        return self.elements

    @property
    def triangulation(self):
        return Triangulation(
                self.memmap_points[:, 0],
                self.memmap_points[:, 1],
                self.memmap_elements)

    @property
    def dst_crs(self):
        return self._dst_crs

    @property
    def memmap_points(self):
        return self._memmap_points

    @property
    def memmap_elements(self):
        return self._memmap_elements

    @property
    def tmpfile_points(self):
        return self._tmpfile_points

    @property
    def tmpfile_elements(self):
        return self._tmpfile_elements

    @property
    def tmpfile_shp(self):
        return self._tmpfile_shp

    @property
    def ndim(self):
        return 2

    @property
    def mshID(self):
        return 'euclidean-mesh'

    @property
    def geom(self):
        geom = jigsaw_msh_t()
        geom.vert2 = self.vert2
        geom.edge2 = self.edge2
        geom.ndim = self.ndim
        geom.mshID = self.mshID
        return geom

    @property
    def vert2(self):
        coords = [([x, y], 0) for x, y in self.coords]
        return np.asarray(coords, dtype=jigsaw_msh_t.VERT2_t)

    @property
    def edge2(self):
        idxs = np.vstack(list(np.where(self.triangulation.neighbors == -1))).T
        edge2 = [([self.elements[i, j], self.elements[i, (j+1) % 3]], 0)
                 for i, j in idxs]
        return np.asarray(edge2, dtype=jigsaw_msh_t.EDGE2_t)

    @property
    def _collection(self):
        try:
            return self.__collection
        except AttributeError:
            polygon_collection = list()
            for raster in self.raster_collection:
                multipolygon = raster(self.zmin, self.zmax)
                for polygon in multipolygon:
                    polygon_collection.append(polygon)
            multipolygon = MultiPolygon(polygon_collection).buffer(0)
            with fiona.open(
                    self.tmpfile_shp.name,
                    'w',
                    driver='ESRI Shapefile',
                    crs=self.dst_crs,
                    schema={
                        'geometry': 'MultiPolygon',
                        'properties': {
                            'zmin': 'float',
                            'zmax': 'float'}}) as dst:
                dst.write({
                    "geometry": mapping(multipolygon),
                    "properties": {
                        "zmin": self.zmin,
                        "zmax": self.zmax}})
            self.__collection = fiona.open(self.tmpfile_shp.name)
            return self.__collection

    @property
    def _memmap_points(self):
        try:
            return self.__memmap_points
        except AttributeError:
            points = np.empty((0, 2))
            for polygon in self.multipolygon:
                points = np.vstack([points, polygon.exterior.coords])
                for interior in polygon.interiors:
                    points = np.vstack([points, interior.coords])
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
            for polygon in self.multipolygon:
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
            for polygon in self.multipolygon:
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
    def _tmpfile_shp(self):
        try:
            return self.__tmpfile_shp
        except AttributeError:
            self.__tmpfile_shp = tempfile.TemporaryDirectory()
            return self.__tmpfile_shp

    @property
    def _raster_collection(self):
        return self.__raster_collection

    @property
    def _simplify_multipolygon(self):
        try:
            return self.__simplify_multipolygon
        except AttributeError:
            return False

    @property
    def _dst_crs(self):
        return self.__dst_crs

    @property
    def _zmin(self):
        return self.__zmin

    @property
    def _zmax(self):
        return self.__zmax

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        self._dst_crs = dst_crs

    @simplify_multipolygon.setter
    def simplify_multipolygon(self, simplify_multipolygon):
        self._simplify_multipolygon = simplify_multipolygon

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        del(self._collection)
        self.__dst_crs = dst_crs

    @_simplify_multipolygon.setter
    def _simplify_multipolygon(self, simplify_multipolygon):
        assert isinstance(simplify_multipolygon, bool)
        self.__simplify_multipolygon = simplify_multipolygon

    @_raster_collection.setter
    def _raster_collection(self, raster_collection):
        # accept additional data types as input
        if not isinstance(raster_collection, RasterCollection):
            raster = raster_collection
            raster_collection = RasterCollection()
            # accepts geomesh.Raster or str object
            if isinstance(raster, (Raster, str)):
                raster_collection.append(raster)
        self.__raster_collection = raster_collection

    @_zmin.setter
    def _zmin(self, zmin):
        if zmin is None:
            del(self._zmin)
        else:
            zmin = float(zmin)
            try:
                if zmin != self.__zmin:
                    del(self._collection)
            except AttributeError:
                pass
            self.__zmin = float(zmin)

    @_zmax.setter
    def _zmax(self, zmax):
        if zmax is None:
            del(self._zmax)
        else:
            zmax = float(zmax)
            try:
                if zmax != self.__zmax:
                    del(self._collection)
            except AttributeError:
                pass
            self.__zmax = float(zmax)

    @_collection.deleter
    def _collection(self):
        try:
            del(self.__collection)
            del(self._triangulation)
        except AttributeError:
            pass

    @_zmin.deleter
    def _zmin(self):
        try:
            del(self.__zmin)
            del(self._collection)
        except AttributeError:
            pass

    @_zmax.deleter
    def _zmax(self):
        try:
            del(self.__zmax)
            del(self._collection)
        except AttributeError:
            pass
