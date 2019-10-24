import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.tri import Triangulation
import tempfile
import fiona
from shapely.geometry import shape, mapping, MultiPolygon, Polygon, LinearRing
from jigsawpy import jigsaw_msh_t
from geomesh.raster import Raster
from geomesh.raster_collection import RasterCollection


class PlanarStraightLineGraph:

    def __init__(
        self,
        raster_collection,
        zmin=None,
        zmax=None,
        dst_crs="EPSG:3395",
        simplify=False
    ):
        self._raster_collection = raster_collection
        self._zmin = zmin
        self._zmax = zmax
        self._dst_crs = dst_crs
        self._simplify = simplify

    # def __call__(self, i=None, **kwargs):
    #     if isinstance(i, int):
    #         return self.raster_collection[i](**kwargs)
    #     else:
    #         raise NotImplementedError

    # def __iter__(self):
    #     for raster in self.raster_collection:
    #         yield raster

    def plot(self, idx, show=False):
        for polygon in self.multipolygon(idx):
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

    def multipolygon(self, idx):
        # if type(idx) is list:
        #     idx = set(idx)
        # elif type(idx) is not set:
        #     idx = set([idx])
        return self._get_multipolygon(idx)

    def geom(self, idx):

        # get multipolygon
        multipolygon = self.multipolygon(idx)

        # extract vertices from multipolygon
        vertices = list()
        for polygon in multipolygon:
            for x, y in polygon.exterior.coords:
                vertices.append((x, y))
            for interior in polygon.interiors:
                for x, y in interior.coords:
                    vertices.append((x, y))

        # generate concave hull
        vertices = np.unique(np.asarray(vertices), axis=0)
        tri = Triangulation(vertices[:, 0], vertices[:, 1])
        mask = np.full((tri.triangles.shape[0],), True)
        centroids = np.vstack(
            [np.sum(tri.x[tri.triangles], axis=1) / 3,
             np.sum(tri.y[tri.triangles], axis=1) / 3]).T
        for polygon in multipolygon:
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
        for polygon in multipolygon:
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

        # refresh element indexing
        tri = Triangulation(tri.x, tri.y, tri.triangles[~mask])

        # create vert2 object
        vert2 = [([x, y], 0) for x, y in np.vstack([tri.x, tri.y]).T]
        vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)

        # create edge2 object
        idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
        edge2 = [([tri.triangles[i, j], tri.triangles[i, (j+1) % 3]], 0)
                 for i, j in idxs]
        edge2 = np.asarray(edge2, dtype=jigsaw_msh_t.EDGE2_t)

        # create geom object
        geom = jigsaw_msh_t()
        geom.ndim = 2
        geom.mshID = 'euclidean-mesh'
        geom.vert2 = vert2
        geom.edge2 = edge2

        return geom

    def vert2(self, i=None):
        pass

    def edge2(self, i=None):
        pass

    def _get_multipolygon(self, idx):
        if idx not in list(self.multipolygon_collection.keys()):
            self._add_multipolygon(idx)
        geom = shape(self.multipolygon_collection.get(idx)["geometry"])
        if isinstance(geom, Polygon):
            return MultiPolygon([geom])
        return MultiPolygon(geom)

    def _add_multipolygon(self, idx):
        multipolygon = self._get_pslg(idx)
        self.multipolygon_collection.close()
        with fiona.open(self.multipolygon_tmpdir.name, 'a') as dst:
            dst.write({
                "id": idx,
                "geometry": mapping(multipolygon),
                "properties": {
                    "zmin": self.zmin,
                    "zmax": self.zmax}})
        self.__multipolygon_collection = fiona.open(
            self.multipolygon_tmpdir.name)

    def _get_pslg(self, idx):
        multipolygon = self._get_zmin_zmax_multipolygon(idx)
        # multipolygon = self._get_clipped_multipolygon(multipolygon)
        return multipolygon

    def _get_zmin_zmax_multipolygon(self, idx):
        raster = self.raster_collection[idx]
        zmin = np.min(raster.values) if self.zmin is None else self.zmin
        zmax = np.max(raster.values) if self.zmax is None else self.zmax
        ax = plt.contourf(
            raster.x, raster.y, raster.values, levels=[zmin, zmax])
        plt.close(plt.gcf())
        # extract linear_rings from plot
        linear_ring_collection = list()
        for path_collection in ax.collections:
            for path in path_collection.get_paths():
                polygons = path.to_polygons(closed_only=True)
                for linear_ring in polygons:
                    linear_ring_collection.append(LinearRing(linear_ring))
        if len(linear_ring_collection) > 1:
            # reorder linear rings from above
            areas = [Polygon(linear_ring).area
                     for linear_ring in linear_ring_collection]
            idx = np.where(areas == np.max(areas))[0][0]
            polygon_collection = list()
            outer_ring = linear_ring_collection.pop(idx)
            path = Path(np.asarray(outer_ring.coords), closed=True)
            while len(linear_ring_collection) > 0:
                inner_rings = list()
                for i, linear_ring in reversed(
                        list(enumerate(linear_ring_collection))):
                    xy = np.asarray(linear_ring.coords)[0, :]
                    if path.contains_point(xy):
                        inner_rings.append(linear_ring_collection.pop(i))
                polygon_collection.append(Polygon(outer_ring, inner_rings))
                if len(linear_ring_collection) > 0:
                    areas = [Polygon(linear_ring).area
                             for linear_ring in linear_ring_collection]
                    idx = np.where(areas == np.max(areas))[0][0]
                    outer_ring = linear_ring_collection.pop(idx)
                    path = Path(np.asarray(outer_ring.coords), closed=True)
            multipolygon = MultiPolygon(polygon_collection)
        else:
            multipolygon = MultiPolygon(
                [Polygon(linear_ring_collection.pop())])
        return multipolygon

    @property
    def crs(self):
        return self.raster_collection.dst_crs

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    # @property
    # def multipolygon(self):
    #     multipolygon = shape(self.collection[0]["geometry"])
    #     if self.simplify:
    #         areas = [polygon.area for polygon in multipolygon]
    #         idx = np.where(areas == np.max(areas))[0][0]
    #         return MultiPolygon([multipolygon[idx]])
    #     return multipolygon

    @property
    def simplify(self):
        return self._simplify

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
        return self.points[:, 1]

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
        try:
            return self.__tmpfile_points
        except AttributeError:
            self.__tmpfile_points = tempfile.NamedTemporaryFile()
            return self.__tmpfile_points

    @property
    def tmpfile_elements(self):
        try:
            return self.__tmpfile_elements
        except AttributeError:
            self.__tmpfile_elements = tempfile.NamedTemporaryFile()
            return self.__tmpfile_elements

    @property
    def collection_tmpdir(self):
        try:
            return self.__collection_tmpdir
        except AttributeError:
            self.__collection_tmpdir = tempfile.TemporaryDirectory()
            return self.__collection_tmpdir

    @property
    def multipolygon_tmpdir(self):
        try:
            return self.__multipolygon_tmpdir
        except AttributeError:
            tmpdir = tempfile.TemporaryDirectory()
            collection = fiona.open(
                tmpdir.name,
                'w',
                driver='ESRI Shapefile',
                crs=self.dst_crs,
                schema={'geometry': 'MultiPolygon',
                        'properties': {'zmin': 'float',
                                       'zmax': 'float'}})
            collection.close()
            self.__multipolygon_tmpdir = tmpdir
            return self.__multipolygon_tmpdir

    @property
    def multipolygon_collection(self):
        return fiona.open(self.multipolygon_tmpdir.name)

    @property
    def raster_collection(self):
        return self._raster_collection

    @property
    def ndim(self):
        return 2

    @property
    def mshID(self):
        return 'euclidean-mesh'

    # @property
    # def vert2(self):
    #     coords = [([x, y], 0) for x, y in self.coords]
    #     return np.asarray(coords, dtype=jigsaw_msh_t.VERT2_t)

    # @property
    # def edge2(self):
    #     idxs = np.vstack(list(np.where(self.triangulation.neighbors == -1))).T
    #     edge2 = [([self.elements[i, j], self.elements[i, (j+1) % 3]], 0)
    #              for i, j in idxs]
    #     return np.asarray(edge2, dtype=jigsaw_msh_t.EDGE2_t)

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
                # raster.close()
            multipolygon = MultiPolygon(polygon_collection).buffer(0)
            with fiona.open(
                    self.collection_tmpdir.name,
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
            self.__collection = fiona.open(self.collection_tmpdir.name)
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
            
                memmap_elements = np.memmap(
                            self.tmpfile_elements.name,
                            dtype=int, mode='r+', shape=shape)
                memmap_elements[:] = tri.triangles[~mask]
                del memmap_elements
                self.__memmap_elements = np.memmap(
                    self.tmpfile_elements.name, dtype=int, mode='r', shape=shape)
                return self.__memmap_elements

    @property
    def _raster_collection(self):
        return self.__raster_collection

    @property
    def _simplify(self):
        try:
            return self.__simplify
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

    @simplify.setter
    def simplify(self, simplify):
        self._simplify = simplify

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        del(self._collection)
        self.__dst_crs = dst_crs

    @_simplify.setter
    def _simplify(self, simplify):
        assert isinstance(simplify, bool)
        self.__simplify = simplify

    @_raster_collection.setter
    def _raster_collection(self, raster_collection):
        # accept additional data types as input
        if not isinstance(raster_collection, RasterCollection):
            raster = raster_collection
            raster_collection = RasterCollection()
            # accepts geomesh.Raster or str object
            if isinstance(raster, (Raster, str)):
                raster_collection.append(raster)
            elif isinstance(raster, list):
                for _raster in raster:
                    raster_collection.append(_raster)
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
        self.__zmin = zmin

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
        self.__zmax = zmax

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
