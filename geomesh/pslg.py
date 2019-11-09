import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.tri import Triangulation
import tempfile
import fiona
from shapely.geometry import shape, mapping, MultiPolygon, Polygon, LinearRing
from jigsawpy import jigsaw_msh_t, savemsh, loadmsh
import geomesh
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

    def _get_geom(self, idx):
        geom = self.geom_collection[idx]
        if geom is None:
            geom = self._generate_geom(self._get_multipolygon(idx))
            self._save_geom(idx, geom)
        else:
            geom = self._load_geom(idx)
        return geom

    @staticmethod
    def _generate_geom(multipolygon):
        # vert2
        vert2 = list()
        for polygon in multipolygon:
            for x, y in polygon.exterior.coords[:-1]:
                vert2.append(((x, y), 0))
            for interior in polygon.interiors:
                for x, y in interior.coords[:-1]:
                    vert2.append(((x, y), 0))
        vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)
        # edge2
        edge2 = list()
        for polygon in multipolygon:
            polygon = [polygon.exterior, *polygon.interiors]
            for linear_ring in polygon:
                _edge2 = list()
                for i in range(len(linear_ring.coords)-2):
                    _edge2.append((i, i+1))
                _edge2.append((_edge2[-1][1], _edge2[0][0]))
                edge2.extend(
                    [(e0+len(edge2), e1+len(edge2)) for e0, e1 in _edge2])
        edge2 = np.asarray(
            [((e0, e1), 0) for e0, e1 in edge2], dtype=jigsaw_msh_t.EDGE2_t)
        # geom
        geom = jigsaw_msh_t()
        geom.ndim = 2
        geom.mshID = 'euclidean-mesh'
        geom.vert2 = vert2
        geom.edge2 = edge2
        return geom

    def _save_geom(self, idx, geom):
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=geomesh.tmpdir, suffix='.msh')
        savemsh(tmpfile.name, geom)
        self._geom_collection[idx] = tmpfile

    def _load_geom(self, idx):
        geom = jigsaw_msh_t()
        loadmsh(self.geom_collection[idx].name, geom)
        return geom

    def _get_multipolygon(self, idx):
        multipolygon = self.multipolygon_collection[idx]
        if multipolygon is None:
            multipolygon = self._generate_multipolygon(idx)
            self._save_multipolygon(idx, multipolygon)
        else:
            with fiona.open(multipolygon.name) as shp:
                multipolygon = shape(shp[0]['geometry'])
                if isinstance(multipolygon, Polygon):
                    multipolygon = MultiPolygon([multipolygon])
        return multipolygon

    def _generate_multipolygon(self, idx):
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

    def _save_multipolygon(self, idx, multipolygon):
        tmpdir = tempfile.TemporaryDirectory(
            prefix=geomesh.tmpdir, suffix='_multipolygon')
        with fiona.open(
                    tmpdir.name,
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
        self._multipolygon_collection[idx] = tmpdir

    @property
    def geom(self):
        try:
            return self.__geom
        except AttributeError:
            self.__geom = self._generate_geom(self.multipolygon)
            return self.__geom

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
    def simplify(self):
        return self._simplify

    @property
    def dst_crs(self):
        return self._dst_crs

    @property
    def ndim(self):
        return 2

    @property
    def multipolygon(self):
        polygon_collection = list()
        for i in range(len(self.raster_collection)):
            multipolygon = self._get_multipolygon(i)
            for polygon in multipolygon:
                polygon_collection.append(polygon)
        return MultiPolygon(polygon_collection).buffer(0)

    @property
    def raster_collection(self):
        return self._raster_collection

    @property
    def multipolygon_collection(self):
        return tuple(self._multipolygon_collection)

    @property
    def geom_collection(self):
        return tuple(self._geom_collection)

    @property
    def _raster_collection(self):
        return self.__raster_collection

    @property
    def _multipolygon_collection(self):
        try:
            return self.__multipolygon_collection
        except AttributeError:
            self.__multipolygon_collection = len(self.raster_collection)*[None]
            return self.__multipolygon_collection

    @property
    def _geom_collection(self):
        try:
            return self.__geom_collection
        except AttributeError:
            self.__geom_collection = len(self.raster_collection)*[None]
            return self.__geom_collection

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
        if zmin is not None:
            assert isinstance(zmin, float)
        self.__zmin = zmin

    @_zmax.setter
    def _zmax(self, zmax):
        if zmax is not None:
            assert isinstance(zmax, float)
        self.__zmax = zmax


