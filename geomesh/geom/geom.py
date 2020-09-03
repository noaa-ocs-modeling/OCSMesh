import logging
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
from functools import lru_cache
from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import MultiPolygon, Polygon

from jigsawpy import jigsaw_msh_t

from ..raster import Raster
from . import types


class Geom:

    __slots__ = ["__geom"]

    __types__ = {
        Raster: types._RasterGeomType,
    }

    def __init__(self, geom, crs=None, ellipsoid=None):
        """
        Input parameters
        ----------------
        geom:
        crs:
            Assigns CRS to geom, required for shapely object.
            Overrides the input geom crs.
        ellipsoid:
            None, False, True, 'WGS84' or '??'
        """
        self._crs = crs
        self._ellipsoid = ellipsoid
        self._geom = geom

    def get_jigsaw_msh_t(self, **kwargs):
        multipolygon = self.get_multipolygon(**kwargs)
        vert2 = list()
        for polygon in multipolygon:
            if np.all(
                    np.asarray(
                        polygon.exterior.coords).flatten() == float('inf')):
                raise NotImplementedError("ellispoidal-mesh")
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
        geom.ndims = +2
        geom.mshID = 'euclidean-mesh' if self._ellipsoid is None \
            else 'ellipsoidal-mesh'
        geom.vert2 = vert2
        geom.edge2 = edge2
        return geom

    def make_plot(
        self,
        ax=None,
        show=False,
    ):

        # spherical plot
        if self._ellipsoid is not None:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for polygon in self.multipolygon:
                coords = np.asarray(polygon.exterior.coords)
                x, y, z = self._geodetic_to_geocentric(
                    self._ellipsoids[self._ellipsoid.upper()],
                    coords[:, 1],
                    coords[:, 0],
                    0.
                    )
                ax.add_collection3d(
                    m3d.art3d.Line3DCollection([np.vstack([x, y, z]).T]),
                    )
        # planar plot
        else:
            for polygon in self.multipolygon:
                plt.plot(*polygon.exterior.xy, color='k')
                for interior in polygon.interiors:
                    plt.plot(*interior.xy, color='r')
        if show:
            if self._ellipsoid is None:
                plt.gca().axis('scaled')
            else:
                radius = self._ellipsoids[self._ellipsoid.upper()][0]
                # ax.set_aspect('equal')
                ax.set_xlim3d([-radius, radius])
                ax.set_xlabel("X")
                ax.set_ylim3d([-radius, radius])
                ax.set_ylabel("Y")
                ax.set_zlim3d([-radius, radius])
                ax.set_zlabel("Z")

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
    def srid(self):
        return self.crs.to_epsg()

    @property
    def crs(self):
        return self._crs

    @property
    def ndims(self):
        return self.geom.ndims

    @property
    def _geom(self):

        return self.__geom

    @_geom.setter
    def _geom(self, geom):
        if type(geom) in self.__types__:
            raise TypeError(f'geom must be one of {self.__types__}')
        geom_t = self.__type___[type(geom)]

        if isinstance(geom_t, Raster):
            geom = geom_t(geom)

        if isinstance(geom_t, (Polygon, MultiPolygon)):
            geom = geom_t(geom, crs=self._crs)

        self.get_multipolygon = geom.get_multipolygon

        self.__geom = geom

    @property
    def _crs(self):
        return self._geom.crs

    @_crs.setter
    def _crs(self, crs):
        self._geom._crs = CRS.from_user_input(crs)

    @property
    def _ellipsoid(self):
        return self._geom.ellipsoid

    @_ellipsoid.setter
    def _ellipsoid(self, ellipsoid):
        self._geom._ellipsoid = ellipsoid

    @property
    def _ellipsoids(self):
        return {
            "WGS84": (6378137, 298.257223563),
            "GRS80": (6378137, 298.257222100882711)
            }

    @property
    @lru_cache(maxsize=None)
    def _logger(self):
        return logging.getLogger(__name__ + '.' + self.__class__.__name__)


def geodetic_to_geocentric(ellipsoid, latitude, longitude, height):
    """Return geocentric (Cartesian) Coordinates x, y, z corresponding to
    the geodetic coordinates given by latitude and longitude (in
    degrees) and height above ellipsoid. The ellipsoid must be
    specified by a pair (semi-major axis, reciprocal flattening).
    https://codereview.stackexchange.com/questions/195933/convert-geodetic-coordinates-to-geocentric-cartesian
    """
    φ = np.deg2rad(latitude)
    λ = np.deg2rad(longitude)
    sin_φ = np.sin(φ)
    a, rf = ellipsoid           # semi-major axis, reciprocal flattening
    e2 = 1 - (1 - 1 / rf) ** 2  # eccentricity squared
    n = a / np.sqrt(1 - e2 * sin_φ ** 2)  # prime vertical radius
    r = (n + height) * np.cos(φ)   # perpendicular distance from z axis
    x = r * np.cos(λ)
    y = r * np.sin(λ)
    z = (n * (1 - e2) + height) * sin_φ
    return x, y, z



    # def transform_to(self, dst_crs):
    #     dst_crs = CRS.from_user_input(dst_crs)
    #     if not self.crs.equals(dst_crs):
    #         # Case shapely.geometry
    #         if isinstance(self._geom, (Polygon, MultiPolygon)):
    #             transformer = Transformer.from_crs(
    #                 self.crs, dst_crs, always_xy=True)
    #             # Case Polygon
    #             if isinstance(self._geom, Polygon):
    #                 geom = ops.transform(transformer.transform, self._geom)
    #             # Case MultiPolygon
    #             elif isinstance(self._geom, MultiPolygon):
    #                 polygon_collection = list()
    #                 for polygon in self._geom:
    #                     polygon_collection.append(
    #                         ops.transform(transformer.transform, polygon))
    #                 outer = polygon_collection.pop(0)
    #                 geom = MultiPolygon([outer, *polygon_collection])
    #         else:
    #             raise NotImplementedError("")
    #         self._geom = geom
    #         self._crs = dst_crs