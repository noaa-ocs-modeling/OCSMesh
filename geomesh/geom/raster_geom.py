import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
from pyproj import CRS

from jigsawpy import jigsaw_msh_t

from geomesh.raster import Raster
from geomesh.geom.base import BaseGeom


class RasterGeom(BaseGeom):


    def __init__(self, raster, zmin=None, zmax=None, **kwargs):
        self._raster = raster
        # TODO: Make these properties with setter to test values
        # maybe in the base class?
        self._zmin = zmin
        self._zmax = zmax


    def get_multipolygon(self, **kwargs):
        return self._raster.get_multipolygon(**kwargs)


    @property
    def backend(self):
        return self._raster


    @property
    def _raster(self):
        return self.__raster


    @_raster.setter
    def _raster(self, raster):
        assert isinstance(raster, Raster)
        self.__raster = raster


    @property
    def geom(self):
        '''Return a jigsaw_msh_t object representing the geometry'''

        # TODO: Add zmin/zmax/window, etc.
        kwargs = {'zmin': self._zmin, 'zmax': self._zmax}
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
                    [(e0+len(edge2), e1+len(edge2))
                        for e0, e1 in _edge2])
        edge2 = np.asarray(
            [((e0, e1), 0) for e0, e1 in edge2],
            dtype=jigsaw_msh_t.EDGE2_t)
        # geom
        geom = jigsaw_msh_t()
        geom.ndims = +2
        geom.mshID = 'euclidean-mesh'
        # TODO: Does raster mean it's not ellipsoid?
#        geom.mshID = 'euclidean-mesh' if self._ellipsoid is None \
#            else 'ellipsoidal-mesh'
        geom.vert2 = vert2
        geom.edge2 = edge2
        return geom


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
    def _crs(self):
        return self.backend.crs


    @_crs.setter
    def _crs(self, crs):
        self.backend._crs = CRS.from_user_input(crs)


#    @property
#    def _ellipsoid(self):
#        return self._geom.ellipsoid
#
#
#    @_ellipsoid.setter
#    def _ellipsoid(self, ellipsoid):
#        self._geom._ellipsoid = ellipsoid
#
#
#    @property
#    def _ellipsoids(self):
#        return {
#            "WGS84": (6378137, 298.257223563),
#            "GRS80": (6378137, 298.257222100882711)
#            }

    def make_plot(
        self,
        ax=None,
        show=False,
    ):


        # TODO: This funciton doesn't work due to disabling ellipsoid

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
