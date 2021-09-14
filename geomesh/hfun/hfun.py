from ocsmesh.hfun.base import BaseHfun
from ocsmesh.hfun.raster import HfunRaster
from ocsmesh.hfun.mesh import HfunMesh
from ocsmesh.hfun.collector import HfunCollector
from ocsmesh.mesh.mesh import EuclideanMesh2D
from ocsmesh.raster import Raster


class Hfun:

    def __new__(cls, hfun, **kwargs):
        """
        Input parameters
        ----------------
        hfun: Object used to define and compute mesh size function.
        """

        if isinstance(hfun, Raster):
            return HfunRaster(hfun, **kwargs)

        elif isinstance(hfun, EuclideanMesh2D):
            return HfunMesh(hfun)

        elif isinstance(hfun, (list, tuple)):
            return HfunCollector(hfun, **kwargs)

        else:
            raise TypeError(
                f'Argument hfun must be of type {BaseHfun} or a derived type, '
                f'not type {type(hfun)}.')

    @staticmethod
    def is_valid_type(hfun_object):
        return isinstance(hfun_object, BaseHfun)
