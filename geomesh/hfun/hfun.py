from geomesh.raster import Raster
from geomesh.hfun.base import BaseHfun
from geomesh.hfun.raster import HfunRaster


class Hfun:

    def __new__(cls, hfun, **kwargs):
        """
        Input parameters
        ----------------
        hfun: Object used to define and compute mesh size function.
        """

        if isinstance(hfun, Raster):
            return HfunRaster(hfun, **kwargs)

        else:
            raise TypeError(
                f'Argument hfun must be of type {BaseHfun} or a derived type, '
                f'not type {type(hfun)}.')

    @staticmethod
    def is_valid_type(hfun_object):
        return isinstance(hfun_object, BaseHfun)
