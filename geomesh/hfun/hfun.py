from functools import lru_cache
import logging
import pathlib
import tempfile


from ..raster import Raster
from . import types

tmpdir = pathlib.Path(tempfile.gettempdir()) / 'geomesh'
tmpdir.mkdir(exist_ok=True)


class Hfun:

    __slots__ = ["__hfun"]

    __types__ = {
        Raster: types._HfunRaster
    }

    def __init__(
            self,
            hfun,
            geom=None,
            hmin=None,
            hmax=None,
            ellipsoid=None,
            verbosity=0,
            interface='cmdsaw',
            nprocs=None,
    ):
        self._hfun = hfun
        self._geom = geom
        self._hmin = hmin
        self._hmax = hmax
        self._ellipsoid = ellipsoid
        self._verbosity = verbosity
        self._interface = interface
        self._nprocs = nprocs

    def get_mesh(self, geom=None):
        return self._hfun.get_mesh(geom=geom)

    @property
    def _hfun(self):
        return self.__hfun

    @_hfun.setter
    def _hfun(self, hfun):

        if type(hfun) in self.__types__:
            raise TypeError(f'hfun must be one of {self.__types__}')

        hfun = self.__type___[type(hfun)](hfun)

        if hasattr(hfun, 'contourf'):
            self.contourf = hfun.contourf

        self.add_contour = hfun.add_contour
        self.add_feature = hfun.add_feature
        self.add_subtidal_flow_limiter = hfun.add_subtidal_flow_limiter
        # self.add_constant_range = hfun.add_constant_floodplain
        self.__hfun = hfun

    @property
    def _nprocs(self):
        return self._hfun._nprocs

    @_nprocs.setter
    def _nprocs(self, nprocs):
        self._hfun._nprocs = nprocs

    @property
    def _hmin(self):
        return self._hfun._hmin

    @_hmin.setter
    def _hmin(self, hmin):
        self._hfun._hmin = hmin

    @property
    def _hmax(self):
        return self._hfun._hmax

    @_hmax.setter
    def _hmax(self, hmax):
        self._hfun._hmax = hmax

    @property
    @lru_cache(maxsize=None)
    def _logger(self):
        return logging.getLogger(__name__ + '.' + self.__class__.__name__)
