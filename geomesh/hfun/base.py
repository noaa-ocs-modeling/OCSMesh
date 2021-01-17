import abc
import logging
import numpy as np
from functools import lru_cache


class BaseHfun(abc.ABC):

    @property
    @abc.abstractmethod
    def hfun(self):
        '''Return a jigsaw_msh_t object representing the mesh size'''
        raise NotImplementedError

    @property
    def scaling(self):
        # TODO: Hardcoded for now
        return "absolute"

    @property
    def nprocs(self):
        return self._nprocs

    @property
    def hmin(self):
        return self._hmin

    @property
    def hmax(self):
        return self._hmax

    @property
    def hmin_is_absolute_limit(self):
        # TODO: Provide setter
        return True

    @property
    def hmax_is_absolute_limit(self):
        # TODO: Provide setter
        return True

    @property
    def _nprocs(self):
        return np.abs(self.__nprocs)

    @_nprocs.setter
    def _nprocs(self, nprocs):
        nprocs = cpu_count() if nprocs == -1 else nprocs
        nprocs = 1 if nprocs is None else nprocs
        self.__nprocs = nprocs

    @property
    def _hmin(self):
        return self.__hmin

    @_hmin.setter
    def _hmin(self, hmin):
        self.__hmin = hmin

    @property
    def _hmax(self):
        return self.__hmax

    @_hmax.setter
    def _hmax(self, hmax):
        self.__hmax = hmax

    @property
    @lru_cache(maxsize=None)
    def _logger(self):
        return logging.getLogger(__name__ + '.' + self.__class__.__name__)

    @abc.abstractmethod
    def add_contour(
            self,
            level: float,
            target_size: float,
            expansion_rate: float
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def add_feature(self, feature, target_size, expansion_rate):
        raise NotImplementedError

    @abc.abstractmethod
    def add_subtidal_flow_limiter(
            self, hmin=None, upper_bound=None, lower_bound=None):
        raise NotImplementedError
