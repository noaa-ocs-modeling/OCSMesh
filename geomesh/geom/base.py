import abc
import logging
from functools import lru_cache


class BaseGeom(abc.ABC):


    @abc.abstractmethod
    def get_multipolygon(self):
        raise NotImplementedError


    @property
    @abc.abstractmethod
    def backend(self):
        raise NotImplementedError


    @property
    @abc.abstractmethod
    def geom(self):
        '''Return a jigsaw_msh_t object representing the geometry'''
        raise NotImplementedError


    @property
    @lru_cache(maxsize=None)
    def _logger(self):
        return logging.getLogger(__name__ + '.' + self.__class__.__name__)
