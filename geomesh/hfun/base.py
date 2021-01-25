from abc import ABC, abstractmethod

from jigsawpy import jigsaw_msh_t  # type: ignore[import]

from geomesh.logger import Logger


class BaseHfun(ABC):

    logger = Logger()

    @property
    def hfun(self) -> jigsaw_msh_t:
        '''Return a jigsaw_msh_t object representing the mesh size'''
        return self._get_jigsaw_msh_t('hfun')

    @property
    def hmat(self) -> jigsaw_msh_t:
        '''Return a jigsaw_msh_t object representing the mesh size'''
        return self._get_jigsaw_msh_t('hmat')

    @abstractmethod
    def get_hmat(self) -> jigsaw_msh_t:
        '''Abstract method to generate hmat object.'''

    @abstractmethod
    def get_hfun(self) -> jigsaw_msh_t:
        '''Abstract method to generate hfun object.'''

    def _get_jigsaw_msh_t(self, hfun_type, **kwargs) -> jigsaw_msh_t:
        '''Returns a :class:jigsawpy.jigsaw_msh_t object representing the
        geometry constrained by the arguments.'''
        assert hfun_type in ['hfun', 'hmat']
        return getattr(self, f'get_{hfun_type}')(**kwargs)
