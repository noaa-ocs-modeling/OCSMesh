from abc import ABC, abstractmethod

from jigsawpy import jigsaw_msh_t  # type: ignore[import]


class BaseHfun(ABC):

    @abstractmethod
    def msh_t(self) -> jigsaw_msh_t:
        '''Abstract method to generate hfun object.'''
