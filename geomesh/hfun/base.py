from abc import ABC, abstractmethod

from jigsawpy import jigsaw_msh_t  # type: ignore[import]

from geomesh.logger import Logger


class BaseHfun(ABC):

    logger = Logger()

    @abstractmethod
    def msh_t(self) -> jigsaw_msh_t:
        '''Abstract method to generate hfun object.'''
