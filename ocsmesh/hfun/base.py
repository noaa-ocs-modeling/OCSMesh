"""This module defines the base class for all size function types
"""

from abc import ABC, abstractmethod

from jigsawpy import jigsaw_msh_t  # type: ignore[import]


class BaseHfun(ABC):
    """Abstract base class used to construct OCSMesh "hfun" objects.

    This abstract class defines the interface that all size function
    objects are expected to have in OCSMesh. All size function
    subclasses need to inherit this base class.

    Methods
    -------
    msh_t()
        Returns the `jigsawpy` vertex-edge representation of the mesh.
    """


    @abstractmethod
    def msh_t(self) -> jigsaw_msh_t:
        '''Abstract method to generate hfun object.'''
