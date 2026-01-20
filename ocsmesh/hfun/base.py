"""This module defines the base class for all size function types
"""

import warnings
from abc import ABC, abstractmethod

from ocsmesh.internal import MeshData  # type: ignore[import]


class BaseHfun(ABC):
    """Abstract base class used to construct OCSMesh "hfun" objects.

    This abstract class defines the interface that all size function
    objects are expected to have in OCSMesh. All size function
    subclasses need to inherit this base class.

    Methods
    -------
    meshdata()
        Returns the `MeshData` vertex-edge representation of the mesh.
    """

    def msh_t(self, *args, **kwargs) -> MeshData:
        '''Abstract method to generate hfun object.'''
        warnings.warn("Use meshdata(...) instead!", DeprecationWarning)
        return self.meshdata(*args, **kwargs)


    @abstractmethod
    def meshdata(self) -> MeshData:
        '''Abstract method to generate hfun object.'''
