"""This module defines a base type for mesh objects.
"""
import warnings

import numpy as np
import numpy.typing as npt

from ocsmesh.internal import MeshData

class BaseMesh:
    """Base class for mesh types in OCSMesh.

    Attributes
    ----------
    meshdata : MeshData
        Refernece to underlying mesh object.
    coord : array-like
        Coordinates of mesh nodes.

    Methods
    -------
    """

    @property
    def msh_t(self) -> MeshData:
        warnings.warn("Use meshdata(...) instead!", DeprecationWarning)
        return self.meshdata


    @property
    def meshdata(self) -> MeshData:
        """Read-only property returning reference to the mesh data

        Notes
        -----
        The property is read-only, however the returned value is not.
        That means if the user mutates the returned MeshData
        object, it will affect the mesh.
        """

        return self._meshdata

    @property
    def coord(self) -> npt.NDArray[np.float32]:
        """Read-only property for coordinates of mesh points

        Raises
        ------
        If the number of mesh dimensions is not 2 or 3.

        Notes
        -----
        The property is read-only, however the returned value is not.
        That means if the user mutates the returned numpy array
        object, it will affect the mesh.
        """

        return self.meshdata.coords
