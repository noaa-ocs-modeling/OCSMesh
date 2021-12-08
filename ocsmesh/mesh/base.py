"""This module defines a base type for mesh objects.
"""

import numpy as np
import numpy.typing as npt
from jigsawpy import jigsaw_msh_t

class BaseMesh:
    """Base class for mesh types in OCSMesh.

    Attributes
    ----------
    msh_t : jigsaw_msh_t
        Refernece to underlying jigsaw mesh object.
    coord : array-like
        Coordinates of mesh nodes.

    Methods
    -------
    """

    @property
    def msh_t(self) -> jigsaw_msh_t:
        """Read-only property returning reference to the jigsawpy mesh object

        Notes
        -----
        The property is read-only, however the returned value is not.
        That means if the user mutates the returned jigsaw_msh_t
        object, it will affect the mesh.
        """

        return self._msh_t

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

        if self.msh_t.ndims == 2: # pylint: disable=R1705
            return self.msh_t.vert2['coord']
        elif self.msh_t.ndims == 3:
            return self.msh_t.vert3['coord']

        raise ValueError(f'Unhandled mesh dimensions {self.msh_t.ndims}.')
