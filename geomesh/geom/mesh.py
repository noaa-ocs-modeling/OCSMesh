import os
from typing import Union

# from jigsawpy import jigsaw_msh_t  # type: ignore[import]
# import matplotlib.pyplot as plt  # type: ignore[import]
# import mpl_toolkits.mplot3d as m3d  # type: ignore[import]
# import numpy as np  # type: ignore[import]
# from shapely import ops  # type: ignore[import]

from geomesh.geom.base import BaseGeom
from geomesh.mesh import Mesh


class MeshDescriptor:

    def __set__(self, obj, val: Union[Mesh, str, os.PathLike]):

        if isinstance(val, (str, os.PathLike)):  # type: ignore[misc]
            val = Mesh.open(val)

        if not isinstance(val, Mesh):
            raise TypeError(f'Argument mesh must be of type {Mesh}, {str} '
                            f'or {os.PathLike}, not type {type(val)}')

        obj.__dict__['mesh'] = val

    def __get__(self, obj, val):
        return obj.__dict__['mesh']


class MeshGeom(BaseGeom):

    _mesh = MeshDescriptor()

    def __init__(self, mesh: Union[Mesh, str, os.PathLike]):
        """
        Input parameters
        ----------------
        mesh:
            Input object used to compute the output mesh hull.
        """
        self._mesh = mesh

    def get_multipolygon(self):
        self.mesh.get_multipolygon(self)

    @property
    def mesh(self):
        return self._mesh
