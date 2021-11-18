import os
from typing import Union

from ocsmesh.geom.base import BaseGeom
from ocsmesh.mesh.base import BaseMesh
from ocsmesh.mesh.mesh import Mesh

# from jigsawpy import jigsaw_msh_t  # type: ignore[import]
# import matplotlib.pyplot as plt  # type: ignore[import]
# import mpl_toolkits.mplot3d as m3d  # type: ignore[import]
# import numpy as np  # type: ignore[import]
# from shapely import ops  # type: ignore[import]



class MeshDescriptor:

    def __set__(self, obj, val: Union[BaseMesh, str, os.PathLike]):

        if isinstance(val, (str, os.PathLike)):  # type: ignore[misc]
            val = Mesh.open(val)

        if not isinstance(val, BaseMesh):
            raise TypeError(f'Argument mesh must be of type {Mesh}, {str} '
                            f'or {os.PathLike}, not type {type(val)}')

        obj.__dict__['mesh'] = val

    def __get__(self, obj, val):
        return obj.__dict__['mesh']


class MeshGeom(BaseGeom):

    _mesh = MeshDescriptor()

    def __init__(self, mesh: Union[BaseMesh, str, os.PathLike]):
        """
        Input parameters
        ----------------
        mesh:
            Input object used to compute the output mesh hull.
        """
        self._mesh = mesh

    def get_multipolygon(self, **kwargs):
        # TODO: What if there's no tria, e.g. Mesh object is
        # created  from geom.msh_t() return value
        return self.mesh.hull.multipolygon()

    @property
    def mesh(self):
        return self._mesh

    @property
    def crs(self):
        return self._mesh.crs
