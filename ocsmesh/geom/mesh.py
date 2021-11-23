"""This module defines mesh based geometry class
"""

import os
from typing import Union

# from jigsawpy import jigsaw_msh_t  # type: ignore[import]
# import matplotlib.pyplot as plt  # type: ignore[import]
# import mpl_toolkits.mplot3d as m3d  # type: ignore[import]
# import numpy as np  # type: ignore[import]
from shapely.geometry import MultiPolygon

from ocsmesh.geom.base import BaseGeom
from ocsmesh.mesh.mesh import Mesh
from ocsmesh.mesh.base import BaseMesh


class MeshDescriptor:
    """Descriptor class for storing handle to input mesh"""

    def __set__(self, obj, val: Union[BaseMesh, str, os.PathLike]):

        if isinstance(val, (str, os.PathLike)):  # type: ignore[misc]
            val = Mesh.open(val)

        if not isinstance(val, BaseMesh):
            raise TypeError(f'Argument mesh must be of type {Mesh}, {str} '
                            f'or {os.PathLike}, not type {type(val)}')

        obj.__dict__['mesh'] = val

    def __get__(self, obj, objtype=None) -> BaseMesh:
        return obj.__dict__['mesh']


class MeshGeom(BaseGeom):
    """Mesh based geometry.

    Create a geometry based on an input mesh. All the calculations
    for polygon are done on the underlying mesh object.

    Attributes
    ----------
    mesh : BaseMesh
        Reference to the underlying mesh object used to create
        the geometry
    crs : CRS
        CRS of the underlying mesh

    Methods
    -------
    get_multipolygon(**kwargs)
        Returns `shapely` object representation of the geometry
    msh_t(**kwargs)
        Returns the `jigsawpy` vertex-edge representation of the geometry

    Notes
    -----
    This class is a handy tool for reusing existing mesh extent
    for geometry (i.e. domain) definition and then remeshing that
    domain.
    """

    _mesh = MeshDescriptor()

    def __init__(
            self,
            mesh: Union[BaseMesh, str, os.PathLike]
            ) -> None:
        """ Initialize a mesh based geometry object

        Parameters
        ----------
        mesh:
            Input object used to compute the output mesh hull.
        """

        self._mesh = mesh

    def get_multipolygon(self, **kwargs) -> MultiPolygon:
        """Returns the `shapely` representation of the geometry

        Calculates and returns the `MultiPolygon` representation of
        the geometry.

        Parameters
        ----------
        **kwargs : dict, optional
            Currently unused for this class, needed for generic API
            support

        Returns
        -------
        MultiPolygon
            Calculated polygon from mesh based on the element boundaries
        """

        # TODO: What if there's no tria, e.g. Mesh object is
        # created from geom.msh_t() return value
        return self.mesh.hull.multipolygon()

    @property
    def mesh(self):
        """Read-only attribute for reference to the source mesh"""
        return self._mesh

    @property
    def crs(self):
        """Read-only attribute returning the CRS of the underlying mesh"""
        return self._mesh.crs
