"""This module defines the size function factory class

Instead of importing specific size function types, users can import
factory `Hfun` object from this module which creates the correct
type of size function based on the input arguments passed to it
"""

from typing import Any

from ocsmesh.hfun.base import BaseHfun
from ocsmesh.hfun.raster import HfunRaster
from ocsmesh.hfun.mesh import HfunMesh
from ocsmesh.mesh.mesh import EuclideanMesh2D
from ocsmesh.raster import Raster
from ocsmesh.hfun.collector import HfunCollector, CanCreateHfun


class Hfun:
    """ Size function object factory

    Factory class that creates and returns concrete mesh size function
    object based on the type of the inputs.

    Methods
    -------
    is_valid_type(hfun_object)
        Static method to check if an object is a valid size function
        type

    Notes
    -----
    The object created when using this class is not an instance of
    this class or a subclass of it. It is a subclass of BaseHfun
    instead.
    """


    # pylint: disable=no-member

    def __new__(cls, hfun: CanCreateHfun, **kwargs: Any) -> BaseHfun:
        """
        Parameters
        ----------
        hfun : CanCreateHfun
            Object to create the mesh size function from. The type of
            this object determines the created size function object
            type
        **kwargs : dict, optional
            Keyword arguments passed to the constructor of the
            correct size function type
        """

        if isinstance(hfun, Raster): # pylint: disable=R1705
            return HfunRaster(hfun, **kwargs)

        elif isinstance(hfun, EuclideanMesh2D):
            return HfunMesh(hfun)

        elif isinstance(hfun, (list, tuple)):
            return HfunCollector(hfun, **kwargs)

        else:
            raise TypeError(
                f'Argument hfun has the wrong type, type {type(hfun)}.')

    @staticmethod
    def is_valid_type(hfun_object: Any) -> bool:
        """Checks if an object is a valid size function type"""
        return isinstance(hfun_object, BaseHfun)
