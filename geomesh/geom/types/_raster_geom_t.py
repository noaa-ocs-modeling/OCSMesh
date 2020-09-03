from ...raster import Raster
from ._base import _BaseGeomType


class _RasterGeomType(_BaseGeomType):

    __slots__ = ["__raster"]

    def __init__(self, raster, **kwargs):
        self._raster = raster

    def get_multipolygon(self, **kwargs):
        return self._raster.get_multipolygon(**kwargs)

    @property
    def _geom(self):
        return self.__raster

    @property
    def _raster(self):
        return self.__raster

    @_raster.setter
    def _raster(self, raster):
        assert isinstance(raster, Raster)
        self.__raster = raster
