from osgeo import osr


class _SpatialReference(object):

    def __init__(self, SpatialReference):
        self._SpatialReference = SpatialReference

    @property
    def SpatialReference(self):
        return self._SpatialReference

    @property
    def _SpatialReference(self):
        return self.__SpatialReference

    @SpatialReference.setter
    def SpatialReference(self, SpatialReference):
        raise NotImplementedError

    @_SpatialReference.setter
    def _SpatialReference(self, SpatialReference):
        if SpatialReference is not None:
            assert isinstance(SpatialReference, osr.SpatialReference)
        self.__SpatialReference = SpatialReference
