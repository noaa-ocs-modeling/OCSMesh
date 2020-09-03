import abc


class _BaseGeomType(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_multipolygon(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _geom(self):
        raise NotImplementedError
