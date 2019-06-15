from geomesh._SpatialReference import _SpatialReference


class Hfun(_SpatialReference):

    def __init__(self):
        super(Hfun, self).__init__(None)
        self.__DatasetCollection = list()

    @property
    def _DatasetCollection(self):
        return self.__DatasetCollection
