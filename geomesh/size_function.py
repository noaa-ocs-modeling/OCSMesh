import matplotlib.pyplot as plt
from jigsawpy import jigsaw_msh_t
from geomesh.gdal_dataset_collection import GdalDatasetCollection
from geomesh.gdal_tools import GdalTools


class SizeFunction(GdalTools):

    def __init__(self, SpatialReference=3395):
        super(SizeFunction, self).__init__()
        self._GdalDatasetCollection = GdalDatasetCollection()
        self._SpatialReference = SpatialReference

    def add_dataset(self, path):
        self._GdalDatasetCollection.add_dataset(path, hfun=True)

    @property
    def shoreline(self):
        for i, dataset in enumerate(self._GdalDatasetCollection):
            dataset.SpatialReference = self.SpatialReference
            plt.contour(dataset.x, dataset.y, dataset.values, levels=[0.])
            plt.show()
            dataset.reset()

    @property
    def hfun(self):
        hfun = jigsaw_msh_t()
        for dataset in self._GdalDatasetCollection._hfun:
            dataset.SpatialReference = self.SpatialReference
        return hfun

    @property
    def SpatialReference(self):
        return self.__SpatialReference

    @property
    def _SpatialReference(self):
        return self.__SpatialReference

    @property
    def _mshID(self):
        return "euclidean-mesh"

    @property
    def _hfun_scal(self):
        return 'absolute'

    @_SpatialReference.setter
    def _SpatialReference(self, SpatialReference):
        self.__SpatialReference = self.sanitize_SpatialReference(
            SpatialReference)
