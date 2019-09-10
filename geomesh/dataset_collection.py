from geomesh.gdal_dataset import GdalDataset
from geomesh import gdal_tools


class DatasetCollection:

    def __init__(self, SpatialReference=3395):
        self._SpatialReference = SpatialReference
        self.__container = list()

    def __iter__(self):
        for gdal_dataset in self.__container:
            gdal_dataset.SpatialReference = self.SpatialReference
            yield gdal_dataset

    def add_dataset(self, gdal_dataset):
        if isinstance(gdal_dataset, str):
            gdal_dataset = GdalDataset(GdalDataset)
        else:
            assert isinstance(gdal_dataset, GdalDataset)
        exist = False
        for _ in self.__container:
            if gdal_dataset.path == _.path:
                exist = True
                break
        if not exist:
            self.__container.append(gdal_dataset)

    @property
    def SpatialReference(self):
        return self.__SpatialReference

    @property
    def _SpatialReference(self):
        return self.__SpatialReference

    @_SpatialReference.setter
    def _SpatialReference(self, SpatialReference):
        self.__SpatialReference = gdal_tools.sanitize_SpatialReference(
            SpatialReference)
