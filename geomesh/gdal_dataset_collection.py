from geomesh.gdal_dataset import GdalDataset


class GdalDatasetCollection:

    __container = list()

    def __iter__(self):
        for gdal_dataset in self.__container:
            yield gdal_dataset

    def add_dataset(self, path, pslg=True, hfun=True):
        exist = False
        for pos, gdal_dataset in enumerate(self.__container):
            if path == gdal_dataset.path:
                exist = True
                break
        if not exist:
            try:
                pos += 1
            except UnboundLocalError:
                pos = 0
            gdal_dataset = GdalDataset(path)
        else:
            gdal_dataset = self.__container.pop(pos)
        if pslg:
            gdal_dataset._pslg = True
        if hfun:
            gdal_dataset._hfun = True
        self.__container.insert(pos, gdal_dataset)

    @property
    def _pslg(self):
        for dataset in self:
            if dataset._pslg:
                yield dataset

    @property
    def _hfun(self):
        for dataset in self:
            if dataset._hfun:
                yield dataset


# alias export
DatasetCollection = GdalDatasetCollection
