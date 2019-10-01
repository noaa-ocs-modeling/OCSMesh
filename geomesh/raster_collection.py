from pathlib import Path
from geomesh.raster import Raster


class RasterCollection:

    def __init__(self, dst_crs=None):
        self._dst_crs = dst_crs

    def __iter__(self):
        for raster in self.container:
            yield raster

    def __getitem__(self, i):
        return self.container[i]

    def add_dataset(self, raster):
        # TODO: hacky way of avoiding repeats
        if isinstance(raster, (str, Path)):
            raster = Raster(raster)
        else:
            assert isinstance(raster, Raster)
        exist = False
        for _ in self.container:
            if raster.path == _.path:
                exist = True
                break
        if not exist:
            raster.dst_crs = self.dst_crs
            self._container.append(raster)

    @property
    def dst_crs(self):
        return self._dst_crs

    @property
    def container(self):
        return tuple(self._container)

    @property
    def _container(self):
        try:
            return self.__container
        except AttributeError:
            self.__container = list()
            return self.__container

    @property
    def _dst_crs(self):
        return self.__dst_crs

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        self._dst_crs = dst_crs

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        for raster in self.container:
            raster.dst_crs = dst_crs
        self.__dst_crs = dst_crs
