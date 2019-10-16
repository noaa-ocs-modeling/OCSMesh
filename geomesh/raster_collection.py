from pathlib import Path
from geomesh.raster import Raster


class RasterCollection:

    def __init__(self, files=[], dst_crs="EPSG:3395"):
        self._dst_crs = dst_crs
        files = list(files)
        for file in files:
            self.append(file)

    def __iter__(self):
        for raster in self.container:
            yield raster

    def __getitem__(self, i):
        return self.container[i]

    def append(self, raster):
        if isinstance(raster, (str, Path)):
            raster = Raster(raster)
        else:
            assert isinstance(raster, Raster)
        for item in self.container:
            if raster.path == item.path:
                return
        raster.set_dst_crs(self.dst_crs)
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
