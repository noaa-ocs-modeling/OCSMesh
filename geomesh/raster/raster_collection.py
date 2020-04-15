from pathlib import Path
from functools import lru_cache
from pyproj import CRS
# from shapely.geometry import MultiPolygon, Polygon
from geomesh.raster import Raster


class RasterCollection:

    def __init__(self, dst_crs="EPSG:3395"):
        self.dst_crs = dst_crs

    def __iter__(self):
        for raster in self._container:
            raster = Raster(
                raster['path'],
                crs=raster['crs'],
                dst_crs=self.dst_crs
                )
            yield raster

    def __getitem__(self, i):
        return Raster(
                self._container[i]['path'],
                self._container[i]['crs'],
                dst_crs=self.dst_crs
                )

    def __len__(self):
        return len(self._container)

    def append(self, raster, crs=None):
        if isinstance(raster, (str, Path)):
            pass
        elif isinstance(raster, Raster):
            crs = raster.crs
            raster = raster.path
        else:
            msg = f"Unrecognized input type: {raster}"
            raise Exception(msg)
        for item in self._container:
            if raster == item['path']:
                return
        self._container.append({'path': raster, 'crs': crs})

    # def get_multipolygon(self, zmin=None, zmax=None):
    #     polygon_collection = list()
    #     for raster in self.container:
    #         multipolygon = raster.get_multipolygon(zmin, zmax)
    #         for polygon in multipolygon:
    #             polygon_collection.append(polygon)
    #     multipolygon = MultiPolygon(polygon_collection).buffer(0)
    #     if isinstance(multipolygon, Polygon):
    #         multipolygon = MultiPolygon([multipolygon])
    #     return multipolygon

    @property
    def dst_crs(self):
        return self.__dst_crs

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        if dst_crs is not None:
            dst_crs = CRS.from_user_input(dst_crs)
        self.__dst_crs = dst_crs

    @property
    @lru_cache
    def _container(self):
        return []
