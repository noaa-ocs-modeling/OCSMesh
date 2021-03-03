from pathlib import Path
from typing import Union

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

class Patch:
    
    def __init__(self,
                 shape: Union[None, MultiPolygon, Polygon] = None,
                 shapefile: Union[None, str, Path] = None
                 ):

        if not(shape or shapefile):
            raise ValueError(
                "No patch input provided")

        self._shape = None
        self._shapefile = Path(shapefile)
        if isinstance(shape, Polygon):
            self._shape = MultiPolygon([shape])
        elif isinstance(shape, MultiPolygon):
            self._shape = shape
        elif shape != None:
            raise TypeError(
                f"Type of shape input must be either {MultiPolygon}"
                f" or {Polygon}")

    def get_multipolygon(self) -> MultiPolygon:

        if self._shape:
            return self._shape

        elif self._shapefile.is_file():
            gdf = gpd.read_file(self._shapefile)
            dst_crs = CRS.from_user_input("EPSG:4326")
            print("GDF", gdf.crs)
            if gdf.crs != dst_crs:
                print("SHAPECRS", gdf.crs)
                gdf = gdf.to_crs(dst_crs)
            multipolygon = MultiPolygon([i for i in gdf.geometry])
            
            return multipolygon, dst_crs
