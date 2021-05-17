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
            
        elif not self._shapefile.is_file():
            raise ValueError(
                f"Not shape input for patch definition")


    def get_multipolygon(self) -> MultiPolygon:

        if self._shape:
            dst_crs = CRS.from_user_input("EPSG:4326")
            return self._shape, dst_crs

        elif self._shapefile.is_file():
            gdf = gpd.read_file(self._shapefile)
            poly_list = list()
            for shp in gdf.geometry:
                if isinstance(shp, Polygon):
                    poly_list.append(shp)
                elif isinstance(shp, MultiPolygon):
                    poly_list.extend([pl for pl in shp.geoms])

            multipolygon = MultiPolygon(poly_list)
            
            return multipolygon, gdf.crs

        raise ValueError(
            f"Error retrieving shape information for patch")
