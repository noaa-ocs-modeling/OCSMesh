from pathlib import Path
from typing import Union

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

class Patch:

    def __init__(self,
                 shape: Union[None, MultiPolygon, Polygon] = None,
                 shape_crs: CRS = CRS.from_user_input("EPSG:4326"),
                 shapefile: Union[None, str, Path] = None
                 ):

        if not(shape or shapefile):
            raise ValueError(
                "No patch input provided")

        # crs input is only for shape, shapefile needs to provide
        # its own crs
        self._shape_crs = shape_crs
        self._shape = None
        self._shapefile = Path(shapefile if shapefile else "")
        if isinstance(shape, Polygon):
            self._shape = MultiPolygon([shape])

        elif isinstance(shape, MultiPolygon):
            self._shape = shape

        elif shape is not None:
            raise TypeError(
                f"Type of shape input must be either {MultiPolygon}"
                f" or {Polygon}")

        elif not self._shapefile.is_file():
            raise ValueError(
                "Not shape input for patch definition")


    def get_multipolygon(self) -> MultiPolygon:

        if self._shape: # pylint: disable=R1705
            return self._shape, self._shape_crs

        elif self._shapefile.is_file():
            gdf = gpd.read_file(self._shapefile)
            poly_list = []
            for shp in gdf.geometry:
                if isinstance(shp, Polygon):
                    poly_list.append(shp)
                elif isinstance(shp, MultiPolygon):
                    poly_list.extend(list(shp.geoms))

            multipolygon = MultiPolygon(poly_list)

            return multipolygon, gdf.crs

        raise ValueError(
            "Error retrieving shape information for patch")
