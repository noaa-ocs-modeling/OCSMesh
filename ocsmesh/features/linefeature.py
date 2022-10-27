from pathlib import Path
from typing import Union

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString

class LineFeature:

    def __init__(self,
                 shape: Union[None, MultiLineString, LineString] = None,
                 shape_crs: CRS = CRS.from_user_input("EPSG:4326"),
                 shapefile: Union[None, str, Path] = None
                 ):

        if not(shape or shapefile):
            raise ValueError("No line feature input provided")


        # crs input is only for shape, shapefile needs to have
        # its own crs
        self._shape_crs = shape_crs
        self._shape = None
        self._shapefile = Path(shapefile if shapefile else "")
        if isinstance(shape, LineString):
            self._shape = MultiLineString([shape])

        elif isinstance(shape, MultiLineString):
            self._shape = shape

        elif shape is not None:
            raise TypeError(
                f"Type of shape input must be either {MultiLineString}"
                f" or {LineString}")

        elif not (self._shapefile.is_file() or self._shapefile.is_dir()):
            raise ValueError(
                "Not shape input for line feature definition")


    def get_multiline(self) -> MultiLineString:

        if self._shape: # pylint: disable=R1705
            return self._shape, self._shape_crs

        elif self._shapefile.is_file() or self._shapefile.is_dir():
            gdf = gpd.read_file(self._shapefile)
            if gdf.crs is None:
                raise ValueError("Shapefile doesn't have any CRS")
            line_list = []
            for shp in gdf.geometry:
                if isinstance(shp, LineString):
                    line_list.append(shp)
                elif isinstance(shp, MultiLineString):
                    line_list.extend(list(shp.geoms))

            multiline = MultiLineString(line_list)

            return multiline, gdf.crs

        raise ValueError(
            "Error retrieving shape information for line feature")
