from geomesh.raster import Raster
from geomesh.geom.base import BaseGeom
from geomesh.geom.raster_geom import RasterGeom

class Geom:
    """
    Factory class that creates and returns correct object type
    based on the input type
    """
    
    def __new__(cls, geom, crs=None, ellipsoid=None,
                zmin=None, zmax=None):
        """
        Input parameters
        ----------------
        geom:
        crs:
            Assigns CRS to geom, required for shapely object.
            Overrides the input geom crs.
        ellipsoid:
            None, False, True, 'WGS84' or '??'
        zmin: 
            Minimum height/depth to be meshed
        zmax: 
            Maximum height/depth to be meshed
        """

        # TODO: Apply CRS and Ellipsoid if not NONE
        if isinstance(geom, Raster):
            return RasterGeom(geom, zmin, zmax)
        else:
            raise NotImplementedError(
                f"Geom type {type(geom)} is not supported!")

    @staticmethod
    def is_valid_type(geom_object):
        return isinstance(geom_object, BaseGeom)
