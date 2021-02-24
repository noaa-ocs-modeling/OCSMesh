from pathlib import Path

class Contour:
    def __init__(self, level=None, sources=list(), shapefile=None):

        # Either based on shape or the source-level
        if shapefile and Path(shapefile).is_file():
            raise NotImplementedError(
                "Contour based on shapefiles are not supported yet!")

        elif level != None:
            self._level = level
            self._sources = list()
            for source in sources:
                self.add_source(source)

        else:
            raise ValueError("Input is not sufficient to define a contour")

    @property
    def has_source(self):
        return len(self._sources) > 0

    def add_source(self, source_object):
        src_class = type(source_object).__name__
        if src_class not in ("Raster", "RasterGeom", "HfunRaster"):
            # TODO: Accepts meshhfun?
            raise TypeError("")

        self._sources.append(source_object)

    def iter_contours(self):
        #if shapefile
        #else:
        for source in self._sources:
            src_class = type(source).__name__
            if src_class == "Raster":
                contour = source.get_contour(self._level)
                crs = source.crs
            elif src_class in ("RasterGeom", "HfunRaster"):
                contour = source.raster.get_contour(self._level)
                crs = source.raster.crs
            else:
                raise TypeError("")

            yield contour, crs
