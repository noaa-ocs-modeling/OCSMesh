from pathlib import Path
from abc import ABC, abstractmethod

class ContourBase(ABC):

    def __init__(self, sources=None, shapefile=None):
        if sources is None:
            sources = []
        if sources and shapefile:
            raise ValueError(
                "Both sources and shapefile cannot be specified at the same time!")

        # Either based on shape or the source-level
        if shapefile and Path(shapefile).is_file():
            raise NotImplementedError(
                "Contour based on shapefiles are not supported yet!")

        self._sources = []
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
        for source in sources:
            self.add_source(source)
        self._shapefile = shapefile

    @property
    def has_source(self):
        return len(self._sources) > 0

    @property
    def sources(self):
        return self._sources

    @property
    def shapefile(self):
        return self._shapefile

    def add_source(self, source_object):
        src_class = type(source_object).__name__
        if src_class not in ("Raster", "RasterGeom", "HfunRaster"):
            # TODO: Accepts meshhfun?
            raise TypeError("")

        self._sources.append(source_object)

    def iter_contours(self):
        if self._shapefile:
            yield self._get_contour_from_shapefile(self._shapefile)
        for source in self._sources:
            yield self._get_contour_from_source(source)

    @abstractmethod
    def _get_contour_from_source(self, source):
        pass

#    @abstractmethod
    def _get_contour_from_shapefile(self, shapefile):
        # TODO: Support shapefile?
        raise NotImplementedError(
                "Contour based on shapefiles are not supported yet!")

    @property
    @abstractmethod
    def level(self):
        return 0


class Contour(ContourBase):

    def __init__(self, level=None, sources=None, shapefile=None):

        if sources is None:
            sources = []

        super().__init__(sources, shapefile)
        self._level = level

    def _get_contour_from_source(self, source):
        src_class = type(source).__name__
        if src_class == "Raster":
            contour = source.get_contour(self._level)
            crs = source.crs
        elif src_class in ("RasterGeom", "HfunRaster"):
            contour = source.raster.get_contour(self._level)
            crs = source.raster.crs
        else:
            raise TypeError("")

        return contour, crs

    @property
    def level(self):
        return self._level


class FilledContour(ContourBase):

    def __init__(self,
                 level0=None,
                 level1=None,
                 sources=None,
                 max_contour_defn : Contour = None,
                 shapefile=None):

        if sources is None:
            sources = []

        super().__init__(sources, shapefile)
        if max_contour_defn:
            self._level0 = None
            self._level1 = max_contour_defn.level
            # NOTE: Overriding the existing sources or shapefiles
            self._sources = max_contour_defn.sources
            self._shapefile = max_contour_defn.shapefile
        else:
            self._level0 = level0
            self._level1 = level1

    def _get_contour_from_source(self, source):
        z_info = {}
        if self._level0 is not None:
            z_info['zmin'] = self._level0
        if self._level1 is not None:
            z_info['zmax'] = self._level1

        src_class = type(source).__name__
        if src_class == "Raster":
            contour = source.get_multipolygon(**z_info)
            crs = source.crs
        elif src_class in ("RasterGeom", "HfunRaster"):
            contour = source.raster.get_multipolygon(**z_info)
            crs = source.raster.crs
        else:
            raise TypeError("")

        return contour, crs

    @property
    def level(self):
        return self._level0, self._level1
