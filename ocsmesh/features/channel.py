class Channel:

    def __init__(self, level=0, width=1000, tolerance=50, sources=None):

        if sources is None:
            sources = []

        # Even a tolerance of 1 for simplifying polygon for channel
        # calculations is much faster than no simplification. 50
        # is much faster than 1. The reason is in simplify we don't
        # preserve topology

        self._level = level
        self._width = width # and less
        self._tolerance = tolerance # to simplify

        self._sources = []
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
        for source in sources:
            self.add_source(source)

    @property
    def has_source(self):
        return len(self._sources) > 0

    @property
    def sources(self):
        return self._sources

    def add_source(self, source_object):
        src_class = type(source_object).__name__
        if src_class not in ("Raster", "RasterGeom", "HfunRaster"):
            # TODO: Accepts meshhfun?
            raise TypeError("")

        self._sources.append(source_object)

    def iter_channels(self):
        for source in self._sources:
            ch, crs = self._get_contour_from_source(source)
            if ch is None:
                continue
            yield ch, crs

    @property
    def level(self):
        return self._level

    @property
    def width(self):
        return self._width

    def _get_contour_from_source(self, source):

        src_class = type(source).__name__
        if src_class == "Raster":
            channels = source.get_channels(
                    self._level, self._width, self._tolerance)
            crs = source.crs
        elif src_class in ("RasterGeom", "HfunRaster"):
            channels = source.raster.get_channels(
                    self._level, self._width, self._tolerance)
            crs = source.raster.crs
        else:
            raise TypeError("")

        return channels, crs
