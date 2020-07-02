import tempfile
import logging
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import pathlib
import geomesh
from geomesh.figures import _figure

tmpdir = pathlib.Path(tempfile.gettempdir()) / 'geomesh'
tmpdir.mkdir(exist_ok=True)


class Hfun:

    def __init__(
        self,
        hfun,
        geom=None,
        hmin=None,
        hmax=None,
        ellipsoid=None,
        verbosity=0,
        interface='cmdsaw',
    ):
        self._hfun = hfun
        self._geom = geom
        self._hmin = hmin
        self._hmax = hmax
        self._ellipsoid = ellipsoid
        self._verbosity = verbosity
        self._interface = interface

    def add_contour(
        self,
        level,
        target_size,
        expansion_rate,
        hmin=None,
        hmax=None,
        n_jobs=-1,
    ):
        msg = f'{self.path}:add_contour({level})'
        self._logger.debug(msg)
        raise NotImplementedError
        # tree = self._get_raster_level_kdtree(level)
        # xt, yt = np.meshgrid(raster.x, raster.y)
        # xt = xt.flatten()
        # yt = yt.flatten()
        # xy_target = np.vstack([xt, yt]).T
        # values, _ = tree.query(xy_target, n_jobs=n_jobs)
        # values = expansion_rate*target_size*values + target_size
        # values = values.reshape(raster.values.shape)
        # if hmin is not None:
        #     values[np.where(values < hmin)] = hmin
        # if hmax is not None:
        #     values[np.where(values > hmax)] = hmax
        # outband = np.minimum(outband, values)
        # return outband

    @_figure
    def contourf(self, **kwargs):
        plt.contourf(
                self.xgrid,
                self.ygrid,
                self.zgrid,
                levels=kwargs.pop("levels", 256),
                cmap=kwargs.pop('cmap', 'jet'),
                vmin=kwargs.pop('vmin', np.min(self.zgrid)),
                vmax=kwargs.pop('vmax', np.max(self.zgrid)),
            )

    @property
    def xgrid(self):
        try:
            return self._src.x()
        except AttributeError:
            return []

    @property
    def ygrid(self):
        try:
            return self._src.y()
        except AttributeError:
            return []

    @property
    def zgrid(self):
        try:
            return self._src.values(1, masked=True)
        except AttributeError:
            return []

    @property
    def _hfun(self):
        return self.__hfun

    @_hfun.setter
    def _hfun(self, hfun):
        msg = 'hfun must be one of '
        msg += "geomesh.raster.Raster "
        assert isinstance(
            hfun,
            (
                # Geom,
                # geomesh.mesh.Mesh,
                # jigsaw_msh_t,
                geomesh.Raster,
                # RasterCollection,
                # float,
                # int
             )
        )

        if isinstance(hfun, geomesh.Raster):
            self._raster = hfun

        self.__hfun = hfun

    @property
    def _raster(self):
        return self.__raster

    @_raster.setter
    def _raster(self, raster):
        assert isinstance(raster, geomesh.Raster)
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=str(tmpdir.resolve())+'/hfun_'
            )
        kwargs = raster._src.meta.copy()
        nodata = np.finfo(rasterio.float32).max
        kwargs.update({
            "dtype": rasterio.float32,
            "nodata": nodata
            })
        with rasterio.open(tmpfile.name, 'w', **kwargs) as dst:
            dst.write_band(1, np.full(raster.shape, nodata))
        self._tmpfile = tmpfile
        self.__raster = raster

    @property
    def _src(self):
        try:
            return self.__src
        except AttributeError:
            pass

    @_src.setter
    def _src(self, tmpfile):
        self.__src = geomesh.Raster(tmpfile.name)

    @property
    def _geom(self):
        return self.__geom

    @_geom.setter
    def _geom(self, geom):
        if geom is not None:
            from geomesh.geom import Geom
            assert isinstance(geom, Geom)
        self.__geom = geom

    @property
    def _tmpfile(self):
        try:
            return self.__tmpfile
        except AttributeError:
            pass

    @_tmpfile.setter
    def _tmpfile(self, tmpfile):
        self._src = tmpfile
        self.__tmpfile = tmpfile

    @property
    @lru_cache(maxsize=None)
    def _logger(self):
        return logging.getLogger(__name__ + '.' + self.__class__.__name__)
