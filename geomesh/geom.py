import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import tempfile
import fiona
from functools import lru_cache
from pyproj import Proj, CRS, Transformer
import pathlib
from shapely.geometry import (
    MultiPolygon,
    Polygon,
    shape,
    mapping,
    LinearRing,
    box
    )
from shapely.ops import transform
from jigsawpy import jigsaw_msh_t, savemsh, loadmsh
from geomesh import tmpdir
from geomesh import mesh
from geomesh import utils
from geomesh.raster import Raster, RasterCollection


class Geom:

    def __init__(
        self,
        geom,
        zmin=None,
        zmax=None,
        crs=None,
        dst_crs=None,
        radii=None,
        unit='meter',
        clip=None,
        buffer=None,
    ):
        """
        Input parameters
        ----------------
        geom:
            Can be Raster, RasterCollection, jigsaw_msh_t, Polygon or
            MultiPolygon

        zmin:
            Used to clip geom to a minimum depth.

        zmax:
            Used to clip geom to a maximum depth.

        crs:
            Assigns CRS to geom, required for shapely object.
            Overrides the input geom crs.

        dst_crs:
            CRS of output geom

        clip:
            (DeprecationWarning) Used to clip input geom. Might be removed.

        radii:
            If applied, treats the input parameters as ellipsoidal

        unit:
            Units for radii, zmin and zmax

        """
        self._geom = geom
        self._zmin = zmin
        self._zmax = zmax
        self._crs = crs
        self._dst_crs = dst_crs
        self._radii = radii
        self._unit = unit
        self._clip = clip
        self._buffer = buffer

    def __add__(self, other):
        """
        Creates a new Geom by buffering the existing geom with another Geom
        instance.
        """
        polygon_collection = list()
        for polygon in self.multipolygon:
            polygon_collection.append(polygon)
        for polygon in other.multipolygon:
            polygon_collection.append(polygon)
        multipolygon = MultiPolygon(polygon_collection).buffer(0)
        assert isinstance(multipolygon, MultiPolygon)
        return Geom(multipolygon, crs=self.crs)

    def make_plot(
        self,
        ax=None,
        show=False,
    ):
        for polygon in self.multipolygon:
            plt.plot(*polygon.exterior.xy, color='k')
            for interior in polygon.interiors:
                plt.plot(*interior.xy, color='r')
        if show:
            plt.gca().axis('scaled')
            plt.show()
        return plt.gca()

    def triplot(
        self,
        show=False,
        linewidth=0.07,
        color='black',
        alpha=0.5,
        **kwargs
    ):
        plt.triplot(
            self.triangulation,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            **kwargs
            )
        if show:
            plt.axis('scaled')
            plt.show()

    @property
    def multipolygon(self):

        if isinstance(self._geom, RasterCollection):
            polygon_collection = list()
            for i in range(len(self.raster_collection)):
                multipolygon = self._get_raster_multipolygon(i)
                for polygon in multipolygon:
                    polygon_collection.append(polygon)
            multipolygon = MultiPolygon(polygon_collection).buffer(0)
            if isinstance(multipolygon, Polygon):
                multipolygon = MultiPolygon([multipolygon])

        elif isinstance(self._geom, Raster):
            multipolygon = self._get_raster_multipolygon(0)

        elif isinstance(self._geom, MultiPolygon):
            multipolygon = self._geom

        else:
            msg = 'ERROR: Must Implement multipolygon attribute for geom type '
            msg += f'{type(self._geom)}.'
            raise NotImplementedError(msg)
        if self.crs.srs != self._dst_crs.srs:
            multipolygon = self._transform_multipolygon(
                multipolygon, self._crs, self._dst_crs)

        if self._clip is not None:
            multipolygon = self._clip.intersection(multipolygon)

        if self._buffer is not None:
            polygon_collection = list()
            for polygon in multipolygon:
                polygon_collection.append(polygon)
            for multipolygon in self._buffer:
                for polygon in multipolygon:
                    polygon_collection.append(polygon)
            multipolygon = MultiPolygon(polygon_collection).buffer(0)
        if isinstance(multipolygon, Polygon):
            multipolygon = MultiPolygon([multipolygon])
        return multipolygon

    @property
    def geom(self):
        """
        This is the global geom
        """
        try:
            name = self.__geom_tmpfile.name
            geom = jigsaw_msh_t()
            loadmsh(name, geom)
            return geom

        except AttributeError:

            # jigsawpy.msh_t.jigsaw_msh_t
            if isinstance(self._geom, jigsaw_msh_t):
                self._logger.debug('self.geom:AttributeError:jigsaw_msh_t')
                self._geom.edge2 = utils.edge2_from_msh_t(self._geom)
                geom = self._geom

            elif isinstance(self._geom, (Polygon, MultiPolygon)):
                msg = 'self.geom:AttributeError:(Polygon, MultiPolygon)'
                self._logger.debug(msg)
                geom = self._get_geom_from_shapely()

            # RasterCollection
            elif isinstance(self._geom, (Raster, RasterCollection)):
                msg = 'self.geom:AttributeError:(Raster, RasterCollection)'
                self._logger.debug(msg)
                geom = self._get_geom_from_raster()

            else:
                self._logger.debug('self.geom:AttributeError:Undefined')
                msg = f"Undefined handler for geom type {self._geom}"
                raise NotImplementedError(msg)

            tmpfile = tempfile.NamedTemporaryFile(
                prefix=tmpdir, suffix='.msh')
            savemsh(tmpfile.name, geom)
            self.__geom_tmpfile = tmpfile

            if self._radii is not None:
                geom.radii = self._radii
                geom.mshID = 'ellipsoidal-mesh'
            return geom

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    @property
    def crs(self):
        return CRS.from_user_input(self._crs)

    @property
    def proj(self):
        return Proj(self.crs)

    @property
    def srs(self):
        return self.proj.srs

    @property
    def ndims(self):
        return self.geom.ndims

    @property
    @lru_cache
    def raster_collection(self):
        if isinstance(self._geom, RasterCollection):
            return self._geom

        elif isinstance(self._geom, Raster):
            return [self._geom]

    @property
    def _logger(self):
        try:
            return self.__logger
        except AttributeError:
            self.__logger = logging.getLogger(
                __name__ + '.' + self.__class__.__name__)
            return self.__logger

    def _get_raster_multipolygon(self, id):

        feature = None
        with fiona.open(self._multipolygon_tmpfile, 'r') as shp:
            for _feature in shp:
                if _feature['id'] == id:
                    feature = _feature
                    break

        if feature is None:
            multipolygon = self._generate_raster_multipolygon(id)
            self._save_raster_multipolygon(id, multipolygon)

        else:
            multipolygon = shape(feature['geometry'])
            if isinstance(multipolygon, Polygon):
                multipolygon = MultiPolygon([multipolygon])

        return multipolygon

    def _generate_raster_multipolygon(self, id):
        raster = self.raster_collection[id]

        # bounding box based
        if self.zmin is None and self.zmax is None:
            # check if it is masked
            if raster.read_masks(1).any():
                # get multipolygon from mask
                mask = raster.read_masks(1)
                values = raster.values.copy()
                values[np.where(~mask)] = -1
                values[np.where(mask)] = 1
                ax = plt.contourf(raster.x, raster.y, values, levels=[0, 1])
                plt.close(plt.gcf())
                multipolygon = self._get_multipolygon_from_axes(ax)
            # full bbox
            else:
                msg = f'_generate_raster_multipolygon({id})'
                msg += ':Input is full bbox.'
                self._logger.debug(msg)
                x0 = raster.bbox.xmin
                x1 = raster.bbox.xmax
                y0 = raster.bbox.ymin
                y1 = raster.bbox.ymax
                multipolygon = MultiPolygon([box(x0, y0, x1, y1)])
        # contour based
        else:
            msg = f'_generate_raster_multipolygon({id}) Computing contours.'
            self._logger.debug(msg)
            zmin = np.min(raster.values) if self.zmin is None else self.zmin
            zmax = np.max(raster.values) if self.zmax is None else self.zmax
            ax = plt.contourf(
                raster.x, raster.y, raster.values, levels=[zmin, zmax])
            plt.close(plt.gcf())
            multipolygon = self._get_multipolygon_from_axes(ax)

        if self._clip is not None:
            multipolygon = self._clip.intersection(multipolygon)
            if isinstance(multipolygon, Polygon):
                multipolygon = MultiPolygon([multipolygon])
            msg = f"Clipping returned a {type(multipolygon)} "
            msg += f"instead of the expected {MultiPolygon}. "
            msg += "This usually means the polygons are at different CRS."
            assert isinstance(multipolygon, MultiPolygon), msg

        if self._buffer is not None:
            polygon_collection = list()
            for polygon in multipolygon:
                polygon_collection.append(polygon)
            for multipolygon in self._buffer:
                for polygon in multipolygon:
                    polygon_collection.append(polygon)
            multipolygon = MultiPolygon(polygon_collection).buffer(0)
            if isinstance(multipolygon, Polygon):
                multipolygon = MultiPolygon([multipolygon])
            msg = f"Buffer returned a {type(multipolygon)} "
            msg += f"instead of the expected {MultiPolygon}. "
            msg += "This usually means the polygons are at different CRS."
            assert isinstance(multipolygon, MultiPolygon), msg
        return multipolygon

    def _save_raster_multipolygon(self, id, multipolygon):
        with fiona.open(self._multipolygon_tmpfile, 'a') as shp:
            shp.write({
                    "geometry": mapping(multipolygon),
                    "id": id,
                    "properties": {
                        "zmin": self.zmin,
                        "zmax": self.zmax}})

    def _get_raster_geom(self, idx):
        geom = self._geom_tmpfile_ptrs[idx]
        if geom is None:
            geom = self._generate_raster_geom(idx)
        else:
            geom = self._load_raster_geom(idx)
        return geom

    def _generate_raster_geom(self, idx):
        multipolygon = self._get_raster_multipolygon(idx)
        geom = utils.multipolygon_to_geom(multipolygon)
        self._save_raster_geom(idx, geom)
        return geom

    def _save_raster_geom(self, idx, geom):
        tmpfile = tempfile.NamedTemporaryFile(
            prefix=tmpdir, suffix='.msh')
        savemsh(tmpfile.name, geom)
        self._geom_tmpfile_ptrs[idx] = tmpfile

    def _load_raster_geom(self, idx):
        geom = jigsaw_msh_t()
        loadmsh(self._geom_tmpfile_ptrs[idx].name, geom)
        return geom

    # def _clip_multipolygon(self, multipolygon):
    #     _multipolygon = multipolygon.intersection(self.clip)
    #     if isinstance(multipolygon, Polygon):
    #         _multipolygon = MultiPolygon([multipolygon])
    #     if isinstance(multipolygon, GeometryCollection):
    #         return multipolygon
    #     return _multipolygon

    @staticmethod
    def _transform_multipolygon(multipolygon, src_crs, dst_crs):
        if isinstance(src_crs, str):
            src_crs = CRS.from_user_input(src_crs)
        if isinstance(dst_crs, str):
            dst_crs = CRS.from_user_input(dst_crs)
        if dst_crs.srs != src_crs.srs:
            transformer = Transformer.from_crs(
                src_crs, dst_crs, always_xy=True)
            polygon_collection = list()
            for polygon in multipolygon:
                polygon_collection.append(
                    transform(transformer.transform, polygon))
            outer = polygon_collection.pop(0)
            multipolygon = MultiPolygon([outer, *polygon_collection])
        return multipolygon

    @staticmethod
    def _transform_polygon(polygon, src_crs, dst_crs):
        if isinstance(src_crs, str):
            src_crs = CRS.from_user_input(src_crs)
        if isinstance(dst_crs, str):
            dst_crs = CRS.from_user_input(dst_crs)
        if dst_crs.srs != src_crs.srs:
            transformer = Transformer.from_crs(
                src_crs, dst_crs, always_xy=True)
            polygon = transform(transformer.transform, polygon)
        return polygon

    def _get_geom_from_shapely(self):
        # shapely.geometry.Polygon
        # if isinstance(self._geom, Polygon):
        #     multipolygon = self._clip_multipolygon(
        #         MultiPolygon([self._geom]))

        # shapely.geometry.MultiPolygon
        if isinstance(self._geom, MultiPolygon):
            multipolygon = self._geom

        if self.crs.srs != self._dst_crs.srs:
            multipolygon = self._transform_multipolygon(
                multipolygon, self._crs, self._dst_crs)
        geom = utils.multipolygon_to_geom(multipolygon)
        return geom

    def _get_geom_from_raster(self):

        if isinstance(self._geom, RasterCollection):
            self._logger.debug("_get_geom_from_raster():RasterCollection")
            polygon_collection = list()
            for idx in range(len(self._geom)):
                multipolygon = self._get_raster_multipolygon(idx)
                for polygon in multipolygon:
                    polygon_collection.append(polygon)
            multipolygon = MultiPolygon(polygon_collection).buffer(0)

        elif isinstance(self._geom, Raster):
            self._logger.debug("_get_geom_from_raster():Raster")
            multipolygon = self._get_raster_multipolygon(0)

        if isinstance(multipolygon, Polygon):
            msg = "_get_geom_from_raster(): Polygon to multipolygon typecast"
            self._logger.debug(msg)
            multipolygon = MultiPolygon([multipolygon])

        return utils.multipolygon_to_geom(multipolygon)

    def _get_multipolygon_from_geom(self, geom):
        """
        geom can be jigsaw_msh_t, Polygon, MultiPolygon or Mesh
        """
        pass

    def _get_multipolygon_from_axes(self, ax):
        # extract linear_rings from plot
        linear_ring_collection = list()
        for path_collection in ax.collections:
            for path in path_collection.get_paths():
                polygons = path.to_polygons(closed_only=True)
                for linear_ring in polygons:
                    if linear_ring.shape[0] > 3:
                        linear_ring_collection.append(
                            LinearRing(linear_ring))
        if len(linear_ring_collection) > 1:
            # reorder linear rings from above
            areas = [Polygon(linear_ring).area
                     for linear_ring in linear_ring_collection]
            idx = np.where(areas == np.max(areas))[0][0]
            polygon_collection = list()
            outer_ring = linear_ring_collection.pop(idx)
            path = Path(np.asarray(outer_ring.coords), closed=True)
            while len(linear_ring_collection) > 0:
                inner_rings = list()
                for i, linear_ring in reversed(
                        list(enumerate(linear_ring_collection))):
                    xy = np.asarray(linear_ring.coords)[0, :]
                    if path.contains_point(xy):
                        inner_rings.append(linear_ring_collection.pop(i))
                polygon_collection.append(Polygon(outer_ring, inner_rings))
                if len(linear_ring_collection) > 0:
                    areas = [Polygon(linear_ring).area
                             for linear_ring in linear_ring_collection]
                    idx = np.where(areas == np.max(areas))[0][0]
                    outer_ring = linear_ring_collection.pop(idx)
                    path = Path(np.asarray(outer_ring.coords), closed=True)
            multipolygon = MultiPolygon(polygon_collection)
        else:
            multipolygon = MultiPolygon(
                [Polygon(linear_ring_collection.pop())])
        return multipolygon

    @property
    def _geom(self):
        return self.__geom

    @property
    def _dst_crs(self):
        return self.__dst_crs

    # @property
    # def _crs(self):
    #     return self.__crs

    @property
    def _zmin(self):
        return self.__zmin

    @property
    def _zmax(self):
        return self.__zmax

    @property
    def _clip(self):
        return self.__clip

    @property
    def _buffer(self):
        return self.__buffer

    @property
    def _radii(self):
        return self.__radii

    @property
    @lru_cache
    def _geom_tmpfile_ptrs(self):

        if isinstance(self._geom, RasterCollection):
            return len(self.raster_collection)*[None]

        elif isinstance(self._geom, RasterCollection):
            return [None]

    @property
    @lru_cache
    def _multipolygon_tmpfile(self):
        self.__multipolygon_tmpdir = tempfile.TemporaryDirectory(
            prefix=tmpdir, suffix='_multipolygon')
        with fiona.open(
            pathlib.Path(self.__multipolygon_tmpdir.name).resolve(),
            'w',
            driver='ESRI Shapefile',
            crs=self._dst_crs.srs,
            schema={
                'geometry': 'MultiPolygon',
                'id': 'int',
                'properties': {
                    'zmin': 'float',
                    'zmax': 'float'}}) as _:
            pass
        _tmpdir = pathlib.Path(self.__multipolygon_tmpdir.name)
        return _tmpdir / f'{_tmpdir.name}.shp'

    @_geom.setter
    def _geom(self, geom):
        types = (jigsaw_msh_t,
                 mesh.mesh.Mesh,
                 Polygon,
                 MultiPolygon,
                 Raster,
                 RasterCollection)
        msg = f"ERROR: Input must be of one types: {types}"
        assert isinstance(geom, types), msg
        # cast Polygon input to MultiPolygon
        if isinstance(geom, Polygon):
            geom = MultiPolygon([geom])
        self.__geom = geom

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):

        if dst_crs is not None:
            dst_crs = CRS.from_user_input(dst_crs)

            if isinstance(self._geom, RasterCollection):
                self._geom.dst_crs = dst_crs

            elif isinstance(self._geom, Raster):
                self._geom.warp(dst_crs)

        else:

            if isinstance(self._geom, RasterCollection):
                dst_crs = self._geom.dst_crs

            elif isinstance(self._geom, Raster):
                dst_crs = self._geom.crs

        self.__dst_crs = dst_crs

    @_clip.setter
    def _clip(self, clip):
        if clip is not None:
            assert isinstance(clip, (Polygon, MultiPolygon))
            clip = self._transform_multipolygon(clip, self.crs, self._dst_crs)
        self.__clip = clip

    @_buffer.setter
    def _buffer(self, buffer):
        if buffer is not None:
            assert isinstance(buffer, (Polygon, MultiPolygon))
            buffer = self._transform_multipolygon(
                    buffer, self.crs, self._dst_crs)
        self.__buffer = buffer

    @_zmin.setter
    def _zmin(self, zmin):
        if zmin is not None:
            assert isinstance(zmin, (int, float))
        self.__zmin = zmin

    @_zmax.setter
    def _zmax(self, zmax):
        if zmax is not None:
            assert isinstance(zmax, (int, float))
        self.__zmax = zmax

    @_radii.setter
    def _radii(self, radii):
        if radii is not None:
            radii = np.array(radii, dtype=jigsaw_msh_t.REALS_t)
            msg = "radii must be of shape 3"
            assert radii.shape[0] == 3, msg
        self.__radii = radii
