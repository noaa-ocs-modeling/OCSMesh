import matplotlib.pyplot as plt
from matplotlib import tri
# from matplotlib import Path
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import griddata
# from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
# from osgeo import ogr
# from jigsawpy import jigsaw_msh_t
# from pysheds.grid import Grid
# import tempfile
# from shapely import geometry
# import fiona
# import rasterio
# from geomesh.mesh import TriMesh
from geomesh.pslg import PlanarStraightLineGraph
# from geomesh.dataset_collection import DatasetCollection
# from geomesh import gdal_tools


class SizeFunction:

    def __init__(self, pslg, dst_crs="EPSG:3395"):
        self._pslg = pslg
        self._dst_crs = dst_crs

    def make_plot(self, show=False):
        plt.tricontourf(self.mpl_tri, self.values)
        plt.gca().axis('scaled')
        plt.colorbar()
        if show:
            plt.show()

    def add_contour(
        self,
        level,
        target_size,
        expansion_rate=0.2,
        hmin=None,
        hmax=None
    ):
        level = float(level)
        target_size = float(target_size)
        expansion_rate = float(expansion_rate)
        for i, raster in enumerate(self.raster_collection):
            ax = plt.contour(raster.x, raster.y, raster.values, levels=[level])
            plt.close(plt.gcf())
            xy = list()
            for path_collection in ax.collections:
                for path in path_collection.get_paths():
                    xy = [*xy, *path.vertices.tolist()]
            xt, yt = np.meshgrid(raster.x, raster.y)
            xt = xt.flatten()
            yt = yt.flatten()
            xy_target = np.vstack([xt, yt]).T
            tree = cKDTree(np.asarray(xy))
            values, _ = tree.query(xy_target, n_jobs=-1)
            values = expansion_rate*target_size*values + target_size
            values = values.reshape(raster.values.shape)
            if hmin is not None:
                values[np.where(values < hmin)] = hmin
            if hmax is not None:
                values[np.where(values > hmax)] = hmax
            raster.add_band("SIZE_FUNCTION", values)

    def add_subtidal_flow_limiter(self, hmin=None, hmax=None):
        """
        https://wiki.fvcom.pml.ac.uk/doku.php?id=configuration%3Agrid_scale_considerations
        """
        for i, raster in enumerate(self.raster_collection):
            dx = np.abs(raster.src.transform[0])
            dy = np.abs(raster.src.transform[4])
            dx, dy = np.gradient(raster.values, dx, dy)
            dh = np.sqrt(dx**2 + dy**2)
            dh = np.ma.masked_equal(dh, 0.)
            values = np.abs((1./3.)*(raster.values/dh))
            values = values.filled(np.max(values))
            if hmin is not None:
                values[np.where(values < hmin)] = hmin
            if hmax is not None:
                values[np.where(values > hmax)] = hmax
            raster.add_band("SIZE_FUNCTION", values)

    def _set_triangulation(self):
        vertices = np.empty((0, 2), float)
        minval = float("inf")
        for raster in self.raster_collection:
            band = np.full(raster.shape, float("inf"))
            for i in range(1, raster.count + 1):
                if raster.tags(i)['BAND_TYPE'] == "SIZE_FUNCTION":
                    band = np.minimum(band, raster.read(i))
            raster.add_band("SIZE_FUNCTION_FINALIZED", band)
            minval = np.min([minval, np.min(band)])
            band_id = raster.count
            target_res = np.min(band)
            if raster.dx < target_res or raster.dy < target_res:
                x, y, band = raster.resampled(band_id, target_res, target_res)
            else:
                x, y = raster.x, raster.y
            band = np.ma.masked_equal(
                band.astype(raster.dtype(band_id)),
                raster.nodataval(band_id))
            ax = plt.contour(x, y, band, levels=self.levels)
            plt.close(plt.gcf())
            for i, path_collection in enumerate(ax.collections):
                for path in path_collection.get_paths():
                    if ax.levels[i] not in [self.pslg.zmin, self.pslg.zmax]:
                        vertices = np.vstack([vertices, path.vertices])
        vertices = np.vstack([vertices, self.pslg.coords])
        mpl_tri = tri.Triangulation(vertices[:, 0], vertices[:, 1])
        # mask option 1,
        mask = np.full(
            (1, mpl_tri.triangles.shape[0]),
            raster.nodataval(band_id)).flatten()
        centroids = np.vstack(
            [np.sum(mpl_tri.x[mpl_tri.triangles], axis=1) / 3,
             np.sum(mpl_tri.y[mpl_tri.triangles], axis=1) / 3]).T
        values = np.full(
            (1, vertices.shape[0]),
            raster.nodataval(band_id)).flatten()
        for raster in self.raster_collection:
            raster.mask(self.pslg.multipolygon)
            bbox = raster.bbox
            f = RectBivariateSpline(
                raster.x,
                np.flip(raster.y),
                np.flipud(raster.read(band_id)).T,
                bbox=[bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax],
                kx=1, ky=1)
            idxs = np.where(np.logical_and(
                                np.logical_and(
                                    bbox.xmin <= centroids[:, 0],
                                    bbox.xmax >= centroids[:, 0]),
                                np.logical_and(
                                    bbox.ymin <= centroids[:, 1],
                                    bbox.ymax >= centroids[:, 1])))[0]
            mask[idxs] = f.ev(centroids[idxs, 0], centroids[idxs, 1])
            idxs = np.where(np.logical_and(
                                np.logical_and(
                                    bbox.xmin <= vertices[:, 0],
                                    bbox.xmax >= vertices[:, 0]),
                                np.logical_and(
                                    bbox.ymin <= vertices[:, 1],
                                    bbox.ymax >= vertices[:, 1])))[0]
            values[idxs] = f.ev(vertices[idxs, 0], vertices[idxs, 1])
        condition = values >= minval
        idx = np.where(condition)
        _idx = np.where(~condition)
        values[_idx] = griddata(
                    (mpl_tri.x[idx], mpl_tri.y[idx]),
                    values[idx],
                    (mpl_tri.x[_idx], mpl_tri.y[_idx]),
                    method='nearest')
        mask = np.ma.masked_equal(
            mask.astype(raster.dtype(band_id)),
            raster.nodataval(band_id)).mask
        mpl_tri = tri.Triangulation(
            vertices[:, 0], vertices[:, 1], triangles=mpl_tri.triangles[~mask])
        # plt.tripcolor(mpl_tri, values, vmin=50, vmax=250.)
        # plt.colorbar()
        # plt.triplot(mpl_tri, linewidth=0.07, color='k', alpha=0.5)
        # plt.gca().axis("scaled")
        # plt.show(block=False)
        # breakpoint()
        self.__mpl_tri = mpl_tri
        self.__values = values

    @property
    def pslg(self):
        return self._pslg

    @property
    def raster_collection(self):
        return self.pslg._raster_collection

    @property
    def schema(self):
        return self._schema

    @property
    def values(self):
        try:
            return self.__values
        except AttributeError:
            self._set_triangulation()
            return self.__values

    @property
    def scaling(self):
        return self._scaling

    @property
    def dst_crs(self):
        return self._dst_crs

    @property
    def levels(self):
        return self._levels

    @property
    def _pslg(self):
        return self.__pslg

    @property
    def _scaling(self):
        try:
            return self.__scaling
        except AttributeError:
            self._scaling = "absolute"
            return self.__scaling

    @property
    def _levels(self):
        try:
            return self.__levels
        except AttributeError:
            return 256

    @property
    def _dst_crs(self):
        return self.__dst_crs

    # @property
    # def _size_function_collection(self):
    #     try:
    #         return self.__size_function_collection
    #     except AttributeError:
    #         self.__size_function_collection = list()
    #         return self.__size_function_collection

    # @property
    # def _dataset(self):
    #     try:
    #         return self.__dataset
    #     except AttributeError:
    #         transform, width, height = warp.calculate_default_transform(
    #             src.crs, self.dst_crs, src.width, src.height, *src.bounds)
    #         kwargs = src.meta.copy()
    #         kwargs.update({
    #             'crs': self.dst_crs,
    #             'transform': transform,
    #             'width': width,
    #             'height': height
    #         })
    #         fname = self.tiff.name
    #         dst = rasterio.open(fname, 'w', **kwargs)
    #         for i in range(1, src.count + 1):
    #             rasterio.warp.reproject(
    #                 source=rasterio.band(src, i),
    #                 destination=rasterio.band(dst, i),
    #                 src_transform=src.transform,
    #                 src_crs=src.crs,
    #                 dst_transform=transform,
    #                 dst_crs=dst_crs,
    #                 # resampling=<Resampling.nearest: 0>,
    #                 num_threads=multiprocessing.cpu_count()-1,
    #                 )
    #         self._dataset = rasterio.open(fname)
    #         return self.__dataset

    # @property
    # def _tiff(self):
    #     try:
    #         return self.__tiff
    #     except AttributeError:
    #         self.__tiff = tempfile.NamedTemporaryFile()
    #         return self.__tiff

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = scaling

    @levels.setter
    def levels(self, levels):
        self._levels = levels

    @dst_crs.setter
    def dst_crs(self, dst_crs):
        self._dst_crs = dst_crs

    @_scaling.setter
    def _scaling(self, scaling):
        assert scaling in ["absolute", "relative"]
        self.__scaling = scaling

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        self.pslg.dst_crs = dst_crs
        self.__dst_crs = dst_crs

    @_pslg.setter
    def _pslg(self, pslg):
        assert isinstance(pslg, PlanarStraightLineGraph)
        self.__pslg = pslg

    @_levels.setter
    def _levels(self, levels):
        assert isinstance(levels, int)
        assert levels > 3
        self.__levels = levels
    # @_dataset.setter
    # def _dataset(self, dataset):
    #     self.__dataset = dataset



    



















    # def __iter__(self):
    #     for level, target_size, MultiLineString in self.features:
    #         yield {
    #             "level": level,
    #             "target_size": target_size,
    #             "MultiLineString": MultiLineString}

    # def add_watershed(self, target_size, pour_point, **kwargs):
    #     # default dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    #     # dirmap => (N, NE, E, SE, S, SW, W, NW)
    #     raise NotImplementedError
    #     for dem in self.pslg:
    #         grid = Grid.from_raster(dem.path, 'dem')
    #         grid.fill_depressions(data='dem', out_name='flooded_dem')
    #         grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
    #         dirmap = (360., 45., 90., 135., 180., 225., 270., 315.)
    #         grid.flowdir(
    #             data='inflated_dem',
    #             out_name='dir',
    #             dirmap=dirmap,
    #             routing='dinf',
    #             pits=0,
    #             flats=-1
    #             )
    #         plt.imshow(
    #             grid.view('dir'),
    #             interpolation='nearest',
    #             cmap='twilight_shifted',
    #             vmin=0.,
    #             vmax=360.
    #             )
    #         values = grid.view('dem')
    #         idx = np.where(values == np.max(values))
    #         grid.catchment(
    #             data='dir',
    #             x=idx[1][0],
    #             y=idx[0][0],
    #             out_name='catch',
    #             recursionlimit=15000,
    #             xytype='index'
    #             )
    #         grid.clip_to('catch')
    #         plt.imshow(grid.view('catch'))
    #         plt.show()
    #         plt.close(plt.gcf())
    #         grid.clip_to('dir')
    #         grid.accumulation(data='catch', out_name='acc')
    #         branches = grid.extract_river_network('catch', 'acc')
    #         for branch in branches['features']:
    #             line = np.asarray(branch['geometry']['coordinates'])
    #             plt.plot(line[:, 0], line[:, 1])
    #         plt.show()
    #     self._add_feature()


    # def make_plot(self, show=False):
    #     plt.tricontourf(self.mpl_tri, self.values)
    #     plt.gca().axis('scaled')
    #     plt.colorbar()
    #     if show:
    #         plt.show()

    # def _add_feature(self, target_size, MultiLineString):
    #     del(self._points)
    #     self._features.append((target_size, MultiLineString))

    # @property
    # def Dataset(self):
    #     return self._Dataset

    # @property
    # def planar_straight_line_graph(self):
    #     return self.pslg

    # @property
    # def pslg(self):
    #     return self._pslg

    # @property
    # def hmin(self):
    #     return self._hmin

    # @property
    # def hmax(self):
    #     return self._hmax

    # @property
    # def SpatialReference(self):
    #     return self._SpatialReference

    # @property
    # def mpl_tri(self):
    #     return self._mpl_tri

    # @property
    # def mpl_tri_mask(self):
    #     return self._mpl_tri_mask

    # @property
    # def x(self):
    #     return self._x

    # @property
    # def y(self):
    #     return self._y

    # @property
    # def xy(self):
    #     return self._xy

    # @property
    # def elements(self):
    #     return self._elements

    # @property
    # def values(self):
    #     return self._values

    # @property
    # def elevation(self):
    #     return self._elevation

    # @property
    # def points(self):
    #     return self._points

    # @property
    # def features(self):
    #     return tuple(self._features)


    # @property
    # def vert2(self):
    #     vert2 = list()
    #     for x, y, z in self.points:
    #         vert2.append(((x, y), 0))
    #     return np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)

    # @property
    # def tria3(self):
    #     tria3 = list()
    #     triangles = self.mpl_tri.triangles[np.where(~self.mpl_tri_mask)]
    #     mpl_tri = Triangulation(self.x, self.y, triangles=triangles)
    #     for indices in mpl_tri.triangles:
    #         tria3.append((tuple(indices), 0))
    #     return np.asarray(tria3, dtype=jigsaw_msh_t.TRIA3_t)

    # @property
    # def hfun_value(self):
    #     return np.asarray(self.values.tolist(), dtype=jigsaw_msh_t.REALS_t)

    # @property
    # def hfun(self):
    #     hfun = jigsaw_msh_t()
    #     hfun.vert2 = self.vert2
    #     hfun.tria3 = self.tria3
    #     hfun.value = self.hfun_value
    #     hfun.ndim = 2
    #     hfun.mshID = "euclidean-mesh"
    #     return hfun

    # @property
    # def _pslg(self):
    #     return self.__pslg

    # @property
    # def _hmin(self):
    #     return self.__hmin

    # @property
    # def _hmax(self):
    #     return self.__hmax

    # @property
    # def _x(self):
    #     return self.mpl_tri.x

    # @property
    # def _y(self):
    #     return self.mpl_tri.y

    # @property
    # def _xy(self):
    #     return np.vstack([self.x, self.y]).T

    # @property
    # def _elements(self):
    #     return self.mpl_tri.triangles

    # @property
    # def _Dataset(self):
    #     try:
    #         return self.__Dataset
    #     except AttributeError:
    #         directory = tempfile.mkdtemp()
    #         print(directory)
    #         BREAKME
    #         # def array2raster(
    #         #     newRasterfn,
    #         #     rasterOrigin,
    #         #     pixelWidth,
    #         #     pixelHeight,
    #         #     array
    #         # ):
    #         #     cols = array.shape[1]
    #         #     rows = array.shape[0]
    #         #     originX = rasterOrigin[0]
    #         #     originY = rasterOrigin[1]

    #         #     driver = gdal.GetDriverByName('GTiff')
    #         #     outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    #         #     outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    #         #     outband = outRaster.GetRasterBand(1)
    #         #     outband.WriteArray(array)
    #         #     outRasterSRS = osr.SpatialReference()
    #         #     outRasterSRS.ImportFromEPSG(4326)
    #         #     outRaster.SetProjection(outRasterSRS.ExportToWkt())
    #         #     outband.FlushCache()
    #         # gdal.Open()
    #         newRasterfn = directory + '/{}.tif'.format(uuid.uuid())
    #         driver = gdal.GetDriverByName('GTiff')

    #         outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    #         return self.__Dataset

    # @property
    # def _values(self):
    #     try:
    #         return self.__values
    #     except AttributeError:
    #         values = np.full((self.elevation.size,), np.nan)
    #         initial_i = self.pslg.values.size
    #         for target_size, _MultiLineString in self.features:
    #             final_i = initial_i
    #             for _LineString in _MultiLineString:
    #                 final_i += _LineString.GetPointCount()
    #             values[initial_i:final_i] = target_size
    #             initial_i = final_i
    #         idxs = np.where(np.isnan(values))
    #         not_idxs = np.where(~np.isnan(values))
    #         pad_values = griddata(
    #             (self.x[not_idxs], self.y[not_idxs]),
    #             values[not_idxs],
    #             (self.x[idxs], self.y[idxs]),
    #             method='nearest')
    #         values[idxs] = pad_values
    #         self._values = values
    #         return self.__values

    # @property
    # def _elevation(self):
    #     return self.points[:, 2]

    # @property
    # def _points(self):
    #     try:
    #         return self.__points
    #     except AttributeError:
    #         self._points = self.pslg.points
    #         return self.__points

    # @property
    # def _SpatialReference(self):
    #     return self.__SpatialReference

    # @property
    # def _mpl_tri(self):
    #     try:
    #         return self.__mpl_tri
    #     except AttributeError:
    #         self._mpl_tri = self.points
    #         self.__mpl_tri.set_mask(self.mpl_tri_mask)
    #         return self.__mpl_tri


    # @property
    # def _features(self):
    #     try:
    #         return self.__features
    #     except AttributeError:
    #         self._features = list()
    #         return self.__features



    # @property
    # def _hfun(self):
    #     try:
    #         return self.__hfun
    #     except AttributeError:
    #         self._hfun = jigsaw_msh_t()
    #         return self.__hfun



    # @_pslg.setter
    # def _pslg(self, pslg):
    #     assert isinstance(pslg, PlanarStraightLineGraph)
    #     self.__pslg = pslg

    # @_hmin.setter
    # def _hmin(self, hmin):
    #     if hmin is not None:
    #         hmin = float(hmin)
    #         assert hmin > 0. and hmin < float("inf")
    #     self.__hmin = hmin

    # @_features.setter
    # def _features(self, features):
    #     del(self._points)
    #     self.__features = features

    # @_hmax.setter
    # def _hmax(self, hmax):
    #     if hmax is not None:
    #         hmax = float(hmax)
    #         assert hmax > 0. and hmax < float("inf")
    #     self.__hmax = hmax

    # @_mpl_tri.setter
    # def _mpl_tri(self, points):
    #     self.__mpl_tri = Triangulation(points[:, 0], points[:, 1])

    # @_points.setter
    # def _points(self, points):
    #     for target_size, _MultiLineString in self.features:
    #         for _LineString in _MultiLineString:
    #             _points = np.asarray(_LineString.GetPoints())
    #             points = np.vstack([points, _points])
    #     self.__points = points

    # @_values.setter
    # def _values(self, values):
    #     del(self._points)
    #     self.__values = values

    # @_mpl_tri_mask.setter
    # def _mpl_tri_mask(self, mask):
    #     centroid_x = np.sum(
    #         self.mpl_tri.x[self.mpl_tri.triangles], axis=1) / 3
    #     centroid_y = np.sum(
    #         self.mpl_tri.y[self.mpl_tri.triangles], axis=1) / 3
    #     centroids = np.vstack([centroid_x, centroid_y]).T
    #     for _Polygon in self.pslg.MultiPolygon:
    #         for i, _LinearRing in enumerate(_Polygon):
    #             mpl_path = Path(
    #                 np.asarray(_LinearRing.GetPoints())[:, :2],
    #                 closed=True)
    #             if i == 0:
    #                 mask = np.logical_and(
    #                     mask, ~mpl_path.contains_points(centroids))
    #             else:
    #                 mask = np.logical_or(
    #                     mask, mpl_path.contains_points(centroids))
    #     self.__mpl_tri_mask = mask

    # @_SpatialReference.setter
    # def _SpatialReference(self, SpatialReference):
    #     SpatialReference = gdal_tools.sanitize_SpatialReference(
    #         SpatialReference)
    #     self.__SpatialReference = SpatialReference

    # @_points.deleter
    # def _points(self):
    #     try:
    #         del(self.__points)
    #         del(self._values)
    #         del(self._mpl_tri)
    #     except AttributeError:
    #         pass

    # @_values.deleter
    # def _values(self):
    #     try:
    #         del(self.__values)
    #     except AttributeError:
    #         pass

    # @_mpl_tri.deleter
    # def _mpl_tri(self):
    #     try:
    #         del(self.__mpl_tri)
    #         del(self._mpl_tri_mask)
    #     except AttributeError:
    #         pass

    # @_mpl_tri_mask.deleter
    # def _mpl_tri_mask(self):
    #     try:
    #         del(self.__mpl_tri_mask)
    #     except AttributeError:
    #         pass
