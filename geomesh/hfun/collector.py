import os
import gc
import logging
import warnings
import tempfile
import numpy as np
from functools import reduce
from pathlib import Path
from time import time
from multiprocessing import Pool, cpu_count
from copy import copy, deepcopy
from typing import Union, Sequence, List

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection, box
from shapely import ops
from jigsawpy import jigsaw_msh_t
from rasterio.transform import from_origin, Affine
from rasterio.warp import reproject, Resampling
import rasterio
import utm

from geomesh import utils
from geomesh.hfun.base import BaseHfun
from geomesh.hfun.raster import HfunRaster
from geomesh.hfun.mesh import HfunMesh
from geomesh.mesh.mesh import Mesh
from geomesh.raster import Raster
from geomesh.features.contour import Contour
from geomesh.features.patch import Patch

_logger = logging.getLogger(__name__)

class RefinementContourInfoCollector:

    def __init__(self):
        self._contours_info = dict()

    def add(self, contour_defn, **size_info):
        self._contours_info[contour_defn] = size_info

    def __iter__(self):
        for defn, info in self._contours_info.items():
            yield defn, info




class RefinementContourCollector:

    def __init__(self, contours_info):
        self._contours_info = contours_info
        self._container: List[Union[Tuple, None]] = []

    def calculate(self, source_list, out_path):

        out_dir = Path(out_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        file_counter = 0
        pid = os.getpid()
        for contour_defn, size_info in self._contours_info:
            if not contour_defn.has_source:
                # Copy so that in case of a 2nd run the no-source 
                # contour still gets all current sources
                contour_defn = copy(contour_defn)
                for source in source_list:
                    contour_defn.add_source(source)

            for contour, crs in contour_defn.iter_contours():
                file_counter = file_counter + 1
                feather_path = out_dir / f"contour_{pid}_{file_counter}.feather"
                crs_path = out_dir / f"crs_{pid}_{file_counter}.json"
                gpd.GeoDataFrame(
                    { 'geometry': [contour],
                      'expansion_rate': size_info['expansion_rate'],
                      'target_size': size_info['target_size'],
                    },
                    crs=crs).to_feather(feather_path)
                gc.collect()
                with open(crs_path, 'w') as fp:
                    fp.write(crs.to_json())
                self._container.append((feather_path, crs_path))


    def __iter__(self):
        for raster_data in self._container:
            feather_path, crs_path = raster_data
            gdf = gpd.read_feather(feather_path)
            with open(crs_path) as fp:
                gdf.set_crs(CRS.from_json(fp.read()))
            yield gdf




class ConstantValueContourInfoCollector:

    def __init__(self):
        self._contours_info = dict()

    def add(self, src_idx, contour_defn0, contour_defn1, value):
        srcs = tuple(src_idx) if src_idx is not None else None
        self._contours_info[
                (srcs, contour_defn0, contour_defn1)] = value 

    def __iter__(self):
        for defn, info in self._contours_info.items():
            yield defn, info



class RefinementPatchInfoCollector:

    def __init__(self):
        self._patch_info = dict()

    def add(self, patch_defn, **size_info):
        self._patch_info[patch_defn] = size_info

    def __iter__(self):
        for defn, info in self._patch_info.items():
            yield defn, info



class FlowLimiterInfoCollector:

    def __init__(self):
        self._flow_lim_info = list()

    def add(self, src_idx, hmin, hmax, upper_bound, lower_bound):

        srcs = tuple(src_idx) if src_idx is not None else None
        self._flow_lim_info.append(
                (src_idx, hmin, hmax, upper_bound, lower_bound))

    def __iter__(self):

        for src_idx, hmin, hmax, ub, lb in self._flow_lim_info:
            yield src_idx, hmin, hmax, ub, lb




class HfunCollector(BaseHfun):

    def __init__(
            self,
            in_list: Sequence[
                Union[str, Raster, Mesh, HfunRaster, HfunMesh]],
            base_mesh: Mesh = None,
            hmin: float = None,
            hmax: float = None,
            nprocs: int = None,
            verbosity: int = 0,
            method: str = 'exact'
            ):

        # NOTE: Input Hfuns and their Rasters can get modified

        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs

        self._applied = False
        self._size_info = dict(hmin=hmin, hmax=hmax)
        self._nprocs = nprocs
        self._hfun_list = list()
        self._method = method
        # NOTE: Base mesh has to have a crs otherwise HfunMesh throws
        # exception
        self._base_mesh = None
        if base_mesh:
            self._base_mesh = HfunMesh(base_mesh)
        self._contour_info_coll = RefinementContourInfoCollector()
        self._contour_coll = RefinementContourCollector(
                self._contour_info_coll)
        self._const_val_contour_coll = ConstantValueContourInfoCollector()
        self._refine_patch_info_coll = RefinementPatchInfoCollector()
        self._flow_lim_coll = FlowLimiterInfoCollector()

        self._type_chk(in_list)

        # TODO: Interpolate max size on base mesh basemesh?
        #
        # TODO: CRS considerations
        
        for in_item in in_list:
            # Add supports(ext) to each hfun type?

            if isinstance(in_item, BaseHfun):
                hfun = in_item

            elif isinstance(in_item, Raster):
                if self._base_mesh:
                    in_item.clip(self._base_mesh.mesh.get_bbox(crs=in_item.crs))
                hfun = HfunRaster(in_item, **self._size_info)

            elif isinstance(in_item, EuclideanMesh2D):
                hfun = HfunMesh(in_item)

            elif isinstance(in_item, str):
                if in_item.endswith('.tif'):
                    raster = Raster(in_item)
                    if self._base_mesh:
                        raster.clip(self._base_mesh.mesh.get_bbox(crs=raster.crs))
                    hfun = HfunRaster(raster, **self._size_info)

                elif in_item.endswith(
                        ('.14', '.grd', '.gr3', '.msh', '.2dm')):
                    mesh = Mesh.open(path)
                    hfun = HfunMesh(mesh)

                else:
                    raise TypeError("Input file extension not supported!")

            self._hfun_list.append(hfun)


    def msh_t(self) -> jigsaw_msh_t:

        composite_hfun = jigsaw_msh_t()

        if self._method == 'exact':
            self._apply_features()

            with tempfile.TemporaryDirectory() as temp_dir:
                hfun_path_list = self._write_hfun_to_disk(temp_dir)
                composite_hfun = self._get_hfun_composite(hfun_path_list)
        

        elif self._method == 'fast':

            with tempfile.TemporaryDirectory() as temp_dir:
                rast = self._create_big_raster(temp_dir)
                hfun = self._apply_features_fast(rast)
                composite_hfun = self._get_hfun_composite_fast(hfun)

        else:
            raise ValueError(f"Invalid method specified: {self._method}")

        return composite_hfun


    def add_contour(
            self,
            level: Union[List[float], float] = None,
            expansion_rate: float = 0.01,
            target_size: float = None,
            contour_defn: Contour = None,
    ):
        # Always lazy
        self._applied = False

        levels = list()
        if isinstance(level, (list, tuple)):
            levels.extend(level)
        else:
            levels.append(level)


        contour_defns = list()
        if contour_defn == None:
            for level in levels:
                contour_defns.append(Contour(level=level))

        elif not isinstance(contour_defn, Contour):
            raise TypeError(
                f"Contour definition must be of type {Contour} not"
                f" {type(contour_defn)}!")

        elif level != None:
            msg = "Level is ignored since a contour definition is provided!"
            warnings.warn(msg)
            _logger.info(msg)

        else:
            contour_defns.append(contour_defn)

        for contour_defn in contour_defns:
            self._contour_info_coll.add(
                contour_defn, 
                expansion_rate=expansion_rate,
                target_size=target_size)

    def add_subtidal_flow_limiter(
            self,
            hmin=None,
            hmax=None,
            upper_bound=None,
            lower_bound=None,
            source_index: Union[List[int], int, None] = None):

        self._applied = False

        if source_index != None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]

        # TODO: Checks on hmin/hmax, etc?
            
        self._flow_lim_coll.add(
            source_index,
            hmin=hmin,
            hmax=hmin,
            upper_bound=upper_bound,
            lower_bound=lower_bound)


    def add_constant_value(
            self, value,
            lower_bound=None,
            upper_bound=None,
            source_index: Union[List[int], int, None] =None):


        self._applied = False

        contour_defn0 = None
        contour_defn1 = None
        if lower_bound != None and not np.isinf(lower_bound):
            contour_defn0 = Contour(level=lower_bound)
        if upper_bound != None and not np.isinf(upper_bound):
            contour_defn1 = Contour(level=upper_bound)

        if source_index != None and not isinstance(source_index, (tuple, list)):
            source_index = [source_index]
        self._const_val_contour_coll.add(
            source_index, contour_defn0, contour_defn1, value)


    def add_patch(
            self,
            shape: Union[MultiPolygon, Polygon] = None,
            patch_defn: Patch = None,
            shapefile: Union[None, str, Path] = None,
            expansion_rate: float = None,
            target_size: float = None,
    ):

        self._applied = False

        if not patch_defn:
            if shape:
                patch_defn = Patch(shape=shape)

            elif shapefile:
                patch_defn = Patch(shapefile=shapefile)

        self._refine_patch_info_coll.add(
            patch_defn,
            expansion_rate=expansion_rate,
            target_size=target_size)


    def _type_chk(self, input_list):
        ''' Check the input type for constructor '''
        valid_types = (str, Raster, Mesh, HfunRaster, HfunMesh)
        if not all(isinstance(item, valid_types) for item in input_list):
            raise TypeError(
                f'Input list items must be of type' 
                f' {", ".join(str(i) for i in valid_types)},'
                f' or a derived type.')

    def _apply_features(self):

        if not self._applied:
            self._apply_contours()
            self._apply_flow_limiters()
            self._apply_const_val()
            self._apply_patch()

        self._applied = True

    def _apply_contours(self, apply_to=None):

        # TODO: Consider CRS before applying to different hfuns
        #
        # TODO: Can add_feature be added to non-raster hfun?

        # NOTE: for parallelization make sure a single hfun is NOT
        # passed to multiple processes

        contourable_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        if apply_to is None:
            apply_to = contourable_list

        with tempfile.TemporaryDirectory() as temp_path:
            with Pool(processes=self._nprocs) as p:
                self._contour_coll.calculate(contourable_list, temp_path)
                counter = 0
                for hfun in apply_to:
                    for gdf in self._contour_coll:
                        for row in gdf.itertuples():
                            _logger.debug(row)
                            shape = row.geometry
                            if isinstance(shape, GeometryCollection):
                                continue
                            # NOTE: CRS check is done AFTER
                            # GeometryCollection check because
                            # gdf.to_crs results in an error in case
                            # of empty GeometryCollection
                            if not gdf.crs.equals(hfun.crs):
                                _logger.info(f"Reprojecting feature...")
                                transformer = Transformer.from_crs(
                                    gdf.crs, hfun.crs, always_xy=True)
                                shape = ops.transform(
                                        transformer.transform, shape)
                            counter = counter + 1
                            hfun.add_feature(**{
                                'feature': shape,
                                'expansion_rate': row.expansion_rate,
                                'target_size': row.target_size,
                                'proc_pool': p
                            })
            p.join()
            # hfun objects cause issue with pickling
            # -> cannot be passed to pool
#            with Pool(processes=self._nprocs) as p:
#                p.starmap(
#                    _apply_contours_worker,
#                    [(hfun, self._contour_coll, self._nprocs)
#                     for hfun in apply_to])

    def _apply_flow_limiters(self):

        if self._method == 'fast':
            raise NotImplementedError(
                "This function does not suuport fast hfun method")

        contourable_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        for in_idx, hfun in enumerate(contourable_list):
            for src_idx, hmin, hmax, zmax, zmin in self._flow_lim_coll:
                if src_idx != None and in_idx not in src_idx:
                    continue
                if hmin is None:
                    hmin = self._size_info['hmin']
                if hmax is None:
                    hmax = self._size_info['hmax']
                hfun.add_subtidal_flow_limiter(hmin, hmax, zmax, zmin)


    def _apply_const_val(self):

        if self._method == 'fast':
            raise NotImplementedError(
                "This function does not suuport fast hfun method")

        contourable_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        for in_idx, hfun in enumerate(contourable_list):
            for (src_idx, ctr0, ctr1), const_val in self._const_val_contour_coll:
                if src_idx != None and in_idx not in src_idx:
                    continue
                level0 = None
                level1 =  None
                if ctr0 != None:
                    level0 = ctr0.level
                if ctr1 != None:
                    level1 = ctr1.level
                hfun.add_constant_value(const_val, level0, level1)


    def _apply_patch(self, apply_to=None):

        contourable_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        if apply_to is None:
            apply_to = contourable_list

        # TODO: Parallelize
        for hfun in apply_to:
            for patch_defn, size_info in self._refine_patch_info_coll:
                shape, crs = patch_defn.get_multipolygon()
                if hfun.crs != crs:
                    transformer = Transformer.from_crs(
                        crs, hfun.crs, always_xy=True)
                    shape = ops.transform(
                            transformer.transform, shape)

                hfun.add_patch(
                        shape, nprocs=self._nprocs, **size_info)


    def _write_hfun_to_disk(self, out_path):

        out_dir = Path(out_path)
        path_list = list()
        file_counter = 0
        pid = os.getpid()
        bbox_list = list()

        # TODO: Ask for user input to consider size from mesh?
        hfun_list = self._hfun_list[::-1]
        if self._base_mesh:
            self._base_mesh.size_from_mesh()
            hfun_list = [*self._hfun_list[::-1], self._base_mesh]

        # Last user input item has the highest priority (its trias
        # are not dropped) so process in reverse order
        for hfun in hfun_list:
            # TODO: Calling msh_t() on HfunMesh more than once causes
            # issue right now due to change in crs of internal Mesh

            # To avoid removing verts and trias from mesh hfuns
            hfun_mesh = deepcopy(hfun.msh_t())
            # If no CRS info, we assume EPSG:4326
            if hasattr(hfun_mesh, "crs"):
                dst_crs = CRS.from_user_input("EPSG:4326")
                if hfun_mesh.crs != dst_crs:
                    utils.reproject(hfun_mesh, dst_crs)

            # Get all previous bbox and clip to resolve overlaps
            # removing all tria that have NODE in bbox because it's
            # faster and so we can resolve all overlaps
            _logger.info(f"Removing bounds from hfun mesh...")
            for ibox in bbox_list:
                hfun_mesh = utils.clip_mesh_by_shape(
                    hfun_mesh,
                    ibox,
                    use_box_only=True,
                    fit_inside=True,
                    inverse=True)

            if not len(hfun_mesh.vert2):
                _logger.debug("Hfun ignored due to overlap")
                continue

            # Check hfun_mesh.value against hmin & hmax
            hmin = self._size_info['hmin']
            hmax = self._size_info['hmax']
            if hmin:
                hfun_mesh.value[hfun_mesh.value < hmin] = hmin
            if hmax:
                hfun_mesh.value[hfun_mesh.value > hmax] = hmax

            mesh = Mesh(hfun_mesh)
            bbox_list.append(mesh.get_bbox(crs="EPSG:4326"))
            file_counter = file_counter + 1
            _logger.info(f'write mesh {file_counter} to file...')
            file_path = out_dir / f'hfun_{pid}_{file_counter}.2dm'
            mesh.write(file_path, format='2dm')
            path_list.append(file_path)
            _logger.info('Done writing 2dm file.')
            del mesh
            gc.collect()
        return path_list

                        

    def _get_hfun_composite(self, hfun_path_list):

        collection = []
        _logger.info('Reading 2dm hfun files...')
        start = time()
        for path in hfun_path_list:
            collection.append(Mesh.open(path, crs='EPSG:4326'))
        _logger.info(f'Reading 2dm hfun files took {time()-start}.')

        # NOTE: Overlaps are taken care of in the write stage

        coord = list()
        index = list()
        value = list()
        offset = 0
        for hfun in collection:
            index.append(hfun.tria3['index'] + offset)
            coord.append(hfun.coord)
            value.append(hfun.value)
            offset += hfun.coord.shape[0]

        composite_hfun = jigsaw_msh_t()
        composite_hfun.mshID = 'euclidean-mesh'
        composite_hfun.ndims = 2

        composite_hfun.vert2 = np.array(
                [(coord, 0) for coord in np.vstack(coord)],
                dtype=jigsaw_msh_t.VERT2_t)
        composite_hfun.tria3 = np.array(
                [(index, 0) for index in np.vstack(index)],
                dtype=jigsaw_msh_t.TRIA3_t)
        composite_hfun.value = np.array(
                np.vstack(value),
                dtype=jigsaw_msh_t.REALS_t)

        composite_hfun.crs = CRS.from_user_input("EPSG:4326")

        # NOTE: In the end we need to return in a CRS that
        # uses meters as units. UTM based on the center of
        # the bounding box of the hfun is used
        # Up until now all calculation was in EPSG:4326
        utils.msh_t_to_utm(composite_hfun)

        return composite_hfun


    def _create_big_raster(self, out_path):

        out_dir = Path(out_path)
        out_rast = out_dir / 'big_raster.tif'

        rast_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        all_bounds = list()
        for hfun_in in rast_hfun_list:
            all_bounds.append(hfun_in.get_bbox(crs='EPSG:4326').bounds)
        all_bounds = np.array(all_bounds)

        x0, y0 = np.min(all_bounds[:, [0, 1]], axis=0)
        x1, y1 = np.max(all_bounds[:, [2, 3]], axis=0)

        _, _, number, letter = utm.from_latlon(
                (y0 + y1)/2, (x0 + x1)/2)
        utm_crs = CRS(
            proj='utm', zone=f'{number}{letter}', ellps='WGS84')
        transformer = Transformer.from_crs(
                'EPSG:4326', utm_crs, always_xy=True)

        box_epsg4326 = box(x0, y0, x1, y1)
        poly_utm = ops.transform(transformer.transform, Polygon(box_epsg4326))
        x0, y0, x1, y1 = poly_utm.bounds

        # TODO: What if no hmin? -> use smallest raster res!
        g_hmin = self._size_info['hmin']
        res = g_hmin / 2
        shape0 = int(np.ceil(abs(x1 - x0) / res))
        shape1 = int(np.ceil(abs(y1 - y0) / res))

        # NOTE: Upper-left vs lower-left origin
        # (this only works for upper-left)
        transform = from_origin(x0 - res / 2, y1 + res / 2, res, res)

        rast_profile = {
                'driver': 'GTiff',
                'dtype': np.float32,
                'width': shape0,
                'height': shape1,
                'crs': utm_crs,
                'transform': transform,
                'count': 1,
        }
        with rasterio.open(str(out_rast), 'w', **rast_profile) as dst:
            # For places where raster is DEM is not provided it's
            # assumed deep ocean for contouring purposes
            dst.write(
                np.full((shape0, shape1), -99999, dtype=np.float32), 1)

            # Reproject if needed (for now only needed if constant
            # value levels or subtidal limiters are added)
            for in_idx, hfun in enumerate(rast_hfun_list):
                ignore = True
                for (src_idx, _, _), _ in self._const_val_contour_coll:
                    if src_idx is None or in_idx in src_idx:
                        ignore = False
                        break
                for src_idx, _, _, _, _ in self._flow_lim_coll:
                    if src_idx is None or in_idx in src_idx:
                        ignore = False
                        break
                if ignore:
                    continue

                # NOTE: Last one implicitely has highest priority in
                # case of overlap
                reproject(
                    source=rasterio.band(hfun.raster.src, 1),
                    destination=rasterio.band(dst, 1),
                    resampling=Resampling.nearest,
                    init_dest_nodata=False, # To avoid overwrite
                    num_threads=self._nprocs)



        return Raster(out_rast)

    def _apply_features_fast(self, big_raster):
        
        # NOTE: Caching applied doesn't work here since we apply
        # everything on a temporary big raster
        hfun = HfunRaster(big_raster, **self._size_info)
        self._apply_contours([hfun])
        self._apply_flow_limiters_fast(hfun)
        self._apply_const_val_fast(hfun)
        self._apply_patch([hfun])

        return hfun

    def _apply_flow_limiters_fast(self, big_hfun):

        for src_idx, hmin, hmax, zmax, zmin in self._flow_lim_coll:
            # TODO: Account for source index
            if hmin is None:
                hmin = self._size_info['hmin']
            if hmax is None:
                hmax = self._size_info['hmax']

            # To avoid sharp gradient where no raster is projected
            if zmin is None:
                zmin = -99990
            else:
                zmin = max(zmin, -99990)

            big_hfun.add_subtidal_flow_limiter(hmin, hmax, zmax, zmin)

    def _apply_const_val_fast(self, big_hfun):

        for (src_idx, ctr0, ctr1), const_val in self._const_val_contour_coll:
            # TODO: Account for source index
            level0 = None
            level1 =  None
            if ctr0 != None:
                level0 = ctr0.level
            if ctr1 != None:
                level1 = ctr1.level
            big_hfun.add_constant_value(const_val, level0, level1)


    def _get_hfun_composite_fast(self, big_hfun):

        # In fast method all DEM hfuns have more priority than all
        # other inputs
        dem_hfun_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]
        nondem_hfun_list = [
            i for i in self._hfun_list if not isinstance(i, HfunRaster)]

        epsg4326 = CRS.from_user_input("EPSG:4326")

        dem_box_list = list()
        for hfun in dem_hfun_list:
            dem_box_list.append(hfun.get_bbox(crs=epsg4326))


        # Calculate multipoly and clip big hfun
        dem_gdf = gpd.GeoDataFrame(
                geometry=dem_box_list, crs=epsg4326)
        big_cut_shape = dem_gdf.unary_union
        big_msh_t = big_hfun.msh_t()
        if hasattr(big_msh_t, "crs"):
            if not epsg4326.equals(big_msh_t.crs):
                utils.reproject(big_msh_t, epsg4326)
        big_msh_t = utils.clip_mesh_by_shape(
            big_msh_t,
            big_cut_shape,
            use_box_only=False,
            fit_inside=False)


        # TODO: User option to ignore base mesh
        hfun_list = nondem_hfun_list[::-1]
        if self._base_mesh:
            self._base_mesh.size_from_mesh()
            hfun_list = [*nondem_hfun_list[::-1], self._base_mesh]

        index = [big_msh_t.tria3['index']]
        coord = [big_msh_t.vert2['coord']]
        value = [big_msh_t.value]
        offset = coord[-1].shape[0]
        nondem_box_list = list()
        for hfun in hfun_list:
            nondem_msh_t = deepcopy(hfun.msh_t())
            if hasattr(nondem_msh_t, "crs"):
                if not epsg4326.equals(nondem_msh_t.crs):
                    utils.reproject(nondem_msh_t, epsg4326)
            nondem_bbox = hfun.get_bbox(crs=epsg4326)
            # In fast method all DEM hfuns have more priority than all
            # other inputs
            nondem_msh_t = utils.clip_mesh_by_shape(
                nondem_msh_t,
                big_cut_shape,
                use_box_only=False,
                fit_inside=True,
                inverse=True)
            for ibox in nondem_box_list:
                nondem_msh_t = utils.clip_mesh_by_shape(
                    nondem_msh_t,
                    ibox,
                    use_box_only=True,
                    fit_inside=True,
                    inverse=True)

            nondem_box_list.append(nondem_bbox)

            index.append(nondem_msh_t.tria3['index'] + offset)
            coord.append(nondem_msh_t.vert2['coord'])
            value.append(nondem_msh_t.value)
            offset += coord[-1].shape[0]

        composite_hfun = jigsaw_msh_t()
        composite_hfun.mshID = 'euclidean-mesh'
        composite_hfun.ndims = 2

        composite_hfun.vert2 = np.array(
                [(coord, 0) for coord in np.vstack(coord)],
                dtype=jigsaw_msh_t.VERT2_t)
        composite_hfun.tria3 = np.array(
                [(index, 0) for index in np.vstack(index)],
                dtype=jigsaw_msh_t.TRIA3_t)
        composite_hfun.value = np.array(
                np.vstack(value),
                dtype=jigsaw_msh_t.REALS_t)

        # TODO: Get user input for wether to force hmin and hmax on
        # final hfun (which includes non-raster and basemesh sizes)
        hmin = self._size_info['hmin']
        hmax = self._size_info['hmax']
        if hmin:
            composite_hfun.value[composite_hfun.value < hmin] = hmin
        if hmax:
            composite_hfun.value[composite_hfun.value > hmax] = hmax

        composite_hfun.crs = epsg4326

        # NOTE: In the end we need to return in a CRS that
        # uses meters as units. UTM based on the center of
        # the bounding box of the hfun is used
        # Up until now all calculation was in EPSG:4326
        utils.msh_t_to_utm(composite_hfun)

        return composite_hfun
