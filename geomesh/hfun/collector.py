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
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
from jigsawpy import jigsaw_msh_t

from geomesh.hfun.base import BaseHfun
from geomesh.hfun.raster import HfunRaster
from geomesh.hfun.mesh import HfunMesh
from geomesh.mesh.mesh import Mesh
from geomesh.raster import Raster
from geomesh.features.contour import Contour

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
        self._contours_info[
                (tuple(src_idx), contour_defn0, contour_defn1)] = value 

    def __iter__(self):
        for defn, info in self._contours_info.items():
            yield defn, info




class HfunCollector(BaseHfun):

    def __init__(
            self,
            in_list: Sequence[
                Union[str, Raster, Mesh, HfunRaster, HfunMesh]],
            base_mesh: Mesh,
            hmin: float = None,
            hmax: float = None,
            nprocs: int = None,
            verbosity: int = 0,
            ):

        # NOTE: Input Hfuns and their Rasters can get modified

        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs

        self._applied = False
        self._size_info = dict(hmin=hmin, hmax=hmax)
        self._nprocs = nprocs
        self._hfun_list = list()
        # NOTE: Base mesh has to have a crs otherwise HfunMesh throws
        # exception
        self._base_mesh = HfunMesh(base_mesh)
        self._contour_info_coll = RefinementContourInfoCollector()
        self._contour_coll = RefinementContourCollector(
                self._contour_info_coll)
        self._const_val_contour_coll = ConstantValueContourInfoCollector()

        self._type_chk(in_list)

        # TODO: Interpolate max size on base mesh basemesh?
        #
        # TODO: CRS considerations
        
        for in_item in in_list:
            # Add supports(ext) to each hfun type?

            if isinstance(in_item, BaseHfun):
                hfun = in_item

            elif isinstance(in_item, Raster):
                in_item.clip(self._base_mesh.mesh.get_bbox(crs=in_item.crs))
                hfun = HfunRaster(in_item, **self._size_info)

            elif isinstance(in_item, EuclideanMesh2D):
                hfun = HfunMesh(in_item)

            elif isinstance(in_item, str):
                if in_item.endswith('.tif'):
                    raster = Raster(in_item)
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

        self._apply_features()

        composite_hfun = jigsaw_msh_t()
        with tempfile.TemporaryDirectory() as temp_dir:
            hfun_path_list = self._write_hfun_to_disk(temp_dir)
            composite_hfun = self._get_hfun_composite(hfun_path_list)
        composite_hfun.crs = CRS.from_user_input("EPSG:4326")
        return composite_hfun


    def add_contour(
            self,
            level: Union[List[float], float] = None,
            expansion_rate: float = None,
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


    def add_constant_value(
            self, value,
            lower_bound=None,
            upper_bound=None,
            source_index: Union[List[int], int, None] =None):

        # TODO: Add sources arg?

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


    def add_patch(self, shape):

        self._applied = False

        raise NotImplementedError(
            "Patch is not implemented for collector hfun!")

    def _type_chk(self, input_list):
        ''' Check the input type for constructor '''
        valid_types = (str, Raster, Mesh, HfunRaster, HfunMesh)
        if not all(isinstance(item, valid_types) for item in input_list):
            raise TypeError(
                f'Input list items must be of type {", ".join(valid_types)}'
                f', or a derived type.')

    def _apply_features(self):

        if not self._applied:
            self._apply_contours()
            self._apply_const_val()
            #self._apply_patch()

        self._applied = True

    def _apply_contours(self):

        # TODO: Consider CRS before applying to different hfuns
        #
        # TODO: Can add_feature be added to non-raster hfun?

        # NOTE: for parallelization make sure a single hfun is NOT
        # passed to multiple processes

        contourable_list = [
            i for i in self._hfun_list if isinstance(i, HfunRaster)]

        with tempfile.TemporaryDirectory() as temp_path:
            self._contour_coll.calculate(contourable_list, temp_path)
            counter = 0
            for hfun in contourable_list:
                for gdf in self._contour_coll:
                    for row in gdf.itertuples():
                        _logger.debug(row)
                        if isinstance(row.geometry, GeometryCollection):
                            continue
                        counter = counter + 1
                        hfun.add_feature(**{
                            'feature': row.geometry,
                            'expansion_rate': row.expansion_rate,
                            'target_size': row.target_size,
                            'nprocs': self._nprocs
                        })
            # hfun objects cause issue with pickling
            # -> cannot be passed to pool
#            with Pool(processes=self._nprocs) as p:
#                p.starmap(
#                    _apply_contours_worker,
#                    [(hfun, self._contour_coll, self._nprocs)
#                     for hfun in contourable_list])

    def _apply_const_val(self):

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


    def _apply_patch(self):
        raise NotImplementedError(
            "Patch is not implemented for collector hfun!")

    def _write_hfun_to_disk(self, out_path):

        out_dir = Path(out_path)
        path_list = list()
        file_counter = 0
        pid = os.getpid()
        bbox_list = list()
        # TODO: Should basemesh be included?
        # Last user input item has the highest priority (its trias
        # are not dropped) so process in reverse order
        for hfun in self._hfun_list[::-1]:
            # TODO: Calling msh_t() on HfunMesh more than once causes
            # issue right now due to change in crs of internal Mesh

            # To avoid removing verts and trias from mesh hfuns
            hfun_mesh = deepcopy(hfun.msh_t())
            # If no CRS info, we assume EPSG:4326
            if hasattr(hfun_mesh, "crs"):
                dst_crs = CRS.from_user_input("EPSG:4326")
                if hfun_mesh.crs != dst_crs:
                    xx = hfun_mesh.vert2["coord"][:, 0]
                    yy = hfun_mesh.vert2["coord"][:, 1]
                    transformer = Transformer.from_crs(
                        hfun_mesh.crs, dst_crs, always_xy=True)
                    transformed_crd = np.vstack(
                            transformer.transform(xx, yy)).T
                    hfun_mesh.crs = dst_crs
                    hfun_mesh.vert2 = np.array(
                            [(coo, 0) for coo in np.vstack(transformed_crd)],
                            dtype=jigsaw_msh_t.VERT2_t)

            # Get all previous bbox and clip to resolve overlaps
            # removing all tria that have NODE in bbox because it's
            # faster and so we can resolve all overlaps
            _logger.info(f"Removing bounds from hfun mesh...")
            for bounds in bbox_list:

                xmin, ymin, xmax, ymax = bounds

                cnn = hfun_mesh.tria3['index']
                crd = hfun_mesh.vert2['coord']
                _logger.info(f"# tria3: {len(cnn)}")

                start = time()
                in_box_idx_1 = np.arange(len(crd))[crd[:, 0] > xmin]
                in_box_idx_2 = np.arange(len(crd))[crd[:, 0] < xmax]
                in_box_idx_3 = np.arange(len(crd))[crd[:, 1] > ymin]
                in_box_idx_4 = np.arange(len(crd))[crd[:, 1] < ymax]
                in_box_idx = reduce(
                    np.intersect1d,
                    (in_box_idx_1, in_box_idx_2, in_box_idx_3, in_box_idx_4))
                _logger.info(f"Find drop verts took {time() - start}")

                start = time()
                drop_tria = np.all(
                    np.isin(cnn.ravel(), in_box_idx).reshape(cnn.shape),
                    1)
                _logger.info(f"Find drop trias took {time() - start}")

                _logger.info(f"Dropping {np.sum(drop_tria)} triangles...")
                start = time()
                new_cnn_unfinished = cnn[np.logical_not(drop_tria), :]
                _logger.info(f"Getting Unfinished CNN took {time() - start}")

                start = time()
                lookup_table = {
                    index: i for i, index
                    in enumerate(sorted(np.unique(new_cnn_unfinished.flatten())))}
                new_cnn = np.array([list(map(lambda x: lookup_table[x], element))
                                      for element in new_cnn_unfinished])
                new_crd = crd[list(lookup_table.keys()), :]
                value = hfun_mesh.value[list(lookup_table.keys()), :]
                
                _logger.info(f"# tria3: {len(new_cnn)}")

                hfun_mesh.value = value
                hfun_mesh.vert2 = np.array(
                    [(coo, 0) for coo in new_crd], dtype=jigsaw_msh_t.VERT2_t)
                hfun_mesh.tria3 = np.array(
                    [(con, 0) for con in new_cnn], dtype=jigsaw_msh_t.TRIA3_t)

                _logger.info(f"Getting new CRD and CNN took {time() - start}")

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
            bbox_list.append(mesh.get_bbox(crs="EPSG:4326").bounds)
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

        return composite_hfun


def _apply_contours_worker(hfun, contour_coll, nprocs):
    # HfunRaster object cannot be passed to pool due to raster buffers
    # issue with pickling -- TODO: Use hfuninfo objects?
    for gdf in contour_coll:
        for row in gdf.itertuples():
            _logger.debug(row)
            if isinstance(row.geometry, GeometryCollection):
                continue
            hfun.add_feature(**{
                'feature': row.geometry,
                'expansion_rate': row.expansion_rate,
                'target_size': row.target_size,
                'nprocs': nprocs
            })

def _find_exact_elements_to_discard(bbox, possible_elements_to_discard):
    # NOTE: HfunCollector methods cannot be passed to pool due to 
    # raster buffers issue with pickling
    exact_elements_to_discard = set()
    for row in possible_elements_to_discard.itertuples():
        if row.geometry.within(bbox):
            exact_elements_to_discard.add(row.Index)
    return list(exact_elements_to_discard)
