import os
import gc
import logging
import warnings
import tempfile
import numpy as np
from pathlib import Path
from time import time
from multiprocessing import Pool, cpu_count
from copy import copy
from typing import Union, Sequence, List

import geopandas as gpd
from pyproj import CRS
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
        self._contours_info[contour_defn]=size_info

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

        self._size_info = dict(hmin=hmin, hmax=hmax)
        self._nprocs = nprocs
        self._hfun_list = list()
        # NOTE: Base mesh has to have a crs otherwise HfunMesh throws
        # exception
        self._base_mesh = HfunMesh(base_mesh)
        self._contour_info_coll = RefinementContourInfoCollector()
        self._contour_coll = RefinementContourCollector(
                self._contour_info_coll)

        self._type_chk(in_list)

        # TODO: Interpolate max size on base mesh basemesh?
        #
        # TODO: CRS considerations
        # 
        # TODO: Clip by basemesh

        for in_item in in_list:
            # Add supports(ext) to each hfun type?

            if isinstance(in_item, BaseHfun):
                hfun = in_item

            elif isinstance(in_item, Raster):
                hfun = HfunRaster(in_item, **self._size_info)

            elif isinstance(in_item, EuclideanMesh2D):
                hfun = HfunMesh(in_item)

            elif isinstance(in_item, str):
                if in_item.endswith('.tif'):
                    raster = Raster(in_item)
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
        return composite_hfun


    def add_contour(
            self,
            level: Union[List[float], float] = None,
            expansion_rate: float = None,
            target_size: float = None,
            contour_defn: Contour = None,
    ):
        # Always lazy

        if contour_defn == None:
            contour_defn = Contour(level=level)

        elif not isinstance(contour_defn, Contour):
            raise TypeError(
                f"Contour definition must be of type {Contour} not"
                f" {type(contour_defn)}!")

        elif level != None:
            msg = "Level is ignored since a contour definition is provided!"
            warnings.warn(msg)
            _logger.info(msg)

        self._contour_info_coll.add(
            contour_defn, 
            expansion_rate=expansion_rate,
            target_size=target_size)



    def add_patch(self, shape):
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
        self._apply_contours()
        #self._apply_patch()

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

    def _apply_patch(self):
        raise NotImplementedError(
            "Patch is not implemented for collector hfun!")

    def _write_hfun_to_disk(self, out_path):

        out_dir = Path(out_path)
        path_list = list()
        file_counter = 0
        pid = os.getpid()
        for hfun in [self._base_mesh, *self._hfun_list]:
            mesh = hfun.msh_t()
            file_counter = file_counter + 1
            _logger.info(f'write mesh {file_counter} to file...')
            file_path = out_dir / f'hfun_{pid}_{file_counter}.2dm'
            Mesh(mesh).write(file_path, format='2dm')
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


        _logger.info('Figuring out which one is the base hfun...')
        start = time()
        areas = [mp.area for mp in
                 [mesh.hull.multipolygon() for mesh in collection]]
        # TODO: What about other hfuns overlap?
        base_hfun = collection.pop(
            np.where(areas == np.max(areas))[0][0])
        _logger.info(f'Found base hfun in {time()-start} seconds.')

        _logger.info('Generating base hfun geodataframe...')
        start = time()
        elements = base_hfun.elements.geodataframe()
        _logger.info(f'geodataframe generation took {time()-start}.')

        _logger.info('Generating base hfun rtree index...')
        start = time()
        elements_r_index = elements.sindex
        _logger.info(f'base_hfun rtree index gen took {time()-start}.')

        _logger.info(
            'Using r-tree indexing to find possible elements to discard...')
        start = time()
        possible_elements_to_discard = set()
        for hfun in collection:
            bounds = hfun.get_bbox(crs=base_hfun.crs).bounds
            for index in list(elements_r_index.intersection(bounds)):
                possible_elements_to_discard.add(index)
        del elements_r_index
        gc.collect()
        possible_elements_to_discard = elements.iloc[list(
            possible_elements_to_discard)]

        _logger.info(
            f'Found possible elements to discard in {time()-start} seconds.')

        _logger.info('Finding exact elements to discard...')
        start = time()
        with Pool(processes=self._nprocs) as pool:
            result = pool.starmap(
                    _find_exact_elements_to_discard,
                    [(hfun.get_bbox(crs=base_hfun.crs),
                      possible_elements_to_discard)
                     for hfun in collection]
                )
        to_keep = elements.loc[elements.index.difference(
            [item for sublist in result for item in sublist])].index
        _logger.info(
            f'Found exact elements to discard in {time()-start} seconds.')
        del elements
        gc.collect()

        final_tria = base_hfun.tria3['index'][to_keep, :]
        del to_keep
        gc.collect()

        lookup_table = {index: i for i, index
                        in enumerate(sorted(np.unique(final_tria.flatten())))}
        coord = [base_hfun.coord[list(lookup_table.keys()), :]]
        index = [np.array([list(map(lambda x: lookup_table[x], element))
                          for element in final_tria])]
        value = [base_hfun.value[list(lookup_table.keys()), :]]
        offset = coord[-1].shape[0]
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
    # issue with pickling
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
    exact_elements_to_discard = set()
    for row in possible_elements_to_discard.itertuples():
        if row.geometry.within(bbox):
            exact_elements_to_discard.add(row.Index)
    return list(exact_elements_to_discard)
