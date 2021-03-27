import os
import gc
import logging
import warnings
import tempfile
import numpy as np
from numbers import Number
from functools import reduce
from pathlib import Path
from time import time
from multiprocessing import Pool, cpu_count
from copy import copy
from typing import Union, Sequence, List, Tuple

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon
from jigsawpy import jigsaw_msh_t

from geomesh.mesh import Mesh
from geomesh.raster import Raster
from geomesh.geom.base import BaseGeom
from geomesh.geom.raster import RasterGeom
from geomesh.geom.shapely import MultiPolygonGeom, PolygonGeom
from geomesh.geom.mesh import MeshGeom
from geomesh.features.contour import FilledContour, Contour
from geomesh.features.patch import Patch
from geomesh.ops import combine_geometry

_logger = logging.getLogger(__name__)

class ContourPatchInfoCollector:
    def __init__(self):
        self._contour_patch_info = list()

    def add(self, contour_defn, patch_defn):
        self._contour_patch_info.append((contour_defn, patch_defn))

    def __iter__(self):
        for ctr_defn, ptch_defn in self._contour_patch_info:
            yield ctr_defn, ptch_defn


class GeomCollector(BaseGeom):

    def __init__(
            self,
            in_list: Sequence[
                Union[str, Raster, RasterGeom, MeshGeom,
                      MultiPolygonGeom, PolygonGeom]],
            base_mesh: Mesh = None,
            zmin: float = None,
            zmax: float = None,
            nprocs: int = None,
            chunk_size: int = None,
            overlap: int = None,
            verbosity: int = 0,
            ):

        # TODO: Like hfun collector and ops, later move the geom
        # combine functionality here and just call it from ops instead
        # of the other way around

        # For shapely and potentially mesh geom there's no priority
        # definition, they are just unioned with whatever the rest
        # of the input results in

        # store all the info pass
        # pass dem tmpfile or store address of each series
        # (global, patchcontour) to ops and get results and store
        # in the end combine all results (unary_union)

        # NOTE: Input Hfuns and their Rasters can get modified

        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs


        self._elev_info = dict(zmin=zmin, zmax=zmax)
        self._nprocs = nprocs
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._geom_list = list()
        # NOTE: Base mesh has to have a crs otherwise MeshGeom throws
        # exception
        self._base_mesh = base_mesh
        self._contour_patch_info_coll = ContourPatchInfoCollector()

        self._type_chk(in_list)

        # TODO: CRS considerations -- geom combine doesn't necessarily
        # return EPSG:4326 (unlike hfun collector msh_t)
        self._crs = 'EPSG:4326'
        
        for in_item in in_list:
            # Add supports(ext) to each hfun type?

            if isinstance(in_item, BaseGeom):
                geom = in_item

            elif isinstance(in_item, Raster):
                if self._base_mesh:
                    in_item.clip(self._base_mesh.get_bbox(crs=in_item.crs))
                geom = RasterGeom(in_item, **self._elev_info)

            elif isinstance(in_item, str):
                if in_item.endswith('.tif'):
                    raster = Raster(in_item)
                    if self._base_mesh:
                        raster.clip(self._base_mesh.get_bbox(crs=raster.crs))
                    geom = RasterGeom(raster, **self._elev_info)

                elif in_item.endswith(
                        ('.14', '.grd', '.gr3', '.msh', '.2dm')):
                    geom = MeshGeom(path)

                else:
                    raise TypeError("Input file extension not supported!")

            self._geom_list.append(geom)


    def get_multipolygon(self, **kwargs) -> MultiPolygon:
        '''Returns a :class:shapely.geometry.MultiPolygon object representing
        the geometry constrained by the arguments.'''

        # For now we don't need to do any calculations here, the
        # ops will take care of extracting everything. Later the logic
        # in ops needs to move here (like hfun collector)

        # Since raster geoms are stateless, the polygons should be
        # calculated everytime

        mp = None
        with tempfile.TemporaryDirectory() as temp_dir:
            feather_files = list()

            temp_path = Path(temp_dir)

            mesh_multipoly = None
            if self._base_mesh:
                mesh_multipoly = self._base_mesh.hull.multipolygon()
            feather_files.append(self._extract_global_boundary(
                temp_path, mesh_multipoly))
            feather_files.extend(self._extract_features(
                temp_path, mesh_multipoly))

            # TODO: Make sure all calcs are in EPSG:4326
            gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
            for f in feather_files:
                gdf = gdf.append(gpd.read_feather(f))
            mp = MultiPolygon(
                [geom for geom in gdf.unary_union.geoms])

        return mp

    def add_patch(self,
            shape: Union[MultiPolygon, Polygon] = None,
            level: Union[Tuple[float, float], float] = None,
            contour_defn: Union[FilledContour, Contour] = None,
            patch_defn: Patch = None,
            shapefile: Union[None, str, Path] = None,
            ):

        # Always lazy

        if not contour_defn:
            level0 = None
            level1 = None
            if isinstance(level, tuple) and len(level) == 2:
                level0 = level[0]
                level1 = level[1]
            elif isinstance(level, Number):
                level0 = None
                level1 = level
            else:
                raise ValueError(
                    "Level must be specified either by min and max values"
                    " or by only max value ")

            contour_defn = FilledContour(level=level)
            level0, level1

        elif isinstance(contour_defn, Contour):
            contour_defn = FilledContour(max_contour_defn=contour_defn)

        elif not isinstance(contour_defn, FilledContour):
            raise TypeError(
                f"Filled contour definition must be of type"
                f" {FilledContour} not {type(contour_defn)}!")

        elif level != None:
            msg = "Level is ignored since a contour definition is provided!"
            warnings.warn(msg)
            _logger.info(msg)

        if not patch_defn:
            if shape:
                patch_defn = Patch(shape=shape)

            elif shapefile:
                patch_defn = Patch(shapefile=shapefile)

        elif not isinstance(patch_defn, Patch):
            raise TypeError(
                f"Patch definition must be of type {Patch} not"
                f" {type(patch_defn)}!")


        # If patch defn is None it means the patch applies to 
        # all the sources of the accompanying contour
        self._contour_patch_info_coll.add(
            contour_defn, patch_defn)


    def _type_chk(self, input_list):
        ''' Check the input type for constructor '''
        valid_types = (str, Raster, RasterGeom, MultiPolygonGeom, PolygonGeom)
        if not all(isinstance(item, valid_types) for item in input_list):
            raise TypeError(
                f'Input list items must be of type {", ".join(valid_types)}'
                f', or a derived type.')

    def _get_raster_sources(self):
        raster_types = (RasterGeom, Raster)
        rasters = [
            i for i in self._geom_list if isinstance(i, raster_types)]
        return rasters

    def _get_raster_source_files(self):
        
        rasters = self._get_raster_sources()
        return self._get_raster_files_from_source(rasters)

    def _get_raster_files_from_source(self, rasters):
        raster_files = list()
        for r in rasters:
            if isinstance(r, Raster):
                raster_files.append(r.path)
            elif isinstance(r, RasterGeom):
                raster_files.append(r.raster.path)
            
        return raster_files
        
    def _get_non_raster_sources(self):
        raster_types = (RasterGeom, Raster)
        non_rasters = [
            i for i in self._geom_list if not isinstance(i, raster_types)]
        return non_rasters

    def _extract_global_boundary(self, out_dir, mesh_multipoly):

        out_path = Path(out_dir)

        geom_path = out_path / 'global_boundary.feather'

        raster_files = self._get_raster_source_files()
        zmin = self._elev_info['zmin']
        zmax = self._elev_info['zmax']
        _logger.info("Extracting global boundaries")
        combine_geometry(
            raster_files, geom_path, "feather",
            None, mesh_multipoly, False,
            zmin, zmax,
            self._chunk_size, self._overlap,
            self._nprocs)

        return geom_path

    def _extract_features(self, out_dir, mesh_multipoly):

        feather_files = list()
        feather_files.extend(self._apply_patch(out_dir, mesh_multipoly))

        return feather_files

    def _apply_patch(self, out_dir, mesh_multipoly):

        out_path = Path(out_dir)


        raster_files = self._get_raster_source_files()
        zmin = self._elev_info['zmin']
        zmax = self._elev_info['zmax']

        feather_files = list()
        for e, (ctr_defn, ptch_defn) in enumerate(self._contour_patch_info_coll):
            
            patch_zmin, patch_zmax = ctr_defn.level
            if not patch_zmin:
                patch_zmin = zmin
            if not patch_zmax:
                patch_zmax = zmax

            patch_raster_files = raster_files
            if ctr_defn.has_source:
                patch_rasters = ctr_defn.sources
                patch_raster_files = self._get_raster_files_from_source(
                        patch_rasters)

            # Pass patch shape instead of base mesh
            # See explanation in add_patch
            _logger.info("Extracting patch contours")
            combine_poly = mesh_multipoly
            if ptch_defn:
                patch_mp, crs = ptch_defn.get_multipolygon()
                gdf_patch = gpd.GeoDataFrame(
                        {'geometry': patch_mp}, crs=crs)
                if crs != CRS.from_user_input("EPSG:4326"):
                    gdf_patch = gdf_patch.to_crs("EPSG:4326")
                combine_poly = MultiPolygon([
                    geom for geom in gdf_patch.geometry])
            geom_path = out_path / f'patch_{os.getpid()}_{e}.feather'
            combine_geometry(
                patch_raster_files, geom_path, "feather",
                None, combine_poly, True,
                patch_zmin, patch_zmax,
                self._chunk_size, self._overlap,
                self._nprocs)

            if geom_path.is_file():
                feather_files.append(geom_path)

        return feather_files
