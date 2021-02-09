import gc
import logging
from multiprocessing import Pool, Lock, cpu_count
import os
import pathlib
import tempfile
import warnings
from typing import Union, Sequence, Tuple, List

import geopandas as gpd
from jigsawpy import jigsaw_msh_t, savemsh, savevtk
import numpy as np
from shapely import ops
from shapely.geometry import box, Polygon, MultiPolygon, LinearRing


from geomesh import Raster, Geom
from geomesh.mesh.mesh import Mesh


_logger = logging.getLogger(__name__)


class GeomCombine:

    _base_mesh_lock = Lock()
    def __init__(
            self,
            dem_files: Sequence[Union[str, os.PathLike]],
            out_file: Union[str, os.PathLike],
            out_format: str = "shapefile",
            mesh_file: Union[str, os.PathLike, None] = None,
            zmin: Union[float, None] = None,
            zmax: Union[float, None] = None,
            chunk_size: Union[int, None] = None,
            overlap: Union[int, None] = None,
            nprocs: int = -1):


        self._base_exterior = None

        self._operation_info = dict(
            dem_files=dem_files,
            out_file=out_file,
            out_format=out_format,
            mesh_file=mesh_file,
            zmin=zmin,
            zmax=zmax,
            chunk_size=chunk_size,
            overlap=overlap,
            nprocs=nprocs)

    def run(self):

        dem_files = self._operation_info['dem_files']
        out_file = self._operation_info['out_file']
        out_format = self._operation_info['out_format']
        mesh_file = self._operation_info['mesh_file']
        zmin = self._operation_info['zmin']
        zmax = self._operation_info['zmax']
        chunk_size = self._operation_info['chunk_size']
        overlap = self._operation_info['overlap']
        nprocs = self._operation_info['nprocs']

        nprocs = cpu_count() if nprocs == -1 else nprocs

        base_mult_poly = None
        if mesh_file and pathlib.Path(mesh_file).is_file():
            _logger.info("Creating mesh object from file...")
            base_mesh = Mesh.open(mesh_file)
            _logger.info("Done")

            _logger.info("Getting mesh hull polygons...")
            base_mult_poly = base_mesh.hull.multipolygon()
            _logger.info("Done")

            # NOTE: This needs to happen once and before any
            # modification to basemesh happens (due to overlap
            # w/ DEM, etc.). Exterior of base mesh is used for
            # raster clipping
            if isinstance(base_mult_poly, Polygon):
                base_mult_poly = MultiPolygon([base_mult_poly])
            self._base_exterior = MultiPolygon(
                    [i for i in ops.polygonize(
                        [poly.exterior for poly in base_mult_poly])])


        z_info = dict()
        if zmin is not None:
            z_info['zmin'] = zmin
        if zmax is not None:
            z_info['zmax'] = zmax

        poly_files_coll = list()
        _logger.info(f"Number of processes: {nprocs}")
        out_dir = pathlib.Path(out_file).parent
        out_dir.mkdir(exist_ok=True, parents=True)
        with tempfile.TemporaryDirectory(dir=out_dir) as temp_dir, \
                tempfile.NamedTemporaryFile() as base_file:

            if base_mult_poly:
                base_mesh_path = base_file.name
                gpd.GeoDataFrame(
                        {'geometry': base_mult_poly}).to_feather(
                            base_mesh_path)
            else:
                base_mesh_path = None
            base_mult_poly = None

            if nprocs > 1:
                n_proc_dem = nprocs
                if nprocs > len(dem_files):
                    n_proc_dem = len(dem_files)
                parallel_args = list()
                for dem_file in dem_files:
                    parallel_args.append(
                        (base_mesh_path, temp_dir, dem_file,
                         z_info, chunk_size, overlap))
                with Pool(processes=n_proc_dem) as p:
                    poly_files_coll = p.starmap(
                        self._parallel_get_polygon_worker, parallel_args)
            else:
                poly_files_coll = self._serial_get_polygon(
                    base_mesh_path, temp_dir, dem_files,
                    z_info, chunk_size, overlap)

            _logger.info("Generating final boundary polygon...")
            poly_coll = list()

            rasters_gdf = gpd.GeoDataFrame(
                    columns=['geometry'],
                    crs='EPSG:4326'
                )
            for feather_f in poly_files_coll:
                rasters_gdf = rasters_gdf.append(
                    gpd.read_feather(feather_f), ignore_index=True)

            # unary union of raster geoms
            _logger.info('Generate unary union of raster geoms...')
            poly_coll.append(MultiPolygon(
                [geom for geom in rasters_gdf.unary_union.geoms]))

            if base_mesh_path is not None:
                
                base_mult_poly = [i for i in gpd.read_feather(
                        base_mesh_path).geometry]
                poly_coll.extend(base_mult_poly)
            fin_mult_poly = ops.unary_union(poly_coll)

            _logger.info("Done")

            if not fin_mult_poly:
                # This should really happen in real-world scenarios!
                if not poly_coll:
                    raise ValueError("No polynomials to work with!")

                fin_mult_poly = ops.unary_union(poly_coll)

        if isinstance(fin_mult_poly, Polygon):
            fin_mult_poly = MultiPolygon([fin_mult_poly])

        # TODO: Consider projection(?)
        self._write_to_file(
                out_format, out_file, fin_mult_poly, 'EPSG:4326')

        self._base_exterior = None


    def _serial_get_polygon(
            self,
            base_mesh_path: Union[str, None],
            temp_dir: Union[str, os.PathLike],
            dem_files: Sequence[Union[str, os.PathLike]],
            z_info: dict = dict(),
            chunk_size: Union[int, None] = None,
            overlap: Union[int, None] = None):


        _logger.info("Getting DEM info")
        poly_coll = list()
        for dem_path in dem_files:
            _logger.info(f"Processing {dem_path} ...")
            if not pathlib.Path(dem_path).is_file():
                warnings.warn(f"File {dem_path} not found!")
                _logger.debug(f"File {dem_path} not found!")
                continue

            # Calculate Polygon
            _logger.info("Loading raster from file...")
            rast = Raster(
                    dem_path,
                    chunk_size=chunk_size,
                    overlap=overlap)

            _logger.info("Clipping to basemesh size if needed...")
            rast_box = box(*rast.src.bounds)
            if base_mesh_path is not None:
                # NOTE: We use the exterior from the earlier calc
                if not rast_box.within(self._base_exterior):
                    _logger.info("Needs clipping...")
                    rast.clip(self._base_exterior)
                    rast_box = box(*rast.src.bounds)

            _logger.info("Creating geom from raster...")
            geom = Geom(rast)

            _logger.info("Getting polygons from geom...")
            geom_mult_poly = geom.get_multipolygon(**z_info)

            if base_mesh_path is not None:
                _logger.info("Subtract DEM bounds from base mesh polygons...")
                self._base_mesh_lock.acquire()
                try:
                    base_mult_poly = MultiPolygon(
                            [i for i in gpd.read_feather(
                                base_mesh_path).geometry])
                    base_mult_poly = base_mult_poly.difference(
                            rast_box)
                    if isinstance(base_mult_poly, Polygon):
                        base_mult_poly = MultiPolygon([base_mult_poly])
                    gpd.GeoDataFrame(
                            {'geometry': base_mult_poly}).to_feather(
                                base_mesh_path)
                finally:
                    self._base_mesh_lock.release()

            if isinstance(geom_mult_poly, Polygon):
                geom_mult_poly = MultiPolygon([geom_mult_poly])
            temp_path = (
                    pathlib.Path(temp_dir)
                    / f'{pathlib.Path(dem_path).name}.shp')

            try:
                gpd.GeoDataFrame({'geometry': geom_mult_poly}
                        ).to_feather(temp_path)
                poly_coll.append(temp_path)
            except:
                warnings.warn(f"Error writing shapefile for {temp_path}")

            # Multipolygon takes a lot of memory
            del geom_mult_poly
            gc.collect(2)

        return poly_coll


    def _parallel_get_polygon_worker(
            self,
            base_mesh_path: str,
            temp_dir: Union[str, os.PathLike],
            dem_file: Union[str, os.PathLike],
            z_info: dict = dict(),
            chunk_size: Union[int, None] = None,
            overlap: Union[int, None] = None):

        poly_coll = self._serial_get_polygon(
            base_mesh_path, temp_dir, [dem_file],
            z_info, chunk_size, overlap)

        # Only one item passed to serial code
        return poly_coll[0]


    def _linearring_to_vert_edge(
            self,
            coords: List[Tuple[float, float]],
            edges: List[Tuple[int, int]],
            lin_ring: LinearRing):

        '''From shapely LinearRing get coords and edges'''

        # NOTE: This function mutates coords and edges

        # TODO: Move to utils?
        idx_b = len(coords)
        coords.extend(coord for coord in lin_ring.coords)
        # Last coord is the same as first in a ring
        coords.pop()
        idx_e = len(coords) - 1
        n_idx = len(coords)

        edges.extend([
            (i, (i + 1) % n_idx + idx_b * ((i + 1) // n_idx))
            for i in range(idx_b, idx_e + 1)])


    def _write_to_file(
            self, out_format, out_file, multi_polygon, crs):

        _logger.info(f"Writing for file ({out_format}) ...")

        # TODO: Check for correct extension on out_file
        if out_format == "shapefile":
            gpd.GeoDataFrame(
                    {'geometry': multi_polygon},
                    crs=crs
                    ).to_file(out_file)

        elif out_format in ("jigsaw", "vtk"):
            msh = jigsaw_msh_t()
            msh.ndims = +2
            msh.mshID = 'euclidean-mesh'

            coords = list()
            edges = list()
            for polygon in multi_polygon:
                self._linearring_to_vert_edge(
                        coords, edges, polygon.exterior)
                for interior in polygon.interiors:
                    self._linearring_to_vert_edge(
                            coords, edges, interior)

            msh.vert2 = np.array(
                [(i, 0) for i in coords],
                dtype=jigsaw_msh_t.VERT2_t)
            msh.edge2 = np.array(
                [(i, 0) for i in edges],
                dtype=jigsaw_msh_t.EDGE2_t)


            if out_format == "jigsaw":
                savemsh(out_file, msh)

            elif out_format == "vtk":
                savevtk(out_file, msh)

        else:
            raise NotImplementedError(f"Output type {out_format} is not supported")

        _logger.info("Done")
