import logging
import os
import pathlib
from multiprocessing import cpu_count
from typing import Union, Sequence, List

from pyproj import CRS
from jigsawpy import savemsh, savevtk

from ocsmesh.raster import Raster
from ocsmesh.hfun.hfun import Hfun
from ocsmesh.mesh.mesh import Mesh
from ocsmesh import utils

_logger = logging.getLogger(__name__)


class HfunCombine:

    def __init__(
            self,
            dem_files: Sequence[Union[str, os.PathLike]],
            out_file: Union[str, os.PathLike],
            out_format: str = "shapefile",
            mesh_file: Union[str, os.PathLike, None] = None,
            hmin: Union[float, None] = None,
            hmax: Union[float, None] = None,
            contours: List[List[float]] = None,
            constants: List[List[float]] = None,
            chunk_size: Union[int, None] = None,
            overlap: Union[int, None] = None,
            method: str = 'exact',
            nprocs: int = -1):


        self._base_exterior = None

        self._operation_info = dict(
            dem_files=dem_files,
            out_file=out_file,
            out_format=out_format,
            mesh_file=mesh_file,
            hmin=hmin,
            hmax=hmax,
            contours=contours,
            constants=constants,
            chunk_size=chunk_size,
            overlap=overlap,
            method=method,
            nprocs=nprocs)

    def run(self):

        dem_files = self._operation_info['dem_files']
        out_file = self._operation_info['out_file']
        out_format = self._operation_info['out_format']
        mesh_file = self._operation_info['mesh_file']
        hmin = self._operation_info['hmin']
        hmax = self._operation_info['hmax']
        contours = self._operation_info['contours']
        constants = self._operation_info['constants']
        chunk_size = self._operation_info['chunk_size']
        overlap = self._operation_info['overlap']
        method = self._operation_info['method']
        nprocs = self._operation_info['nprocs']

        nprocs = cpu_count() if nprocs == -1 else nprocs

        out_dir = pathlib.Path(out_file).parent
        out_dir.mkdir(exist_ok=True, parents=True)

        logging.info("Loading base mesh...")
        base_mesh = None
        if mesh_file:
            base_mesh = Mesh.open(mesh_file, crs="EPSG:4326")

        rast_list = []
        for dem_path in dem_files:

            logging.info(f"Loading raster {dem_path}...")
            rast = Raster(
                    dem_path, chunk_size=chunk_size, overlap=overlap)
            rast_list.append(rast)

        # Create Hfun
        logging.info("Creating Hfun from rasters...")
        hfun_collector = Hfun(
                rast_list, base_mesh=base_mesh,
                hmin=hmin, hmax=hmax, nprocs=nprocs, method=method)

        for contour in contours:
            logging.info("Adding contour refinement...")
            if len(contour) > 3:
                raise ValueError(
                    "Invalid format for contour specification."
                    " It should be level [expansion target-size].")

            level, expansion_rate, target_size = [
                    *contour, *[None]*(3-len(contour))]

            if level is None:
                raise ValueError(
                    "Invalid format for contour specification."
                    " It should be level [expansion target-size].")
            if expansion_rate is None:
                expansion_rate = 0.1
            if target_size is None:
                target_size = hmin

            hfun_collector.add_contour(
                level, expansion_rate, target_size)

        for lower_bound, target_size in constants:
            hfun_collector.add_constant_value(
                    value=target_size, lower_bound=lower_bound)

        self._write_to_file(
                out_format, out_file, hfun_collector, 'EPSG:4326')


    def _write_to_file(
            self, out_format, out_file, hfun_collector, crs):

        _logger.info(f"Writing for file ({out_format}) ...")

        # NOTE: Combined mesh from collector is always in EPSG:4326
        jig_hfun = hfun_collector.msh_t()
        dst_crs = CRS.from_user_input(crs)
        if jig_hfun.crs != dst_crs:
            _logger.info(f"Reprojecting hfun to ({crs}) ...")
            utils.reproject(jig_hfun, dst_crs)

        # TODO: Check for correct extension on out_file
        if out_format in ("jigsaw", "vtk"):
            if out_format == "jigsaw":
                savemsh(out_file, jig_hfun)

            elif out_format == "vtk":
                savevtk(out_file, jig_hfun)

        elif out_format in ['2dm', 'sms']:
            # TODO: How to specify crs in 2dm file?
            mesh = Mesh(jig_hfun)
            mesh.write(out_file, format='2dm')

        else:
            raise NotImplementedError(f"Output type {out_format} is not supported")

        _logger.info("Done")
