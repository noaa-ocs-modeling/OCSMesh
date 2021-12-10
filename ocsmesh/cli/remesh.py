#!/usr/bin/env python
import sys
import gc
import logging
from pathlib import Path
from copy import deepcopy

import jigsawpy
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from ocsmesh import Raster, Geom, Hfun, Mesh
from ocsmesh import utils


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S'
    )
#logging.getLogger().setLevel(logging.DEBUG)
#logging.getLogger().setLevel(logging.INFO)

_logger = logging.getLogger(__name__)

class RemeshByDEM:

    @property
    def script_name(self):
        return 'remesh_by_dem'

    def __init__(self, sub_parser):

        # e.g
        # ./remesh \
        #   --mesh <MESH_ADD> \
        #   --output <OUT_FILE_ADDR> \
        #   --hmin <GLOBAL_MIN_SIZE> \
        #   --contour <LEVEL0 [EXPAND0 SIZE0]> \
        #   --contour <LEVEL1 [EXPAND1 SIZE1]> \
        #   --contour <LEVELN [EXPANDN SIZEN]> \
        #   --constant <ABOVELEVELA SIZEA> \
        #   --constant <ABOVELEVELB SIZEB> \
        #   --zmax <MAX_ELEV_DEM> \
        #   <DEM_FILES>*.tif

        this_parser = sub_parser.add_parser(
            self.script_name,
            help="Locally refine an existing input mesh based on input DEMs")

        this_parser.add_argument('--mesh', required=True, type=Path)
        this_parser.add_argument('--mesh-crs', default='EPSG:4326')


        this_parser.add_argument(
            '--contour',
            action='append', nargs='+', type=float, dest='contours',
            metavar='CONTOUR_DEFN', default=[],
            help="Each contour's (level, [expansion, target])"
                 " to be applied on all size functions in collector")
        this_parser.add_argument(
            '--constant',
            action='append', nargs=2, type=float, dest='constants',
            metavar='CONST_DEFN', default=[],
            help="Specify constant mesh size above a given contour level"
                 " by passing (lower_bound, target_size) for each constant")
        this_parser.add_argument('--hmin', type=float, default=250)
        this_parser.add_argument('--zmax', type=float, default=0)
        this_parser.add_argument(
            '--clip-by-base', action='store_true',
            help='Flag to clip input DEMs using base mesh polygon')

        this_parser.add_argument('--geom', type=Path)
        this_parser.add_argument('--hfun', type=Path)
        this_parser.add_argument('--hfun-crs', default='EPSG:4326')

        this_parser.add_argument('-s', '--sieve', type=float)
        this_parser.add_argument(
            '--interpolate', nargs='+', type=Path, default=[],
            help="To interpolate from depth of DEMs not involved in"
                 " the remeshing process")

        this_parser.add_argument('-o', '--output', type=Path)
        this_parser.add_argument('-f', '--output-format', default='2dm')
        this_parser.add_argument('-k', '--keep-intermediate', action='store_true')
        this_parser.add_argument('--nprocs', type=int, default=-1)

        this_parser.add_argument('dem', nargs='+', type=Path)


    @staticmethod
    def _read_geom_hfun(geom_file, hfun_file, hfun_crs):
        _logger.info("Read geom and hfun from disk")
        _logger.info("Readng geometry...")
        gdf_geom = gpd.read_file(geom_file)
        poly_list = []
        for i in gdf_geom.geometry:
            if isinstance(i, MultiPolygon):
                poly_list.extend(i)
            elif isinstance(i, Polygon):
                poly_list.append(i)
        geom = Geom(MultiPolygon(poly_list), crs=gdf_geom.crs)
        _logger.info("Done")

        _logger.info("Readng size function...")
        if hfun_crs is None:
            hfun_crs = "EPSG:4326"
        hfun = Hfun(Mesh.open(hfun_file, crs=hfun_crs))
        _logger.info("Done")

        return geom, hfun


    def run(self, args):

        # Get inputs
        base_path = args.mesh
        mesh_crs = args.mesh_crs

        dem_paths = args.dem
        contours = args.contours
        constants = args.constants
        hmin = args.hmin
        zmax = args.zmax
        clip_by_mesh = args.clip_by_base

        geom_file = args.geom
        hfun_file = args.hfun
        hfun_crs = args.hfun_crs

        sieve = args.sieve
        interp = args.interpolate

        out_path = args.output
        out_format = args.output_format
        write_intermediate = args.keep_intermediate
        nprocs = args.nprocs

        # Process inputs
        contour_defns = []
        for contour in contours:
            if len(contour) > 3:
                raise ValueError(
                    "Invalid format for contour specification."
                    " It should be level [expansion target-size].")
            level, expansion_rate, target_size = [
                    *contour, *[None]*(3-len(contour))]
            contour_defns.append((level, expansion_rate, target_size))

        constant_defns = []
        for lower_bound, target_size in constants:
            constant_defns.append((lower_bound, target_size))


        if out_path is None:
            out_path = base_path.parent / ('remeshed.' + out_format)
        out_path.parent.mkdir(exist_ok=True, parents=True)

        nprocs = -1 if nprocs is None else nprocs

        geom_rast_list = []
        hfun_rast_list = []
        interp_rast_list = []

        # Low priority interpolation. e.g. user remeshes based on NCEI
        # but still wants to interpolate GEBCO on mesh
        for dem in interp:
            interp_rast_list.append(Raster(dem))

        # NOTE: Region of interest is calculated from geom_rast_list
        # so they're needed even if hfun and geom are read from file
        _logger.info("Read DEM files")
        for dem_path in dem_paths:
            geom_rast_list.append(Raster(dem_path))
            hfun_rast_list.append(Raster(dem_path))
            interp_rast_list.append(Raster(dem_path))

        _logger.info("Read base mesh")
        if mesh_crs is None:
            mesh_crs = "EPSG:4326"
        init_mesh = Mesh.open(str(base_path), crs=mesh_crs)
        # TODO: Cleanup isolates?

        log_calculation = True
        # Read geometry and hfun from files if provided
        if (geom_file and hfun_file
                and geom_file.is_file() and hfun_file.is_file()):
            geom, hfun = self._read_geom_hfun(geom_file, hfun_file, hfun_crs)
            log_calculation = False

        # Create geometry and hfun from inputs
        elif dem_paths:

            _logger.info("Creating geometry object")
            geom_base_mesh = None
            geom_inputs = geom_rast_list
            if clip_by_mesh:
                _logger.info("Clip raster data by base mesh")
                geom_base_mesh = deepcopy(init_mesh)
            else:
                _logger.info("Union raster data with base mesh")
                geom_inputs = [deepcopy(init_mesh), *geom_rast_list]
            geom = Geom(
                    geom_inputs,
                    base_mesh=geom_base_mesh,
                    zmax=zmax,
                    nprocs=nprocs)


            # NOTE: Instead of passing base mesh to be used as boundary,
            # it is passed as an hfun itself
            hfun_base_mesh = Hfun(deepcopy(init_mesh))
            _logger.info("Calculating size function from mesh...")
            hfun_base_mesh.size_from_mesh()
            _logger.info("Done")
            hfun = Hfun(
                [hfun_base_mesh, *hfun_rast_list],
                hmin=hmin,
                hmax=np.max(hfun_base_mesh.msh_t().value),
                nprocs=nprocs)

            for level, expansion_rate, target_size in contour_defns:
                if expansion_rate is None:
                    expansion_rate = 0.1
                if target_size is None:
                    target_size = hmin
                _logger.info(f"Adding contour to collector:"
                             f" {level} {expansion_rate} {target_size}")
                hfun.add_contour(
                    level, expansion_rate, target_size)

            for lower_bound, target_size in constant_defns:
                hfun.add_constant_value(
                        value=target_size, lower_bound=lower_bound)


            if write_intermediate:
                _logger.info("Calculating final geometry")
                poly_geom = geom.get_multipolygon()

                _logger.info("Writing geom to disk")
                gpd.GeoDataFrame(
                        {'geometry': gpd.GeoSeries(poly_geom)},
                        crs=geom.crs
                    ).to_file(str(out_path)+'.geom.shp')
                del poly_geom
                gc.collect()

                _logger.info("Calculating final size function")
                jig_hfun = hfun.msh_t()

                _logger.info("Writing hfun to disk")
                # This writes in EPSG:4326
                Mesh(jig_hfun).write(
                    str(out_path)+'.hfun.2dm',
                    format='2dm', overwrite=True)
                del jig_hfun
                gc.collect()

                # Read back from file to avoid recalculation of hfun
                # and geom
                geom, hfun = self._read_geom_hfun(
                    str(out_path) + '.geom.shp',
                    str(out_path) + '.hfun.2dm',
                    "EPSG:4326")

                log_calculation = False
        else:
            raise ValueError(
                "Input not valid to initialize geom and hfun")


        if log_calculation:
            # NOTE: If intermediate files are written then we calculated
            # this in previous section
            _logger.info("Calculating final geometry")
        jig_geom = geom.msh_t()
        if log_calculation:
            # NOTE: If intermediate files are written then we calculated
            # this in previous section
            _logger.info("Calculating final size function")
        jig_hfun = hfun.msh_t()

        jig_init = init_mesh.msh_t

        _logger.info("Projecting geometry to be in meters unit")
        utils.msh_t_to_utm(jig_geom)
        _logger.info("Projecting size function to be in meters unit")
        utils.msh_t_to_utm(jig_hfun)
        _logger.info("Projecting initial mesh to be in meters unit")
        utils.msh_t_to_utm(jig_init)


        # pylint: disable=C0325
        if not (jig_geom.crs == jig_hfun.crs == jig_init.crs):
            raise ValueError(
                "Converted UTM CRS for geometry, hfun and init mesh"
                "is not equivalent")


        _logger.info("Calculate remeshing region of interest")
        # Prep for Remeshing
        boxes = [i.get_bbox(crs=jig_geom.crs) for i in geom_rast_list]
        region_of_interest = MultiPolygon(boxes)
        roi_bnds = region_of_interest.bounds
        roi_s = max(roi_bnds[2] - roi_bnds[0], roi_bnds[3] - roi_bnds[1])

        _logger.info("Clip mesh by inverse of region of interest")
        fixed_mesh_w_hole = utils.clip_mesh_by_shape(
            jig_init, region_of_interest, fit_inside=True, inverse=True)

        _logger.info(
                "Get all initial mesh vertices in the region of interest")
        vert_idx_to_refin = utils.get_verts_in_shape(
            jig_hfun, region_of_interest)

        fixed_mesh_w_hole.point['IDtag'][:] = -1
        fixed_mesh_w_hole.edge2['IDtag'][:] = -1

        refine_opts = jigsawpy.jigsaw_jig_t()
        refine_opts.hfun_scal = "absolute"
        refine_opts.hfun_hmin = np.min(jig_hfun.value)
        refine_opts.hfun_hmax = np.max(jig_hfun.value)
        refine_opts.mesh_dims = +2
        # Mesh becomes TOO refined on exact boundaries from DEM
#    refine_opts.mesh_top1 = True
#    refine_opts.geom_feat = True

        jig_remeshed = jigsawpy.jigsaw_msh_t()
        jig_remeshed.ndims = +2

        _logger.info("Remeshing...")
        # Remeshing
        jigsawpy.lib.jigsaw(
                refine_opts,
                jig_geom,
                jig_remeshed,
                init=fixed_mesh_w_hole,
                hfun=jig_hfun)
        jig_remeshed.crs = fixed_mesh_w_hole.crs
        _logger.info("Done")

        if jig_remeshed.tria3['index'].shape[0] == 0:
            _err = 'ERROR: Jigsaw returned empty mesh.'
            _logger.error(_err)
            raise ValueError(_err)

        # TODO: This is irrelevant right now since output file is
        # always is EPSG:4326, enable when APIs for remeshing is added
#    if out_crs is not None:
#        utils.reproject(jig_remeshed, out_crs)

        _logger.info('Finalizing mesh...')
        utils.finalize_mesh(jig_remeshed, sieve)

        _logger.info("Interpolating depths on mesh...")
        # Interpolation
        utils.interpolate_euclidean_mesh_to_euclidean_mesh(
                jig_init, jig_remeshed)
        final_mesh = Mesh(jig_remeshed)
        final_mesh.interpolate(interp_rast_list, nprocs=nprocs)
        _logger.info("Done")


        _logger.info("Writing final mesh to disk...")
        # This writes EPSG:4326 to file, whatever the crs of the object
        final_mesh.write(str(out_path), format=out_format, overwrite=True)
        _logger.info("Done")
