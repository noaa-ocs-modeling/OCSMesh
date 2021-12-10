#!/bin/env python3
import gc
import sys
import pathlib
import logging

import geopandas as gpd
from shapely.geometry import MultiPolygon

from ocsmesh import Raster, Geom, Hfun, JigsawDriver
from ocsmesh.mesh.mesh import Mesh
from ocsmesh.geom.shapely import MultiPolygonGeom
from ocsmesh.hfun.mesh import HfunMesh
from ocsmesh.features.contour import Contour
from ocsmesh.mesh.parsers import sms2dm
from ocsmesh.utils import msh_t_to_2dm


logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S'
    )
#logging.getLogger().setLevel(logging.DEBUG)
#logging.getLogger().setLevel(logging.INFO)


class MeshUpgrader:

    @property
    def script_name(self):
        return 'mesh_upgrader'

    def __init__(self, sub_parser):
        # e.g
        # ./multi_dem_mesh_2.py \
        #       --basemesh data/prusvi/mesh/PRUSVI_COMT.14 \
        #       --demlo gebco_2020_n90.0_s0.0_w-90.0_e0.0.tif \
        #       --demhi ncei19_*.tif \
        #       --out demo/final_mesh.2dm

        this_parser = sub_parser.add_parser(
            self.script_name,
            help="Create a new mesh based on a base-mesh and input DEMs.")

        this_parser.add_argument('--basemesh', required=True)
        this_parser.add_argument('--demlo', nargs='*', required=True)
        this_parser.add_argument('--demhi', nargs='*', required=True)
        this_parser.add_argument('--out', required=True)

    def run(self, args):

        logging.info(args)

        base_path = pathlib.Path(args.basemesh)
        demlo_paths = args.demlo
        demhi_paths = args.demhi
        out_path = pathlib.Path(args.out)

        out_path.parent.mkdir(exist_ok=True, parents=True)

        base_mesh_4_hfun = Mesh.open(base_path, crs="EPSG:4326")
        base_mesh_4_geom = Mesh.open(base_path, crs="EPSG:4326")

        geom_rast_list = []
        hfun_rast_list = []
        hfun_hirast_list = []
        hfun_lorast_list = []
        interp_rast_list = []
        for dem_path in demlo_paths:
            hfun_lorast_list.append(Raster(dem_path))
            interp_rast_list.append(Raster(dem_path))

        for dem_path in demhi_paths:
            geom_rast_list.append(Raster(dem_path))
            hfun_hirast_list.append(Raster(dem_path))
            interp_rast_list.append(Raster(dem_path))


        hfun_rast_list = [*hfun_lorast_list, *hfun_hirast_list]

        geom = Geom(
            geom_rast_list, base_mesh=base_mesh_4_geom,
            zmax=15, nprocs=4)

        hfun = Hfun(
            hfun_rast_list, base_mesh=base_mesh_4_hfun,
            hmin=30, hmax=15000, nprocs=4)

        ## Add contour refinements at 0 separately for GEBCO and NCEI
        ctr1 = Contour(level=0, sources=hfun_hirast_list)
        hfun.add_contour(None, 1e-3, 30, contour_defn=ctr1)

        ctr2 = Contour(level=0, sources=hfun_lorast_list)
        hfun.add_contour(None, 1e-2, 500, contour_defn=ctr2)

        ## Add constant values from 0 to inf on hi-res rasters
        hfun.add_constant_value(30, 0, source_index=list(range(len(demhi_paths))))


        # Calculate geom
        geom_mp = geom.get_multipolygon()
        # Write to disk
        gpd.GeoDataFrame(
                {'geometry': geom_mp},
                crs="EPSG:4326"
                ).to_file(str(out_path) + '.geom.shp')
        del geom_mp

        # Calculate hfun
        hfun_msh_t = hfun.msh_t()
        # Write to disk
        sms2dm.writer(
                msh_t_to_2dm(hfun_msh_t),
                str(out_path) + '.hfun.2dm',
                True)
        del hfun_msh_t


        # Read back stored values to pass to mesh driver
        read_gdf = gpd.read_file(str(out_path) + '.geom.shp')
        geom_from_disk = MultiPolygonGeom(
            MultiPolygon(list(read_gdf.geometry)),
            crs=read_gdf.crs)

        read_hfun = Mesh.open(str(out_path) + '.hfun.2dm', crs="EPSG:4326")
        hfun_from_disk = HfunMesh(read_hfun)

        jigsaw = JigsawDriver(geom_from_disk, hfun=hfun_from_disk, initial_mesh=None)
        jigsaw.verbosity = 1

        ## Execute mesher (processing of geom and hfun happens here)
        mesh = jigsaw.run()

        ## Free-up memory
        del read_gdf
        del geom_from_disk
        del read_hfun
        del hfun_from_disk
        gc.collect()

        mesh.write(str(out_path) + '.raw.2dm', format='2dm', overwrite=True)

        ## Interpolate DEMs on the mesh
        mesh.interpolate(interp_rast_list, nprocs=4)

        ## Output
        mesh.write(out_path, format='2dm', overwrite=True)
