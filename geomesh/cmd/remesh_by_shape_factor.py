#!/usr/bin/env python
from pathlib import Path
from copy import deepcopy
import logging, sys

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, box, Polygon
from shapely.ops import polygonize
import jigsawpy

from geomesh import Raster, Geom, Hfun, Mesh
from geomesh import utils

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S'
    )
#logging.getLogger().setLevel(logging.DEBUG)
#logging.getLogger().setLevel(logging.INFO)

_logger = logging.getLogger(__name__)

# Enable KML driver
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

def main(args):

    # Get inputs
    mesh_file = args.mesh
    mesh_crs = args.mesh_crs

    shape_path = args.shape

    refine_upstream = args.upstream
    refine_factor = args.factor

    sieve = args.sieve
    interp = args.interpolate

    out_path = args.output
    out_format = args.output_format
    nprocs = args.nprocs

    interp_rast_list = list()
    for dem in interp:
        interp_rast_list.append(Raster(dem))

    mesh = Mesh.open(mesh_file, crs=mesh_crs)

    geom = Geom(deepcopy(mesh))
    geom_jig = geom.msh_t()

    hfun = Hfun(deepcopy(mesh))
    hfun.size_from_mesh()

    # DO NOT DEEPCOPY
    hfun_jig = hfun.msh_t()

    utils.reproject(geom_jig, hfun_jig.crs)

    init_jig = None

    # If there's an input shape, refine in the shape, otherwise
    # refine the whole domain by the factor
    if shape_path:
        gdf_shape = gpd.read_file(shape_path)

        gdf_shape = gdf_shape.to_crs(mesh.crs)

        mesh_poly = mesh.hull.multipolygon()
        gdf_mesh_poly = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(mesh_poly), crs=mesh.crs)

        gdf_to_refine = gpd.overlay(
                gdf_mesh_poly, gdf_shape, how='intersection')

        gdf_diff = gpd.overlay(
                gdf_mesh_poly, gdf_shape, how='difference')
        diff_polys = list()
        for geom in gdf_diff.geometry:
            if isinstance(geom, Polygon):
                diff_polys.append(geom)
            elif isinstance(geom, MultiPolygon):
                diff_polys.extend(geom)

        if refine_upstream:
            # TODO: Check for multipolygon and single polygon in multi assumption
            area_ref = 0.05 * np.sum([i.area for i in gdf_to_refine.geometry])
            upstream_polys = list()
            for ipoly in diff_polys:
                if ipoly.area < area_ref:
                    upstream_polys.append(ipoly)
            if upstream_polys:
                gdf_upstream = gpd.GeoDataFrame(
                        geometry=gpd.GeoSeries(upstream_polys),
                        crs=gdf_diff.crs)

                gdf_to_refine = gpd.overlay(
                        gdf_upstream, gdf_to_refine, how='union')


        gdf_to_refine = gdf_to_refine.to_crs(hfun_jig.crs)
        refine_polys = gdf_to_refine.unary_union

        init_jig = deepcopy(mesh.msh_t)
        utils.reproject(init_jig, hfun_jig.crs)
        utils.clip_mesh_by_shape(
                init_jig,
                refine_polys,
                fit_inside=True,
                inverse=True,
                in_place=True)

        # Fix elements in the inital mesh that are NOT clipped by refine
        # polygon
        init_jig.vert2['IDtag'][:] = -1

        # Reduce hfun by factor in refinement area
        vert_in = utils.get_verts_in_shape(hfun_jig, refine_polys)
        # Modifying in-place
        hfun_jig.value[vert_in] = (
                hfun_jig.value.take(vert_in, axis=0) / refine_factor)


    else:
        hfun_jig.value[:] = hfun_jig.value / refine_factor


    if not (geom_jig.crs == hfun_jig.crs
            and (init_jig and init_jig.crs == hfun_jig.crs)):
        raise ValueError(
            f"CRS for geometry, hfun and init mesh is not the same")

    opts = jigsawpy.jigsaw_jig_t()
    opts.hfun_scal = "absolute"
    opts.hfun_hmin = hfun.hmin
    opts.hfun_hmax = hfun.hmax
    opts.mesh_dims = +2

    remesh_jig = jigsawpy.jigsaw_msh_t()
    remesh_jig.mshID = 'euclidean-mesh'                           
    remesh_jig.ndims = 2                                          
    remesh_jig.crs = init_jig.crs

    jigsawpy.lib.jigsaw(
            opts, geom_jig, remesh_jig, init=init_jig, hfun=hfun_jig)

    utils.finalize_mesh(remesh_jig, sieve)

    # Interpolate from inpu mesh and DEM if any
    utils.interpolate_euclidean_mesh_to_euclidean_mesh(
            mesh.msh_t, remesh_jig)
    final_mesh = Mesh(remesh_jig)
    if interp_rast_list:
        final_mesh.interpolate(interp_rast_list, nprocs=nprocs)

    # Write to disk
    final_mesh.write(
            str(out_path), format=out_format, overwrite=True)


if __name__ == "__main__":
    # e.g 
    # ./remesh_by_shape_factor \
    #   --output <OUT_PATH> \
    #   --shape <REFINEMENT_SHAPE_PATH> \
    #   --factor <REFINEMENT_FACTOR> \
    #   --upstream \
    #   --in-crs <MESH_CRS> \
    #   <MESH_FILE> \

    parser = argparse.ArgumentParser()

    parser.add_argument('--in-crs', default='EPSG:4326')
    parser.add_argument('--shape', type=str)

    parser.add_argument('-u', '--upstream', action='store_true')
    parser.add_argument('--factor', default=2, type=float)
    parser.add_argument('--constant', default=200, type=float)
    parser.add_argument('-s', '--sieve', type=float)

    parser.add_argument(
        '--interpolate', nargs='+', type=Path, default=list(),
        help="To interpolate from depth of DEMs not involved in"
             " the remeshing process")

    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('-f', '--output-format', default='2dm')
    parser.add_argument('--nprocs', type=int, default=-1)

    parser.add_argument('mesh', required=True, type=Path)

    args = parser.parse_args()
    main(args)
