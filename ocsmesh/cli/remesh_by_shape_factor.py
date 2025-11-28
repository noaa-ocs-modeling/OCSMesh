#!/usr/bin/env python
from pathlib import Path
from copy import deepcopy
import logging
import sys

from fiona.drvsupport import supported_drivers
import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform
import jigsawpy
from pyproj import Transformer

from ocsmesh import Raster, Geom, Hfun, Mesh
from ocsmesh import utils
from ocsmesh.engine.factory import get_mesh_engine

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S'
    )
#logging.getLogger().setLevel(logging.DEBUG)
#logging.getLogger().setLevel(logging.INFO)

_logger = logging.getLogger(__name__)

# Enable KML driver
#from stackoverflow issue 72960340
supported_drivers['KML'] = 'rw'
supported_drivers['LIBKML'] = 'rw'

class RemeshByShape:

    @property
    def script_name(self):
        return 'remesh_by_shape'

    def __init__(self, sub_parser):

        this_parser = sub_parser.add_parser(
            self.script_name,
            help="Locally refine an existing input mesh based"
                 + " on input DEMs within the region specified"
                 + " by input shape and cut-off depth.")

        this_parser.add_argument('--in-crs', default='EPSG:4326')
        this_parser.add_argument('--shape', type=str)

        this_parser.add_argument('-u', '--upstream', action='store_true')
        this_parser.add_argument(
            '--cutoff', default=-250, type=float,
            help="Refinement cutoff depth in meters (positive up)")
        this_parser.add_argument('--factor', default=2, type=float)
        this_parser.add_argument(
            '--contour',
            action='append', nargs='+', type=float, dest='contours',
            metavar='CONTOUR_DEFN', default=[],
            help="Each contour's (level, [expansion, target])"
                 " to be applied on all size functions in collector")
        this_parser.add_argument(
            '--patch',
            action='append', nargs=3, type=float, dest='patches',
            metavar='PATCH_DEFN', default=[],
            help="Specify patch mesh size above a given contour level"
                 " by passing (lower_bound, expansion, target_size)"
                 " for each patch")
        this_parser.add_argument(
            '--constant',
            action='append', nargs=2, type=float, dest='constants',
            metavar='CONST_DEFN', default=[],
            help="Specify constant mesh size above a given contour level"
                 " by passing (lower_bound, target_size) for each constant")
        this_parser.add_argument('-s', '--sieve', type=float)

        this_parser.add_argument(
            '--interpolate', nargs='+', type=Path, default=[],
            help="To interpolate from depth of DEMs not involved in"
                 " the remeshing process")

        this_parser.add_argument('-o', '--output', type=Path)
        this_parser.add_argument('-f', '--output-format', default='2dm')
        this_parser.add_argument('--nprocs', type=int, default=-1)

        this_parser.add_argument('mesh', type=Path)


    def run(self, args):

        # Get inputs
        mesh_file = args.mesh
        mesh_crs = args.mesh_crs

        shape_path = args.shape

        refine_upstream = args.upstream
        refine_factor = args.factor
        refine_cutoff = args.cutoff

        contours = args.contours
        patches = args.patches
        constants = args.constants

        sieve = args.sieve
        interp = args.interpolate

        out_path = args.output
        out_format = args.output_format
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

        patch_defns = []
        for lower_bound, target_size in patches:
            patch_defns.append((lower_bound, expansion_rate, target_size))

        constant_defns = []
        for lower_bound, target_size in constants:
            constant_defns.append((lower_bound, target_size))


        interp_rast_list = []
        for dem in interp:
            interp_rast_list.append(Raster(dem))

        mesh = Mesh.open(mesh_file, crs=mesh_crs)

        geom = Geom(deepcopy(mesh))
        geom_shape = geom.geoseries()

        initial_hfun = Hfun(deepcopy(mesh))
        initial_hfun.size_from_mesh()

        # DO NOT DEEPCOPY
        initial_hfun_meshdata = initial_hfun.meshdata()
        ref_crs = initial_hfun_meshdata.crs

        utils.reproject(geom_shape, ref_crs)

        init_meshdata = None

        # If there's an input shape, refine in the shape, otherwise
        # refine the whole domain by the factor
        if shape_path or refine_cutoff is not None:

            mesh_poly = mesh.hull.multipolygon()
            gdf_mesh_poly = gpd.GeoDataFrame(
                    geometry=gpd.GeoSeries(mesh_poly), crs=mesh.crs)

            if shape_path:
                gdf_shape = gpd.read_file(shape_path)

                gdf_shape = gdf_shape.to_crs(mesh.crs)


                gdf_to_refine = gpd.overlay(
                        gdf_mesh_poly, gdf_shape, how='intersection')

                gdf_diff = gpd.overlay(
                        gdf_mesh_poly, gdf_shape, how='difference')
                diff_polys = []
                for geom in gdf_diff.geometry:
                    if isinstance(geom, Polygon):
                        diff_polys.append(geom)
                    elif isinstance(geom, MultiPolygon):
                        diff_polys.extend(geom)

                if refine_upstream:
                    # TODO: Check for multipolygon and single polygon in multi assumption
                    area_ref = 0.05 * np.sum(
                            [i.area for i in gdf_to_refine.geometry])
                    upstream_polys = []
                    for ipoly in diff_polys:
                        if ipoly.area < area_ref:
                            upstream_polys.append(ipoly)
                    if upstream_polys:
                        gdf_upstream = gpd.GeoDataFrame(
                                geometry=gpd.GeoSeries(upstream_polys),
                                crs=gdf_diff.crs)

                        gdf_to_refine = gpd.overlay(
                                gdf_upstream, gdf_to_refine, how='union')
            else:
                gdf_to_refine = gdf_mesh_poly

            gdf_to_refine = gdf_to_refine.to_crs(ref_crs)

            if refine_cutoff is not None:
                cutoff_mp = mesh.get_multipolygon(zmin=refine_cutoff)
                cutoff_gdf = gpd.GeoDataFrame(
                        geometry=gpd.GeoSeries(cutoff_mp), crs=mesh.crs)
                cutoff_gdf = cutoff_gdf.to_crs(ref_crs)
                gdf_to_refine = gpd.overlay(
                        gdf_to_refine, cutoff_gdf, how='intersection')

            refine_polys = gdf_to_refine.union_all()

            # Initial mesh for the refinement (all except refinement area)
            init_meshdata = deepcopy(mesh.meshdata)
            utils.reproject(init_meshdata, ref_crs)
            utils.clip_mesh_by_shape(
                    init_meshdata,
                    refine_polys,
                    fit_inside=True,
                    inverse=True,
                    in_place=True)

            # Fix elements in the inital mesh that are NOT clipped by refine
            # polygon
            init_meshdata.vert2['IDtag'][:] = -1

            # Preparing refinement size function
            vert_in = utils.get_verts_in_shape(initial_hfun_meshdata, refine_polys)
            # Reduce hfun by factor in refinement area; modifying in-place

            refine_hfun_meshdata = utils.clip_mesh_by_shape(
                initial_hfun_meshdata, refine_polys, fit_inside=False)

            utils.clip_mesh_by_shape(
                initial_hfun_meshdata, refine_polys,
                fit_inside=True, inverse=True, in_place=True)


        else:
            # Refine the whole domain by factor
            refine_hfun_meshdata = deepcopy(initial_hfun_meshdata)

        # Prepare refinement size function with additional criteria
        refine_hfun_meshdata.values = refine_hfun_meshdata.values / refine_factor

        hfun_refine = Hfun(Mesh(deepcopy(refine_hfun_meshdata)))

        transformer = Transformer.from_crs(
                mesh.crs, ref_crs, always_xy=True)
        for level, expansion_rate, target_size in contour_defns:
            if expansion_rate is None:
                expansion_rate = 0.1
            if target_size is None:
                target_size = np.min(refine_hfun_meshdata.values)

            refine_ctr = mesh.get_contour(level=level)
            refine_ctr = transform(transformer.transform, refine_ctr)

            hfun_refine.add_feature(
                    refine_ctr, expansion_rate, target_size,
                    nprocs=nprocs)

        for lower_bound, expansion_rate, target_size in patch_defns:
            refine_mp = mesh.get_multipolygon(zmin=lower_bound)
            refine_mp = transform(transformer.transform, refine_mp)

            hfun_refine.add_patch(
                    refine_mp, expansion_rate, target_size, nprocs)

        for lower_bound, target_size in constant_defns:
            refine_mp = mesh.get_multipolygon(zmin=lower_bound)
            refine_mp = transform(transformer.transform, refine_mp)

            hfun_refine.add_patch(
                    refine_mp, None, target_size, nprocs)

        refine_hfun_meshdata = hfun_refine.meshdata()
        utils.reproject(refine_hfun_meshdata, ref_crs)

        final_hfun_meshdata = utils.merge_meshdata(
                initial_hfun_meshdata, refine_hfun_meshdata,
                out_crs=ref_crs,
                drop_by_bbox=False)

        if not (geom_shape.crs == ref_crs
                and (init_meshdata and init_meshdata.crs == ref_crs)):
            raise ValueError(
                "CRS for geometry, hfun and init mesh is not the same")

        # TODO: Redo after imlementing remesh
#        engine = get_mesh_engine(mesh_engine, **mesh_options)
        meshdata_remeshed = engine.remesh(
            meshdata_init,
            gpd.GeoSeries(region_of_interest, crs=...),
            meshdata_hfun
        )
#        opts = jigsawpy.jigsaw_jig_t()
#        opts.hfun_scal = "absolute"
#        opts.hfun_hmin = np.min(final_hfun_jig.value)
#        opts.hfun_hmax = np.max(final_hfun_jig.value)
#        opts.mesh_dims = +2
#
#        remesh_jig = jigsawpy.jigsaw_msh_t()
#        remesh_jig.mshID = 'euclidean-mesh'
#        remesh_jig.ndims = 2
#        remesh_jig.crs = init_jig.crs
#
#        jigsawpy.lib.jigsaw(
#                opts, geom_jig, remesh_jig,
#                init=init_jig,
#                hfun=final_hfun_jig)

        utils.finalize_mesh(remesh_jig, sieve)

        # Interpolate from inpu mesh and DEM if any
        utils.interpolate_euclidean_mesh_to_euclidean_mesh(
                mesh.meshdata, remesh_meshdata)
        final_mesh = Mesh(remesh_meshdata)
        if interp_rast_list:
            final_mesh.interpolate(interp_rast_list, nprocs=nprocs)

        # Write to disk
        final_mesh.write(
                str(out_path), format=out_format, overwrite=True)
