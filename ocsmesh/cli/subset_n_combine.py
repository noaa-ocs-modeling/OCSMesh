#!/bin/env python3
from copy import deepcopy
import logging
from pathlib import Path
import sys
from time import time

import geopandas as gpd
import jigsawpy
from jigsawpy.msh_t import jigsaw_msh_t
from jigsawpy.jig_t import jigsaw_jig_t
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection
from shapely.ops import polygonize, unary_union

from ocsmesh import Raster, Geom, Mesh, Hfun, utils

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S'
    )
#logging.getLogger().setLevel(logging.DEBUG)
#logging.getLogger().setLevel(logging.INFO)

_logger = logging.getLogger(__name__)

class SubsetAndCombine:

    @property
    def script_name(self):
        return 'subset_n_combine'

    def __init__(self, sub_parser):

        this_parser = sub_parser.add_parser(
            self.script_name,
            help="Subset and combine the two input meshes based on the region of interset")

        this_parser.add_argument(
                "--rasters", nargs='+', type=Path,
                help="List of rasters used for calculating depth cutoff")
        this_parser.add_argument(
                "-o", "--out", type=Path, metavar="DIRECTORY",
                help="Output directory")
        this_parser.add_argument(
                "-a", "--outall", action='store_true',
                help="Flag to indicate all intermediate shapes to be written to disk")
        this_parser.add_argument(
                "-c", "--crs", default="epsg:4326",
                help="Input mesh CRS and output CRS [experimental]")
        this_parser.add_argument(
                "-d", "--cutoff", type=float, default=-200,
                help="Cut-off depth (positive up) for calculating"
                     + " region subsetted from fine mesh")
        this_parser.add_argument(
                "-s", "--isotach-speed", metavar="WIND_SPEED",
                choices=[34, 50, 64], type=int, default=34,
                help="Isotach (34, 50 or 64) to use if region of interset is storm track")
        this_parser.add_argument(
                "-n", "--adv-buffer-n-layers", metavar="N_LAYERS",
                type=int, default=9,
                help="Number of buffer layers to mesh between fine and coarse mesh")
        this_parser.add_argument(
                "--adv-upstream-area-max", metavar="AREA",
                type=float, default=100,
                help="Advanced option: maximum segment area for calculating upstream")
        this_parser.add_argument(
                "--adv-rel-island-area-min", metavar="AREA",
                type=float, default=1/100,
                help="Advanced option: minimum segment area for"
                     + " dropping islands during shape cleanup")
        this_parser.add_argument(
                "fine_mesh", type=Path,
                help="Path to the fine mesh")
        this_parser.add_argument(
                "coarse_mesh", type=Path,
                help="Path to the coarse mesh")
        this_parser.add_argument(
                "region_of_interset", type=Path,
                help="Path to the shape file for region of interest or storm track")

    def run(self, args):

        # Currently only used for clipping!
        pathlist_raster = args.rasters

        path_fine = args.fine_mesh
        path_coarse = args.coarse_mesh

        # e.g. track - test it's shapefile
        roi = args.region_of_interset

        out_dir = args.out

        # TODO: Clarify what "crs" refers to (it's both input and output!)
        crs = CRS.from_user_input(args.crs)

        wind_speed = args.isotach_speed
        cutoff_elev = args.cutoff

        # 9 layers: fine mesh coarser than coarse mesh in some areas!
        num_buffer_layers = args.adv_buffer_n_layers

        upstream_size_max = args.adv_upstream_area_max
        rel_island_area_min = args.adv_rel_island_area_min

        out_all = args.outall

        self._main(
            pathlist_raster, path_fine, path_coarse, roi,
            cutoff_elev, upstream_size_max,
            wind_speed, num_buffer_layers, rel_island_area_min,
            out_dir, crs, out_all)

    def _get_largest_polygon(self, mpoly):
        if isinstance(mpoly, Polygon):
            return mpoly

        return sorted(mpoly.geoms, key=lambda i: i.area)[-1]

    def _get_polygons_smaller_than(self, mpoly, size):
        if isinstance(mpoly, Polygon):
            if mpoly.area <= size:
                return mpoly
            return Polygon()

        areas = np.array([p.area for p in mpoly.geoms])

        return MultiPolygon(np.extract(areas < size, mpoly.geoms).tolist())



    def _get_region_bounded_by_isotach(self, track_file, wind_speed=34):

        gdf_region_of_interset = gpd.read_file(track_file)
        if "RADII" in gdf_region_of_interset.columns:
            gdf_specific_isotach = gdf_region_of_interset[
                    gdf_region_of_interset.RADII.astype(int) == wind_speed]
            isotach_exterior_polygon = MultiPolygon(
                list(polygonize(list(gdf_specific_isotach.exterior))))
            region_of_interest = isotach_exterior_polygon

        else:
            region_of_interest = gdf_region_of_interset.unary_union


        return region_of_interest

    def _calculate_clipping_polygon(
            self,
            pathlist_raster,
            region_of_interest, crs,
            cutoff_elev=-200,
            upstream_size_max=100,
            upstream_poly_list=None):

        rasters = [Raster(raster_file) for raster_file in pathlist_raster]
        geom_cutoff = Geom(
                rasters, zmin=cutoff_elev,
                base_shape=region_of_interest,
                base_shape_crs=crs)
        clip_poly_draft = geom_cutoff.get_multipolygon()

        # Add back upstream
        poly_upstreams = []
        if upstream_poly_list is not None:
            for poly in upstream_poly_list:
                poly_upstreams.append(
                    self._get_polygons_smaller_than(
                        poly.difference(clip_poly_draft),
                        upstream_size_max))

        # Islands are not of intereset when clipping high and low-res meshes
        clip_poly = utils.remove_holes(unary_union([
            clip_poly_draft, *poly_upstreams]))

        return clip_poly



    def _get_polygon_from_geom_collection(self, shape):

        if not isinstance(shape, GeometryCollection):
            raise ValueError(
                "Expected a GeometryCollection, received {}".format(
                    type(shape)))

        polygon_list = []
        for g in shape.geoms:
            if isinstance(g, Polygon):
                polygon_list.append(g)
            elif isinstance(g, MultiPolygon):
                polygon_list.extend(g.geoms)
        if len(polygon_list) == 0:
            raise ValueError("No polygon in the intersection!")

        return MultiPolygon(polygon_list)

    def _add_one_mesh_layer_to_polygon(
            self,
            polygon,
            fine_mesh, fine_polygon,
            coarse_mesh, coarse_polygon):

        fine_clip_plus_layer = utils.clip_mesh_by_shape(
                fine_mesh, fine_polygon,
                fit_inside=False, adjacent_layers=1)
        poly_1_lyr_hires = utils.get_mesh_polygons(
                fine_clip_plus_layer).difference(fine_polygon)

        coarse_clip_plus_layer = utils.clip_mesh_by_shape(
                coarse_mesh, coarse_polygon,
                fit_inside=False, adjacent_layers=1)
        poly_1_lyr_lowres = utils.get_mesh_polygons(
                coarse_clip_plus_layer).difference(coarse_polygon)

        # Add the one layer to the buffer to make boundaries shape conformal
        return unary_union(
                [polygon, poly_1_lyr_hires, poly_1_lyr_lowres])


    def _add_overlap_to_polygon(self, mesh, polygon):

        # Find segment to add to polygon
        clipped_add = utils.clip_mesh_by_shape(
                mesh, polygon,
                fit_inside=False,
                inverse=False,
                check_cross_edges=True)
        poly_add = utils.get_mesh_polygons(clipped_add)
        new_polygon = unary_union([polygon, poly_add])

        # Inverse of above
        clipped_mesh = utils.clip_mesh_by_shape(
                mesh, polygon,
                fit_inside=True,
                inverse=True,
                check_cross_edges=True
                )

        return new_polygon, clipped_mesh

    def _calculate_mesh_size_function(self, hires_mesh_clip, lores_mesh_clip, crs):

        # calculate mesh size for clipped bits
        hfun_hires = Hfun(Mesh(deepcopy(hires_mesh_clip)))
        hfun_hires.size_from_mesh()
        hfun_lowres = Hfun(Mesh(deepcopy(lores_mesh_clip)))
        hfun_lowres.size_from_mesh()

#    _logger.info("Writing hfun clips...")
#    start = time()
#    hfun_hires.mesh.write(str(out_dir / "hfun_fine.2dm"), format="2dm", overwrite=True)
#    hfun_lowres.mesh.write(str(out_dir / "hfun_coarse.2dm"), format="2dm", overwrite=True)
#    _logger.info(f"Done in {time() - start} sec")

        jig_hfun = utils.merge_msh_t(
            hfun_hires.msh_t(), hfun_lowres.msh_t(),
            out_crs=crs)#jig_geom) ### TODO: CRS MUST BE == GEOM_MSHT CRS

        return jig_hfun


    def _generate_mesh_for_buffer_region(
            self, buffer_polygon, jig_hfun, crs):

        seam = Geom(buffer_polygon, crs=crs)

        jig_geom = seam.msh_t()

        # IMPORTANT: Setting these to -1 results in NON CONFORMAL boundary
#    jig_geom.vert2['IDtag'][:] = -1
#    jig_geom.edge2['IDtag'][:] = -1

        jig_init = deepcopy(seam.msh_t())
        jig_init.vert2['IDtag'][:] = -1
        jig_init.edge2['IDtag'][:] = -1

        # Calculate length of all edges on geom
        geom_edges = jig_geom.edge2['index']
        geom_coords = jig_geom.vert2['coord'][geom_edges, :]
        geom_edg_lens = np.sqrt(np.sum(
            np.power(np.abs(np.diff(geom_coords, axis=1)), 2),
            axis=2)).squeeze()

        # TODO: Use marche to calculate mesh size in the area between
        # the two regions?

        _logger.info("Meshing...")
        start = time()
        opts = jigsaw_jig_t()
        opts.hfun_scal = "absolute"
        opts.hfun_hmin = np.min(geom_edg_lens)
        opts.hfun_hmax = np.max(geom_edg_lens)
#    opts.hfun_hmin = np.min(jig_hfun.value.ravel())
#    opts.hfun_hmax = np.max(jig_hfun.value.ravel())
        opts.optm_zip_ = False
        opts.optm_div_ = False
        opts.mesh_dims = +2
        opts.mesh_rad2 = 1.05
#    opts.mesh_rad2 = 2.0

        jig_mesh = jigsaw_msh_t()
        jig_mesh.mshID = 'euclidean-mesh'
        jig_mesh.ndims = 2
        jig_mesh.crs = jig_geom.crs

        jigsawpy.lib.jigsaw(
            opts, jig_geom, jig_mesh,
#        hfun=jig_hfun,
            init=jig_init
            )

        jig_mesh.value = np.zeros((jig_mesh.vert2.shape[0], 1))
        self._transform_mesh(jig_mesh, crs)

        return jig_mesh


    def _transform_mesh(self, mesh, out_crs):

        transformer = Transformer.from_crs(
            mesh.crs, out_crs, always_xy=True)

        coords = mesh.vert2['coord']

        # pylint: disable=E0633
        coords[:, 0], coords[:, 1] = transformer.transform(
                coords[:, 0], coords[:, 1])
        mesh.vert2['coord'][:] = coords
        mesh.crs = out_crs


    def _merge_all_meshes(self, crs, jig_mesh, jig_clip_lowres, jig_clip_hires):

        # Combine mesh
        # 1. combine hires and low-res
        jig_old = utils.merge_msh_t(
            jig_clip_lowres, jig_clip_hires,
            drop_by_bbox=False, out_crs=crs)

        # 2. Create kdtree for boundary of combined mesh
        old_mesh_bdry_edges = utils.get_boundary_edges(jig_old)
        old_mesh_bdry_verts = np.unique(old_mesh_bdry_edges)
        old_mesh_bdry_coords = jig_old.vert2['coord'][old_mesh_bdry_verts]

        tree_old = cKDTree(old_mesh_bdry_coords)

        # 3. Find shared boundary nodes from the tree and bdry nodes
        new_mesh_bdry_edges = utils.get_boundary_edges(jig_mesh)
        new_mesh_bdry_verts = np.unique(new_mesh_bdry_edges)
        new_mesh_bdry_coords = jig_mesh.vert2['coord'][new_mesh_bdry_verts]

        tree_new = cKDTree(new_mesh_bdry_coords)

        neigh_idxs = tree_old.query_ball_tree(tree_new, r=1e-6)

        # 4. Create a map for shared nodes
        map_idx_shared = {}
        for idx_tree_old, neigh_idx_list in enumerate(neigh_idxs):
            num_match = len(neigh_idx_list)
            if num_match == 0:
                continue
            if num_match > 1:
                raise ValueError("More than one node match on boundary!")

            idx_tree_new = neigh_idx_list[0]

            map_idx_shared[new_mesh_bdry_verts[idx_tree_new]] = old_mesh_bdry_verts[idx_tree_old]

        # 5. Combine seam into the rest replacing the index for shared nodes
        #    with the ones from tree
        mesh_types = {
            'tria3': 'TRIA3_t',
            'quad4': 'QUAD4_t',
            'hexa8': 'HEXA8_t'
        }

        coord = []
        elems = {k: [] for k in mesh_types}
        value = []
        offset = 0

        for k in mesh_types:
            elems[k].append(getattr(jig_old, k)['index'] + offset)
        coord.append(jig_old.vert2['coord'])
        value.append(jig_old.value)
        offset += coord[-1].shape[0]

        # Drop shared vertices and update element cnn based on map and dropped offset
        mesh_orig_idx = np.arange(len(jig_mesh.vert2))
        mesh_shrd_idx = np.unique(list(map_idx_shared.keys()))
        mesh_renum_idx = np.setdiff1d(mesh_orig_idx, mesh_shrd_idx)
        map_to_combined_idx = {
            index: i + offset for i, index in enumerate(mesh_renum_idx)}
        map_to_combined_idx.update(map_idx_shared)

        for k in mesh_types:
            cnn = getattr(jig_mesh, k)['index']
            # If it's empty executing list comprehension results in a
            # (0,) shape instead of (0, 4)
            if cnn.shape[0] == 0:
                elems[k].append(cnn)
                continue

            elems[k].append(np.array([[map_to_combined_idx[x]
                                      for x in  elm] for elm in cnn]))

        coord.append(jig_mesh.vert2['coord'][mesh_renum_idx, :])
        value.append(jig_mesh.value[mesh_renum_idx])

        # Putting it all together
        composite_mesh = jigsaw_msh_t()
        composite_mesh.mshID = 'euclidean-mesh'
        composite_mesh.ndims = 2

        composite_mesh.vert2 = np.array(
                [(crd, 0) for crd in np.vstack(coord)],
                dtype=jigsaw_msh_t.VERT2_t)
        composite_mesh.value = np.array(
                np.vstack(value),
                dtype=jigsaw_msh_t.REALS_t)
        for k, v in mesh_types.items():
            setattr(composite_mesh, k, np.array(
                [(cnn, 0) for cnn in np.vstack(elems[k])],
                dtype=getattr(jigsaw_msh_t, v)))

        composite_mesh.crs = crs

        return composite_mesh, mesh_shrd_idx


    def _write_polygon(self, polygon, crs, path):
        gdf  = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygon), crs=crs)
        gdf.to_file(path)

    def _write_jigsaw(self, jigsaw_mesh, path):
        mesh_obj = Mesh(jigsaw_mesh)
        mesh_obj.write(path, format="2dm", overwrite=True)

    def _write_outputs(
            self,
            out_dir,
            out_all,
            crs,
            poly_clip_hires,
            poly_clip_lowres,
            poly_seam,
            poly_clipper,
            mesh_fine,
            mesh_coarse,
            jig_buffer_mesh,
            jig_clip_hires,
            jig_clip_lowres,
            jig_combined_mesh,
            buffer_shrd_idx
            ):

        if out_all:
            _logger.info("Writing shapes...")
            start = time()
            self._write_polygon(poly_clip_hires, crs, out_dir/"hires")
            self._write_polygon(poly_clip_lowres, crs, out_dir/"lowres")
            self._write_polygon(poly_seam, crs, out_dir/"seam")
            self._write_polygon(poly_clipper, crs, out_dir/"clip")
            _logger.info(f"Done in {time() - start} sec")

            _logger.info("Writing input mesh (copy)...")
            start = time()
            mesh_fine.write(str(out_dir / "fine.2dm"), format="2dm", overwrite=True)
            mesh_coarse.write(str(out_dir / "coarse.2dm"), format="2dm", overwrite=True)
            _logger.info(f"Done in {time() - start} sec")

            _logger.info("Writing mesh...")
            start = time()
            self._write_jigsaw(jig_buffer_mesh, out_dir/"seam_mesh.2dm")
            self._write_jigsaw(jig_clip_hires, out_dir/"hires_mesh.2dm")
            self._write_jigsaw(jig_clip_lowres, out_dir/"lowres_mesh.2dm")
            _logger.info(f"Done in {time() - start} sec")

        _logger.info("Writing mesh...")
        start = time()
        self._write_jigsaw(jig_combined_mesh, out_dir/"final_mesh.2dm")
        _logger.info(f"Done in {time() - start} sec")


    def _interpolate_values(self, jig_combined_mesh, mesh_fine, mesh_coarse):
        interp_1_msh_t = deepcopy(jig_combined_mesh)
        utils.interpolate_euclidean_mesh_to_euclidean_mesh(
            mesh_fine.msh_t, interp_1_msh_t,
            method='linear', fill_value=np.nan)

        interp_2_msh_t = deepcopy(jig_combined_mesh)
        utils.interpolate_euclidean_mesh_to_euclidean_mesh(
            mesh_coarse.msh_t, interp_2_msh_t,
            method='linear', fill_value=np.nan)


        # Manually get the nan values and only overwrite them!
        mask_1 = np.isnan(interp_1_msh_t.value)
        mask_2 = np.logical_not(np.isnan(interp_2_msh_t.value))
        mask = np.logical_and(mask_1, mask_2)

        jig_combined_mesh.value = interp_1_msh_t.value
        jig_combined_mesh.value[mask] = interp_2_msh_t.value[mask]


    def _main(
            self,
            pathlist_raster, path_fine, path_coarse, track_file,
            cutoff_elev, upstream_size_max,
            wind_speed, num_buffer_layers, rel_island_area_min,
            out_dir, crs, out_all):

        _logger.info("Reading meshes...")
        start = time()
        mesh_fine = Mesh.open(path_fine, crs=crs)
        mesh_coarse = Mesh.open(path_coarse, crs=crs)
        _logger.info(f"Done in {time() - start} sec")


        _logger.info("Calculate mesh polygons...")
        start = time()
        poly_fine = self._get_largest_polygon(mesh_fine.get_multipolygon())
        poly_coarse = self._get_largest_polygon(mesh_coarse.get_multipolygon())
        _logger.info(f"Done in {time() - start} sec")

        _logger.info("Calculate impact area...")
        start = time()
        poly_isotach = self._get_region_bounded_by_isotach(track_file, wind_speed)
        poly_storm_roi = poly_isotach.intersection(poly_fine)

        poly_clipper = self._calculate_clipping_polygon(
            pathlist_raster=pathlist_raster,
            region_of_interest=poly_storm_roi,
            crs=crs,
            cutoff_elev=cutoff_elev,
            upstream_size_max=upstream_size_max,
            upstream_poly_list=[poly_fine])
        _logger.info(f"Done in {time() - start} sec")


        _logger.info("Calculate clipped polygons...")
        jig_clip_hires_0 = utils.clip_mesh_by_shape(
                mesh_fine.msh_t, poly_clipper,
                fit_inside=False)
        # 9 layers buffer because of fine mesh being coarser than coarse
        # in some non US areas!
        jig_clip_lowres_0 = utils.clip_mesh_by_shape(
                mesh_coarse.msh_t, poly_clipper,
                fit_inside=True,
                inverse=True,
                adjacent_layers=9)
        _logger.info(f"Done in {time() - start} sec")

        _logger.info("Calculate clipped polygons...")
        start = time()
        poly_clip_hires_0 = utils.remove_holes(
                utils.get_mesh_polygons(jig_clip_hires_0))
        poly_clip_lowres_0 = utils.get_mesh_polygons(jig_clip_lowres_0)
        _logger.info(f"Done in {time() - start} sec")

        _logger.info("Calculating buffer region...")
        start = time()

        # Remove the two mesh clip regions from the coarse mesh polygon
        poly_seam_0 = poly_coarse.difference(
                unary_union([poly_clip_lowres_0, poly_clip_hires_0]))

        poly_seam_1 = poly_seam_0.intersection(poly_fine)

        # Get rid of non polygon results of the intersection
        poly_seam_2 = poly_seam_1
        if isinstance(poly_seam_1, GeometryCollection):
            poly_seam_2 = self._get_polygon_from_geom_collection(poly_seam_1)

        # Get one layer on each mesh
        poly_seam_3 = self._add_one_mesh_layer_to_polygon(
                poly_seam_2,
                mesh_fine.msh_t, poly_clip_hires_0,
                mesh_coarse.msh_t, poly_clip_lowres_0)


        # Attach overlaps to buffer region (due to 1 layer and upstream)
        poly_seam_4, jig_clip_hires = self._add_overlap_to_polygon(jig_clip_hires_0, poly_seam_3)
        poly_seam_5, jig_clip_lowres = self._add_overlap_to_polygon(jig_clip_lowres_0, poly_seam_4)

        # Cleanup buffer shape
        poly_seam_6 = utils.remove_holes_by_relative_size(
                poly_seam_5, rel_island_area_min)

        poly_seam_7 = utils.drop_extra_vertex_from_polygon(poly_seam_6)
        _logger.info(f"Done in {time() - start} sec")


        _logger.info("Calculate reclipped polygons...")
        start = time()
        poly_clip_hires = utils.remove_holes(
                utils.get_mesh_polygons(jig_clip_hires))
        poly_clip_lowres = utils.get_mesh_polygons(jig_clip_lowres)
        _logger.info(f"Done in {time() - start} sec")

        poly_seam_8 = poly_seam_7.difference(
                    unary_union([poly_clip_hires, poly_clip_lowres]))


        poly_seam = poly_seam_8

        # TODO: Get CRS correctly from geom utm
#    jig_hfun = self._calculate_mesh_size_function(jig_clip_hires, jig_clip_lowres, crs)
        jig_buffer_mesh = self._generate_mesh_for_buffer_region(poly_seam, None, crs)

        _logger.info("Combining meshes...")
        start = time()
        jig_combined_mesh, buffer_shrd_idx = self._merge_all_meshes(
                crs, jig_buffer_mesh, jig_clip_lowres, jig_clip_hires)
        _logger.info(f"Done in {time() - start} sec")

        # NOTE: This call also detects overlap issues
        utils.finalize_mesh(jig_combined_mesh)

        self._interpolate_values(jig_combined_mesh, mesh_fine, mesh_coarse)

        # TODO: Interpolate DEM?

        self._write_outputs(
            out_dir,
            out_all,
            crs,
            poly_clip_hires,
            poly_clip_lowres,
            poly_seam,
            poly_clipper,
            mesh_fine,
            mesh_coarse,
            jig_buffer_mesh,
            jig_clip_hires,
            jig_clip_lowres,
            jig_combined_mesh,
            buffer_shrd_idx
        )
