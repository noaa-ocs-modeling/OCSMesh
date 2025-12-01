#!/bin/env python3
from copy import deepcopy
import logging
from pathlib import Path
import sys
from time import time

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection, MultiPoint
from shapely.ops import polygonize, unary_union, transform

from ocsmesh import Raster, Geom, Mesh, Hfun, utils
from ocsmesh.internal import MeshData
from ocsmesh.engine.factory import get_mesh_engine

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



    def _get_region_of_interest(self, track_file, wind_speed=34):

        # CRS is assumed to be epsg 4326
        crs = 4326
        gdf_roi = gpd.read_file(track_file)
        if not (gdf_roi.crs is None or gdf_roi.crs.equals(crs)):
            gdf_roi = gdf_roi.to_crs(crs)
        if "RADII" in gdf_roi.columns:
            gdf_specific_isotach = gdf_roi[
                    gdf_roi.RADII.astype(int) == wind_speed]
            isotach_exterior_polygon = MultiPolygon(
                list(polygonize(list(gdf_specific_isotach.exterior))))
            region_of_interest = isotach_exterior_polygon

        else:
            region_of_interest = gdf_roi.union_all()


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

        t = Transformer.from_crs(4326, crs, always_xy=True)
        clip_poly_draft = transform(t.transform, clip_poly_draft)

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
                f"Expected a GeometryCollection, received {type(shape)}")

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
                fit_inside=False,
                inverse=True,
                check_cross_edges=True
                )

        return new_polygon, clipped_mesh

    def _calculate_mesh_size_function(
            self,
            buffer_domain,
            hires_mesh_clip,
            lores_mesh_clip,
            buffer_crs
        ):

        assert buffer_crs == hires_mesh_clip.crs == lores_mesh_clip.crs

        # HARDCODED FOR NOW
        approx_elem_per_width = 3

        meshdata_hi = deepcopy(hires_mesh_clip)
        meshdata_lo = deepcopy(lores_mesh_clip)

        crs = buffer_crs
        assert(not buffer_crs.is_geographic)

        # calculate mesh size for clipped bits
        hfun_hi = Hfun(Mesh(meshdata_hi))
        hfun_hi.size_from_mesh()

        hfun_lo = Hfun(Mesh(meshdata_lo))
        hfun_lo.size_from_mesh()

        engine = get_mesh_engine('triangle', opts='p')
        meshdata_cdt = engine.generate(gpd.GeoSeries(buffer_domain))
        meshdata_cdt.crs = crs

        hfun_cdt = Hfun(Mesh(meshdata_cdt))
        hfun_cdt.size_from_mesh()

        hfun_cdt_sz = deepcopy(hfun_cdt.meshdata().values) / approx_elem_per_width
        meshdata_cdt.values = hfun_cdt_sz

        # TODO: Make jigsaw an option
        engine = get_mesh_engine('jigsaw')
        utm_crs = utils.estimate_bounds_utm(buffer_domain.bounds, crs=crs):
        meshdata_domain_rep = engine.generate(
            gpd.GeoSeries(buffer_domain, crs=crs).to_crs(utm_crs),
            meshdata_cdt,
        )
        utils.final_mesh(meshdata_domain_rep)

#        utils.reproject(meshdata_domain_rep, crs)

        pts_2mesh = np.vstack(
            (hfun_hi.meshdata().coords, hfun_lo.meshdata().coords)
        )
        val_2mesh = np.vstack(
            (hfun_hi.meshdata().values, hfun_lo.meshdata().values)
        )
        domain_sz_1 = griddata(
            pts_2mesh, val_2mesh, meshdata_domain_rep.coords, method='linear'
        )
        domain_sz_2 = griddata(
            pts_2mesh, val_2mesh, meshdata_domain_rep.coords, method='nearest'
        )
        domain_sz = domain_sz_1.copy()
        domain_sz[np.isnan(domain_sz_1)] = domain_sz_2[np.isnan(domain_sz_1)]

        meshdata_domain_rep.values = domain_sz

        return meshdata_domain_rep


    def _generate_mesh_for_buffer_region(
            self, buffer_polygon, meshdata_hfun_buffer, buffer_crs):

        crs = buffer_crs
        assert(not buffer_crs.is_geographic)
        assert(buffer_crs == meshdata_hfun_buffer.crs)

        # TODO: Make jigsaw an option
        engine = get_mesh_engine('jigsaw')
        utm_crs = utils.estimate_bounds_utm(buffer_polygon.bounds, crs=crs):
        meshdata_buf_apprx = engine.generate(
            gpd.GeoSeries(buffer_polygon, crs=crs).to_crs(utm_crs),
            meshdata_hfun_buffer,
        )
        utils.final_mesh(meshdata_buf_apprx)

        # If vertices are too close to buffer geom boundary,
        # it's going to cause issues (thin elements)
#        if meshdata_buf_apprx.crs != hfun_buffer.crs:
#            utils.reproject(meshdata_buf_apprx, hfun_buffer.crs)
        seed_mesh = utils.clip_mesh_by_shape(
            meshdata_buf_apprx, 
            buffer_polygon.buffer(-min(meshdata_hfun_buffer)))
        )

#        utils.reproject(meshdata_buf_apprx, buffer_crs)
        engine = get_mesh_engine('triangle', opts='p')
        meshdata = engine.generate(
            gpd.GeoSeries(buffer_polygon),
            seed=seed_mesh
        )
        meshdata_buffer.crs = crs

#        utils.reproject(meshdata_buffer, buffer_crs)

        return meshdata_buffer


    def _transform_mesh(self, mesh, out_crs):

        transformer = Transformer.from_crs(
            mesh.crs, out_crs, always_xy=True)

        coords = mesh.coords

        # pylint: disable=E0633
        coords[:, 0], coords[:, 1] = transformer.transform(
                coords[:, 0], coords[:, 1])
        mesh.coords[:] = coords
        mesh.crs = out_crs


    def _merge_all_meshes(self, crs, meshdata_mesh, meshdata_clip_lowres, meshdata_clip_hires):

        # Combine mesh
        # 1. combine hires and low-res
        meshdata_old = utils.merge_meshdata(
            meshdata_clip_lowres, meshdata_clip_hires,
            drop_by_bbox=False, out_crs=crs)

        # 2. Create kdtree for boundary of combined mesh
        old_mesh_bdry_edges = utils.get_boundary_edges(meshdata_old)
        old_mesh_bdry_verts = np.unique(old_mesh_bdry_edges)
        old_mesh_bdry_coords = meshdata_old.coords[old_mesh_bdry_verts]

        tree_old = cKDTree(old_mesh_bdry_coords)

        # 3. Find shared boundary nodes from the tree and bdry nodes
        new_mesh_bdry_edges = utils.get_boundary_edges(meshdata_mesh)
        new_mesh_bdry_verts = np.unique(new_mesh_bdry_edges)
        new_mesh_bdry_coords = meshdata_mesh.coords[new_mesh_bdry_verts]

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
            'tria': int,
            'quad': int
        }

        coord = []
        elems = {k: [] for k in mesh_types}
        value = []
        offset = 0

        for k in mesh_types:
            elems[k].append(getattr(meshdata_old, k) + offset)
        coord.append(meshdata_old.coords)
        value.append(meshdata_old.values)
        offset += coord[-1].shape[0]

        # Drop shared vertices and update element cnn based on map and dropped offset
        mesh_orig_idx = np.arange(len(meshdata_mesh.coords))
        mesh_shrd_idx = np.unique(list(map_idx_shared.keys()))
        mesh_renum_idx = np.setdiff1d(mesh_orig_idx, mesh_shrd_idx)
        map_to_combined_idx = {
            index: i + offset for i, index in enumerate(mesh_renum_idx)}
        map_to_combined_idx.update(map_idx_shared)

        for k in mesh_types:
            cnn = getattr(meshdata_mesh, k)
            # If it's empty executing list comprehension results in a
            # (0,) shape instead of (0, 4)
            if cnn.shape[0] == 0:
                elems[k].append(cnn)
                continue

            elems[k].append(np.array([[map_to_combined_idx[x]
                                      for x in  elm] for elm in cnn]))

        coord.append(meshdata_mesh.coords[mesh_renum_idx, :])
        value.append(meshdata_mesh.values[mesh_renum_idx])

        # Putting it all together
        composite_mesh = MeshData(
            coords=np.vstack(coord),
            values=np.vstack(value),
            crs=crs,
            **elems,
        )

        return composite_mesh, mesh_shrd_idx


    def _write_polygon(self, polygon, crs, path):
        gdf  = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygon), crs=crs)
        gdf.to_file(path)

    def _write_meshdata(self, meshdata_mesh, path):
        mesh_obj = Mesh(meshdata_mesh)
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
            meshdata_buffer_mesh,
            meshdata_clip_hires,
            meshdata_clip_lowres,
            meshdata_combined_mesh,
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
            self._write_meshdata(meshdata_buffer_mesh, out_dir/"seam_mesh.2dm")
            self._write_meshdata(meshdata_clip_hires, out_dir/"hires_mesh.2dm")
            self._write_meshdata(meshdata_clip_lowres, out_dir/"lowres_mesh.2dm")
            _logger.info(f"Done in {time() - start} sec")

        _logger.info("Writing mesh...")
        start = time()
        self._write_meshdata(meshdata_combined_mesh, out_dir/"final_mesh.2dm")
        _logger.info(f"Done in {time() - start} sec")


    def _interpolate_values(self, meshdata_combined_mesh, mesh_fine, mesh_coarse):
        interp_1_meshdata = deepcopy(meshdata_combined_mesh)
        utils.interpolate_euclidean_mesh_to_euclidean_mesh(
            mesh_fine.meshdata, interp_1_meshdata,
            method='linear', fill_value=np.nan)

        interp_2_meshdata = deepcopy(meshdata_combined_mesh)
        utils.interpolate_euclidean_mesh_to_euclidean_mesh(
            mesh_coarse.meshdata, interp_2_meshdata,
            method='linear', fill_value=np.nan)


        # Manually get the nan values and only overwrite them!
        mask_1 = np.isnan(interp_1_meshdata.values)
        mask_2 = np.logical_not(np.isnan(interp_2_meshdata.values))
        mask = np.logical_and(mask_1, mask_2)

        meshdata_combined_mesh.values = interp_1_meshdata.values
        meshdata_combined_mesh.values[mask] = interp_2_meshdata.values[mask]


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



        _logger.info("Calculate impact area...")
        start = time()

        # poly_isotach is in EPSG:4326
        poly_isotach = self._get_region_of_interest(
            track_file, wind_speed
        )
        utm = utils.estimate_bounds_utm(poly_isotach.bounds, 4326)

        # Transform all inputs to UTM:
        t1 = Transformer.from_crs(4326, utm, always_xy=True)
        poly_isotach = transform(t1.transform, poly_isotach)
        utils.reproject(mesh_fine.meshdata, utm)
        utils.reproject(mesh_coarse.meshdata, utm)


        _logger.info("Calculate mesh polygons...")
        start = time()
        poly_fine = self._get_largest_polygon(mesh_fine.get_multipolygon())
        poly_coarse = self._get_largest_polygon(mesh_coarse.get_multipolygon())
        _logger.info(f"Done in {time() - start} sec")

        poly_storm_roi = poly_isotach.intersection(poly_fine)

        poly_clipper = self._calculate_clipping_polygon(
            pathlist_raster=pathlist_raster,
            region_of_interest=poly_storm_roi,
            crs=utm,
            cutoff_elev=cutoff_elev,
            upstream_size_max=upstream_size_max,
            upstream_poly_list=[poly_fine])
        _logger.info(f"Done in {time() - start} sec")


        _logger.info("Calculate clipped polygons...")
        meshdata_clip_hires_0 = utils.clip_mesh_by_shape(
                mesh_fine.meshdata, poly_clipper,
                fit_inside=False)
        meshdata_clip_lowres_0 = utils.clip_mesh_by_shape(
                mesh_coarse.meshdata, poly_clipper,
                fit_inside=True,
                inverse=True,
                adjacent_layers=num_buffer_layers)
        _logger.info(f"Done in {time() - start} sec")

        _logger.info("Calculate clipped polygons...")
        start = time()
        poly_clip_hires_0 = utils.remove_holes(
                utils.get_mesh_polygons(meshdata_clip_hires_0))
        poly_clip_lowres_0 = utils.get_mesh_polygons(meshdata_clip_lowres_0)
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
                mesh_fine.meshdata, poly_clip_hires_0,
                mesh_coarse.meshdata, poly_clip_lowres_0)


        # Attach overlaps to buffer region (due to 1 layer and upstream)
        poly_seam_4, meshdata_clip_hires_1 = self._add_overlap_to_polygon(meshdata_clip_hires_0, poly_seam_3)
        poly_seam_5, meshdata_clip_lowres_1 = self._add_overlap_to_polygon(meshdata_clip_lowres_0, poly_seam_4)

        # Cleanup buffer shape
        poly_seam_6 = utils.remove_holes_by_relative_size(
                poly_seam_5, rel_island_area_min)

        poly_seam_7 = utils.drop_extra_vertex_from_polygon(poly_seam_6)
        _logger.info(f"Done in {time() - start} sec")


        _logger.info("Calculate reclipped polygons...")
        start = time()
        meshdata_clip_hires = meshdata_clip_hires_1
        poly_clip_hires = utils.remove_holes(
                utils.get_mesh_polygons(meshdata_clip_hires))

        meshdata_clip_lowres = utils.clip_mesh_by_shape(
            meshdata_clip_lowres_1, poly_clip_hires,
            fit_inside=False, inverse=True)
        poly_clip_lowres = utils.get_mesh_polygons(meshdata_clip_lowres)
        _logger.info(f"Done in {time() - start} sec")

        poly_seam_8 = poly_seam_7.difference(
                    unary_union([poly_clip_hires, poly_clip_lowres]))


        poly_seam = poly_seam_8

        meshdata_hfun_buffer = self._calculate_mesh_size_function(
            poly_seam, meshdata_clip_hires, meshdata_clip_lowres, utm
        )
        meshdata_buffer_mesh = self._generate_mesh_for_buffer_region(
            poly_seam, meshdata_hfun_buffer, utm
        )

        _logger.info("Combining meshes...")
        start = time()
        meshdata_combined_mesh, buffer_shrd_idx = self._merge_all_meshes(
                utm, meshdata_buffer_mesh, meshdata_clip_lowres, meshdata_clip_hires)
        _logger.info(f"Done in {time() - start} sec")

        # NOTE: This call also detects overlap issues
        utils.finalize_mesh(meshdata_combined_mesh)

        self._interpolate_values(meshdata_combined_mesh, mesh_fine, mesh_coarse)

        # TODO: Interpolate DEM?

        self._write_outputs(
            out_dir,
            out_all,
            utm,
            poly_clip_hires,
            poly_clip_lowres,
            poly_seam,
            poly_clipper,
            mesh_fine,
            mesh_coarse,
            meshdata_buffer_mesh,
            meshdata_clip_hires,
            meshdata_clip_lowres,
            meshdata_combined_mesh,
            buffer_shrd_idx
        )
