import pathlib
from time import time
from copy import deepcopy

import numpy as np
from scipy.spatial import cKDTree
import geopandas as gpd
from pyproj import CRS, Transformer
from meshpy.triangle import build, MeshInfo
import jigsawpy
from jigsawpy.msh_t import jigsaw_msh_t
from jigsawpy.jig_t import jigsaw_jig_t
from shapely.geometry import (
        MultiPolygon, Polygon, GeometryCollection, box,
        CAP_STYLE, JOIN_STYLE)
from shapely.ops import polygonize, unary_union, transform
import pygmsh

from ocsmesh import Raster, Geom, Mesh, utils, Hfun


def remove_holes(poly):
    if isinstance(poly, MultiPolygon):
        return MultiPolygon([remove_holes(p) for p in poly.geoms])

    if poly.interiors:
        return Polygon(poly.exterior)

    return poly


def remove_holes_by_relative_size(poly, rel_size):
    if isinstance(poly, MultiPolygon):
        return MultiPolygon([
            remove_holes_by_relative_size(p, rel_size) for p in poly.geoms])

    if poly.interiors:
        ref_area = poly.area
        new_interiors = [
                intr for intr in poly.interiors
                if Polygon(intr).area / ref_area > rel_size] 
        return Polygon(poly.exterior, new_interiors)

    return poly

def get_largest_polygon(mpoly):
    if isinstance(mpoly, Polygon):
        return mpoly

    return sorted(mpoly.geoms, key=lambda i: i.area)[-1]

def get_polygons_smaller_than(mpoly, size):
    if isinstance(mpoly, Polygon):
        if mpoly.area <= size:
            return mpoly
        return Polygon()

    areas = np.array([p.area for p in mpoly.geoms])

    return MultiPolygon(np.extract(areas < size, mpoly.geoms).tolist())



def get_region_bounded_by_isotach(track_file, wind_speed=34):

    gdf_all_isotachs = gpd.read_file(track_file)
    gdf_specific_isotach = gdf_all_isotachs[
            gdf_all_isotachs.RADII.astype(int) == wind_speed]
    isotach_exterior_polygon = MultiPolygon([
            i for i in polygonize(
                [ext for ext in gdf_specific_isotach.exterior])
            ])
    
    return isotach_exterior_polygon

def calculate_clipping_polygon(
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
                get_polygons_smaller_than(
                    poly.difference(clip_poly_draft),
                    upstream_size_max))

    # Islands are not of intereset when clipping high and low-res meshes
    clip_poly = remove_holes(unary_union([
        clip_poly_draft, *poly_upstreams]))

    return clip_poly


def clip_and_get_shape(*args, **kwargs):
    clipped_mesh = utils.clip_mesh_by_shape(
            *args, **kwargs)

    return utils.get_mesh_polygons(clipped_mesh)


def get_polygon_from_geom_collection(shape):

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

def add_one_mesh_layer_to_polygon(
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
    

def add_overlap_to_polygon(mesh, polygon):

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

def calculate_mesh_size_function(hires_mesh_clip, lores_mesh_clip, crs):

    # calculate mesh size for clipped bits
    hfun_hires = Hfun(Mesh(deepcopy(hires_mesh_clip)))
    hfun_hires.size_from_mesh()
    hfun_lowres = Hfun(Mesh(deepcopy(lores_mesh_clip)))
    hfun_lowres.size_from_mesh()

#    print("Writing hfun clips...")
#    start = time()
#    hfun_hires.mesh.write(str(out_dir / "hfun_fine.2dm"), format="2dm", overwrite=True)
#    hfun_lowres.mesh.write(str(out_dir / "hfun_coarse.2dm"), format="2dm", overwrite=True)
#    print(f"Done in {time() - start} sec")

    jig_hfun = utils.merge_msh_t(
        hfun_hires.msh_t(), hfun_lowres.msh_t(),
        out_crs=crs)#jig_geom) ### TODO: CRS MUST BE == GEOM_MSHT CRS

    return jig_hfun


def generate_mesh_for_buffer_region(
        buffer_polygon, jig_hfun, crs):

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

    print("Meshing...")
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
    transform_mesh(jig_mesh, crs)

    return jig_mesh


def transform_mesh(mesh, out_crs):

    transformer = Transformer.from_crs(
        mesh.crs, out_crs, always_xy=True)

    coords = mesh.vert2['coord']

    coords[:, 0], coords[:, 1] = transformer.transform(
            coords[:, 0], coords[:, 1])
    mesh.vert2['coord'][:] = coords
    mesh.crs = out_crs


def merge_all_meshes(crs, jig_mesh, jig_clip_lowres, jig_clip_hires):

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
    map_idx_shared = dict()
    for idx_tree_old, neigh_idx_list in enumerate(neigh_idxs):
        num_match = len(neigh_idx_list)
        if num_match == 0:
            continue
        elif num_match > 1:
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


def write_polygon(polygon, crs, path):
    gdf  = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygon), crs=crs)
    gdf.to_file(path)

def write_jigsaw(jigsaw_mesh, path):
    mesh_obj = Mesh(jigsaw_mesh)
    mesh_obj.write(path, format="2dm", overwrite=True)

def write_outputs(
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

    print("Writing shapes...")
    start = time()
    write_polygon(poly_clip_hires, crs, out_dir/"hires")
    write_polygon(poly_clip_lowres, crs, out_dir/"lowres")
    write_polygon(poly_seam, crs, out_dir/"seam")
    write_polygon(poly_clipper, crs, out_dir/"clip")
    print(f"Done in {time() - start} sec")

    print("Writing input mesh (copy)...")
    start = time()
    mesh_fine.write(str(out_dir / "fine.2dm"), format="2dm", overwrite=True)
    mesh_coarse.write(str(out_dir / "coarse.2dm"), format="2dm", overwrite=True)
    print(f"Done in {time() - start} sec")

    print("Writing mesh...")
    start = time()
    write_jigsaw(jig_buffer_mesh, out_dir/"seam_mesh.2dm")
    write_jigsaw(jig_clip_hires, out_dir/"hires_mesh.2dm")
    write_jigsaw(jig_clip_lowres, out_dir/"lowres_mesh.2dm")
    print(f"Done in {time() - start} sec")

#    shared_coords = jig_buffer_mesh.vert2['coord'][buffer_shrd_idx, :]
#    gdf_points_shared = gpd.GeoDataFrame(
#        geometry=gpd.points_from_xy(
#            shared_coords[:, 0], shared_coords[:, 1], crs=crs))
#    gdf_points_shared.to_file(out_dir/"shared_nodes")

    print("Writing mesh...")
    start = time()
    write_jigsaw(jig_combined_mesh, out_dir/"final_mesh.2dm")
    print(f"Done in {time() - start} sec")


def interpolate_values(jig_combined_mesh, mesh_fine, mesh_coarse):
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


def main(pathlist_raster, path_fine, path_coarse,
         track_file,
         cutoff_elev, upstream_size_max,
         wind_speed, num_buffer_layers, rel_island_area_min,
         out_dir, crs):

    # TODO: Clarify what "crs" refers to (it's both input and output!)

    print("Reading meshes...")
    start = time()
    mesh_fine = Mesh.open(path_fine, crs=crs)
    mesh_coarse = Mesh.open(path_coarse, crs=crs)
    print(f"Done in {time() - start} sec")


    print("Calculate mesh polygons...")
    start = time()
    poly_fine = get_largest_polygon(mesh_fine.get_multipolygon())
    poly_coarse = get_largest_polygon(mesh_coarse.get_multipolygon())
    print(f"Done in {time() - start} sec")

    print("Calculate impact area...")
    start = time()
    poly_isotach = get_region_bounded_by_isotach(track_file, wind_speed)
    poly_storm_roi = poly_isotach.intersection(poly_fine)

    poly_clipper = calculate_clipping_polygon(
        pathlist_raster=pathlist_raster,
        region_of_interest=poly_storm_roi,
        crs=crs,
        cutoff_elev=cutoff_elev,
        upstream_size_max=upstream_size_max,
        upstream_poly_list=[poly_fine])
    print(f"Done in {time() - start} sec")
    
    
    print("Calculate clipped polygons...")
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
    print(f"Done in {time() - start} sec")
    
    print("Calculate clipped polygons...")
    start = time()
    poly_clip_hires_0 = remove_holes(
            utils.get_mesh_polygons(jig_clip_hires_0))
    poly_clip_lowres_0 = utils.get_mesh_polygons(jig_clip_lowres_0)
    print(f"Done in {time() - start} sec")

    print("Calculating buffer region...")
    start = time()

    # Remove the two mesh clip regions from the coarse mesh polygon
    poly_seam_0 = poly_coarse.difference(
            unary_union([poly_clip_lowres_0, poly_clip_hires_0]))

    poly_seam_1 = poly_seam_0.intersection(poly_fine)

    # Get rid of non polygon results of the intersection
    poly_seam_2 = poly_seam_1
    if isinstance(poly_seam_1, GeometryCollection):
        poly_seam_2 = get_polygon_from_geom_collection(poly_seam_1)

    # Get one layer on each mesh
    poly_seam_3 = add_one_mesh_layer_to_polygon(
            poly_seam_2,
            mesh_fine.msh_t, poly_clip_hires_0,
            mesh_coarse.msh_t, poly_clip_lowres_0)


    # Attach overlaps to buffer region (due to 1 layer and upstream)
    poly_seam_4, jig_clip_hires = add_overlap_to_polygon(jig_clip_hires_0, poly_seam_3)
    poly_seam_5, jig_clip_lowres = add_overlap_to_polygon(jig_clip_lowres_0, poly_seam_4)

    # Cleanup buffer shape
    poly_seam_6 = remove_holes_by_relative_size(
            poly_seam_5, rel_island_area_min)

    poly_seam_7 = utils.drop_extra_vertex_from_polygon(poly_seam_6)
    print(f"Done in {time() - start} sec")


    print("Calculate reclipped polygons...")
    start = time()
    poly_clip_hires = remove_holes(
            utils.get_mesh_polygons(jig_clip_hires))
    poly_clip_lowres = utils.get_mesh_polygons(jig_clip_lowres)
    print(f"Done in {time() - start} sec")

    poly_seam_8 = poly_seam_7.difference(
                unary_union([poly_clip_hires, poly_clip_lowres]))


    poly_seam = poly_seam_8

    # TODO: Get CRS correctly from geom utm
#    jig_hfun = calculate_mesh_size_function(jig_clip_hires, jig_clip_lowres, crs)
    jig_buffer_mesh = generate_mesh_for_buffer_region(poly_seam, None, crs)

    print("Combining meshes...")
    start = time()
    jig_combined_mesh, buffer_shrd_idx = merge_all_meshes(
            crs, jig_buffer_mesh, jig_clip_lowres, jig_clip_hires)
    print(f"Done in {time() - start} sec")

    # NOTE: This call also detects overlap issues
    utils.finalize_mesh(jig_combined_mesh) 

    interpolate_values(jig_combined_mesh, mesh_fine, mesh_coarse)

    # TODO: Interpolate DEM?

    write_outputs(
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


if __name__ == "__main__":

    pathlist_raster = [
        "/home/smani/workarea/test/windows/DEM/GEBCO/gebco_2020_n90.0_s0.0_w-90.0_e0.0.tif",
        "/home/smani/workarea/test/windows/DEM/GEBCO/gebco_2020_n90.0_s0.0_w-180.0_e-90.0.tif"
        ]
    path_fine = "/home/smani/workarea/test/windows/Grid/HSOFS_250m_v1.0_fixed.14"
    path_coarse = "/home/smani/workarea/test/windows/Grid/WNAT_1km.14"

#    track_file = "/home/smani/WinHome/Desktop/maria_track/windswath.shp"
    track_file = "/home/smani/WinHome/Desktop/florence_track/windswath.shp"

    out_dir = pathlib.Path("/home/smani/WinHome/Desktop")

    crs = CRS.from_epsg(4326)

    cutoff_elev=-200
    upstream_size_max=100
    # 9 layers: fine mesh coarser than coarse mesh in some areas!
    num_buffer_layers = 9
    rel_island_area_min = 1/100
    wind_speed = 34

    main(pathlist_raster, path_fine, path_coarse,
         track_file,
         cutoff_elev, upstream_size_max,
         wind_speed, num_buffer_layers, rel_island_area_min,
         out_dir, crs)
