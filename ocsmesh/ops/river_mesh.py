import logging
import warnings
from itertools import islice

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import triangulate
from shapely import union_all, make_valid

from ocsmesh import utils
from ocsmesh.internal import MeshData

# IMPORTS FROM OPS
# This establishes that river_mesh depends on combine_mesh.
# This is safe as long as combine_mesh does NOT import river_mesh.
from ocsmesh.ops.combine_mesh import triangulate_polygon_s

_logger = logging.getLogger(__name__)

# Constants
ELEM_2D_TYPES = ['tria', 'quad']

# =============================================================================
# Core River Mapper Functions
# =============================================================================
def quadrangulate_rivermapper_arcs(
    arcs_shp,
    _buffer_1: float = 0.001,
    _buffer_2: float = 0.0001,
    _batch_size: int = 1000,
    crs=4326
) -> MeshData:
    '''
    Creates a quadrangular mesh from RiverMapper diagnostic output arcs.

    The process involves:
    1. Processing each river stream individually to form quads between parallel arcs.
    2. Creating local meshes for each stream.
    3. Merging all stream meshes into a single global mesh.
    4. Identifying and removing overlapping regions where streams join.
    5. Performing final quality cleanup (skewed elements, concave quads).

    Parameters
    ----------
    arcs_shp : GeoDataFrame
        Shapefile containing RiverMapper diagnostic arcs.
        Must contain 'river_idx' and 'local_arc_' columns.
    _buffer_1 : float, default=0.001
        (Unused in current logic, kept for API compatibility)
    _buffer_2 : float, default=0.0001
        Buffer size used when removing overlapping regions at river junctions.
    _batch_size : int, default=1000
        Batch size for merging stream meshes to avoid memory spikes.
    crs : int or str, default=4326
        Coordinate reference system for spatial operations.

    Returns
    -------
    MeshData
        The resulting quadrangular mesh.
    '''

    # ---------------------------------------------------------
    # 1. Preparation
    # ---------------------------------------------------------
    # Sort arcs by river index and then by local arc index.
    # This ensures we process streams sequentially and arcs within a stream
    # are ordered correctly (bank-to-bank or flow-aligned).
    shape = arcs_shp.sort_values(['river_idx','local_arc_'],ascending=[True,True])
    meshes = []

    # ---------------------------------------------------------
    # 2. Process Each River Stream
    # ---------------------------------------------------------
    for r_idx in shape['river_idx'].unique():
        quads = []
        coords = [] # Will hold all points for this single stream
        n = 0       # Global index tracker for points in this stream

        try:
            # Get all arcs belonging to the current river
            sub_shape = shape.loc[shape['river_idx'] == r_idx]

            # Extract coordinates from geometries
            geoms = []
            for g in sub_shape.geometry:
                # Convert to numpy array
                c = np.array(g.coords)

                # Fix: Handle 3D coordinates (LINESTRING Z).
                # MeshData strictly requires 2D (X, Y).
                if c.shape[1] > 2:
                    c = c[:, :2]
                geoms.append(c)

            # -----------------------------------------------------
            # 3. Generate Quads between Parallel Arcs
            # -----------------------------------------------------
            # We iterate through the arcs. If we have N arcs, we form quads
            # between Arc i and Arc i+1.
            for ln, g_coords in enumerate(geoms):

                # Check if there is a "next" arc to connect to
                if ln + 1 < len(geoms):
                    current_arc_len = len(g_coords)

                    # Iterate through points in the current arc
                    for nn in range(current_arc_len):
                        # Ensure we don't go out of bounds on the current arc
                        if nn + 1 < current_arc_len:

                            # Increment global node index tracker
                            n = n + 1

                    # Define the 4 nodes of the quad.
                            idx1 = n - 1
                            idx2 = n + current_arc_len - 1
                            idx3 = n + current_arc_len
                            idx4 = n

                            quads.append([idx1, idx2, idx3, idx4])

                    # Increment n once more to account for the last point in the arc
                    # that wasn't a "start" of a quad segment loop
                    n = n + 1

                # Add this arc's points to the master coordinate list
                coords.append(g_coords)

            if not coords:
                continue

            # Stack into (N, 2) array
            coords_np = np.vstack(coords)

            # -----------------------------------------------------
            # 4. Filter Invalid Quads
            # -----------------------------------------------------
            valid_quads = []
            max_idx = len(coords_np) - 1

            for q in quads:
                # Safety check: indices must be within bounds
                if any(idx > max_idx for idx in q):
                    continue

                # Geometry check: A quad must have 4 unique vertices.
                # If vertices overlap, it's degenerate (line or triangle).
                if len(np.unique(coords_np[q], axis=0)) == 4:
                    valid_quads.append(q)

            if not valid_quads:
                continue

            # Create the local mesh for this stream
            # FIX: Must assign CRS so merge_meshdata doesn't fail later
            mesh = MeshData(
                coords=coords_np,
                quad=np.array(valid_quads),
                crs=crs
            )

            # Basic cleanup for this stream
            utils.cleanup_duplicates(mesh)
            meshes.append(mesh)

        except Exception as e:
            warnings.warn(f"Error quadrangulating arcs id: {r_idx}. Error: {e}")

    # Return empty mesh if no streams were processed
    if not meshes:
        return MeshData(coords=np.empty((0,2)), quad=np.empty((0,4)), crs=crs)

    # ---------------------------------------------------------
    # 5. Merge Stream Meshes
    # ---------------------------------------------------------
    # Use batched merging to handle large numbers of streams efficiently.
    merged_all_list = []
    for sub_list in batched(meshes, _batch_size):
        # Merge a batch
        merged = utils.merge_meshdata(*sub_list, drop_by_bbox=False)
        merged_all_list.append(merged)

    # Merge the batches into one final mesh
    final_mesh = utils.merge_meshdata(*merged_all_list, drop_by_bbox=False)

    # ---------------------------------------------------------
    # 6. Remove Overlaps (Junctions)
    # ---------------------------------------------------------
    # Streams often overlap at junctions. We need to cut out these
    # overlapping areas to prevent bad geometry.

    # Get the polygon boundary for every stream mesh
    stream_polys = [utils.get_mesh_polygons(m) for m in meshes]
    gdf_streams = gpd.GeoDataFrame(geometry=stream_polys, crs=crs)

    overlaps = []
    sindex = gdf_streams.sindex

    # Find spatial intersections between different streams
    for i, poly in enumerate(stream_polys):
        # Optimization: Use spatial index to find candidates
        possible_matches_index = list(sindex.intersection(poly.bounds))
        if i in possible_matches_index:
            possible_matches_index.remove(i)

        for j in possible_matches_index:
            # j > i ensures we only check each pair once
            if j > i:
                other = stream_polys[j]
                if poly.intersects(other):
                    # Calculate the intersection polygon
                    overlaps.append(poly.intersection(other))

    if overlaps:
        overlap_gdf = gpd.GeoDataFrame(geometry=overlaps, crs=crs)

        # Buffer the overlap slightly to ensure clean cutting
        # Project to UTM for accurate meter-based buffering if needed
        if overlap_gdf.crs.is_geographic:
            utm_crs = utils.estimate_bounds_utm(overlap_gdf.total_bounds,
                                                overlap_gdf.crs)
            overlap_gdf = overlap_gdf.to_crs(utm_crs).buffer(_buffer_2).to_crs(crs)
        else:
            overlap_gdf = overlap_gdf.buffer(_buffer_2)

        # Merge all overlap shapes into one cut-out mask
        # Using the new Shapely 2.0+ union_all (replaces unary_union)
        removal_shape = union_all(overlap_gdf.geometry)

        # Clip the final mesh to remove these junctions
        final_mesh = utils.clip_mesh_by_shape(
            final_mesh,
            removal_shape,
            inverse=True, # Inverse=True means "Remove this shape"
            check_cross_edges=True
        )

    # ---------------------------------------------------------
    # 7. Final Cleanup
    # ---------------------------------------------------------
    # Fix bad elements and ensure node uniqueness
    # These functions are defined locally in this file
    final_mesh = utils.cleanup_skewed_el(final_mesh)
    final_mesh = utils.cleanup_concave_quads(final_mesh)

    utils.cleanup_duplicates(final_mesh)
    utils.cleanup_isolates(final_mesh)

    return final_mesh


def triangulate_rivermapper_poly(rm_poly):
    '''
    Creates triangulated mesh using the RiverMapper outputs.
    
    Workflow:
    1) Finds slivers (gdf_invalid) on RM shapefile.
    2) Finds all polygons next to slivers.
    3) Removes all polygons next to slivers.
    4) Triangulates the slivers-free RM shapefile.
    5) Finds/Removes non-matching vertices and their neighbor elements.

    Parameters
    ----------
    rm_poly: .shp (gpd) file with the RiverMapper outputs

    Returns
    -------
    MeshData
        River Mesh
    '''
    # Finds slivers (gdf_invalid) on RM shapefile:
    gdf_valid, gdf_invalid = validate_poly(rm_poly)

    # Finds all polygons next to slivers:
    polyneighbors = find_polyneighbors(gdf_valid, gdf_invalid)

    # Removes all polygons next to slivers:
    fixed_rm = gdf_valid[~gdf_valid.index.isin(polyneighbors.index)]

    # Triangulate the slivers-free RM shapefile:
    rm_mesh = triangulate_poly(fixed_rm)

    # Finds/Removes non-matching vertices and their neighbor el:
    validated_rm_mesh = validate_RMmesh(rm_mesh)

    return validated_rm_mesh


def triangulate_poly(rm_poly) -> MeshData:
    '''
    Creates triangulated meshes from gpf with poly and multipoly

    Parameters
    ----------
    rm_poly : .shp (gpd) (e.g.file with the RiverMapper outputs)

    Returns
    -------
    MeshData
        River Mesh
    '''
    rm_poly_triangulated = triangulate_shp(rm_poly)
    rm_mesh = shptri_to_meshsdata(rm_poly_triangulated)
    return rm_mesh


def triangulate_shp(gdf):
    '''
    Fills out the gaps left by the delaunay_within

    Parameters
    ----------
    gdf : gpd of polygons

    Returns
    -------
    gdf : gpd of triangulated polygons
    '''
    shape_tri = [delaunay_within(gdf)]

    diff_geom = gdf.difference(shape_tri[0])
    shape_diff = gpd.GeoDataFrame(geometry=gpd.GeoSeries(diff_geom))
    shape_diff = shape_diff[~shape_diff.is_empty].dropna()

    shape_diff_len = len(shape_diff)

    while shape_diff_len > 0:
        shape_diff_tri = delaunay_within(shape_diff)
        shape_tri.append(shape_diff_tri)

        current_diff_union = union_all(shape_diff.geometry)
        new_tri_union = union_all(shape_diff_tri.geometry)

        remaining = current_diff_union.difference(new_tri_union)

        shape_diff = gpd.GeoDataFrame(geometry=gpd.GeoSeries(remaining))
        shape_diff = shape_diff[~shape_diff.is_empty].dropna().explode(index_parts=True)

        if len(shape_diff) == shape_diff_len:
            break
        shape_diff_len = len(shape_diff)

    shape_final = gpd.GeoDataFrame(pd.concat(shape_tri,
                                             ignore_index=True)).explode(index_parts=True)
    shape_final = shape_final[shape_final.geometry.type == 'Polygon']
    shape_final.reset_index(drop=True, inplace=True)

    return shape_final


def delaunay_within(gdf):
    '''
    Creates the initial delaunay triangles for
    a gpd composed of polygons (only).
    Selects those delaunay triangles that fall within domain.

    Parameters
    ----------
    gdf : gpd of polygons

    Returns
    -------
    gdf : gpd of triangulated polygons
    '''
    tt = []
    indices = []

    # Iterate over items to preserve index alignment for later differencing
    for idx, polygon in gdf['geometry'].items():
        try:
            # shapely.ops.triangulate performs Delaunay triangulation
            tri = [
                triangle for triangle in triangulate(polygon) 
                if triangle.within(polygon)
            ]
            if tri:
                tt.append(MultiPolygon(tri))
                indices.append(idx)
        except Exception:
            # Skip invalid polygons
            pass

    # Create GeoDataFrame with matching indices
    shape_tri = gpd.GeoDataFrame(
        geometry=tt,
        index=indices,
        crs=gdf.crs
    )

    return shape_tri


def shptri_to_meshsdata(triangulated_shp) -> MeshData:
    '''
    Converts a triangulated shapefile to MeshData

    Parameters
    ----------
    triangulated_shp : triangulated gpd

    Returns
    -------
    MeshData
    '''
    coords = []
    verts = []

    for idx, poly in enumerate(triangulated_shp['geometry']):
        x, y = poly.exterior.coords.xy
        pts = np.array([x, y]).T

        if len(pts) == 4:
            coords.append(pts[:-1])
            verts.append(np.array([0, 1, 2]) + 3 * idx)

    if not coords:
        return MeshData(coords=np.empty((0, 2)),
                        tria=np.empty((0, 3),
                                      dtype=int))

    meshsdata = MeshData(
        coords=np.vstack(coords),
        tria=np.vstack(verts)
    )

    utils.cleanup_duplicates(meshsdata)
    utils.cleanup_isolates(meshsdata)
    return meshsdata


def validate_RMmesh(rm_mesh: MeshData) -> MeshData:
    '''
    Takes a mesh triangulated from a polygon and
    removes invalid elements (and their neighbors)

    Parameters
    ----------
    rm_mesh: MeshData
        Triangulated mesh

    Returns
    -------
    MeshData
        River Mesh
    '''
    tri = rm_mesh.tria
    xy = rm_mesh.coords

    poly = []
    if len(tri) > 0:
        v0 = xy[tri[:, 0]]
        v1 = xy[tri[:, 1]]
        v2 = xy[tri[:, 2]]

        for i in range(len(tri)):
            p = [v0[i], v1[i], v2[i], v0[i]]
            poly.append(p)

    rm_mesh_poly = gpd.GeoDataFrame(
        geometry=[Polygon(p) for p in poly],
        crs=4326
    )

    rm_mesh_invalid = rm_mesh_poly.loc[rm_mesh_poly['geometry'].area < 1e-15]
    polyneighbors = find_polyneighbors(rm_mesh_poly, rm_mesh_invalid)

    el2remove_idx = np.concatenate((
        rm_mesh_invalid.index.to_numpy(),
        polyneighbors.index.to_numpy()
    ))

    new_tri = np.delete(tri, el2remove_idx, axis=0)

    meshsdata_valid = MeshData(
        coords=xy,
        tria=new_tri
    )

    utils.cleanup_duplicates(meshsdata_valid)
    utils.cleanup_isolates(meshsdata_valid)

    return meshsdata_valid


# =============================================================================
# Helper Utilities (Validation & Indexing)
# =============================================================================
def validate_poly(gdf):
    '''
    Goes over all polygons in a gpf and applied the make_valid func

    Parameters
    ----------
    gdf : .shp (gpd) (e.g.file with the RiverMapper outputs)

    Returns
    -------
    gdf_valid, gdf_invalid: valid and invalid polygon in the gpdf
    '''
    polys = [make_valid(p) for p in gdf['geometry'].to_list()]

    gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(polys),
        crs=4326
    ).explode(index_parts=True)

    valid_types = ['Polygon', 'MultiPolygon']
    gdf_valid = gdf[gdf.geometry.geom_type.isin(valid_types)].drop_duplicates()
    gdf_invalid = gdf[~gdf.geometry.geom_type.isin(valid_types)].drop_duplicates()

    return gdf_valid, gdf_invalid


def find_polyneighbors(target_gdf, ref_gdf):
    '''
    Finds all polygons in target_gdf that border polygons in ref_gdf

    Parameters
    ----------
    target_gdf: .shp (gpd) 
    ref_gdf: .shp (gpd) 

    Returns
    -------
    sjoin: .shp (gpd) with all target_gdf that
            border polygons in ref_gdf
    '''
    intersects = gpd.sjoin(
        target_gdf, ref_gdf, how='inner', predicate='intersects'
    )
    crosses = gpd.sjoin(
        target_gdf, ref_gdf, how='inner', predicate='crosses'
    )
    touches = gpd.sjoin(
        target_gdf, ref_gdf, how='inner', predicate='touches'
    )
    overlaps = gpd.sjoin(
        target_gdf, ref_gdf, how='inner', predicate='overlaps'
    )
    sjoin = pd.concat([intersects, crosses, touches, overlaps])
    sjoin = sjoin.drop_duplicates()

    return sjoin


def batched(iterable, n):
    '''
    This function is part of itertools for python 3.12+
    This function was added to ensure OCSMesh can run on older python<3.12
    '''
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


# =============================================================================
# Local Cleanup Functions
# =============================================================================
def remesh_holes(meshsdata: MeshData,
                 area_threshold_min: float = .0,
                 area_threshold_max: float =.002
                ) -> MeshData:
    '''
    Remove undesirable island and slivers based on area.

    Parameters
    ----------
    meshsdata : MeshData
    area_threshold_min : float, default=0.0
    area_threshold_max : float, default=0.002

    Returns
    -------
    MeshData
        Mesh with holes remeshed.
    '''
    mesh_poly = utils.get_mesh_polygons(meshsdata)

    # GeoPandas buffering to merge close geometries
    mesh_gdf = gpd.GeoDataFrame(geometry=[mesh_poly],
                                crs=4326).explode(index_parts=True)
    mesh_noholes_poly = utils.remove_holes(union_all(mesh_gdf.geometry))

    mesh_holes_poly = mesh_noholes_poly.difference(mesh_poly)
    if mesh_holes_poly.is_empty:
        return meshsdata

    mesh_holes_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(mesh_holes_poly),
                                      crs=4326).explode(index_parts=True)
    mesh_holes_gdf['area'] = mesh_holes_gdf.geometry.area

    selected_holes = mesh_holes_gdf[
        (mesh_holes_gdf['area'] >= area_threshold_min) &
        (mesh_holes_gdf['area'] <= area_threshold_max)
    ]

    if len(selected_holes) > 0:
        # 1. Carve holes out (ensure they are open)
        carved_mesh = utils.clip_mesh_by_shape(
            meshsdata,
            union_all(selected_holes.geometry),
            adjacent_layers=2,
            inverse=True
        )

        # 2. Triangulate holes
        patch_gdf = selected_holes.copy()
        patch_gdf.geometry = patch_gdf.buffer(0.00001)

        aux_pts_mesh = utils.clip_mesh_by_shape(meshsdata,
                                                union_all(patch_gdf.geometry),
                                                fit_inside=True)

        # Uses the function imported from combine_mesh
        meshsdata_patch = triangulate_polygon_s(
            union_all(patch_gdf.geometry),
            aux_pts=aux_pts_mesh.coords
        )

        # 3. Merge
        mesh_filled = utils.merge_meshdata(
            carved_mesh, meshsdata_patch,
            drop_by_bbox=False
        )

        utils.cleanup_duplicates(mesh_filled)
        utils.cleanup_isolates(mesh_filled)

        return mesh_filled

    return meshsdata
