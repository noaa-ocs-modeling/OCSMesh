import logging
from copy import deepcopy
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, Point
from shapely import  intersection, union_all
from scipy.spatial import cKDTree
from pyproj import CRS

from ocsmesh import utils
from ocsmesh.internal import MeshData
from ocsmesh.engines.factory import get_mesh_engine

_logger = logging.getLogger(__name__)
ELEM_2D_TYPES = ['tria', 'quad']

# =============================================================================
# Main Workflow
# =============================================================================
def merge_overlapping_meshes(
    all_msht: list,
    adjacent_layers: int = 0,
    buffer_size: float = 0.0075,
    buffer_domain: float = 0.002,
    min_int_ang: int = 30,
    hfun_mesh = None,
    crs=4326,
    clip_final = True,
) -> MeshData:
    '''
    Combine meshes that overlap by stitching them together.
    Follows the robust logic of the legacy OCSMesh utils.
    '''
    if not all_msht:
        raise ValueError("No meshes provided.")

    msht_combined = all_msht[0]
    dst_crs = CRS.from_user_input(crs) if crs else None

    for msht in all_msht[1:]:
        # Clip background (msht_combined) by foreground (msht)
        carved_mesh = clip_mesh_by_mesh(
            msht_combined,
            msht,
            adjacent_layers=adjacent_layers,
            buffer_size=buffer_size
        )
        if clip_final is True:
            domain = pd.concat([gpd.GeoDataFrame(geometry=\
                                [utils.get_mesh_polygons(i)],crs=crs) for \
                                i in [msht_combined,msht]])
        utils.cleanup_isolates(carved_mesh)

        if hfun_mesh is None:
            hfun_mesh = deepcopy(msht_combined)

        buff_mesh = create_mesh_from_mesh_diff(
                                               [msht_combined,msht],
                                               carved_mesh,
                                               msht,
                                               min_int_ang=min_int_ang,
                                               buffer_domain=buffer_domain,
                                               hfun_mesh=hfun_mesh,
                                               crs=crs
                                              )

        msht_combined = merge_neighboring_meshes(buff_mesh, carved_mesh, msht)

        if clip_final is True:
            msht_combined = utils.clip_mesh_by_shape(msht_combined,
                                                     domain.union_all(),
                                                     fit_inside=True,
                                                     check_cross_edges=False
                                                    )

        del carved_mesh,buff_mesh,msht

    utils.finalize_mesh(msht_combined)
    if dst_crs:
        msht_combined.crs = dst_crs

    return msht_combined


# =============================================================================
# Core Logic Ports
# =============================================================================
def merge_neighboring_meshes(*all_msht):
    '''
    Combine meshes whose boundaries match.
    Ported from old utils.py to support MeshData.
    Uses KDTree to snap shared boundary nodes.
    '''
    # Start with the first mesh (usually the buffer mesh in the sequence)
    msht_combined = deepcopy(all_msht[0])

    for msht in all_msht[1:]:
        # 1. Identify Boundary Nodes for snapping
        combined_bdry_edges = utils.get_boundary_edges(msht_combined)
        combined_bdry_verts = np.unique(combined_bdry_edges)
        combined_bdry_coords = msht_combined.coords[combined_bdry_verts]

        msht_bdry_edges = utils.get_boundary_edges(msht)
        msht_bdry_verts = np.unique(msht_bdry_edges)
        msht_bdry_coords = msht.coords[msht_bdry_verts]

        # 2. Build Tree & Query
        # Tolerance 1e-8 is safe for Lat/Lon (approx 1mm)
        tree_comb = cKDTree(combined_bdry_coords)
        tree_msht = cKDTree(msht_bdry_coords)

        # Find which nodes in 'msht' match nodes in 'combined'
        neigh_idxs = tree_comb.query_ball_tree(tree_msht, r=1e-8)

        # 3. Create Map (Local Index -> Global Index)
        map_idx_shared = {}
        for idx_tree_comb, neigh_idx_list in enumerate(neigh_idxs):
            if len(neigh_idx_list) == 0:
                continue
            # If multiple matches, just take the first one (standard snapping)
            idx_tree_msht = neigh_idx_list[0]

            local_idx = msht_bdry_verts[idx_tree_msht]
            target_idx = combined_bdry_verts[idx_tree_comb]
            map_idx_shared[local_idx] = target_idx

        # 4. Prepare Arrays for Merging
        coord_list = [msht_combined.coords]
        val_list = [msht_combined.values] if msht_combined.values is not None else []

        # Calculate offset for NEW nodes (nodes that are NOT shared)
        offset = len(msht_combined.coords)

        # Identify which nodes from 'msht' are new
        mesh_orig_idx = np.arange(len(msht.coords))
        mesh_shrd_idx = np.array(list(map_idx_shared.keys()), dtype=int)
        mesh_keep_idx = np.setdiff1d(mesh_orig_idx, mesh_shrd_idx)

        # Build the Node Mapping Array
        # node_map[old_id] = new_id
        node_map = np.zeros(len(msht.coords),dtype=np.int64) - 1#Initialize with -1

        # Map shared nodes to existing indices
        for local, target in map_idx_shared.items():
            node_map[local] = target

        # Map new nodes to offset indices
        # We need a secondary map for the keep_idx to 0..N range
        new_indices = np.arange(len(mesh_keep_idx)) + offset
        node_map[mesh_keep_idx] = new_indices

        # 5. Append New Data
        coord_list.append(msht.coords[mesh_keep_idx])
        if msht.values is not None:
            val_list.append(msht.values[mesh_keep_idx])

        # 6. Update & Append Elements
        for etype in ELEM_2D_TYPES:
            # Existing elements
            current_elems = getattr(msht_combined, etype)

            # New elements (from msht)
            new_elems_raw = getattr(msht, etype)

            if new_elems_raw.size == 0:
                continue

            # Remap the new elements
            new_elems_mapped = node_map[new_elems_raw]

            # Stack
            if current_elems.size == 0:
                setattr(msht_combined, etype, new_elems_mapped)
            else:
                setattr(msht_combined,
                        etype,
                        np.vstack((current_elems, new_elems_mapped)))

        # Update Coords and Values
        msht_combined.coords = np.vstack(coord_list)
        if val_list:
            msht_combined.values = np.vstack(val_list)

    return msht_combined


def create_mesh_from_mesh_diff(
    domain,
    mesh_1: MeshData,
    mesh_2: MeshData,
    crs=4326,
    min_int_ang=None,
    buffer_domain = 0.001,
    hfun_mesh = None
) -> MeshData:
    '''
    Port of old `create_mesh_from_mesh_diff`.
    Calculates Gap -> Triangulates -> Clips back to Gap.
    '''
    if isinstance(domain, (gpd.GeoDataFrame)):
        pass
    if isinstance(domain, (gpd.GeoSeries)):
        domain = gpd.GeoDataFrame(geometry=gpd.GeoSeries(domain))
    if isinstance(domain, Polygon):
        domain = MultiPolygon([domain])
        domain = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[domain])
    if isinstance(domain, MeshData):
        domain = [utils.get_mesh_polygons(domain)]
        domain = gpd.GeoDataFrame(geometry=domain,crs=crs)
    if isinstance(domain, (list)):
        domain = pd.concat([gpd.GeoDataFrame\
                            (geometry=[utils.get_mesh_polygons(i)\
                                    #    .buffer(buffer_domain,join_style=2)
                                       ],
                                        crs=crs) for i in domain])
    if not isinstance(domain, (gpd.GeoDataFrame)):
        raise ValueError("Input shape must be a gpd.GeoDataFrame!")

    domain_buffer = gpd.GeoDataFrame(geometry=[i[-1].geometry.buffer(\
        buffer_domain,join_style=2) for i in domain.iterrows()],crs=crs)
    domain_buffer = domain_buffer.dissolve().explode(index_parts=True)
    domain_buffer.crs = domain_buffer.estimate_utm_crs()
    domain_buffer =domain_buffer.loc[domain_buffer['geometry'].area == \
                                    max(domain_buffer['geometry'].area)]
    domain_buffer.crs = crs
    domain_buffer = gpd.GeoDataFrame(geometry=[domain_buffer.union_all()],
                                    crs=crs)

    mesh_1_poly = utils.get_mesh_polygons(mesh_1)
    mesh_2_poly = utils.get_mesh_polygons(mesh_2)

    poly_buffer = domain_buffer.union_all().difference(
        gpd.GeoDataFrame(
            geometry=[
                mesh_1_poly,
                mesh_2_poly,
            ],
            crs = crs
        ).union_all()
    )
    gdf_full_buffer = gpd.GeoDataFrame(
        geometry = [poly_buffer],crs=crs).explode()

    gap_poly = domain.union_all().\
        difference(mesh_1_poly).\
            difference(mesh_2_poly)


    if hfun_mesh is not None:
        hfun_mesh = utils.clip_mesh_by_shape(hfun_mesh,gap_poly,fit_inside=True)
        boundary = np.unique(utils.get_boundary_edges(hfun_mesh))
        all_nodes = hfun_mesh.coords
        hfun_nodes = np.delete(all_nodes, boundary, axis=0)
        hfun_nodes = MultiPoint(hfun_nodes)
        hfun_nodes = gpd.GeoDataFrame(geometry=
                                gpd.GeoSeries(intersection(
                                hfun_nodes,
                                gap_poly.buffer(-0.0001),
                                )))
    if hfun_mesh is None:
        hfun_nodes=None

    if min_int_ang is None:
        msht_buffer = triangulate_polygon(gdf_full_buffer,
                                          aux_pts=hfun_nodes)
    else:
        area_threshold = 1.0e-15 #to remove slivers
        gdf_full_buffer['area'] = gdf_full_buffer.geometry.area
        gdf_full_buffer = gdf_full_buffer[gdf_full_buffer['area'] >= area_threshold]
        msht_buffer = triangulate_polygon_s(gdf_full_buffer,
                                            min_int_ang=min_int_ang,
                                            aux_pts=hfun_nodes)
    msht_buffer.crs = crs
    gap_poly = gpd.GeoDataFrame(geometry=
                                gpd.GeoSeries(
                                gap_poly.buffer(buffer_domain),
                                ))
    gap_poly = gap_poly.explode().dissolve()

    #msht_buffer=clip_mesh_by_shape(msht_buffer,gap_poly.union_all())

    return msht_buffer

def clip_mesh_by_mesh(
    mesh_to_be_clipped: MeshData,
    mesh_clipper: MeshData,
    inverse: bool = True,
    fit_inside: bool = False,
    check_cross_edges: bool = True,
    adjacent_layers: int = 2,
    buffer_size = None,
    crs=None
) -> MeshData:

    clipper_poly = utils.get_mesh_polygons(mesh_clipper)

    if buffer_size is not None and buffer_size > 0:
        shape = clipper_poly.buffer(buffer_size)
    else:
        shape = clipper_poly

    return utils.clip_mesh_by_shape(
        mesh_to_be_clipped,
        shape=shape,
        inverse=inverse,
        fit_inside=fit_inside,
        check_cross_edges=check_cross_edges,
        adjacent_layers=adjacent_layers
    )


# =============================================================================
# Triangle Wrappers
# =============================================================================
def triangulate_polygon(
    shape,
    aux_pts=None,
    opts='p',
    # type_t=2,
) -> MeshData:
    """
    Triangulate input shape. 
    """
    # 1. Prepare Shape as GeoSeries for the Engine
    if isinstance(shape, (gpd.GeoDataFrame, gpd.GeoSeries)):
        # Engine expects GeoSeries of the flattened shape or raw list
        # To match old behavior, we might want to union them if it's multiple
        shape_geom = shape.union_all()
        shape_series = gpd.GeoSeries([shape_geom])
    elif isinstance(shape, (list, np.ndarray)):
        # Handle list of polygons
        shape_geom = union_all(shape)
        shape_series = gpd.GeoSeries([shape_geom])
    elif isinstance(shape, (Polygon, MultiPolygon)):
        shape_series = gpd.GeoSeries([shape])
    else:
        raise ValueError("Input shape must be convertible to polygon/series!")

    # 2. Prepare Aux Points (Seed) for the Engine
    # The Engine expects a MeshData object as a seed.
    seed_mesh = None
    if aux_pts is not None:
        aux_coords = None

        # Logic to extract coords matches the old function to ensure compatibility
        if isinstance(aux_pts, (gpd.GeoDataFrame, gpd.GeoSeries)):
            if isinstance(aux_pts, gpd.GeoDataFrame):
                gs = aux_pts.geometry
            else:
                gs = aux_pts

            def get_coords(geom):
                if geom is None or geom.is_empty:
                    return []
                if isinstance(geom, Point):
                    return [geom.coords[0]]
                if isinstance(geom, MultiPoint):
                    return [pt.coords[0] for pt in geom.geoms]
                return []

            coords_list = []
            for geom in gs:
                coords_list.extend(get_coords(geom))

            if coords_list:
                aux_coords = np.array(coords_list)

        elif isinstance(aux_pts, (list, np.ndarray)):
            aux_coords = np.array(aux_pts)

        if aux_coords is not None and len(aux_coords) > 0:
            # Create a seed mesh containing only vertices
            seed_mesh = MeshData(coords=aux_coords)

    # 3. Instantiate Engine and Generate
    # We pass 'opts' to the factory via kwargs which go to TriangleOptions
    engine = get_mesh_engine('triangle', opts=opts)

    # Engine.generate expects a GeoSeries for shape and MeshData for seed
    msht = engine.generate(shape_series, seed=seed_mesh)

    if aux_pts is not None:
        utils.cleanup_isolates(msht)

    return msht

def triangulate_polygon_s(
    shape,
    aux_pts=None,
    min_int_ang=30,
) -> MeshData:
    '''
    Triangulate the input shape smoothly (2-pass).
    Follows exact logic of old `triangulate_polygon_s`.
    '''
    # Pass 1: Smooth triangulation (adds Steiner points)
    # Using 'q' switch
    opts_1 = f'pq{min_int_ang}'
    mesh1 = triangulate_polygon(shape, aux_pts=aux_pts, opts=opts_1)

    # Extract Internal Nodes
    # Find boundary edges
    nb = utils.get_boundary_edges(mesh1)

    if nb.size > 0:
        nb_idx = np.unique(nb.ravel())
        all_pts = mesh1.coords
        # Delete boundary nodes, keeping only internal ones
        internal_pts = np.delete(all_pts, nb_idx, axis=0)
    else:
        internal_pts = mesh1.coords

    # Pass 2: Constrained triangulation
    # Use internal nodes as aux_pts, but don't refine further ('p' only)
    mesh2 = triangulate_polygon(shape, aux_pts=internal_pts, opts='p')

    return mesh2
