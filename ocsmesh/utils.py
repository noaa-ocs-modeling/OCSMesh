from collections import defaultdict
from itertools import permutations, islice, combinations
from typing import Union, Dict, Sequence, Tuple, List
from functools import reduce
from multiprocessing import cpu_count, Pool
from copy import deepcopy
import logging
import warnings

from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
import numpy.typing as npt
import rasterio as rio
from pyproj import CRS, Transformer
from scipy.interpolate import RectBivariateSpline, griddata
from scipy import sparse, constants
from scipy.spatial import cKDTree
from shapely import intersection, difference
from shapely.geometry import (
    Polygon, MultiPolygon,
    box, GeometryCollection, Point, MultiPoint,
    LineString, LinearRing
)
from shapely.ops import polygonize, linemerge, unary_union, triangulate
from shapely.validation import make_valid
import geopandas as gpd
import pandas as pd
import utm

from ocsmesh.internal import MeshData

# Updated constants for MeshData attributes
ELEM_2D_TYPES = ['tria', 'quad']

_logger = logging.getLogger(__name__)

def mesh_to_tri(mesh):
    """
    mesh is a MeshData instance.
    """
    return Triangulation(
        mesh.coords[:, 0],
        mesh.coords[:, 1],
        mesh.tria
    )


def cleanup_isolates(mesh):
    used_old_idx = np.array([], dtype='int64')
    
    # Collect all nodes used by elements
    for etype in ELEM_2D_TYPES:
        elem_idx = getattr(mesh, etype).flatten()
        used_old_idx = np.hstack((used_old_idx, elem_idx))
    
    used_old_idx = np.unique(used_old_idx)

    # update coords and values
    old_values = mesh.values
    mesh.coords = mesh.coords[used_old_idx]
    mesh.values = old_values[used_old_idx]


    # Re-map elements
    renum = {old: new for new, old in enumerate(used_old_idx)}
    
    # Update elements
    for etype in ELEM_2D_TYPES:
        elem_idx = getattr(mesh, etype)
        if elem_idx.size == 0:
            continue
            
        elem_new_idx = np.array([renum[i] for i in elem_idx.flatten()])
        elem_new_idx = elem_new_idx.reshape(elem_idx.shape)
        setattr(mesh, etype, elem_new_idx)


def cleanup_duplicates(mesh):
    """Cleanup duplicate nodes and elements

    Notes
    -----
    Elements and nodes are duplicate if they fully overlapping (not
    partially)
    """

    # 1. Clean Nodes
    _, cooidx, coorev = np.unique(
        mesh.coords,
        axis=0,
        return_index=True,
        return_inverse=True
    )
    
    # Save old values before coords update resets them
    old_values = mesh.values
    
    # Update coords (this might reset values to 0 in MeshData)
    mesh.coords = mesh.coords[cooidx]
    
    # Restore values mapped to new unique nodes
    mesh.values = old_values[cooidx]

    # 2. Clean Elements (Renumbering)
    nd_map = dict(enumerate(coorev))
    
    for etype in ELEM_2D_TYPES:
        cnn = getattr(mesh, etype)
        if cnn.size == 0:
            continue

        n_node = cnn.shape[1]
        
        # Map old indices to new unique indices
        cnn_renu = np.array(
            [nd_map[i] for i in cnn.flatten()]
        ).reshape(-1, n_node)

        # Remove duplicate elements
        _, cnnidx = np.unique(
            np.sort(cnn_renu, axis=1),
            axis=0,
            return_index=True
        )
        
        # Assign cleaned elements back
        setattr(mesh, etype, cnn_renu[cnnidx])


def cleanup_folded_bound_el(mesh):
    '''
    delete all boundary elements whose nodes (all 3) are boundary nodes
    '''

    # TODO: Check quads as well

    nb = get_boundary_edges(mesh)
    nb = np.sort(list(set(nb.ravel())))
    coords = mesh.coords
    
    el = {1: index for i, index in enumerate(mesh.tria)}
    
    el_pd = pd.DataFrame.from_dict(el,
                                   orient='index',
                                   columns=['one', 'two','tree'])
    selection = el_pd[el_pd.isin(nb)].dropna().astype(int)
    del_tri = selection.values.tolist()
    
    # preparing the geodataframe for the triangles
    all_gdf = []
    for t in del_tri:
        x,y=[],[]
        for n in t:
            x.append(coords[n][0])
            y.append(coords[n][-1])
        polygon_geom = Polygon(zip(x, y))
        all_gdf.append(gpd.GeoDataFrame(crs='epsg:4326',
                                        geometry=[polygon_geom]))
    
    if not all_gdf:
        return mesh

    clip_tri = gpd.GeoDataFrame(pd.concat(all_gdf,ignore_index=True))
    
    # removing the erroneous elements using the triangles (clip_tri)
    fixed_mesh = clip_mesh_by_shape(
                mesh,
                shape=clip_tri.union_all(),
                inverse=True,
                fit_inside=True,
                check_cross_edges=True,
                adjacent_layers=0,
            )

    return fixed_mesh


def geom_to_multipolygon(mesh):
    vertices = mesh.coords
    idx_ring_coll = index_ring_collection(mesh)
    polygon_collection = []
    for polygon in idx_ring_coll.values():
        exterior = vertices[polygon['exterior'][:, 0], :]
        interiors = []
        for interior in polygon['interiors']:
            interiors.append(vertices[interior[:, 0], :])
        polygon_collection.append(Polygon(exterior, interiors))
    return MultiPolygon(polygon_collection)


def get_boundary_segments(mesh) -> List[LineString]:

    coords = mesh.coords
    boundary_edges = get_boundary_edges(mesh)
    boundary_verts = np.unique(boundary_edges)
    boundary_coords = coords[boundary_verts]

    vert_map = {
            orig: new for new, orig in enumerate(boundary_verts)}
    new_boundary_edges = np.array(
        [vert_map[v] for v in boundary_edges.ravel()]).reshape(
                boundary_edges.shape)

    graph = sparse.lil_matrix(
            (len(boundary_verts), len(boundary_verts)))
    for vert1, vert2 in new_boundary_edges:
        graph[vert1, vert2] = 1

    n_components, labels = sparse.csgraph.connected_components(
            graph, directed=False)

    segments = []
    for i in range(n_components):
        conn_mask = np.any(np.isin(
                new_boundary_edges, np.nonzero(labels == i)),
                axis=1)
        conn_edges = new_boundary_edges[conn_mask]
        this_segment = linemerge(boundary_coords[conn_edges].tolist())
        if not this_segment.is_simple:
            # Pinched nodes also result in non-simple linestring,
            # but they can be handled gracefully, here we are looking
            # for other issues like folded elements
            test_polys = list(polygonize(this_segment))
            if not test_polys:
                raise ValueError(
                    "Mesh boundary crosses itself! Folded element(s)!")
        segments.append(this_segment)

    return segments


def get_mesh_polygons(mesh):
    elm_polys = []
    for elm_type in ELEM_2D_TYPES:
        elems = getattr(mesh, elm_type)
        if elems.size > 0:
            elm_polys.extend(
                [Polygon(mesh.coords[cell]) for cell in elems]
            )

    poly = unary_union(elm_polys)
    if isinstance(poly, Polygon):
        poly = MultiPolygon([poly])

    return poly


def repartition_features(
    linestring: LineString,
    max_verts: int,
):


    if not isinstance(linestring, LineString):
        raise ValueError(
            f"Input shape must be a LineString not {type(linestring)}!"
        )
    if not isinstance(max_verts, int):
        raise ValueError(
            "Maximum number of vertices must be an integer!"
        )

    features = []
    lstr_len = len(linestring.coords)
    if lstr_len > max_verts:
        # NOTE: -1 is because of shared connecting nodes
        list_lens = [max_verts] * ((lstr_len - 1) // (max_verts - 1))
        if (lstr_len - 1) % (max_verts - 1) != 0:
            list_lens += [(lstr_len - 1) % (max_verts - 1) + 1]
        new_idx = np.cumsum(np.array(list_lens) - 1)
        orig_coords = np.array(linestring.coords)
        last_idx = 0
        for idx in new_idx:
            features.append(LineString(orig_coords[last_idx:idx + 1]))
            last_idx = idx
    else:
        features.append(linestring)
    return features


def transform_linestring(
    linestring: LineString,
    target_size: float,
):

    if not isinstance(linestring, LineString):
        raise ValueError(
            f"Input shape must be a LineString not {type(linestring)}!"
        )

    lstr_len = linestring.length
    distances = np.cumsum(np.ones(int(lstr_len // target_size)) * target_size)
    if len(distances) == 0:
        return linestring
    if distances[-1] < lstr_len:
        distances = np.append(distances, lstr_len)

    orig_coords = np.array(linestring.coords)
    lengths = ((orig_coords[1:] - orig_coords[:-1]) ** 2).sum(axis=1) ** 0.5
    cum_len = np.cumsum(lengths)

    assert(max(cum_len) == max(distances))

    # Memory issue due to matrix creation?
    idx = len(cum_len) - (distances[:, None] <= cum_len).sum(axis=1)
    ratio = ((cum_len[idx] - distances) / lengths[idx])[:, None]

    interp_coords = (
        orig_coords[idx] * (ratio) + orig_coords[idx + 1] * (1 - ratio)
    )
    interp_coords = np.vstack((orig_coords[0], interp_coords))
    linestring = LineString(interp_coords)
    return linestring


def needs_sieve(mesh, area=None):
    areas = [polygon.area for polygon in geom_to_multipolygon(mesh)]
    if area is None:
        remove = np.where(areas < np.max(areas))[0].tolist()
    else:
        remove = []
        for idx, patch_area in enumerate(areas):
            if patch_area <= area:
                remove.append(idx)
    if len(remove) > 0:
        return True
    return False




def _get_sieve_mask(mesh, polygons, sieve_area):
    areas = [p.area for p in polygons.geoms]
    if sieve_area is None:
        remove = np.where(areas < np.max(areas))[0].tolist()
    else:
        remove = []
        for idx, patch_area in enumerate(areas):
            if patch_area <= sieve_area:
                remove.append(idx)

    # if the path surrounds the node, these need to be removed.
    vert2_mask = np.full((mesh.coords.shape[0],), False)
    for idx in remove:
        path = Path(polygons.geoms[idx].exterior.coords, closed=True)
        vert2_mask = vert2_mask | path.contains_points(mesh.coords)

    return vert2_mask


def _sieve_by_mask(mesh, sieve_mask):

    # if the path surrounds the node, these need to be removed.
    vert2_mask = sieve_mask.copy()

    # select any connected nodes; these ones are missed by
    # path.contains_point() because they are at the path edges.
    _idxs = np.where(vert2_mask)[0]
    conn_verts = get_surrounding_elem_verts(mesh, _idxs)
    vert2_mask[conn_verts] = True

    # Also, there might be some dangling triangles without neighbors,
    # which are also missed by path.contains_point()
    lone_elem_verts = get_lone_element_verts(mesh)
    vert2_mask[lone_elem_verts] = True

    used_old_idx = np.array([], dtype='int64')
    filter_dict = {}
    for etype in ELEM_2D_TYPES:
        elem_idx = getattr(mesh, etype)
        elem_mask = np.any(vert2_mask[elem_idx], axis=1)

        elem_keep_idx = elem_idx[~elem_mask, :].flatten()
        used_old_idx = np.hstack((used_old_idx, elem_keep_idx))
        filter_dict[etype] = [elem_keep_idx, elem_idx.shape[1]]
    used_old_idx = np.unique(used_old_idx)

    # Update mesh (preserving values)

    old_values = mesh.values
    mesh.coords = mesh.coords[used_old_idx]
    mesh.values = old_values[used_old_idx]

    renum = {old: new for new, old in enumerate(np.unique(used_old_idx))}
    for etype, (elem_keep_idx, topo) in filter_dict.items():
        if len(elem_keep_idx) == 0:
            setattr(mesh, etype, [])
            continue
        elem_new_idx = np.array([renum[i] for i in elem_keep_idx])
        elem_new_idx = elem_new_idx.reshape(-1, topo)
        setattr(mesh, etype, elem_new_idx)


def finalize_mesh(mesh, sieve_area=None):
    cleanup_isolates(mesh)

    while True:
        no_op = True
        pinched_nodes = get_pinched_nodes(mesh)
        if len(pinched_nodes):
            no_op = False
            clip_mesh_by_vertex(
                mesh, pinched_nodes,
                can_use_other_verts=True,
                inverse=True, in_place=True)

        boundary_polys = get_mesh_polygons(mesh)
        sieve_mask = _get_sieve_mask(mesh, boundary_polys, sieve_area)
        if np.sum(sieve_mask):
            no_op = False
            _sieve_by_mask(mesh, sieve_mask)

        if no_op:
            break

    cleanup_isolates(mesh)
    cleanup_duplicates(mesh)



def sieve(mesh, area=None):
    """
    A mesh can consist of multiple separate subdomins on as single structure.
    This functions removes subdomains which are equal or smaller than the
    provided area. Default behaviours is to remove all subdomains except the
    largest one.
    """
    # select the nodes to remove based on multipolygon areas
    multipolygon = geom_to_multipolygon(mesh)
    areas = [polygon.area for polygon in multipolygon]
    if area is None:
        remove = np.where(areas < np.max(areas))[0].tolist()
    else:
        remove = []
        for idx, patch_area in enumerate(areas):
            if patch_area <= area:
                remove.append(idx)

    # if the path surrounds the node, these need to be removed.
    vert2_mask = np.full((mesh.coords.shape[0],), False)
    for idx in remove:
        path = Path(multipolygon[idx].exterior.coords, closed=True)
        vert2_mask = vert2_mask | path.contains_points(mesh.coords)

    return _sieve_by_mask(mesh, vert2_mask)


def sort_edges(edges):
    if len(edges) == 0:
        return edges

    # start ordering the edges into linestrings
    edge_collection = []
    ordered_edges = [edges.pop(-1)]
    e0, e1 = [list(t) for t in zip(*edges)]
    while len(edges) > 0:

        if ordered_edges[-1][1] in e0:
            idx = e0.index(ordered_edges[-1][1])
            ordered_edges.append(edges.pop(idx))

        elif ordered_edges[0][0] in e1:
            idx = e1.index(ordered_edges[0][0])
            ordered_edges.insert(0, edges.pop(idx))

        elif ordered_edges[-1][1] in e1:
            idx = e1.index(ordered_edges[-1][1])
            ordered_edges.append(
                list(reversed(edges.pop(idx))))

        elif ordered_edges[0][0] in e0:
            idx = e0.index(ordered_edges[0][0])
            ordered_edges.insert(
                0, list(reversed(edges.pop(idx))))

        else:
            edge_collection.append(tuple(ordered_edges))
            idx = -1
            ordered_edges = [edges.pop(idx)]

        e0.pop(idx)
        e1.pop(idx)

    # finalize
    if len(edge_collection) == 0 and len(edges) == 0:
        edge_collection.append(tuple(ordered_edges))
    else:
        edge_collection.append(tuple(ordered_edges))

    return edge_collection


def index_ring_collection(mesh):

    # find boundary edges using triangulation neighbors table,
    # see: https://stackoverflow.com/a/23073229/7432462
    boundary_edges = []
    tri = mesh_to_tri(mesh)
    idxs = np.vstack(
        list(np.where(tri.neighbors == -1))).T
    for i, j in idxs:
        boundary_edges.append(
            (int(tri.triangles[i, j]),
                int(tri.triangles[i, (j+1) % 3])))
    init_idx_ring_coll = sort_edges(boundary_edges)
    # sort index_rings into corresponding "polygons"
    areas = []
    vertices = mesh.coords
    for index_ring in init_idx_ring_coll:
        e0, _ = [list(t) for t in zip(*index_ring)]
        areas.append(float(Polygon(vertices[e0, :]).area))

    # maximum area must be main mesh
    idx = areas.index(np.max(areas))
    exterior = init_idx_ring_coll.pop(idx)
    areas.pop(idx)
    _id = 0
    idx_ring_coll = {}
    idx_ring_coll[_id] = {
        'exterior': np.asarray(exterior),
        'interiors': []
        }
    e0, e1 = [list(t) for t in zip(*exterior)]
    path = Path(vertices[e0 + [e0[0]], :], closed=True)
    while len(init_idx_ring_coll) > 0:
        # find all internal rings
        potential_interiors = []
        for i, index_ring in enumerate(init_idx_ring_coll):
            e0, e1 = [list(t) for t in zip(*index_ring)]
            if path.contains_point(vertices[e0[0], :]):
                potential_interiors.append(i)
        # filter out nested rings
        real_interiors = []
        for i, p_interior in reversed(list(enumerate(potential_interiors))):
            _p_interior = init_idx_ring_coll[p_interior]
            check = [init_idx_ring_coll[_]
                     for j, _ in reversed(list(enumerate(potential_interiors)))
                     if i != j]
            has_parent = False
            for _path in check:
                e0, e1 = [list(t) for t in zip(*_path)]
                _path = Path(vertices[e0 + [e0[0]], :], closed=True)
                if _path.contains_point(vertices[_p_interior[0][0], :]):
                    has_parent = True
                    break
            if not has_parent:
                real_interiors.append(p_interior)
        # pop real rings from collection
        for i in reversed(sorted(real_interiors)):
            idx_ring_coll[_id]['interiors'].append(
                np.asarray(init_idx_ring_coll.pop(i)))
            areas.pop(i)
        # if no internal rings found, initialize next polygon
        if len(init_idx_ring_coll) > 0:
            idx = areas.index(np.max(areas))
            exterior = init_idx_ring_coll.pop(idx)
            areas.pop(idx)
            _id += 1
            idx_ring_coll[_id] = {
                'exterior': np.asarray(exterior),
                'interiors': []
                }
            e0, e1 = [list(t) for t in zip(*exterior)]
            path = Path(vertices[e0 + [e0[0]], :], closed=True)
    return idx_ring_coll


def outer_ring_collection(mesh):
    idx_ring_coll = index_ring_collection(mesh)
    exterior_ring_collection = defaultdict()
    for key, ring in idx_ring_coll.items():
        exterior_ring_collection[key] = ring['exterior']
    return exterior_ring_collection


def inner_ring_collection(mesh):
    idx_ring_coll = index_ring_collection(mesh)
    inner_ring_coll = defaultdict()
    for key, rings in idx_ring_coll.items():
        inner_ring_coll[key] = rings['interiors']
    return inner_ring_coll


def get_multipolygon_from_pathplot(ax):
    # extract linear_rings from plot
    linear_ring_collection = []
    multipolygon = None
    for path_collection in ax.collections:
        for path in path_collection.get_paths():
            polygons = path.to_polygons(closed_only=True)
            for linear_ring in polygons:
                if linear_ring.shape[0] > 3:
                    linear_ring_collection.append(
                        LinearRing(linear_ring))
    if len(linear_ring_collection) > 1:
        # reorder linear rings from above
        polygon_collection = []
        while len(linear_ring_collection) > 0:
            areas = [Polygon(linear_ring).area
                     for linear_ring in linear_ring_collection]
            idx = np.where(areas == np.max(areas))[0][0]
            outer_ring = linear_ring_collection.pop(idx)
            path = Path(np.asarray(outer_ring.coords), closed=True)

            inner_rings = []
            for i, linear_ring in reversed(
                    list(enumerate(linear_ring_collection))):
                xy = np.asarray(linear_ring.coords)[0, :]
                if path.contains_point(xy):
                    inner_rings.append(linear_ring_collection.pop(i))
            polygon_collection.append(Polygon(outer_ring, inner_rings))

        multipolygon = MultiPolygon(polygon_collection)
    elif len(linear_ring_collection) != 0:
        multipolygon = MultiPolygon(
            [Polygon(linear_ring_collection.pop())])
    return multipolygon


def signed_polygon_area(vertices):
    # https://code.activestate.com/recipes/578047-area-of-polygon-using-shoelace-formula/
    n = len(vertices)  # of vertices
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return area / 2.0


def vertices_around_vertex(mesh):
    def append(geom):
        for simplex in geom:
            for i, j in permutations(simplex, 2):
                vert_list[i].add(j)
    vert_list = defaultdict(set)
    append(mesh.tria)
    append(mesh.quad)
    return vert_list


def get_surrounding_elem_verts(mesh, in_vert):
    tria = mesh.tria
    quad = mesh.quad
    
    conn_verts = []
    
    # NOTE: np.any is used so that vertices that are not in in_verts
    # triangles but are part of the triangles that include in_verts
    # are considered too
    if tria.size > 0:
        
        mark_tria = np.any(
                (np.isin(tria.ravel(), in_vert).reshape(tria.shape)), 1)
        conn_verts.append(tria[mark_tria, :].ravel())
        
    if quad.size > 0:
        mark_quad = np.any(
                (np.isin(quad.ravel(), in_vert).reshape(quad.shape)), 1)
        conn_verts.append(quad[mark_quad, :].ravel())

    if not conn_verts:
        return np.array([])

    return np.unique(np.concatenate(conn_verts))


def get_lone_element_verts(mesh):

    '''
    Also, there might be some dangling triangles without neighbors,
    which are also missed by path.contains_point()
    '''
    tria = mesh.tria
    quad = mesh.quad
    
    all_indices = []
    if tria.size > 0: all_indices.append(tria.ravel())
    if quad.size > 0: all_indices.append(quad.ravel())
    
    if not all_indices:
        return np.array([])

    # Find vertices that are referred to by all elements once
    unq_verts, counts = np.unique(
        np.concatenate(all_indices),
        return_counts=True)
    once_verts = unq_verts[counts == 1]

    # Find elements composed of only vertices that are used one
    lone_elem_verts_list = []
    
    if tria.size > 0:
        mark_tria = np.all(
                (np.isin(tria.ravel(), once_verts).reshape(tria.shape)), 1)
        lone_elem_verts_list.append(tria[mark_tria, :].ravel())

    if quad.size > 0:
        mark_quad = np.all(
                (np.isin(quad.ravel(), once_verts).reshape(quad.shape)), 1)
        lone_elem_verts_list.append(quad[mark_quad, :].ravel())

    if not lone_elem_verts_list:
        return np.array([])

    return np.unique(np.concatenate(lone_elem_verts_list))

# https://en.wikipedia.org/wiki/Polygon_mesh#Summary_of_mesh_representation
# V-V     All vertices around vertex
# E-F     All edges of a face
# V-F     All vertices of a face
# F-V     All faces around a vertex
# E-V     All edges around a vertex
# F-E     Both faces of an edge
# V-E     Both vertices of an edge
# Flook   Find face with given vertices
def get_verts_in_shape(
        mesh: MeshData,
        shape: Union[box, Polygon, MultiPolygon, gpd.GeoSeries],
        from_box: bool = False,
        num_adjacent: int = 0
        ) -> Sequence[int]:

    shp_series = gpd.GeoSeries(shape)
    if from_box:
        crd = mesh.coords
        xmin, ymin, xmax, ymax = shp_series.total_bounds

        in_box_idx_1 = np.arange(len(crd))[crd[:, 0] > xmin]
        in_box_idx_2 = np.arange(len(crd))[crd[:, 0] < xmax]
        in_box_idx_3 = np.arange(len(crd))[crd[:, 1] > ymin]
        in_box_idx_4 = np.arange(len(crd))[crd[:, 1] < ymax]
        in_box_idx = reduce(
            np.intersect1d, (in_box_idx_1, in_box_idx_2,
                             in_box_idx_3, in_box_idx_4))
        return in_box_idx

    pt_series = gpd.GeoSeries(gpd.points_from_xy(
        mesh.coords[:,0], mesh.coords[:,1]))

    # We need point indices in the shapes, not the shape indices
    # query bulk returns all combination of intersections in case
    # input shape results in multi-row series
    in_shp_idx = pt_series.sindex.query(
            shp_series, predicate="intersects")[1]

    in_shp_idx = select_adjacent(mesh, in_shp_idx, num_layers=num_adjacent)

    return in_shp_idx

def select_adjacent(mesh, in_indices, num_layers):
    selected_indices = in_indices.copy()

    # MeshData is implicitly 2D Euclidean
    for i in range(num_layers - 1):
        elm_dict = {
            etype: getattr(mesh, etype) for etype in ELEM_2D_TYPES}

        mark_func = np.any

        mark_dict = {
            key: mark_func(
                (np.isin(elems.ravel(), selected_indices).reshape(
                    elems.shape)), 1)
            for key, elems in elm_dict.items()}

        picked_elems = {
                key: elm_dict[key][mark_dict[key], :]
                for key in elm_dict}

        selected_indices = np.unique(np.concatenate(
            [pick.ravel() for pick in picked_elems.values()]))

    return selected_indices


def get_incident_edges(
        mesh: MeshData,
        vert_idx_list: Sequence[int],
        ) -> Sequence[Tuple[int, int]]:
    edges = get_mesh_edges(mesh, unique=True)
    test = np.isin(edges, vert_idx_list).any(axis=1)
    return edges[test]


def get_cross_edges(
        mesh: MeshData,
        shape: Union[box, Polygon, MultiPolygon],
        ) -> Sequence[Tuple[int, int]]:

    '''
    Return the list of edges crossing the input shape exterior
    '''

    coords = mesh.vert2['coord']

    coord_dict = {}
    for i, coo in enumerate(coords):
        coord_dict[tuple(coo)] = i

    gdf_shape = gpd.GeoDataFrame(geometry=gpd.GeoSeries(shape))
    exteriors = [pl.exterior for pl in gdf_shape.explode(index_parts=True).geometry]

    # TODO: Reduce domain of search for faster results
    all_edges = get_mesh_edges(mesh, unique=True)
    edge_coords = coords[all_edges, :]
    gdf_edg = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(linemerge(edge_coords.tolist())))

    gdf_x = gpd.sjoin(
            gdf_edg.explode(index_parts=True),
            gpd.GeoDataFrame(geometry=gpd.GeoSeries(exteriors)),
            how='inner', predicate='intersects')

    cut_coords = [
        list(cooseq)
        for cooseq in gdf_x.geometry.apply(lambda i: i.coords).values]

    cut_edges = np.array([
        (coo_list[i], coo_list[i+1])
        for coo_list in cut_coords
        for i in range(len(coo_list)-1) ])

    cut_edge_idx = np.array(
            [coord_dict[tuple(coo)]
             for coo in cut_edges.reshape(-1, 2)]).reshape(
                     cut_edges.shape[:2])

    return cut_edge_idx


def clip_mesh_by_shape(
        mesh: MeshData,
        shape: Union[box, Polygon, MultiPolygon],
        use_box_only: bool = False,
        fit_inside: bool = True,
        inverse: bool = False,
        in_place: bool = False,
        check_cross_edges: bool = False,
        adjacent_layers: int = 0
        ) -> MeshData:


    # NOTE: Checking cross edge is only meaningful when
    # fit inside flag is NOT set
    edge_flag = check_cross_edges and not fit_inside

    # If we want to calculate inverse based on shape, calculating
    # from bbox first results in the wrong result
    if not inverse or use_box_only:

        # First based on bounding box only
        shape_box = box(*shape.bounds)

        # TODO: Optimize for multipolygons (use separate bboxes)
        in_box_idx = get_verts_in_shape(mesh, shape_box, True, adjacent_layers)

        if edge_flag and not inverse:
            x_edge_idx = get_cross_edges(mesh, shape_box)
            in_box_idx = np.append(in_box_idx, np.unique(x_edge_idx))

        mesh = clip_mesh_by_vertex(
                mesh, in_box_idx, not fit_inside, inverse, in_place)

        if use_box_only:
            if edge_flag and inverse:
                x_edge_idx = get_cross_edges(mesh, shape_box)
                mesh = remove_mesh_by_edge(mesh, x_edge_idx, in_place)
            return mesh

    in_shp_idx = get_verts_in_shape(mesh, shape, False, adjacent_layers)

    if edge_flag and not inverse:
        x_edge_idx = get_cross_edges(mesh, shape)
        in_shp_idx = np.append(in_shp_idx, np.unique(x_edge_idx))

    mesh = clip_mesh_by_vertex(
            mesh, in_shp_idx, not fit_inside, inverse, in_place)

    if edge_flag and inverse:
        x_edge_idx = get_cross_edges(mesh, shape)
        mesh = remove_mesh_by_edge(mesh, x_edge_idx, in_place)

    return mesh


def remove_mesh_by_edge(
        mesh: MeshData,
        edges: Sequence[Tuple[int, int]],
        in_place: bool = False
        ) -> MeshData:

    mesh_out = mesh
    if not in_place:
        mesh_out = deepcopy(mesh)

    # NOTE: This method selects more elements than needed as it
    # uses only existance of more than two of the vertices attached
    # to the input edges in the element as criteria.
    edge_verts = np.unique(edges)

    for etype in ELEM_2D_TYPES:
        elems = getattr(mesh, etype)
        if elems.size == 0: continue
        
        # If a given element contains two vertices from
        # a crossing edge, it is selected
        test = np.sum(np.isin(elems, edge_verts), axis=1)
        elems = elems[test < 2]
        setattr(mesh_out, etype, elems)

    return mesh_out


def clip_mesh_by_vertex(
        mesh: MeshData,
        vert_in: Sequence[int],
        can_use_other_verts: bool = False,
        inverse: bool = False,
        in_place: bool = False
        ) -> MeshData:

    coord = mesh.coords

    elm_dict = {
        etype: getattr(mesh, etype) for etype in ELEM_2D_TYPES}

    mark_func = np.all
    if can_use_other_verts:
        mark_func = np.any

    mark_dict = {
        key: mark_func(
            (np.isin(elems.ravel(), vert_in).reshape(
                elems.shape)), 1)
        for key, elems in elm_dict.items()}

    # Whether to return elements found by "in" vertices or return
    # all elements except them
    if inverse:
        mark_dict = {
            key: np.logical_not(mark)
            for key, mark in mark_dict.items()}

    # Find elements based on old vertex index
    elem_draft_dict = {
            key: elm_dict[key][mark_dict[key], :]
            for key in elm_dict}

    crd_old_to_new = {
            index: i for i, index
            in enumerate(sorted(np.unique(np.concatenate(
                    [draft.ravel()
                        for draft in elem_draft_dict.values()]
                    ))))
        }

    elem_final_dict = {
        key: np.array(
            [[crd_old_to_new[x] for x in  element]
             for element in draft])
        for key, draft in elem_draft_dict.items()
    }

    new_coord = coord[list(crd_old_to_new.keys()), :]
    
    # Handle values
    new_values = mesh.values[list(crd_old_to_new.keys())]

    mesh_out = mesh
    if not in_place:
        mesh_out = MeshData(coords=new_coord, values=new_values)
        if hasattr(mesh, "crs"):
            mesh_out.crs = deepcopy(mesh.crs)
    else:
        # In place updates
        mesh_out.coords = new_coord
        mesh_out.values = new_values

    for key in elem_final_dict:
        setattr(mesh_out, key, elem_final_dict[key])

    return mesh_out


def get_mesh_edges(mesh: MeshData, unique=True):
    trias = mesh.tria
    quads = mesh.quad
    
    # Get unique set of edges by rolling connectivity
    # and joining connectivities in 3rd dimension, then sorting
    # to get all edges with lower index first
    all_edges = np.empty(shape=(0, 2), dtype=int)
    for elm_type in [trias, quads]:
        if elm_type.shape[0]:
            edges = np.sort(
                    np.stack(
                        (elm_type, np.roll(elm_type, shift=1, axis=1)),
                        axis=2),
                    axis=2)
            edges = edges.reshape(np.prod(edges.shape[0:2]), 2)
            all_edges = np.vstack((all_edges, edges))

    if unique:
        all_edges = np.unique(all_edges, axis=0)

    return all_edges


def calculate_tria_areas(mesh):
    coord = mesh.coords
    trias = mesh.tria

    tria_coo = coord[
        np.sort(np.stack((trias, np.roll(trias, shift=1, axis=1)),
                         axis=2),
                axis=2)]
    tria_side_components = np.diff(tria_coo, axis=2).squeeze()
    tria_sides = np.sqrt(
            np.sum(np.power(np.abs(tria_side_components), 2),
                   axis=2).squeeze())
    perimeter = np.sum(tria_sides, axis=1) / 2
    perimeter = perimeter.reshape(len(perimeter), 1)
    a_side, b_side, c_side = np.split(tria_sides, 3, axis=1)
    tria_areas = np.sqrt(
            perimeter*(perimeter-a_side)
            * (perimeter-b_side)*(perimeter-c_side)
            ).squeeze()
    return tria_areas


def calculate_edge_lengths(mesh, transformer=None):
    coord = mesh.coords
    if transformer is not None:
        coord = np.vstack(
            transformer.transform(coord[:, 0], coord[:, 1])).T

    # Get unique set of edges by rolling connectivity
    # and joining connectivities in 3rd dimension, then sorting
    # to get all edges with lower index first
    all_edges = get_mesh_edges(mesh, unique=True)

    # ONLY TESTED FOR TRIA AS OF NOW

    # This part of the function is generic for tria and quad

    # Get coordinates for all edge vertices
    edge_coords = coord[all_edges, :]

    # Calculate length of all edges based on acquired coords
    edge_lens = np.sqrt(
            np.sum(
                np.power(
                    np.abs(np.diff(edge_coords, axis=1)), 2)
                ,axis=2)).squeeze()

    edge_dict = defaultdict(float)
    for i, edge in enumerate(all_edges):
        edge_dict[tuple(edge)] = edge_lens[i]

    return edge_dict


def get_boundary_edges(mesh):
    all_edges = get_mesh_edges(mesh, unique=False)
    all_edges, e_cnt = np.unique(all_edges, axis=0, return_counts=True)
    boundary_edges = all_edges[e_cnt == 1]
    return boundary_edges


def get_pinched_nodes(mesh):

    '''
    Find nodes through which fluid cannot flow
    '''

    boundary_edges = get_boundary_edges(mesh)

    # Node indices
    boundary_verts, vb_cnt = np.unique(boundary_edges, return_counts=True)

    # vertices/nodes that have more than 2 boundary edges are pinch
    pinch_verts = boundary_verts[vb_cnt > 2]
    return pinch_verts


def has_pinched_nodes(mesh):

    # Older function: computationally more expensive and missing some
    # nodes

    _inner_ring_collection = inner_ring_collection(mesh)
    all_nodes = []
    for inner_rings in _inner_ring_collection.values():
        for ring in inner_rings:
            all_nodes.extend(np.asarray(ring)[:, 0].tolist())
    u, c = np.unique(all_nodes, return_counts=True)
    if len(u[c > 1]) > 0:
        return True

    return False


def cleanup_pinched_nodes(mesh):

    # Older function: computationally more expensive and missing some
    # nodes

    _inner_ring_collection = inner_ring_collection(mesh)
    all_nodes = []
    for inner_rings in _inner_ring_collection.values():
        for ring in inner_rings:
            all_nodes.extend(np.asarray(ring)[:, 0].tolist())
    u, c = np.unique(all_nodes, return_counts=True)
    mesh.tria = mesh.tria.take(
        np.where(
            ~np.any(np.isin(mesh.tria, u[c > 1]), axis=1))[0],
        axis=0)


def interpolate(src, dst, **kwargs):
    # Assuming euclidean-mesh to euclidean-mesh by default for MeshData
    interpolate_euclidean_mesh_to_euclidean_mesh(src, dst, **kwargs)


def interpolate_euclidean_mesh_to_euclidean_mesh(
        src: MeshData,
        dst: MeshData,
        method='linear',
        fill_value=np.nan
):
    values = griddata(
        src.coords,
        src.values.flatten(),
        dst.coords,
        method=method,
        fill_value=fill_value
    )
    dst.values = values

def tricontourf(
    mesh,
    ax=None,
    show=False,
    figsize=None,
    extend='both',
    colorbar=False,
    **kwargs
):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    # TODO: Add quads!
    tcf = ax.tricontourf(
        mesh.coords[:, 0],
        mesh.coords[:, 1],
        mesh.tria,
        mesh.values.flatten(),
        **kwargs)
    if colorbar:
        plt.colorbar(tcf)
    if show:
        plt.gca().axis('scaled')
        plt.show()
    return ax


def triplot(
    mesh,
    axes=None,
    show=False,
    figsize=None,
    color='k',
    linewidth=0.07,
    **kwargs
):
    if axes is None:
        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111)
    # TODO: Add quads
    axes.triplot(
        mesh.coords[:, 0],
        mesh.coords[:, 1],
        mesh.tria,
        color=color,
        linewidth=linewidth,
        **kwargs)
    if show:
        axes.axis('scaled')
        plt.show()
    return axes


def reproject(
    mesh: MeshData,
    dst_crs: Union[str, CRS]
):
    if mesh.crs is None:
        raise ValueError("Mesh doesn't have a CRS!")
    
    src_crs = mesh.crs
    dst_crs = CRS.from_user_input(dst_crs)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    # pylint: disable=E0633
    x, y = transformer.transform(
        mesh.coords[:, 0], mesh.coords[:, 1])
        
    mesh.coords = np.vstack((x, y)).T
    mesh.crs = dst_crs



def msh_t_to_grd(msh: MeshData) -> Dict:
    warnings.warn("Use meshdata_to_grd instead!", DeprecationWarning)
    return meshdata_to_grd(msh)


def meshdata_to_grd(msh: MeshData) -> Dict:
    src_crs = msh.crs
    coords = msh.coords
    desc = "EPSG:4326"
    if src_crs is not None:
        epsg_4326 = CRS.from_epsg(4326)
        if not src_crs.equals(epsg_4326):
            transformer = Transformer.from_crs(
                src_crs, epsg_4326, always_xy=True)
            coords = np.vstack(
                transformer.transform(coords[:, 0], coords[:, 1])).T

    vals = msh.values.flatten() if msh.values is not None else np.zeros(len(coords))
    # Negate values? Original code did: -msh.value
    nodes = {
        i + 1: [tuple(p.tolist()), -v] for i, (p, v) in
            enumerate(zip(coords, vals))}
            
    elems = {
        i + 1: v + 1 for i, v in enumerate(msh.tria)}
    offset = len(elems)
    elems.update({
        offset + i + 1: v + 1 for i, v in enumerate(msh.quad)})

    return {'description': desc,
            'nodes': nodes,
            'elements': elems}


def grd_to_msh_t(grd: Dict) -> MeshData:
    # Returns MeshData now
    warnings.warn("Use grd_to_meshdata instead!", DeprecationWarning)
    return grd_to_meshdata(grd)


def grd_to_meshdata(grd: Dict) -> MeshData:
    id_to_index = {node_id: index for index, node_id
                   in enumerate(grd['nodes'].keys())}
    
    triangles = [list(map(lambda x: id_to_index[x], element)) for element
                 in grd['elements'].values() if len(element) == 3]
    quads = [list(map(lambda x: id_to_index[x], element)) for element
             in grd['elements'].values() if len(element) == 4]
             
    coords = [coord for coord, _ in grd['nodes'].values()]
    values = [value for _, value in grd['nodes'].values()]

    msh = MeshData(
        coords=coords,
        tria=triangles,
        quad=quads,
        values=values
    )
    
    crs = grd.get('crs')
    if crs is not None:
        msh.crs = CRS.from_user_input(crs)
    return msh


def msh_t_to_2dm(msh: MeshData):
    warnings.warn("Use meshdata_to_2dm instead!", DeprecationWarning)
    return meshdata_to_2dm(msh)

def meshdata_to_2dm(msh: MeshData):
    coords = msh.coords
    src_crs = msh.crs if hasattr(msh, 'crs') else None
    if src_crs is not None:
        epsg_4326 = CRS.from_epsg(4326)
        if not src_crs.equals(epsg_4326):
            transformer = Transformer.from_crs(
                src_crs, epsg_4326, always_xy=True)
            coords = np.vstack(
                transformer.transform(coords[:, 0], coords[:, 1])).T
    
    vals = msh.values.flatten()
    return {
            'ND': {i+1: (coord, vals[i] if not
                         np.isnan(vals[i]) else -99999)
                   for i, coord in enumerate(coords)},
            'E3T': {i+1: index+1 for i, index
                    in enumerate(msh.tria)},
            'E4Q': {i+1: index+1 for i, index
                    in enumerate(msh.quad)}
        }


def sms2dm_to_msh_t(_sms2dm) -> MeshData:
    warnings.warn("Use sms2dm_to_meshdata instead!", DeprecationWarning)
    return sms2dm_to_meshdata(_sms2dm)

def sms2dm_to_meshdata(_sms2dm) -> MeshData:
    id_to_index = {node_id: index for index, node_id
                   in enumerate(_sms2dm['ND'].keys())}
    
    tria = None
    if 'E3T' in _sms2dm:
        tria = [list(map(lambda x: id_to_index[x], element)) for element
                     in _sms2dm['E3T'].values()]
    
    quad = None
    if 'E4Q' in _sms2dm:
        quad = [list(map(lambda x: id_to_index[x], element)) for element
                 in _sms2dm['E4Q'].values()]
                 
    coords = [coord for coord, _ in _sms2dm['ND'].values()]
    values = [value for _, value in _sms2dm['ND'].values()]

    msh = MeshData(
        coords=coords,
        tria=tria,
        quad=quad,
        values=values
    )

    crs = _sms2dm.get('crs')
    if crs is not None:
        msh.crs = CRS.from_user_input(crs)
    return msh


def msh_t_to_utm(msh: MeshData):
    warnings.warn("Use project_to_utm instead!", DeprecationWarning)
    return project_to_utm(msh)
    
def project_to_utm(msh: MeshData):
    utm_crs = estimate_mesh_utm(msh)
    if utm_crs is None:
        return

    transformer = Transformer.from_crs(
        msh.crs, utm_crs, always_xy=True)

    # pylint: disable=E0633
    msh.coords[:, 0], msh.coords[:, 1] = transformer.transform(
            msh.coords[:, 0], msh.coords[:, 1])
    msh.crs = utm_crs


def estimate_bounds_utm(bounds, crs="EPSG:4326"):
    in_crs = CRS.from_user_input(crs)
    if in_crs.is_geographic:
        x0, y0, x1, y1 = bounds
        _, _, number, letter = utm.from_latlon(
                (y0 + y1)/2, (x0 + x1)/2)
        # PyProj 3.2.1 throws error if letter is provided
        utm_crs = CRS(
                proj='utm',
                zone=f'{number}',
                south=(y0 + y1)/2 < 0,
                ellps={
                    'GRS 1980': 'GRS80',
                    'WGS 84': 'WGS84'
                    }[in_crs.ellipsoid.name]
            )
        return utm_crs
    return None


def estimate_mesh_utm(msh: MeshData):
    if hasattr(msh, 'crs'):
        coords = msh.coords
        x0, y0, x1, y1 = (
            np.min(coords[:, 0]), np.min(coords[:, 1]),
            np.max(coords[:, 0]), np.max(coords[:, 1]))
        utm_crs = estimate_bounds_utm((x0, y0, x1, y1), msh.crs)
        return utm_crs

    return None

def get_polygon_channels(polygon, width, simplify=None, join_style=3):

    # Operations are done without any CRS info consideration

    polys_gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(polygon))

    if isinstance(simplify, (int, float)):
        polys_gdf = gpd.GeoDataFrame(
                geometry=polys_gdf.simplify(
                    tolerance=simplify,
                    preserve_topology=False))

    buffer_size = width/2
    buffered_gdf = gpd.GeoDataFrame(
            geometry=polys_gdf.buffer(-buffer_size).buffer(
                    buffer_size,
                    join_style=join_style))

    buffered_gdf = buffered_gdf[~buffered_gdf.is_empty]
    if len(buffered_gdf) == 0:
        # All is channel!
        return polygon
    channels_gdf = gpd.overlay(
            polys_gdf, buffered_gdf, how='difference')

    # Use square - 1/4 circle as cleanup criteria
    channels_gdf = gpd.GeoDataFrame(
           geometry=gpd.GeoSeries(
               [p for i in channels_gdf.geometry
                   for p in i.geoms
                   if p.area > width**2 * (1-np.pi/4)]))


    ret_val = channels_gdf.union_all()
    if isinstance(ret_val, GeometryCollection):
        return None

    if isinstance(ret_val, Polygon):
        ret_val = MultiPolygon([ret_val])

    return ret_val


def merge_msh_t(
        *mesh_list,
        out_crs="EPSG:4326",
        drop_by_bbox=True,
        can_overlap=True,
        check_cross_edges=False):

    warnings.warn("Use merge_meshdata instead!", DeprecationWarning)
    return merge_meshdata(
        *mesh_list,
        out_crs=out_crs,
        drop_by_bbox=drop_by_bbox,
        can_overlap=can_overlap,
        check_cross_edges=check_cross_edges)

def merge_meshdata(
        *mesh_list,
        out_crs="EPSG:4326",
        drop_by_bbox=True,
        can_overlap=True,
        check_cross_edges=False):

    dst_crs = CRS.from_user_input(out_crs)

    coord = []
    elems = {k: [] for k in ELEM_2D_TYPES}
    value = []
    offset = 0

    mesh_shape_list = []
    # Last has the highest priority
    for mesh in mesh_list[::-1]:
        if not dst_crs.equals(mesh.crs):
            # To avoid modifying inputs
            mesh = deepcopy(mesh)
            reproject(mesh, dst_crs)

        if drop_by_bbox:
            x = mesh.coords[:, 0]
            y = mesh.coords[:, 1]
            mesh_shape = box(np.min(x), np.min(y), np.max(x), np.max(y))
        else:
            mesh_shape = get_mesh_polygons(mesh)

        for ishp in mesh_shape_list:
            # NOTE: fit_inside = True w/ inverse = True results
            # in overlap when clipping low-priority mesh
            mesh = clip_mesh_by_shape(
                mesh, ishp,
                use_box_only=drop_by_bbox,
                fit_inside=can_overlap,
                inverse=True,
                check_cross_edges=check_cross_edges)

        mesh_shape_list.append(mesh_shape)

        for k in ELEM_2D_TYPES:
            cnn = getattr(mesh, k)
            if cnn.size > 0:
                elems[k].append(cnn + offset)
        
        coord.append(mesh.coords)
        if mesh.values is not None:
            value.append(mesh.values)
        offset += coord[-1].shape[0]

    # Construct composite MeshData
    # Helper to stack if list is not empty, else return empty
    def safe_stack(lst, dim=2):
        if not lst: return None
        return np.vstack(lst)
    
    composite_mesh = MeshData(
        coords=safe_stack(coord),
        tria=safe_stack(elems['tria']),
        quad=safe_stack(elems['quad']),
        values=safe_stack(value)
    )

    composite_mesh.crs = dst_crs

    return composite_mesh


def add_pool_args(func):
    def wrapper(*args, nprocs=None, pool=None, **kwargs):
        if pool is not None:
            rv = func(*args, **kwargs, pool=pool)
        else:
            # Check nprocs
            nprocs = -1 if nprocs is None else nprocs
            nprocs = cpu_count() if nprocs == -1 else nprocs
            with Pool(processes=nprocs) as new_pool:
                rv = func(*args, **kwargs, pool=new_pool)
            new_pool.join()
        return rv
    return wrapper

def drop_extra_vertex_from_line(lstr: LineString) -> LineString:

    coords = np.array(lstr.coords)

    vecs = coords[1:] - coords[:-1]
    def isnt_zero(i): return np.logical_not(np.isclose(i, 0))
    vec_sizes = np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)
    uvecs = vecs / vec_sizes.reshape(-1, 1)
    uvecs = uvecs[isnt_zero(vec_sizes)]
    nondup_coords = np.vstack([coords[:-1][isnt_zero(vec_sizes)], coords[-1]])

    vec_diffs = np.diff(uvecs, axis=0)
    diff_sizes = np.sqrt(vec_diffs[:, 0] ** 2 + vec_diffs[:, 1] ** 2)

    new_coords = np.vstack(
        [nondup_coords[0],
         nondup_coords[1:-1][isnt_zero(diff_sizes)],
         nondup_coords[-1]
         ])

    return LineString(new_coords)

def drop_extra_vertex_from_polygon(
        mpoly: Union[Polygon, MultiPolygon]) -> MultiPolygon:
    if isinstance(mpoly, Polygon): mpoly = MultiPolygon([mpoly])
    poly_seam_list = []
    for poly in mpoly.geoms:
        extr = drop_extra_vertex_from_line(poly.exterior)
        inters = [
            drop_extra_vertex_from_line(lstr)
            for lstr in poly.interiors]
        poly_seam_list.append(Polygon(extr, inters))
    return MultiPolygon(poly_seam_list)


def remove_holes(
    poly: Union[Polygon, MultiPolygon]
) -> Union[Polygon, MultiPolygon]:
    '''Remove holes from the input polygon(s)

    Given input `Polygon` or `MultiPolygon`, remove all the geometric
    holes and return a new shape object.

    Parameters
    ----------
    poly : Polygon or MultiPolygon
        The input shape from which the holes are removed

    Returns
    -------
    Polygon or MultiPolygon
        The resulting (multi)polygon after removing the holes

    See Also
    --------
    remove_holes_by_relative_size :
        Remove all the whole smaller than given size from the input shape

    Notes
    -----
    For a `Polygon` with no holes, this function returns the original
    object. For `MultiPolygon` with no holes, the return value is a
    `unary_union` of all the underlying `Polygon`s.
    '''

    if isinstance(poly, MultiPolygon):
        return unary_union([remove_holes(p) for p in poly.geoms])
    if not isinstance(poly, Polygon):
        raise ValueError(
            "The input must be either a `Polygon` or `MultiPolygon`:"
            + f"\tType: {type(poly)}"
        )

    if poly.interiors:
        return Polygon(poly.exterior)
    return poly


def remove_holes_by_relative_size(
    poly: Union[Polygon, MultiPolygon],
    rel_size:float = 0.1
) -> Union[Polygon, MultiPolygon]:
    '''Remove holes from the input polygon(s)

    Given input `Polygon` or `MultiPolygon`, remove all the geometric
    holes that are smaller than the input relative size `rel_size`
    and return a new shape object.

    Parameters
    ----------
    poly : Polygon or MultiPolygon
        The input shape from which the holes are removed
    rel_size : float, default=0.1
        Maximum ratio of a hole area to the area of polygon

    Returns
    -------
    Polygon or MultiPolygon
        The resulting (multi)polygon after removing the holes

    See Also
    --------
    remove_holes :
        Remove all the whole from the input shape

    Notes
    -----
    For a `Polygon` with no holes, this function returns the original
    object. For `MultiPolygon` with no holes, the return value is a
    `unary_union` of all the underlying `Polygon`s.

    If `rel_size=1` is specified the result is the same as
    `remove_holes` function, except for the additional cost of
    calculating the areas.
    '''

    if isinstance(poly, MultiPolygon):
        return unary_union([
            remove_holes_by_relative_size(p, rel_size) for p in poly.geoms])

    if not isinstance(poly, Polygon):
        raise ValueError(
            "The input must be either a `Polygon` or `MultiPolygon`:"
            + f"\tType: {type(poly)}"
        )

    if poly.interiors:
        ref_area = poly.area
        new_interiors = [
                intr for intr in poly.interiors
                if Polygon(intr).area / ref_area > rel_size]
        return Polygon(poly.exterior, new_interiors)
    return poly


def get_element_size_courant(
    characteristic_velocity_magnitude: Union[float, npt.NDArray[float]],
    timestep: float,
    target_courant: float = 1
) -> Union[float, npt.NDArray[float]]:

    '''Calculate the element size based on the specified Courant number.

    Calculate the element size based on the specified Courant number
    and input value for timestep and characteristic velocity

    Parameters
    ----------
    target_courant : float
        The Courant number to be achieved by the calculated element size
    characteristic_velocity_magnitude : float or array of floats
        Magnitude of total velocity used for element size calculation
        (:math:`\\frac{m}{sec}`)
    timestep : float
        Timestep size (:math:`seconds`) to

    Returns
    -------
    float or array of floats
        The calculated element size(s) to achieve the given Courant #
    '''

    return characteristic_velocity_magnitude * timestep / target_courant


def can_velocity_be_approximated_by_linear_wave_theory(
    depth: Union[float, npt.NDArray[float]],
    wave_amplitude: float = 2
) -> Union[bool, npt.NDArray[bool]]:
    '''Checks whether the particle velocity can be appoximated.

    Based on the input depth, checks whether or not the velocity can
    be approximated from the linear wave theory

    Parameters
    ----------
    depth : float or array of float
        Depth of the point for which the approximation validation is checked
    wave_amplitude : float, default=2
        Free surface elevation (:math:`meters`)
        from the reference (i.e. wave height)

    Returns
    -------
    bool or array of bool
        Whether or not the value at given input depth can be approximated

    Notes
    -----
    Linear wave theory approximation breaks down when :math:`\\nu \\sim h`
    overland. So this method just returns whether depth is below or over
    depth of `wave_amplitude` magnitude.

    References
    ----------
    Based on OceanMesh2D approximation method
    https://doi.org/10.13140/RG.2.2.21840.61446/2.
    '''

    return depth <= -abs(wave_amplitude)


def estimate_particle_velocity_from_depth(
    depth: Union[float, npt.NDArray[float]],
    wave_amplitude: float = 2
) -> Union[float, npt.NDArray[float]]:

    '''Approximate particle velocity magnitude based on depth of water

    Estimate the value of particle velocity magnitude based on the
    linear wave theory as :math:`\\left|u\\right| = \\nu \\sqrt{\\frac{g}{h}}`
    for ocean and :math:`\\left|u\\right| = \\sqrt{gH}` for overland region
    where :math:`\\nu \\sim h` so instead of linear wave theory we take
    :math:`H \\approx \\nu`

    Parameters
    ----------
    depth : float or array of floats
        The depth of still water (e.g. from DEM)
    wave_amplitude : float
        Wave amplitude for approximation as defined by linear wave theory

    Returns
    -------
    float or array of floats
        Estimated water particle velocity

    References
    ----------
    Based on OceanMesh2D approximation method 
    https://doi.org/10.13140/RG.2.2.21840.61446/2.
    '''

    if not isinstance(depth, np.ndarray):
        depth = np.array(depth)
    dep_shape = depth.shape
    depth = depth.ravel()

    depth_mask = can_velocity_be_approximated_by_linear_wave_theory(
        depth, wave_amplitude)

    velocity = np.zeros_like(depth)
    velocity[depth_mask] = wave_amplitude*np.sqrt(constants.g/np.abs(depth[depth_mask]))
    velocity[~depth_mask] = np.sqrt(constants.g * wave_amplitude)

    return velocity.reshape(dep_shape)


def approximate_courant_number_for_depth(
    depth: Union[float, npt.NDArray[float]],
    timestep: float,
    element_size: Union[float, npt.NDArray[float]],
    wave_amplitude: float = 2
) -> Union[float, npt.NDArray[float]]:
    '''Approximate the Courant number for given depths

    Approximate the value of Courant number for the input depth,
    timestep and element size. The velocity is approximated based on
    the input depth.

    Parameters
    ----------
    depth : float or array of floats
    timestep : float
        Timestep size (:math:`seconds`) to
    element_size : float or array of floats
        Element size(s) to use for Courant number calculation. Must
        be scalar otherwise match the dimension of depth
    wave_amplitude : float, default=2
        Free surface elevation (:math:`meters`) from the
        reference (i.e. wave height)

    Returns
    -------
    float or array of floats
        The approximated Courant number for each input depth
    '''

    if not isinstance(depth, np.ndarray):
        depth = np.array(depth)
    if not isinstance(element_size, np.ndarray):
        element_size = np.array(element_size)

    if np.any(element_size == 0):
        raise ValueError("Element size(s) for Courant number approximation include zero!")

    if depth.shape != element_size.shape:
        raise ValueError("Shapes of depths and sizes arrays don't match!")

    depth_mask = can_velocity_be_approximated_by_linear_wave_theory(
        depth, wave_amplitude)

    particle_velocity = estimate_particle_velocity_from_depth(depth,
                                                              wave_amplitude)
    characteristic_velocity_magnitude = (
        particle_velocity + np.sqrt(constants.g * np.abs(depth))
    )
    # For overland where h < nu the characteristic velocity is 2 * sqrt(g*h)
    characteristic_velocity_magnitude[~depth_mask] = 2 * particle_velocity[~depth_mask]
    return characteristic_velocity_magnitude * timestep / element_size


def create_rectangle_mesh(
    nx,
    ny,
    holes,
    x_extent=None,
    y_extent=None,
    quads=None,
):
    # pylint: disable=W1401
    """
    Note:
        x = x-index
        y = y-index

        holes or quads count starting at 1 from bottom corner square

        node-index(node-id)

              25(26)             29(30)
          5     *---*---*---*---*
                | \ | \ | \ | \ |
          4     *---*---*---*---*
                | \ | \ | \ | \ |
          3     *---*---*---*---*
                | \ |   | \ | \ |
          2     *---*---*---*---*
                | \ | \ | # | \ |
          1     *---*---*---*---*
                | \ | \ | \ | \ |
          0     *---*---*---*---*
              0(1)               4(5)

                0   1   2   3   4
    """

    if x_extent is None:
        x_range = range(nx)
    else:
        x_range = np.linspace(x_extent[0], x_extent[1], nx)

    if y_extent is None:
        y_range = range(ny)
    else:
        y_range = np.linspace(y_extent[0], y_extent[1], ny)

    if quads is None:
        quads = []

    X, Y = np.meshgrid(x_range, y_range)
    verts = np.array(list(zip(X.ravel(), Y.ravel())))
    tria = []
    quad = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            is_quad = (i + 1) + ((nx-1) * j) in quads
            is_hole = (i + 1) + ((nx-1) * j) in holes
            if is_hole:
                continue
            if is_quad:
                quad.append([
                    j * nx + i,
                    j * nx + (i + 1),
                    (j + 1) * nx + (i + 1),
                    (j + 1) * nx + i
                ])
            else: # is tria
                tria.append([j * nx + i, j * nx + (i + 1), (j + 1) * nx + i])
                tria.append([j * nx + (i + 1),
                              (j + 1) * nx + (i + 1),
                              (j + 1) * nx + i])



    # NOTE: Everywhere is above 0 (auto: land) unless modified later
    vals = np.ones((len(verts), 1)) * 10

    mesh = MeshData(
        coords=verts,
        tria=tria,
        quad=quad,
        values=vals
    )
    
    # Drop unused verts (e.g. 4 connected holes)
    cleanup_isolates(mesh) # MeshData might handle this logic if needed
    return mesh


def raster_from_numpy(
    filename,
    data,
    mgrid, # Needs to have ij indexing!
    crs=CRS.from_epsg(4326)
) -> None:
    x = mgrid[0][:, 0]
    y = mgrid[1][0, :]
    res_x = (x[-1] - x[0]) / data.shape[1]
    res_y = (y[-1] - y[0]) / data.shape[0]
    # TODO: Mistake in transformation if x and y extent are different?
    transform = rio.transform.Affine.translation(
        x[0], y[0]
    ) * rio.transform.Affine.scale(res_x, res_y)
    if not isinstance(crs, CRS):
        crs = CRS.from_user_input(crs)

    nbands = 1
    if data.ndim == 3:
        nbands = data.shape[2]
    elif data.ndim != 2:
        raise ValueError("Invalid data dimensions!")

    with rio.open(
        filename,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=nbands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        if isinstance(data, np.ma.MaskedArray):
            dst.nodata = data.fill_value

        data = data.reshape(data.shape[0], data.shape[1], -1)
        for i in range(nbands):
            dst.write(data.take(i, axis=2), i + 1)


def msht_from_numpy(
    coordinates,
    *, # Get everything else as keyword args
    edges=None,
    triangles=None,
    quadrilaterals=None,
    values=None,
    crs=CRS.from_epsg(4326)
) -> MeshData:
    # Renamed function intent, but keeping signature for compat
    warnings.warn("Use MeshData class instead!", DeprecationWarning)
    mesh = MeshData(
        coords=coordinates,
        tria=triangles,
        quad=quadrilaterals,
        values=values,
        crs=crs
    )

    return mesh


def clip_elements_by_index(
    msht: MeshData,
    tria=None,
    quad=None,
    inverse: bool = False) -> MeshData:
    '''
    adapted from:
https://github.com/sorooshmani-noaa/river-in-mesh/tree/main/river_in_mesh/utils
    
    parameters
    ----------
    msht : MeshData
        mesh to beclipped

    tria or quad: array with the element ids to be removed
    inverse = default:false

    returns
    -------
    MeshData
        mesh without skewed elements
    
    notes
    -----
    '''
    new_msht = deepcopy(msht)
    rm_dict = {'tria': tria, 'quad': quad}
    for elm_type, idx in rm_dict.items():
        if idx is None: continue
        elems = getattr(new_msht, elm_type)
        mask = np.ones(elems.shape[0], dtype=bool)
        mask[idx] = False
        if inverse is False:
            setattr(new_msht, elm_type, elems[mask])
        else:
            setattr(new_msht, elm_type, elems[~mask])
    cleanup_isolates(new_msht)
    return new_msht
