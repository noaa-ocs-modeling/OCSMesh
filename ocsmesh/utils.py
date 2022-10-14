from collections import defaultdict
from itertools import permutations
from typing import Union, Dict, Sequence, Tuple
from functools import reduce
from multiprocessing import cpu_count, Pool
from copy import deepcopy

import jigsawpy
from jigsawpy import jigsaw_msh_t  # type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
from matplotlib.tri import Triangulation  # type: ignore[import]
import numpy as np  # type: ignore[import]
import numpy.typing as npt
from pyproj import CRS, Transformer  # type: ignore[import]
from scipy.interpolate import (  # type: ignore[import]
    RectBivariateSpline, griddata)
from scipy import sparse, constants
from shapely.geometry import ( # type: ignore[import]
        Polygon, MultiPolygon,
        box, GeometryCollection, Point, MultiPoint,
        LineString, LinearRing)
from shapely.ops import polygonize, linemerge, unary_union
import geopandas as gpd
import utm


ELEM_2D_TYPES = ['tria3', 'quad4', 'hexa8']

def must_be_euclidean_mesh(func):
    def decorator(mesh, *args, **kwargs):
        if mesh.mshID.lower() != 'euclidean-mesh':
            msg = f"Not implemented for mshID={mesh.mshID}"
            raise NotImplementedError(msg)
        return func(mesh, *args, **kwargs)
    return decorator


def mesh_to_tri(mesh):
    """
    mesh is a jigsawpy.jigsaw_msh_t() instance.
    """
    return Triangulation(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'])


def cleanup_isolates(mesh):

    # For triangle only (TODO: add support for other types)
    node_indexes = np.arange(mesh.vert2['coord'].shape[0])
    used_indexes = np.unique(mesh.tria3['index'])
    vert2_idxs = np.where(
        np.isin(node_indexes, used_indexes, assume_unique=True))[0]

    # Since tria simplex refers to node index which always starts from
    # 0 after removing isolate nodes we can use the map approach
    tria3 = mesh.tria3['index'].flatten()
    renum = {old: new for new, old in enumerate(np.unique(tria3))}
    tria3 = np.array([renum[i] for i in tria3])
    tria3 = tria3.reshape(mesh.tria3['index'].shape)

    mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)
    if len(mesh.value) > 0:
        mesh.value = mesh.value.take(vert2_idxs, axis=0)
    mesh.tria3 = np.asarray(
        [(tuple(indices), mesh.tria3['IDtag'][i])
         for i, indices in enumerate(tria3)],
        dtype=jigsaw_msh_t.TRIA3_t)


def put_edge2(mesh):
    tri = Triangulation(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'])
    mesh.edge2 = np.array(
        [(edge, 0) for edge in tri.edges], dtype=jigsaw_msh_t.EDGE2_t)


def geom_to_multipolygon(mesh):
    vertices = mesh.vert2['coord']
    idx_ring_coll = index_ring_collection(mesh)
    polygon_collection = []
    for polygon in idx_ring_coll.values():
        exterior = vertices[polygon['exterior'][:, 0], :]
        interiors = []
        for interior in polygon['interiors']:
            interiors.append(vertices[interior[:, 0], :])
        polygon_collection.append(Polygon(exterior, interiors))
    return MultiPolygon(polygon_collection)


def get_boundary_segments(mesh):

    coords = mesh.vert2['coord']
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


    # TODO: Copy mesh?
    target_mesh = mesh
    result_polys = []

    # 2-pass find, first find using polygons that intersect non-boundary
    # vertices, then from the rest of the mesh find polygons that
    # intersect any vertex
    for find_pass in range(2):


        coords = target_mesh.vert2['coord']

        if len(coords) == 0:
            continue

        boundary_edges = get_boundary_edges(target_mesh)

        lines = get_boundary_segments(target_mesh)

        poly_gen = polygonize(lines)
        polys = list(poly_gen)
        polys = sorted(polys, key=lambda p: p.area, reverse=True)


        bndry_verts = np.unique(boundary_edges)

        if find_pass == 0:
            non_bndry_verts = np.setdiff1d(
                    np.arange(len(coords)), bndry_verts)
            pnts = MultiPoint(coords[non_bndry_verts])
        else:
            pnts = MultiPoint(coords[bndry_verts])


        # NOTE: This logic requires polygons to be sorted by area
        pass_valid_polys = []
        while len(pnts.geoms) > 0:


            idx = np.random.randint(len(pnts.geoms))
            pnt = pnts.geoms[idx]

            polys_gdf = gpd.GeoDataFrame(
                {'geometry': polys, 'list_index': range(len(polys))})


            res_gdf = polys_gdf[polys_gdf.intersects(pnt)]
            if len(res_gdf) == 0:
                # How is this possible?!
                pnts = MultiPoint([*pnts.geoms[:idx], *pnts.geoms[idx + 1:]])
                if pnts.is_empty:
                    break

                continue

            poly = res_gdf.geometry.iloc[0]
            polys.pop(res_gdf.iloc[0].list_index)



            pass_valid_polys.append(poly)
            pnts = pnts.difference(poly)
            if pnts.is_empty:
                break
            if isinstance(pnts, Point):
                pnts = MultiPoint([pnts])


        result_polys.extend(pass_valid_polys)
        target_mesh = clip_mesh_by_shape(
            target_mesh,
            shape=MultiPolygon(pass_valid_polys),
            inverse=True, fit_inside=True)



    return MultiPolygon(result_polys)


def repartition_features(linestring, max_verts):
    features = []
    if len(linestring.coords) > max_verts:
        new_feat = []
        for segment in list(map(LineString, zip(
                linestring.coords[:-1],
                linestring.coords[1:]))):
            new_feat.append(segment)
            if len(new_feat) == max_verts - 1:
                features.append(linemerge(new_feat))
                new_feat = []
        if len(new_feat) != 0:
            features.append(linemerge(new_feat))
    else:
        features.append(linestring)
    return features


def transform_linestring(
    linestring: LineString,
    target_size: float,
):
    distances = [0.]
    while distances[-1] + target_size < linestring.length:
        distances.append(distances[-1] + target_size)
    distances.append(linestring.length)
    linestring = LineString([
        linestring.interpolate(distance)
        for distance in distances
        ])
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


def put_id_tags(mesh):
    # start enumerating on 1 to avoid issues with indexing on fortran models
    mesh.vert2 = np.array(
        [(coord, id+1) for id, coord in enumerate(mesh.vert2['coord'])],
        dtype=jigsaw_msh_t.VERT2_t
        )
    mesh.tria3 = np.array(
        [(index, id+1) for id, index in enumerate(mesh.tria3['index'])],
        dtype=jigsaw_msh_t.TRIA3_t
        )
    mesh.quad4 = np.array(
        [(index, id+1) for id, index in enumerate(mesh.quad4['index'])],
        dtype=jigsaw_msh_t.QUAD4_t
        )
    mesh.hexa8 = np.array(
        [(index, id+1) for id, index in enumerate(mesh.hexa8['index'])],
        dtype=jigsaw_msh_t.HEXA8_t
        )


def _get_sieve_mask(mesh, polygons, sieve_area):

    # NOTE: Some polygons are ghost polygons (interior)
    areas = [p.area for p in polygons.geoms]
    if sieve_area is None:
        remove = np.where(areas < np.max(areas))[0].tolist()
    else:
        remove = []
        for idx, patch_area in enumerate(areas):
            if patch_area <= sieve_area:
                remove.append(idx)

    # if the path surrounds the node, these need to be removed.
    vert2_mask = np.full((mesh.vert2['coord'].shape[0],), False)
    for idx in remove:
        path = Path(polygons.geoms[idx].exterior.coords, closed=True)
        vert2_mask = vert2_mask | path.contains_points(mesh.vert2['coord'])

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

    # Mask out elements containing the unwanted nodes.
    tria3_mask = np.any(vert2_mask[mesh.tria3['index']], axis=1)

    # Tria and node removal and renumbering indexes ...
    tria3_id_tag = mesh.tria3['IDtag'].take(np.where(~tria3_mask)[0])
    tria3_index = mesh.tria3['index'][~tria3_mask, :].flatten()
    used_indexes = np.unique(tria3_index)
    node_indexes = np.arange(mesh.vert2['coord'].shape[0])
    renum = {old: new for new, old in enumerate(np.unique(tria3_index))}
    tria3_index = np.array([renum[i] for i in tria3_index])
    tria3_index = tria3_index.reshape((tria3_id_tag.shape[0], 3))
    vert2_idxs = np.where(np.isin(node_indexes, used_indexes))[0]

    # update vert2
    mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)

    # update value
    if len(mesh.value) > 0:
        mesh.value = mesh.value.take(vert2_idxs, axis=0)

    # update tria3
    mesh.tria3 = np.array(
        [(tuple(indices), tria3_id_tag[i])
         for i, indices in enumerate(tria3_index)],
        dtype=jigsaw_msh_t.TRIA3_t)


def finalize_mesh(mesh, sieve_area=None):

    cleanup_isolates(mesh)

    while True:
        no_op = True

        pinched_nodes = get_pinched_nodes(mesh)
        if len(pinched_nodes):
            no_op = False
            # TODO drop fewer elements for pinch
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
    put_id_tags(mesh)


def remesh_small_elements(opts, geom, mesh, hfun):

    """
    This function uses all the inputs for a given jigsaw meshing
    process and based on that finds and fixes tiny elements that
    might occur during initial meshing by iteratively remeshing
    """

    # TODO: Implement for quad, etc.


    hmin = np.min(hfun.value)
    equilat_area = np.sqrt(3)/4 * hmin**2
    # List of arbitrary coef of equilateral triangle area for a givven
    # minimum mesh size to come up with a decent cut off.
    coeffs = [0.5, 0.2, 0.1, 0.05]

    fixed_mesh = mesh
    for coef in coeffs:
        tria_areas = calculate_tria_areas(fixed_mesh)
        tiny_sz = coef * equilat_area
        tiny_verts = np.unique(fixed_mesh.tria3['index'][tria_areas < tiny_sz, :].ravel())
        if len(tiny_verts) == 0:
            break
        mesh_clip = clip_mesh_by_vertex(fixed_mesh, tiny_verts, inverse=True)

        fixed_mesh = jigsawpy.jigsaw_msh_t()
        fixed_mesh.mshID = 'euclidean-mesh'
        fixed_mesh.ndims = +2
        if hasattr(mesh, 'crs'):
            fixed_mesh.crs = mesh.crs

        jigsawpy.lib.jigsaw(
            opts, geom, fixed_mesh, init=mesh_clip, hfun=hfun)


    return fixed_mesh


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
    vert2_mask = np.full((mesh.vert2['coord'].shape[0],), False)
    for idx in remove:
        path = Path(multipolygon[idx].exterior.coords, closed=True)
        vert2_mask = vert2_mask | path.contains_points(mesh.vert2['coord'])

    # select any connected nodes; these ones are missed by
    # path.contains_point() because they are at the path edges.
    _idxs = np.where(vert2_mask)[0]
    conn_verts = get_surrounding_elem_verts(mesh, _idxs)
    vert2_mask[conn_verts] = True

    # Also, there might be some dangling triangles without neighbors,
    # which are also missed by path.contains_point()
    lone_elem_verts = get_lone_element_verts(mesh)
    vert2_mask[lone_elem_verts] = True


    # Mask out elements containing the unwanted nodes.
    tria3_mask = np.any(vert2_mask[mesh.tria3['index']], axis=1)

    # Renumber indexes ...
    # isolated node removal does not require elimination of triangles from
    # the table, therefore the length of the indexes is constant.
    # We must simply renumber the tria3 indexes to match the new node indexes.
    # Essentially subtract one, but going from the bottom of the index table
    # to the top.
    used_indexes = np.unique(mesh.tria3['index'])
    node_indexes = np.arange(mesh.vert2['coord'].shape[0])
    tria3_idxs = np.where(~np.isin(node_indexes, used_indexes))[0]
    tria3_id_tag = mesh.tria3['IDtag'].take(np.where(~tria3_mask)[0])
    tria3_index = mesh.tria3['index'][~tria3_mask, :].flatten()
    for idx in reversed(tria3_idxs):
        tria3_index[np.where(tria3_index >= idx)] -= 1
    tria3_index = tria3_index.reshape((tria3_id_tag.shape[0], 3))
    vert2_idxs = np.where(np.isin(node_indexes, used_indexes))[0]

    # update vert2
    mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)

    # update value
    if len(mesh.value) > 0:
        mesh.value = mesh.value.take(vert2_idxs, axis=0)

    # update tria3
    mesh.tria3 = np.array(
        [(tuple(indices), tria3_id_tag[i])
         for i, indices in enumerate(tria3_index)],
        dtype=jigsaw_msh_t.TRIA3_t)


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
    vertices = mesh.vert2['coord']
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
    else:
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
    if mesh.mshID == 'euclidean-mesh':
        def append(geom):
            for simplex in geom['index']:
                for i, j in permutations(simplex, 2):
                    vert_list[i].add(j)
        vert_list = defaultdict(set)
        append(mesh.tria3)
        append(mesh.quad4)
        append(mesh.hexa8)
        return vert_list

    msg = f"Not implemented for mshID={mesh.mshID}"
    raise NotImplementedError(msg)

def get_surrounding_elem_verts(mesh, in_vert):

    '''
    Find vertices of elements connected to input vertices
    '''

    tria = mesh.tria3['index']
    quad = mesh.quad4['index']
    hexa = mesh.hexa8['index']

    # NOTE: np.any is used so that vertices that are not in in_verts
    # triangles but are part of the triangles that include in_verts
    # are considered too
    mark_tria = np.any(
            (np.isin(tria.ravel(), in_vert).reshape(
                tria.shape)), 1)
    mark_quad = np.any(
            (np.isin(quad.ravel(), in_vert).reshape(
                quad.shape)), 1)
    mark_hexa = np.any(
            (np.isin(hexa.ravel(), in_vert).reshape(
                hexa.shape)), 1)

    conn_verts = np.unique(np.concatenate(
        (tria[mark_tria, :].ravel(),
         quad[mark_quad, :].ravel(),
         hexa[mark_hexa, :].ravel())))

    return conn_verts

def get_lone_element_verts(mesh):

    '''
    Also, there might be some dangling triangles without neighbors,
    which are also missed by path.contains_point()
    '''

    tria = mesh.tria3['index']
    quad = mesh.quad4['index']
    hexa = mesh.hexa8['index']

    unq_verts, counts = np.unique(
        np.concatenate((tria.ravel(), quad.ravel(), hexa.ravel())),
        return_counts=True)
    once_verts = unq_verts[counts == 1]

    # NOTE: np.all so that lone elements are found vs elements that
    # have nodes that are used only once
    mark_tria = np.all(
            (np.isin(tria.ravel(), once_verts).reshape(
                tria.shape)), 1)
    mark_quad = np.all(
            (np.isin(quad.ravel(), once_verts).reshape(
                quad.shape)), 1)
    mark_hexa = np.all(
            (np.isin(hexa.ravel(), once_verts).reshape(
                hexa.shape)), 1)

    lone_elem_verts = np.unique(np.concatenate(
        (tria[mark_tria, :].ravel(),
         quad[mark_quad, :].ravel(),
         hexa[mark_hexa, :].ravel())))

    return lone_elem_verts



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
        mesh: jigsaw_msh_t,
        shape: Union[box, Polygon, MultiPolygon],
        from_box: bool = False,
        num_adjacent: int = 0
        ) -> Sequence[int]:

    if from_box:
        crd = mesh.vert2['coord']

        xmin, ymin, xmax, ymax = shape.bounds

        in_box_idx_1 = np.arange(len(crd))[crd[:, 0] > xmin]
        in_box_idx_2 = np.arange(len(crd))[crd[:, 0] < xmax]
        in_box_idx_3 = np.arange(len(crd))[crd[:, 1] > ymin]
        in_box_idx_4 = np.arange(len(crd))[crd[:, 1] < ymax]
        in_box_idx = reduce(
            np.intersect1d, (in_box_idx_1, in_box_idx_2,
                             in_box_idx_3, in_box_idx_4))
        return in_box_idx


    pt_series = gpd.GeoSeries(gpd.points_from_xy(
        mesh.vert2['coord'][:,0], mesh.vert2['coord'][:,1]))
    shp_series = gpd.GeoSeries(shape)

    in_shp_idx = pt_series.sindex.query_bulk(
            shp_series, predicate="intersects")

    in_shp_idx = select_adjacent(mesh, in_shp_idx, num_layers=num_adjacent)

    return in_shp_idx

def select_adjacent(mesh, in_indices, num_layers):

    selected_indices = in_indices.copy()

    if mesh.mshID == 'euclidean-mesh' and mesh.ndims == 2:

        for i in range(num_layers - 1):

            coord = mesh.vert2['coord']

            # TODO: What about edge2
            mesh_types = {
                'tria3': 'TRIA3_t',
                'quad4': 'QUAD4_t',
                'hexa8': 'HEXA8_t'
            }
            elm_dict = {
                key: getattr(mesh, key)['index'] for key in mesh_types}

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


    msg = (f"Not implemented for"
           f" mshID={mesh.mshID} and dim={mesh.ndims}")
    raise NotImplementedError(msg)


@must_be_euclidean_mesh
def get_cross_edges(
        mesh: jigsaw_msh_t,
        shape: Union[box, Polygon, MultiPolygon],
        ) -> Sequence[Tuple[int, int]]:

    '''
    Return the list of edges crossing the input shape exterior
    '''

    coords = mesh.vert2['coord']

    coord_dict = {}
    for i, coo in enumerate(coords):
        coord_dict[tuple(coo)] = i

    gdf_shape = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(shape))
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
        mesh: jigsaw_msh_t,
        shape: Union[box, Polygon, MultiPolygon],
        use_box_only: bool = False,
        fit_inside: bool = True,
        inverse: bool = False,
        in_place: bool = False,
        check_cross_edges: bool = False,
        adjacent_layers: int = 0
        ) -> jigsaw_msh_t:


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
                mesh = remove_mesh_by_edge(
                        mesh, x_edge_idx, in_place)
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
        mesh: jigsaw_msh_t,
        edges: Sequence[Tuple[int, int]],
        in_place: bool = False
        ) -> jigsaw_msh_t:

    mesh_out = mesh
    if not in_place:
        mesh_out = deepcopy(mesh)

    # NOTE: This method selects more elements than needed as it
    # uses only existance of more than two of the vertices attached
    # to the input edges in the element as criteria.
    edge_verts = np.unique(edges)

    for etype in ELEM_2D_TYPES:
        elems = getattr(mesh, etype)['index']
        # If a given element contains to vertices from
        # a crossing edge, it is selected
        test = np.sum(np.isin(elems, edge_verts), axis=1)
        elems = elems[test < 2]
        setattr(mesh_out, etype, np.array(
                [(idx, 0) for idx in elems],
                dtype=getattr(
                    jigsawpy.jigsaw_msh_t, f'{etype.upper()}_t')))

    return mesh_out


def clip_mesh_by_vertex(
        mesh: jigsaw_msh_t,
        vert_in: Sequence[int],
        can_use_other_verts: bool = False,
        inverse: bool = False,
        in_place: bool = False
        ) -> jigsaw_msh_t:


    if mesh.mshID == 'euclidean-mesh' and mesh.ndims == 2:
        coord = mesh.vert2['coord']

        # TODO: What about edge2 if in_place?
        mesh_types = {
            'tria3': 'TRIA3_t',
            'quad4': 'QUAD4_t',
            'hexa8': 'HEXA8_t'
        }
        elm_dict = {
            key: getattr(mesh, key)['index'] for key in mesh_types}

        # Whether elements that include "in"-vertices can be created
        # using vertices other than "in"-vertices
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
        value = np.zeros(shape=(0, 0), dtype=jigsaw_msh_t.REALS_t)
        if len(mesh.value) == len(coord):
            value = mesh.value.take(
                    list(crd_old_to_new.keys()), axis=0).copy()


        mesh_out = mesh
        if not in_place:
            mesh_out = jigsaw_msh_t()
            mesh_out.mshID = mesh.mshID
            mesh_out.ndims = mesh.ndims
            if hasattr(mesh, "crs"):
                mesh_out.crs = deepcopy(mesh.crs)

        mesh_out.value = value

        mesh_out.vert2 = np.array(
            [(coo, 0) for coo in new_coord],
            dtype=jigsaw_msh_t.VERT2_t)

        for key, elem_type in mesh_types.items():
            setattr(
                mesh_out,
                key,
                np.array(
                    [(con, 0) for con in elem_final_dict[key]],
                    dtype=getattr(jigsaw_msh_t, elem_type)))

        return mesh_out

    msg = (f"Not implemented for"
           f" mshID={mesh.mshID} and dim={mesh.ndims}")
    raise NotImplementedError(msg)






@must_be_euclidean_mesh
def get_mesh_edges(mesh: jigsaw_msh_t, unique=True):

    # NOTE: For msh_t type vertex id and index are the same
    trias = mesh.tria3['index']
    quads = mesh.quad4['index']
    hexas = mesh.hexa8['index']

    # Get unique set of edges by rolling connectivity
    # and joining connectivities in 3rd dimension, then sorting
    # to get all edges with lower index first
    all_edges = np.empty(shape=(0, 2), dtype=trias.dtype)
    for elm_type in [trias, quads, hexas]:
        if elm_type.shape[0]:
            edges = np.sort(
                    np.stack(
                        (elm_type, np.roll(elm_type, shift=1, axis=1)),
                        axis=2),
                    axis=2)
            edges = edges.reshape(np.product(edges.shape[0:2]), 2)
            all_edges = np.vstack((all_edges, edges))

    if unique:
        all_edges = np.unique(all_edges, axis=0)

    return all_edges


@must_be_euclidean_mesh
def calculate_tria_areas(mesh):

    coord = mesh.vert2['coord']
    trias = mesh.tria3['index']

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
    # pylint: disable=W0632
    a_side, b_side, c_side = np.split(tria_sides, 3, axis=1)
    tria_areas = np.sqrt(
            perimeter*(perimeter-a_side)
            * (perimeter-b_side)*(perimeter-c_side)
            ).squeeze()
    return tria_areas

@must_be_euclidean_mesh
def calculate_edge_lengths(mesh, transformer=None):

    coord = mesh.vert2['coord']
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


@must_be_euclidean_mesh
def elements(mesh):
    elements_id = []
    elements_id.extend(list(mesh.tria3['IDtag']))
    elements_id.extend(list(mesh.quad4['IDtag']))
    elements_id.extend(list(mesh.hexa8['IDtag']))
    elements_id = range(1, len(elements_id)+1) \
        if len(set(elements_id)) != len(elements_id) else elements_id
    elems = []
    elems.extend(list(mesh.tria3['index']))
    elems.extend(list(mesh.quad4['index']))
    elems.extend(list(mesh.hexa8['index']))
    elems = {
        elements_id[i]: indexes for i, indexes in enumerate(elems)}
    return elems


@must_be_euclidean_mesh
def faces_around_vertex(mesh):
    _elements = elements(mesh)
    length = max(map(len, _elements.values()))
    y = np.array([xi+[-99999]*(length-len(xi)) for xi in _elements.values()])
    faces_around_vert = defaultdict(set)
    for i, coord in enumerate(mesh.vert2['index']):
        # TODO:
        pass
#        np.isin(i, axis=0)
#        faces_around_vert[i].add()

    faces_around_vert = defaultdict(set)


def get_boundary_edges(mesh):

    '''
    Find internal and external boundaries of mesh
    '''

    coord = mesh.vert2['coord']

    all_edges = get_mesh_edges(mesh, unique=False)

    # Simplexes (list of node indices)
    all_edges, e_cnt = np.unique(all_edges, axis=0, return_counts=True)
    shared_edges = all_edges[e_cnt == 2]
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
    mesh.tria3 = mesh.tria3.take(
        np.where(
            ~np.any(np.isin(mesh.tria3['index'], u[c > 1]), axis=1))[0],
        axis=0)


def interpolate(src: jigsaw_msh_t, dst: jigsaw_msh_t, **kwargs):
    if src.mshID == 'euclidean-grid' and dst.mshID == 'euclidean-mesh':
        interpolate_euclidean_grid_to_euclidean_mesh(src, dst, **kwargs)
    elif src.mshID == 'euclidean-mesh' and dst.mshID == 'euclidean-mesh':
        interpolate_euclidean_mesh_to_euclidean_mesh(src, dst, **kwargs)
    else:
        raise NotImplementedError(
            f'Not implemented type combination: source={src.mshID}, '
            f'dest={dst.mshID}')


def interpolate_euclidean_mesh_to_euclidean_mesh(
        src: jigsaw_msh_t,
        dst: jigsaw_msh_t,
        method='linear',
        fill_value=np.nan
):
    values = griddata(
        src.vert2['coord'],
        src.value.flatten(),
        dst.vert2['coord'],
        method=method,
        fill_value=fill_value
    )
    dst.value = np.array(
        values.reshape(len(values), 1), dtype=jigsaw_msh_t.REALS_t)


def interpolate_euclidean_grid_to_euclidean_mesh(
        src: jigsaw_msh_t,
        dst: jigsaw_msh_t,
        bbox=None,
        kx=3,
        ky=3,
        s=0
):
    values = RectBivariateSpline(
        src.xgrid,
        src.ygrid,
        src.value.T,
        bbox=bbox or [None, None, None, None],
        kx=kx,
        ky=ky,
        s=s
        ).ev(
        dst.vert2['coord'][:, 0],
        dst.vert2['coord'][:, 1])
    dst.value = np.array(
        values.reshape((values.size, 1)),
        dtype=jigsaw_msh_t.REALS_t)


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
    tcf = ax.tricontourf(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'],
        mesh.value.flatten(),
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
    axes.triplot(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'],
        color=color,
        linewidth=linewidth,
        **kwargs)
    if show:
        axes.axis('scaled')
        plt.show()
    return axes


def reproject(
        mesh: jigsaw_msh_t,
        dst_crs: Union[str, CRS]
):
    src_crs = mesh.crs
    dst_crs = CRS.from_user_input(dst_crs)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    # pylint: disable=E0633
    x, y = transformer.transform(
        mesh.vert2['coord'][:, 0], mesh.vert2['coord'][:, 1])
    mesh.vert2 = np.array(
        [([x[i], y[i]], mesh.vert2['IDtag'][i]) for i
         in range(len(mesh.vert2['IDtag']))],
        dtype=jigsaw_msh_t.VERT2_t)
    mesh.crs = dst_crs


def limgrad(mesh, dfdx, imax=100):
    """
    See https://github.com/dengwirda/mesh2d/blob/master/hjac-util/limgrad.m
    for original source code.
    """
    tri = mesh_to_tri(mesh)
    xy = np.vstack([tri.x, tri.y]).T
    edge = tri.edges
    dx = np.subtract(xy[edge[:, 0], 0], xy[edge[:, 1], 0])
    dy = np.subtract(xy[edge[:, 0], 1], xy[edge[:, 1], 1])
    elen = np.sqrt(dx**2+dy**2)
    ffun = mesh.value.flatten()
    aset = np.zeros(ffun.shape)
    ftol = np.min(ffun) * np.sqrt(np.finfo(float).eps)
    # precompute neighbor table
    point_neighbors = defaultdict(set)
    for simplex in tri.triangles:
        for i, j in permutations(simplex, 2):
            point_neighbors[i].add(j)
    # iterative smoothing
    for _iter in range(1, imax+1):
        aidx = np.where(aset == _iter-1)[0]
        if len(aidx) == 0.:
            break
        active_idxs = np.argsort(ffun[aidx])
        for active_idx in active_idxs:
            adjacent_edges = point_neighbors[active_idx]
            for adj_edge in adjacent_edges:
                if ffun[adj_edge] > ffun[active_idx]:
                    fun1 = ffun[active_idx] + elen[active_idx] * dfdx
                    if ffun[adj_edge] > fun1+ftol:
                        ffun[adj_edge] = fun1
                        aset[adj_edge] = _iter
                else:
                    fun2 = ffun[adj_edge] + elen[active_idx] * dfdx
                    if ffun[active_idx] > fun2+ftol:
                        ffun[active_idx] = fun2
                        aset[active_idx] = _iter
    if not _iter < imax:
        msg = f'limgrad() did not converge within {imax} iterations.'
        raise Exception(msg)
    return ffun


def msh_t_to_grd(msh: jigsaw_msh_t) -> Dict:

    src_crs = msh.crs if hasattr(msh, 'crs') else None
    coords = msh.vert2['coord']
    desc = "EPSG:4326"
    if src_crs is not None:
        # TODO: Support non EPSG:4326 CRS
#        desc = src_crs.to_string()
        epsg_4326 = CRS.from_epsg(4326)
        if not src_crs.equals(epsg_4326):
            transformer = Transformer.from_crs(
                src_crs, epsg_4326, always_xy=True)
            coords = np.vstack(
                transformer.transform(coords[:, 0], coords[:, 1])).T

    nodes = {
        i + 1: [tuple(p.tolist()), v] for i, (p, v) in
            enumerate(zip(coords, -msh.value))}
    # NOTE: Node IDs are node index + 1
    elems = {
        i + 1: v + 1 for i, v in enumerate(msh.tria3['index'])}
    offset = len(elems)
    elems.update({
        offset + i + 1: v + 1 for i, v in enumerate(msh.quad4['index'])})

    return {'description': desc,
            'nodes': nodes,
            'elements': elems}


def grd_to_msh_t(_grd: Dict) -> jigsaw_msh_t:

    msh = jigsaw_msh_t()
    msh.ndims = +2
    msh.mshID = 'euclidean-mesh'
    id_to_index = {node_id: index for index, node_id
                   in enumerate(_grd['nodes'].keys())}
    triangles = [list(map(lambda x: id_to_index[x], element)) for element
                 in _grd['elements'].values() if len(element) == 3]
    quads = [list(map(lambda x: id_to_index[x], element)) for element
             in _grd['elements'].values() if len(element) == 4]
    msh.vert2 = np.array([(coord, 0) for coord, _ in _grd['nodes'].values()],
                         dtype=jigsaw_msh_t.VERT2_t)
    msh.tria3 = np.array([(index, 0) for index in triangles],
                         dtype=jigsaw_msh_t.TRIA3_t)
    msh.quad4 = np.array([(index, 0) for index in quads],
                         dtype=jigsaw_msh_t.QUAD4_t)
    value = [value for _, value in _grd['nodes'].values()]
    msh.value = np.array(np.array(value).reshape((len(value), 1)),
                         dtype=jigsaw_msh_t.REALS_t)
    crs = _grd.get('crs')
    if crs is not None:
        msh.crs = CRS.from_user_input(crs)
    return msh


def msh_t_to_2dm(msh: jigsaw_msh_t):
    coords = msh.vert2['coord']
    src_crs = msh.crs if hasattr(msh, 'crs') else None
    if src_crs is not None:
        epsg_4326 = CRS.from_epsg(4326)
        if not src_crs.equals(epsg_4326):
            transformer = Transformer.from_crs(
                src_crs, epsg_4326, always_xy=True)
            coords = np.vstack(
                transformer.transform(coords[:, 0], coords[:, 1])).T
    return {
            'ND': {i+1: (coord, msh.value[i, 0] if not
                         np.isnan(msh.value[i, 0]) else -99999)
                   for i, coord in enumerate(coords)},
            'E3T': {i+1: index+1 for i, index
                    in enumerate(msh.tria3['index'])},
            'E4Q': {i+1: index+1 for i, index
                    in enumerate(msh.quad4['index'])}
        }


def sms2dm_to_msh_t(_sms2dm: Dict) -> jigsaw_msh_t:
    msh = jigsaw_msh_t()
    msh.ndims = +2
    msh.mshID = 'euclidean-mesh'
    id_to_index = {node_id: index for index, node_id
                   in enumerate(_sms2dm['ND'].keys())}
    if 'E3T' in _sms2dm:
        triangles = [list(map(lambda x: id_to_index[x], element)) for element
                     in _sms2dm['E3T'].values()]
        msh.tria3 = np.array([(index, 0) for index in triangles],
                             dtype=jigsaw_msh_t.TRIA3_t)
    if 'E4Q' in _sms2dm:
        quads = [list(map(lambda x: id_to_index[x], element)) for element
                 in _sms2dm['E4Q'].values()]
        msh.quad4 = np.array([(index, 0) for index in quads],
                             dtype=jigsaw_msh_t.QUAD4_t)
    msh.vert2 = np.array([(coord, 0) for coord, _ in _sms2dm['ND'].values()],
                         dtype=jigsaw_msh_t.VERT2_t)
    value = [value for _, value in _sms2dm['ND'].values()]
    msh.value = np.array(np.array(value).reshape((len(value), 1)),
                         dtype=jigsaw_msh_t.REALS_t)
    crs = _sms2dm.get('crs')
    if crs is not None:
        msh.crs = CRS.from_user_input(crs)
    return msh

@must_be_euclidean_mesh
def msh_t_to_utm(msh):
    utm_crs = estimate_mesh_utm(msh)
    if utm_crs is None:
        return

    transformer = Transformer.from_crs(
        msh.crs, utm_crs, always_xy=True)

    coords = msh.vert2['coord']

    # pylint: disable=E0633
    coords[:, 0], coords[:, 1] = transformer.transform(
            coords[:, 0], coords[:, 1])
    msh.vert2['coord'][:] = coords
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

@must_be_euclidean_mesh
def estimate_mesh_utm(msh):
    if hasattr(msh, 'crs'):
        coords = msh.vert2['coord']
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


    ret_val = channels_gdf.unary_union
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


    dst_crs = CRS.from_user_input(out_crs)

    mesh_types = {
        'tria3': 'TRIA3_t',
        'quad4': 'QUAD4_t',
        'hexa8': 'HEXA8_t'
    }

    coord = []
    elems = {k: [] for k in mesh_types}
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
            x = mesh.vert2['coord'][:, 0]
            y = mesh.vert2['coord'][:, 1]
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


        for k in mesh_types:
            cnn = getattr(mesh, k)
            elems[k].append(cnn['index'] + offset)
        coord.append(mesh.vert2['coord'])
        value.append(mesh.value)
        offset += coord[-1].shape[0]

    composite_mesh = jigsaw_msh_t()
    composite_mesh.mshID = 'euclidean-mesh'
    composite_mesh.ndims = 2

    composite_mesh.vert2 = np.array(
            [(coord, 0) for coord in np.vstack(coord)],
            dtype=jigsaw_msh_t.VERT2_t)
    composite_mesh.value = np.array(
            np.vstack(value),
            dtype=jigsaw_msh_t.REALS_t)
    for k, v in mesh_types.items():
        setattr(composite_mesh, k, np.array(
            [(cnn, 0) for cnn in np.vstack(elems[k])],
            dtype=getattr(jigsaw_msh_t, v)))

    composite_mesh.crs = dst_crs

    return composite_mesh



def add_pool_args(func):
    def wrapper(*args, nprocs=None, pool=None, **kwargs):

        # TODO: Modify docstring?
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

    def is_zero(i):
        return np.isclose(i, 0)
    def isnt_zero(i):
        return np.logical_not(np.isclose(i, 0))

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

    if isinstance(mpoly, Polygon):
        mpoly = MultiPolygon([mpoly])
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
        Magnitude of total velocity used for element size calculation (:math:`\\frac{m}{sec}`)
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
        Free surface elevation (:math:`meters`) from the reference (i.e. wave height)

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
    Based on OceanMesh2D approximation method https://doi.org/10.13140/RG.2.2.21840.61446/2.
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
    Based on OceanMesh2D approximation method https://doi.org/10.13140/RG.2.2.21840.61446/2.
    '''

    if not isinstance(depth, np.ndarray):
        depth = np.array(depth)
    dep_shape = depth.shape
    depth = depth.ravel()

    depth_mask = can_velocity_be_approximated_by_linear_wave_theory(
        depth, wave_amplitude)

    velocity = np.zeros_like(depth)
    velocity[depth_mask] = wave_amplitude * np.sqrt(constants.g / np.abs(depth[depth_mask]))
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
        Free surface elevation (:math:`meters`) from the reference (i.e. wave height)

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

    particle_velocity = estimate_particle_velocity_from_depth(depth, wave_amplitude)
    characteristic_velocity_magnitude = (
        particle_velocity + np.sqrt(constants.g * np.abs(depth))
    )
    # For overland where h < nu the characteristic velocity is 2 * sqrt(g*h)
    characteristic_velocity_magnitude[~depth_mask] = 2 * particle_velocity[~depth_mask]
    return characteristic_velocity_magnitude * timestep / element_size
