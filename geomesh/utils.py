from collections import defaultdict
from enum import Enum
from itertools import permutations
from typing import Union, Dict, Sequence
from functools import reduce
from copy import deepcopy

import jigsawpy
from jigsawpy import jigsaw_msh_t  # type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
from matplotlib.tri import Triangulation  # type: ignore[import]
import numpy as np  # type: ignore[import]
from pyproj import CRS, Transformer  # type: ignore[import]
from scipy.interpolate import (  # type: ignore[import]
    RectBivariateSpline, griddata)
from shapely.geometry import Polygon, MultiPolygon, box  # type: ignore[import]
import geopandas as gpd
import utm

from geomesh.mesh.parsers import grd, sms2dm


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
    _index_ring_collection = index_ring_collection(mesh)
    polygon_collection = list()
    for polygon in _index_ring_collection.values():
        exterior = vertices[polygon['exterior'][:, 0], :]
        interiors = list()
        for interior in polygon['interiors']:
            interiors.append(vertices[interior[:, 0], :])
        polygon_collection.append(Polygon(exterior, interiors))
    return MultiPolygon(polygon_collection)


def needs_sieve(mesh, area=None):
    areas = [polygon.area for polygon in geom_to_multipolygon(mesh)]
    if area is None:
        remove = np.where(areas < np.max(areas))[0].tolist()
    else:
        remove = list()
        for idx, patch_area in enumerate(areas):
            if patch_area <= area:
                remove.append(idx)
    if len(remove) > 0:
        return True
    else:
        return False


def put_IDtags(mesh):
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


def finalize_mesh(mesh, sieve_area=None):
    cleanup_isolates(mesh)
    pinched_nodes = get_pinched_nodes(mesh)
    while needs_sieve(mesh, sieve_area) or len(pinched_nodes):
        clip_mesh_by_vertex(
            mesh, pinched_nodes,
            can_use_other_verts=True, inverse=True, in_place=True)
        sieve(mesh, sieve_area)
        pinched_nodes = get_pinched_nodes(mesh)
    cleanup_isolates(mesh)
    put_IDtags(mesh)


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
        if not len(tiny_verts):
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
        remove = list()
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
    tria3_IDtag = mesh.tria3['IDtag'].take(np.where(~tria3_mask)[0])
    tria3_index = mesh.tria3['index'][~tria3_mask, :].flatten()
    for idx in reversed(tria3_idxs):
        tria3_index[np.where(tria3_index >= idx)] -= 1
    tria3_index = tria3_index.reshape((tria3_IDtag.shape[0], 3))
    vert2_idxs = np.where(np.isin(node_indexes, used_indexes))[0]

    # update vert2
    mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)

    # update value
    if len(mesh.value) > 0:
        mesh.value = mesh.value.take(vert2_idxs, axis=0)

    # update tria3
    mesh.tria3 = np.array(
        [(tuple(indices), tria3_IDtag[i])
         for i, indices in enumerate(tria3_index)],
        dtype=jigsaw_msh_t.TRIA3_t)


def sort_edges(edges):

    if len(edges) == 0:
        return edges

    # start ordering the edges into linestrings
    edge_collection = list()
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
    boundary_edges = list()
    tri = mesh_to_tri(mesh)
    idxs = np.vstack(
        list(np.where(tri.neighbors == -1))).T
    for i, j in idxs:
        boundary_edges.append(
            (int(tri.triangles[i, j]),
                int(tri.triangles[i, (j+1) % 3])))
    index_ring_collection = sort_edges(boundary_edges)
    # sort index_rings into corresponding "polygons"
    areas = list()
    vertices = mesh.vert2['coord']
    for index_ring in index_ring_collection:
        e0, e1 = [list(t) for t in zip(*index_ring)]
        areas.append(float(Polygon(vertices[e0, :]).area))

    # maximum area must be main mesh
    idx = areas.index(np.max(areas))
    exterior = index_ring_collection.pop(idx)
    areas.pop(idx)
    _id = 0
    _index_ring_collection = dict()
    _index_ring_collection[_id] = {
        'exterior': np.asarray(exterior),
        'interiors': []
        }
    e0, e1 = [list(t) for t in zip(*exterior)]
    path = Path(vertices[e0 + [e0[0]], :], closed=True)
    while len(index_ring_collection) > 0:
        # find all internal rings
        potential_interiors = list()
        for i, index_ring in enumerate(index_ring_collection):
            e0, e1 = [list(t) for t in zip(*index_ring)]
            if path.contains_point(vertices[e0[0], :]):
                potential_interiors.append(i)
        # filter out nested rings
        real_interiors = list()
        for i, p_interior in reversed(list(enumerate(potential_interiors))):
            _p_interior = index_ring_collection[p_interior]
            check = [index_ring_collection[_]
                     for j, _ in reversed(list(enumerate(potential_interiors)))
                     if i != j]
            has_parent = False
            for _path in check:
                e0, e1 = [list(t) for t in zip(*_path)]
                _path = Path(vertices[e0 + [e0[0]], :], closed=True)
                if _path.contains_point(vertices[_p_interior[0][0], :]):
                    has_parent = True
            if not has_parent:
                real_interiors.append(p_interior)
        # pop real rings from collection
        for i in reversed(sorted(real_interiors)):
            _index_ring_collection[_id]['interiors'].append(
                np.asarray(index_ring_collection.pop(i)))
            areas.pop(i)
        # if no internal rings found, initialize next polygon
        if len(index_ring_collection) > 0:
            idx = areas.index(np.max(areas))
            exterior = index_ring_collection.pop(idx)
            areas.pop(idx)
            _id += 1
            _index_ring_collection[_id] = {
                'exterior': np.asarray(exterior),
                'interiors': []
                }
            e0, e1 = [list(t) for t in zip(*exterior)]
            path = Path(vertices[e0 + [e0[0]], :], closed=True)
    return _index_ring_collection


def outer_ring_collection(mesh):
    _index_ring_collection = index_ring_collection(mesh)
    outer_ring_collection = defaultdict()
    for key, ring in _index_ring_collection.items():
        outer_ring_collection[key] = ring['exterior']
    return outer_ring_collection


def inner_ring_collection(mesh):
    _index_ring_collection = index_ring_collection(mesh)
    inner_ring_collection = defaultdict()
    for key, rings in _index_ring_collection.items():
        inner_ring_collection[key] = rings['interiors']
    return inner_ring_collection


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
                    vertices_around_vertex[i].add(j)
        vertices_around_vertex = defaultdict(set)
        append(mesh.tria3)
        append(mesh.quad4)
        append(mesh.hexa8)
        return vertices_around_vertex
    else:
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

    else:

        pt_series = gpd.GeoSeries(gpd.points_from_xy(
            mesh.vert2['coord'][:,0], mesh.vert2['coord'][:,1]))
        shp_series = gpd.GeoSeries(shape)

        in_shp_idx = pt_series.sindex.query_bulk(
                shp_series, predicate="intersects")

        return in_shp_idx


def clip_mesh_by_shape(
        mesh: jigsaw_msh_t,
        shape: Union[box, Polygon, MultiPolygon],
        use_box_only: bool = False,
        fit_inside: bool = True,
        inverse: bool = False,
        ) -> jigsaw_msh_t:


    # If we want to calculate inverse based on shape, calculating
    # from bbox first results in the wrong result
    if not inverse or use_box_only:

        # First based on bounding box only
        shape_box = box(*shape.bounds)

        # TODO: Optimize for multipolygons (use separate bboxes)
        in_box_idx = get_verts_in_shape(mesh, shape_box, True)

        mesh = clip_mesh_by_vertex(
                mesh, in_box_idx, not fit_inside, inverse)

        if use_box_only:
            return mesh

    in_shp_idx = get_verts_in_shape(mesh, shape, False)

    mesh = clip_mesh_by_vertex(
            mesh, in_shp_idx, not fit_inside, inverse)

    return mesh


def clip_mesh_by_vertex(
        mesh: jigsaw_msh_t,
        vert_in: Sequence[int],
        can_use_other_verts: bool = False,
        inverse: bool = False,
        in_place: bool = False
        ) -> jigsaw_msh_t:

    if mesh.mshID == 'euclidean-mesh' and mesh.ndims == 2:
        coord = mesh.vert2['coord']
        trias = mesh.tria3['index']
        quads = mesh.quad4['index']
        hexas = mesh.hexa8['index']

        # Whether elements that include "in"-vertices can be created
        # using vertices other than "in"-vertices
        mark_func = np.all
        if can_use_other_verts:
            mark_func = np.any

        mark_tria = mark_func(
                (np.isin(trias.ravel(), vert_in).reshape(
                    trias.shape)), 1)
        mark_quad = mark_func(
                (np.isin(quads.ravel(), vert_in).reshape(
                    quads.shape)), 1)
        mark_hexa = mark_func(
                (np.isin(hexas.ravel(), vert_in).reshape(
                    hexas.shape)), 1)

        # Whether to return elements found by "in" vertices or return
        # all elements except them
        if inverse:
            mark_tria = np.logical_not(mark_tria)
            mark_quad = np.logical_not(mark_quad)
            mark_hexa = np.logical_not(mark_hexa)

        # Find elements based on old vertex index
        new_trias_unfinished = trias[mark_tria, :]
        new_quads_unfinished = quads[mark_quad, :]
        new_hexas_unfinished = hexas[mark_hexa, :]

        crd_old_to_new = {
                index: i for i, index
                in enumerate(sorted(np.unique(np.concatenate(
                        (new_trias_unfinished.ravel(),
                         new_quads_unfinished.ravel(),
                         new_hexas_unfinished.ravel())
                        ))))
            }

        new_trias = np.array([
                [crd_old_to_new[x] for x in  element]
                    for element in new_trias_unfinished])
        new_quads = np.array([
                [crd_old_to_new[x] for x in  element]
                    for element in new_quads_unfinished])
        new_hexas = np.array([
                [crd_old_to_new[x] for x in  element]
                    for element in new_hexas_unfinished])

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

        # TODO: What about edge2 if in_place?
        mesh_out.vert2 = np.array(
            [(coo, 0) for coo in new_coord],
            dtype=jigsaw_msh_t.VERT2_t)
        mesh_out.tria3 = np.array(
            [(con, 0) for con in new_trias],
            dtype=jigsaw_msh_t.TRIA3_t)
        mesh_out.quad4 = np.array(
            [(con, 0) for con in new_quads],
            dtype=jigsaw_msh_t.TRIA3_t)
        mesh_out.hexa8 = np.array(
            [(con, 0) for con in new_hexas],
            dtype=jigsaw_msh_t.TRIA3_t)
        return mesh_out

    msg = (f"Not implemented for"
           f" mshID={mesh.mshID} and dim={mesh.ndims}")
    raise NotImplementedError(msg)




def must_be_euclidean_mesh(f):
    def decorator(mesh):
        if mesh.mshID.lower() != 'euclidean-mesh':
            msg = f"Not implemented for mshID={mesh.mshID}"
            raise NotImplementedError(msg)
        return f(mesh)
    return decorator

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
    p = np.sum(tria_sides, axis=1) / 2
    p = p.reshape(len(p), 1)
    a, b, c = np.split(tria_sides, 3, axis=1)
    tria_areas = np.sqrt(p*(p-a)*(p-b)*(p-c)).squeeze()
    return tria_areas

@must_be_euclidean_mesh
def calculate_edge_lengths(mesh):

    # Taken from size_from_mesh method of hfun-mesh

    coord = mesh.vert2['coord']

    # NOTE: For msh_t type vertex id and index are the same
    trias = mesh.tria3['index']
    quads = mesh.quad4['index']

    # Get unique set of edges by rolling connectivity
    # and joining connectivities in 3rd dimension, then sorting
    # to get all edges with lower index first
    all_edges = np.empty(shape=(0, 2), dtype=trias.dtype)
    if trias.shape[0]:
        edges = np.sort(
                np.stack(
                    (trias, np.roll(trias, shift=1, axis=1)),
                    axis=2),
                axis=2)
        edges = np.unique(
                edges.reshape(np.product(edges.shape[0:2]), 2), axis=0)
        all_edges = np.vstack((all_edges, edges))
    if quads.shape[0]:
        edges = np.sort(
                np.stack(
                    (quads, np.roll(quads, shift=1, axis=1)),
                    axis=2),
                axis=2)
        edges = np.unique(
                edges.reshape(np.product(edges.shape[0:2]), 2), axis=0)
        all_edges = np.vstack((all_edges, edges))

    all_edges = np.unique(all_edges, axis=0)

    # ONLY TESTED FOR TRIA AS OF NOW

    # This part of the function is generic for tria and quad
    
    # Get coordinates for all edge vertices
    edge_coords = coord[edges, :]

    # Calculate length of all edges based on acquired coords
    edge_lens = np.sqrt(
            np.sum(
                np.power(
                    np.abs(np.diff(edge_coords, axis=1)), 2)
                ,axis=2)).squeeze()

    edge_dict = defaultdict(float)
    for en, edge in enumerate(edges):
        edge_dict[tuple(edge)] = edge_lens[en]

    return edge_dict


@must_be_euclidean_mesh
def elements(mesh):
    elements_id = list()
    elements_id.extend(list(mesh.tria3['IDtag']))
    elements_id.extend(list(mesh.quad4['IDtag']))
    elements_id.extend(list(mesh.hexa8['IDtag']))
    elements_id = range(1, len(elements_id)+1) \
        if len(set(elements_id)) != len(elements_id) else elements_id
    elements = list()
    elements.extend(list(mesh.tria3['index']))
    elements.extend(list(mesh.quad4['index']))
    elements.extend(list(mesh.hexa8['index']))
    elements = {
        elements_id[i]: indexes for i, indexes in enumerate(elements)}
    return elements


@must_be_euclidean_mesh
def faces_around_vertex(mesh):
    _elements = elements(mesh)
    length = max(map(len, _elements.values()))
    y = np.array([xi+[-99999]*(length-len(xi)) for xi in _elements.values()])
    print(y)
    faces_around_vertex = defaultdict(set)
    for i, coord in enumerate(mesh.vert2['index']):
        np.isin(i, axis=0)
        faces_around_vertex[i].add()

    faces_around_vertex = defaultdict(set)

def get_pinched_nodes(mesh):

    '''
    Find nodes through which fluid cannot flow
    '''

    coord = mesh.vert2['coord']

    # NOTE: For msh_t type vertex id and index are the same
    trias = mesh.tria3['index']
    quads = mesh.quad4['index']
    hexas = mesh.hexa8['index']

    # Get unique set of edges by rolling connectivity
    # and joining connectivities in 3rd dimension, then sorting
    # to get all edges with lower index first
    all_edges = np.empty(shape=(0, 2), dtype=trias.dtype)
    if trias.shape[0]:
        edges = np.sort(
                np.stack(
                    (trias, np.roll(trias, shift=1, axis=1)),
                    axis=2),
                axis=2)
        edges = edges.reshape(np.product(edges.shape[0:2]), 2)
        all_edges = np.vstack((all_edges, edges))
    if quads.shape[0]:
        edges = np.sort(
                np.stack(
                    (quads, np.roll(quads, shift=1, axis=1)),
                    axis=2),
                axis=2)
        edges = edges.reshape(np.product(edges.shape[0:2]), 2)
        all_edges = np.vstack((all_edges, edges))
    if hexas.shape[0]:
        edges = np.sort(
                np.stack(
                    (hexas, np.roll(hexas, shift=1, axis=1)),
                    axis=2),
                axis=2)
        edges = edges.reshape(np.product(edges.shape[0:2]), 2)
        all_edges = np.vstack((all_edges, edges))

    # Simplexes (list of node indices)
    all_edges, e_cnt = np.unique(all_edges, axis=0, return_counts=True)
    shared_edges = all_edges[e_cnt == 2]
    boundary_edges = all_edges[e_cnt == 1]

    # Node indices
    boundary_verts, vb_cnt = np.unique(boundary_edges, return_counts=True)

    # vertices/nodes that have more than 2 boundary edges are pinch
    pinch_verts = boundary_verts[vb_cnt > 2]

    return pinch_verts


def has_pinched_nodes(mesh):

    # Older function: computationally more expensive and missing some
    # nodes

    _inner_ring_collection = inner_ring_collection(mesh)
    all_nodes = list()
    for inner_rings in _inner_ring_collection.values():
        for ring in inner_rings:
            all_nodes.extend(np.asarray(ring)[:, 0].tolist())
    u, c = np.unique(all_nodes, return_counts=True)
    if len(u[c > 1]) > 0:
        return True
    else:
        return False


def cleanup_pinched_nodes(mesh):

    # Older function: computationally more expensive and missing some
    # nodes

    _inner_ring_collection = inner_ring_collection(mesh)
    all_nodes = list()
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
        bbox=[None, None, None, None],
        kx=3,
        ky=3,
        s=0
):
    values = RectBivariateSpline(
        src.xgrid,
        src.ygrid,
        src.value.T,
        bbox=bbox,
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
    if src_crs is not None:
        EPSG_4326 = CRS.from_epsg(4326)
        if not src_crs.equals(EPSG_4326):
            transformer = Transformer.from_crs(
                src_crs, EPSG_4326, always_xy=True)
            coords = np.vstack(
                transformer.transform(coords[:, 0], coords[:, 1])).T

    desc = "EPSG:4326"
    nodes = {
        i + 1: [tuple(p.tolist()), v] for i, (p, v) in
            enumerate(zip(coords, -msh.value))}
    # NOTE: Node IDs are node index + 1
    elements = {
        i + 1: v + 1 for i, v in enumerate(msh.tria3['index'])}
    offset = len(elements)
    elements.update({
        offset + i + 1: v + 1 for i, v in enumerate(msh.quad4['index'])})

    return {'description': desc,
            'nodes': nodes,
            'elements': elements}


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
        EPSG_4326 = CRS.from_epsg(4326)
        if not src_crs.equals(EPSG_4326):
            transformer = Transformer.from_crs(
                src_crs, EPSG_4326, always_xy=True)
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
    if hasattr(msh, 'crs') and msh.crs.is_geographic:
        coords = msh.vert2['coord']
        x0, y0, x1, y1 = (
            np.min(coords[:, 0]), np.min(coords[:, 1]),
            np.max(coords[:, 0]), np.max(coords[:, 1]))
        _, _, number, letter = utm.from_latlon(
                (y0 + y1)/2, (x0 + x1)/2)
        utm_crs = CRS(
                proj='utm',
                zone=f'{number}{letter}',
                ellps={
                    'GRS 1980': 'GRS80',
                    'WGS 84': 'WGS84'
                    }[msh.crs.ellipsoid.name]
            )
        transformer = Transformer.from_crs(
            msh.crs, utm_crs, always_xy=True)

        coords[:, 0], coords[:, 1] = transformer.transform(
                coords[:, 0], coords[:, 1])
        msh.vert2['coord'][:] = coords
        msh.crs = utm_crs
