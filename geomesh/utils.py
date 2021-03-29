from collections import defaultdict
from enum import Enum
from itertools import permutations
from typing import Union, Dict

from jigsawpy import jigsaw_msh_t  # type: ignore[import]
from matplotlib.path import Path  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
from matplotlib.tri import Triangulation  # type: ignore[import]
import numpy as np  # type: ignore[import]
from pyproj import CRS, Transformer  # type: ignore[import]
from scipy.interpolate import (  # type: ignore[import]
    RectBivariateSpline, griddata)
from shapely.geometry import Polygon, MultiPolygon  # type: ignore[import]

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
    node_indexes = np.arange(mesh.vert2['coord'].shape[0])
    used_indexes = np.unique(mesh.tria3['index'])
    vert2_idxs = np.where(
        np.isin(node_indexes, used_indexes, assume_unique=True))[0]
    tria3_idxs = np.where(
        ~np.isin(node_indexes, used_indexes, assume_unique=True))[0]
    tria3 = mesh.tria3['index'].flatten()
    for idx in reversed(tria3_idxs):
        _idx = np.where(tria3 >= idx)
        tria3[_idx] = tria3[_idx] - 1
    tria3 = tria3.reshape(mesh.tria3['index'].shape)
    mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)
    if len(mesh.value) > 0:
        mesh.value = mesh.value.take(vert2_idxs)
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
    while needs_sieve(mesh) or has_pinched_nodes(mesh):
        cleanup_pinched_nodes(mesh)
        sieve(mesh, sieve_area)
    # cleanup_isolates(mesh)
    put_IDtags(mesh)


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
    _node_neighbors = vertices_around_vertex(mesh)
    _idxs = np.where(vert2_mask)[0]
    for _idx in _idxs:
        vert2_mask[list(_node_neighbors[_idx])] = True

    # Also, there might be some dangling triangles without neighbors, which are
    # also missed by path.contains_point()
    for idx, neighbors in _node_neighbors.items():
        if len(neighbors) <= 2:
            vert2_mask[idx] = True

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
        mesh.value = mesh.value.take(vert2_idxs)

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


# https://en.wikipedia.org/wiki/Polygon_mesh#Summary_of_mesh_representation
# V-V     All vertices around vertex
# E-F     All edges of a face
# V-F     All vertices of a face
# F-V     All faces around a vertex
# E-V     All edges around a vertex
# F-E     Both faces of an edge
# V-E     Both vertices of an edge
# Flook   Find face with given vertices


def must_be_euclidean_mesh(f):
    def decorator(mesh):
        if mesh.mshID.lower() != 'euclidean-mesh':
            msg = f"Not implemented for mshID={mesh.mshID}"
            raise NotImplementedError(msg)
        return f(mesh)
    return decorator


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


def has_pinched_nodes(mesh):
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
    **kwargs
):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    ax.tricontourf(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'],
        mesh.value.flatten(),
        **kwargs)
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
