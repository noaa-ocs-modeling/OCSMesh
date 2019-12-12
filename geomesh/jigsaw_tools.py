from collections import defaultdict
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import LinearRing, Polygon, MultiPolygon
from jigsawpy import jigsaw_msh_t


def mesh_to_tri(mesh):
    """
    mesh is a jigsawpy.jigsaw_msh_t() instance.
    """
    assert isinstance(mesh, jigsaw_msh_t)
    return Triangulation(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'])


def cleanup_isolates(mesh):
    assert isinstance(mesh, jigsaw_msh_t)
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
    _mesh = jigsaw_msh_t()
    _mesh.ndims = 2
    _mesh.vert2 = mesh.vert2.take(vert2_idxs, axis=0)
    _mesh.tria3 = np.asarray(
        [(tuple(indices), mesh.tria3['IDtag'][i])
         for i, indices in enumerate(tria3)],
        dtype=jigsaw_msh_t.TRIA3_t)
    return _mesh


def multipolygon_to_geom(multipolygon):
    assert isinstance(multipolygon, MultiPolygon)
    vert2 = list()
    for polygon in multipolygon:
        if np.all(
                np.asarray(polygon.exterior.coords).flatten() == float('inf')):
            msg = "PSLG seems to correspond to ellipsoidal mesh "
            msg += "which has not yet been implemented."
            raise NotImplementedError(msg)
        for x, y in polygon.exterior.coords[:-1]:
            vert2.append(((x, y), 0))
        for interior in polygon.interiors:
            for x, y in interior.coords[:-1]:
                vert2.append(((x, y), 0))
    vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)
    # edge2
    edge2 = list()
    for polygon in multipolygon:
        polygon = [polygon.exterior, *polygon.interiors]
        for linear_ring in polygon:
            _edge2 = list()
            for i in range(len(linear_ring.coords)-2):
                _edge2.append((i, i+1))
            _edge2.append((_edge2[-1][1], _edge2[0][0]))
            edge2.extend(
                [(e0+len(edge2), e1+len(edge2)) for e0, e1 in _edge2])
    edge2 = np.asarray(
        [((e0, e1), 0) for e0, e1 in edge2], dtype=jigsaw_msh_t.EDGE2_t)
    # geom
    geom = jigsaw_msh_t()
    geom.ndims = +2
    geom.mshID = 'euclidean-mesh'
    geom.vert2 = vert2
    geom.edge2 = edge2
    return geom


def geom_to_multipolygon(msh_t):
    assert isinstance(msh_t, jigsaw_msh_t)
    assert msh_t.ndims == 2
    tri = Triangulation(
        msh_t.vert2['coord'][:, 0],
        msh_t.vert2['coord'][:, 1],
        msh_t.tria3['index'])

    # get boundary_edges from triangulation
    boundary_edges = list()
    idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
    for i, j in idxs:
        boundary_edges.append(
            (tri.triangles[i, j], tri.triangles[i, (j+1) % 3]))
    boundary_edges = np.asarray(boundary_edges)

    # get index_ring_collection from boundary_edges
    index_ring_collection = list()
    ordered_edges = [boundary_edges[-1, :]]
    boundary_edges = np.delete(boundary_edges, -1, axis=0)
    while boundary_edges.shape[0] > 0:
        try:
            idx = np.where(
                boundary_edges[:, 0]
                == ordered_edges[-1][1])[0][0]
            ordered_edges.append(boundary_edges[idx, :])
            boundary_edges = np.delete(boundary_edges, idx, axis=0)
        except IndexError:
            index_ring_collection.append(
                np.asarray(ordered_edges))
            ordered_edges = [boundary_edges[-1, :]]
            boundary_edges = np.delete(boundary_edges, -1, axis=0)

    # get linear_ring_collection from index_ring_collection
    linear_ring_collection = list()
    for index_ring in index_ring_collection:
        coord = msh_t.vert2['coord'][index_ring[:, 0], :]
        coord = np.vstack([coord, coord[0, :]])
        linear_ring_collection.append(LinearRing(coord))

    # convert linear_ring_collection to multipolygon
    assert len(linear_ring_collection) > 0
    if len(linear_ring_collection) > 1:
        areas = [Polygon(linear_ring).area
                 for linear_ring in linear_ring_collection]
        idx = np.where(areas == np.max(areas))[0][0]
        polygon_collection = list()
        outer_ring = linear_ring_collection.pop(idx)
        path = Path(np.asarray(outer_ring.coords), closed=True)
        while len(linear_ring_collection) > 0:
            inner_rings = list()
            for i, linear_ring in reversed(
                    list(enumerate(linear_ring_collection))):
                xy = np.asarray(linear_ring.coords)[0, :]
                if path.contains_point(xy):
                    inner_rings.append(linear_ring_collection.pop(i))
            polygon_collection.append(Polygon(outer_ring, inner_rings))
            if len(linear_ring_collection) > 0:
                areas = [Polygon(linear_ring).area
                         for linear_ring in linear_ring_collection]
                idx = np.where(areas == np.max(areas))[0][0]
                outer_ring = linear_ring_collection.pop(idx)
                path = Path(np.asarray(outer_ring.coords), closed=True)
        multipolygon = MultiPolygon(polygon_collection)
    else:
        multipolygon = MultiPolygon(
            [Polygon(linear_ring_collection.pop())])

    # guarantee typecasting to multipolygon
    if isinstance(multipolygon, Polygon):
        multipolygon = MultiPolygon([multipolygon])

    # plot to check
    # for polygon in multipolygon:
    #     plt.plot(*polygon.exterior.xy, color='k')
    #     for interior in polygon.interiors:
    #         plt.plot(*interior.xy, color='r')
    # plt.show()

    return multipolygon


def edge2_from_msh_t(msh_t):
    assert isinstance(msh_t, jigsaw_msh_t)
    assert msh_t.ndims == 2
    tri = Triangulation(
        msh_t.vert2['coord'][:, 0],
        msh_t.vert2['coord'][:, 1],
        msh_t.tria3['index'])
    return np.array(
        [(edge, 0) for edge in tri.edges], dtype=jigsaw_msh_t.EDGE2_t)


def interpolate_hmat(mesh, hmat, method='spline', **kwargs):
    assert isinstance(mesh, jigsaw_msh_t)
    assert isinstance(hmat, jigsaw_msh_t)
    assert method in ['spline', 'linear', 'nearest']
    if method == 'spline':
        values = RectBivariateSpline(
            hmat.xgrid,
            hmat.ygrid,
            hmat.value.T,
            **kwargs
            ).ev(
            mesh.vert2['coord'][:, 0],
            mesh.vert2['coord'][:, 1])
        mesh.value = np.array(
            values.reshape((values.size, 1)),
            dtype=jigsaw_msh_t.REALS_t)
        return mesh
    else:
        raise NotImplementedError("Only 'spline' method is available")


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
    ax=None,
    show=False,
    figsize=None,
    color='k',
    linewidth=0.07,
    **kwargs
):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    ax.triplot(
        mesh.vert2['coord'][:, 0],
        mesh.vert2['coord'][:, 1],
        mesh.tria3['index'],
        color=color,
        linewidth=linewidth,
        **kwargs)
    if show:
        ax.axis('scaled')
        plt.show()
    return ax


def limgrad(tri, values, dfdx, imax=100):
    """
    See https://github.com/dengwirda/mesh2d/blob/master/hjac-util/limgrad.m
    for original source code.
    """
    xy = np.vstack([tri.x, tri.y]).T
    edge = tri.edges
    dx = np.subtract(xy[edge[:, 0], 0], xy[edge[:, 1], 0])
    dy = np.subtract(xy[edge[:, 0], 1], xy[edge[:, 1], 1])
    elen = np.sqrt(dx**2+dy**2)
    ffun = values.flatten()
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
