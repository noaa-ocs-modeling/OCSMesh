from functools import lru_cache
from multiprocessing import Pool, cpu_count
import os
import pathlib
from typing import Union, List
import warnings

import geopandas as gpd
from jigsawpy import jigsaw_msh_t, savemsh, loadmsh, savevtk
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from matplotlib.tri import Triangulation
import numpy as np
from pyproj import CRS, Transformer
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon, box, LineString, LinearRing, MultiPolygon


from geomesh import utils
from geomesh.raster import Raster
from geomesh.mesh.base import BaseMesh
from geomesh.mesh.parsers import grd, sms2dm


class Rings:

    def __init__(self, mesh: 'EuclideanMesh'):
        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self):
        tri = self.mesh.elements.triangulation()
        idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
        boundary_edges = []
        for i, j in idxs:
            boundary_edges.append(
                (tri.triangles[i, j], tri.triangles[i, (j+1) % 3]))
        sorted_rings = sort_rings(edges_to_rings(boundary_edges),
                                  self.mesh.coord)
        data = []
        for bnd_id, rings in sorted_rings.items():
            coords = self.mesh.coord[rings['exterior'][:, 0], :]
            geometry = LinearRing(coords)
            data.append({
                    "geometry": geometry,
                    "bnd_id": bnd_id,
                    "type": 'exterior'
                })
            for interior in rings['interiors']:
                coords = self.mesh.coord[interior[:, 0], :]
                geometry = LinearRing(coords)
                data.append({
                    "geometry": geometry,
                    "bnd_id": bnd_id,
                    "type": 'interior'
                })
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self):
        return self().loc[self()['type'] == 'exterior']

    def interior(self):
        return self().loc[self()['type'] == 'interior']


class Edges:

    def __init__(self, mesh: 'EuclideanMesh'):
        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self) -> gpd.GeoDataFrame:
        data = []
        for ring in self.mesh.hull.rings().itertuples():
            coords = ring.geometry.coords
            for i in range(1, len(coords)):
                data.append({
                    "geometry": LineString([coords[i-1], coords[i]]),
                    "bnd_id": ring.bnd_id,
                    "type": ring.type})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self):
        return self().loc[self()['type'] == 'exterior']

    def interior(self):
        return self().loc[self()['type'] == 'interior']


class Hull:

    def __init__(self, mesh: 'EuclideanMesh'):
        self.mesh = mesh
        self.rings = Rings(mesh)
        self.edges = Edges(mesh)

    @lru_cache(maxsize=1)
    def __call__(self):
        data = []
        for bnd_id in np.unique(self.rings()['bnd_id'].tolist()):
            exterior = self.rings().loc[
                (self.rings()['bnd_id'] == bnd_id) &
                (self.rings()['type'] == 'exterior')]
            interiors = self.rings().loc[
                (self.rings()['bnd_id'] == bnd_id) &
                (self.rings()['type'] == 'interior')]
            data.append({
                    "geometry": Polygon(
                        exterior.iloc[0].geometry.coords,
                        [row.geometry.coords for _, row
                            in interiors.iterrows()]),
                    "bnd_id": bnd_id
                })
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self):
        data = []
        for exterior in self.rings().loc[
                self.rings()['type'] == 'exterior'].itertuples():
            data.append({"geometry": Polygon(exterior.geometry.coords)})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def interior(self):
        data = []
        for interior in self.rings().loc[
                self.rings()['type'] == 'interior'].itertuples():
            data.append({"geometry": Polygon(interior.geometry.coords)})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def implode(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"geometry": MultiPolygon([polygon.geometry for polygon
                                       in self().itertuples()])},
            crs=self.mesh.crs)

    def multipolygon(self) -> MultiPolygon:
        mp = self.implode().iloc[0].geometry
        if isinstance(mp, Polygon):
            mp = MultiPolygon([mp])
        return mp

    def triangulation(self):
        triangles = self.msh_t.tria3['index'].tolist()
        for quad in self.msh_t.quad4['index']:
            triangles.extend([
                [quad[0], quad[1], quad[3]],
                [quad[1], quad[2], quad[3]]
            ])
        return Triangulation(self.coord[:, 0], self.coord[:, 1], triangles)


class Nodes:

    def __init__(self, mesh: "EuclideanMesh"):
        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self):
        return {i+1: coord for i, coord in enumerate(self.coords())}

    def id(self):
        return list(self().keys())

    def index(self):
        return np.arange(len(self()))

    def coords(self):
        return self.mesh.coord

    def values(self):
        return self.mesh.values

    def get_index_by_id(self, id):
        return self.id_to_index[id]

    def get_id_by_index(self, index: int):
        return self.index_to_id[index]

    @property
    def id_to_index(self):
        if not hasattr(self, '_id_to_index'):
            self._id_to_index = {node_id: index for index, node_id
                                 in enumerate(self().keys())}
        return self._id_to_index

    @property
    def index_to_id(self):
        if not hasattr(self, '_index_to_id'):
            self._index_to_id = {index: node_id for index, node_id
                                 in enumerate(self().keys())}
        return self._index_to_id

    # def get_indexes_around_index(self, index):
    #     indexes_around_index = self.__dict__.get('indexes_around_index')
    #     if indexes_around_index is None:
    #         def append(geom):
    #             for simplex in geom:
    #                 for i, j in permutations(simplex, 2):
    #                     indexes_around_index[i].add(j)
    #         indexes_around_index = defaultdict(set)
    #         append(self.gr3.elements.triangles())
    #         append(self.gr3.elements.quads())
    #         self.__dict__['indexes_around_index'] = indexes_around_index
    #     return list(indexes_around_index[index])


class Elements:

    def __init__(self, mesh: "EuclideanMesh"):
        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self):
        elements = {i+1: index+1 for i, index
                    in enumerate(self.mesh.msh_t.tria3['index'])}
        elements.update({i+len(elements)+1: index+1 for i, index
                         in enumerate(self.mesh.msh_t.quad4['index'])})
        return elements

    @lru_cache(maxsize=1)
    def id(self):
        return list(self().keys())

    @lru_cache(maxsize=1)
    def index(self):
        return np.arange(len(self()))

    def array(self):
        rank = int(max(map(len, self().values())))
        array = np.full((len(self()), rank), -1)
        for i, element in enumerate(self().values()):
            row = np.array(list(map(self.mesh.nodes.get_index_by_id, element)))
            array[i, :len(row)] = row
        return np.ma.masked_equal(array, -1)

    @lru_cache(maxsize=1)
    def triangles(self):
        return np.array(
            [list(map(self.mesh.nodes.get_index_by_id, element))
             for element in self().values()
             if len(element) == 3])

    @lru_cache(maxsize=1)
    def quads(self):
        return np.array(
            [list(map(self.mesh.nodes.get_index_by_id, element))
             for element in self().values()
             if len(element) == 4])

    def triangulation(self):
        triangles = self.triangles().tolist()
        for quad in self.quads():
            # TODO: Not tested.
            triangles.append([quad[0], quad[1], quad[3]])
            triangles.append([quad[1], quad[2], quad[3]])
        return Triangulation(
            self.mesh.coord[:, 0],
            self.mesh.coord[:, 1],
            triangles)

    def geodataframe(self):
        data = []
        for id, element in self().items():
            data.append({
                'geometry': Polygon(
                    self.mesh.coord[list(
                        map(self.mesh.nodes.get_index_by_id, element))]),
                'id': id})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)


class EuclideanMesh(BaseMesh):

    def __init__(self, mesh: jigsaw_msh_t):
        if not isinstance(mesh, jigsaw_msh_t):
            raise TypeError(f'Argument mesh must be of type {jigsaw_msh_t}, '
                            f'not type {type(mesh)}.')
        if mesh.mshID != 'euclidean-mesh':
            raise ValueError(f'Argument mesh has property mshID={mesh.mshID}, '
                             "but expected 'euclidean-mesh'.")
        if not hasattr(mesh, 'crs'):
            warnings.warn('Input mesh has no CRS information.')
            mesh.crs = None
        else:
            if not isinstance(mesh.crs, CRS):
                raise ValueError(f'crs property must be of type {CRS}, not '
                                 f'type {type(mesh.crs)}.')

        self._msh_t = mesh

    def write(self, path: Union[str, os.PathLike], overwrite: bool = False,
              format='grd'):
        path = pathlib.Path(path)
        if path.exists() and overwrite is not True:
            raise IOError(
                f'File {str(path)} exists and overwrite is not True.')
        if format == 'grd':
            grd.write(utils.msh_t_to_grd(self.msh_t), path, overwrite)

        elif format == '2dm':
            sms2dm.writer(utils.msh_t_to_2dm(self.msh_t), path, overwrite)

        elif format == 'msh':
            savemsh(self.msh_t, path)

        elif format == 'vtk':
            savevtk(self.msh_t, path)

        else:
            raise ValueError(f'Unhandled format {format}.')

    @property
    def tria3(self):
        return self.msh_t.tria3

    @property
    def triangles(self):
        return self.msh_t.tria3['index']

    @property
    def quad4(self):
        return self.msh_t.quad4

    @property
    def quads(self):
        return self.msh_t.quad4['index']

    @property
    def crs(self):
        return self.msh_t.crs

    @property
    def hull(self):
        if not hasattr(self, '_hull'):
            self._hull = Hull(self)
        return self._hull

    @property
    def nodes(self):
        if not hasattr(self, '_nodes'):
            self._nodes = Nodes(self)
        return self._nodes

    @property
    def elements(self):
        if not hasattr(self, '_elements'):
            self._elements = Elements(self)
        return self._elements


class EuclideanMesh2D(EuclideanMesh):

    def __init__(self, mesh: jigsaw_msh_t):
        super().__init__(mesh)
        if mesh.ndims != +2:
            raise ValueError(f'Argument mesh has property ndims={mesh.ndims}, '
                             "but expected ndims=2.")

        if len(self.msh_t.value) == 0:
            self.msh_t.value = np.array(
                np.full((self.vert2['coord'].shape[0], 1), np.nan))

    def get_bbox(
            self,
            crs: Union[str, CRS] = None,
            output_type: str = None
    ) -> Union[Polygon, Bbox]:
        output_type = 'polygon' if output_type is None else output_type
        xmin, xmax = np.min(self.coord[:, 0]), np.max(self.coord[:, 0])
        ymin, ymax = np.min(self.coord[:, 1]), np.max(self.coord[:, 1])
        crs = self.crs if crs is None else crs
        if crs is not None:
            if not self.crs.equals(crs):
                transformer = Transformer.from_crs(
                    self.crs, crs, always_xy=True)
                (xmin, xmax), (ymin, ymax) = transformer.transform(
                    (xmin, xmax), (ymin, ymax))
        if output_type == 'polygon':
            return box(xmin, ymin, xmax, ymax)
        elif output_type == 'bbox':
            return Bbox([[xmin, ymin], [xmax, ymax]])
        else:
            raise TypeError(
                'Argument output_type must a string literal \'polygon\' or '
                '\'bbox\'')

    def tricontourf(self, **kwargs):
        return utils.tricontourf(self.msh_t, **kwargs)

    def interpolate(self, raster: Union[Raster, List[Raster]],
                    method='nearest', nprocs=None):

        if isinstance(raster, Raster):
            raster = [raster]

        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs

        with Pool(processes=nprocs) as pool:
            res = pool.starmap(
                _mesh_interpolate_worker,
                [(self.vert2['coord'], self.crs,
                    _raster.tmpfile, _raster.chunk_size)
                 for _raster in raster]
                )

        values = self.msh_t.value.flatten()

        for idxs, _values in res:
            values[idxs] = _values

        self.msh_t.value = np.array(values.reshape((values.shape[0], 1)),
                                    dtype=jigsaw_msh_t.REALS_t)

    @property
    def vert2(self):
        return self.msh_t.vert2

    @property
    def value(self):
        return self.msh_t.value

    @property
    def bbox(self):
        return self.get_bbox()


class Mesh(BaseMesh):
    """Mesh factory"""

    def __new__(self, mesh: jigsaw_msh_t):

        if not isinstance(mesh, jigsaw_msh_t):
            raise TypeError(f'Argument mesh must be of type {jigsaw_msh_t}, '
                            f'not type {type(mesh)}.')

        if mesh.mshID == 'euclidean-mesh':
            if mesh.ndims == 2:
                return EuclideanMesh2D(mesh)
            else:
                raise NotImplementedError(
                    f'mshID={mesh.mshID} + mesh.ndims={mesh.ndims} not '
                    'handled.')

        else:
            raise NotImplementedError(f'mshID={mesh.mshID} not handled.')

    @staticmethod
    def open(path, crs=None):
        try:
            return Mesh(utils.grd_to_msh_t(grd.read(path, crs=crs)))
        except Exception as e:
            if 'not a valid grd file' in str(e):
                pass
            else:
                raise e

        try:
            return Mesh(utils.sms2dm_to_msh_t(sms2dm.read(path, crs=crs)))
        except ValueError:
            pass

        try:
            msh_t = jigsaw_msh_t()
            loadmsh(msh_t, path)
            msh_t.crs = crs
            return Mesh(msh_t)
        except Exception:
            pass

        raise TypeError(
            f'Unable to automatically determine file type for {str(path)}.')


def edges_to_rings(edges):
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


def sort_rings(index_rings, vertices):
    """Sorts a list of index-rings.

    Takes a list of unsorted index rings and sorts them into an "exterior" and
    "interior" components. Any doubly-nested rings are considered exterior
    rings.

    TODO: Refactor and optimize. Calls that use :class:matplotlib.path.Path can
    probably be optimized using shapely.
    """

    # sort index_rings into corresponding "polygons"
    areas = list()
    for index_ring in index_rings:
        e0, e1 = [list(t) for t in zip(*index_ring)]
        areas.append(float(Polygon(vertices[e0, :]).area))

    # maximum area must be main mesh
    idx = areas.index(np.max(areas))
    exterior = index_rings.pop(idx)
    areas.pop(idx)
    _id = 0
    _index_rings = dict()
    _index_rings[_id] = {
        'exterior': np.asarray(exterior),
        'interiors': []
    }
    e0, e1 = [list(t) for t in zip(*exterior)]
    path = Path(vertices[e0 + [e0[0]], :], closed=True)
    while len(index_rings) > 0:
        # find all internal rings
        potential_interiors = list()
        for i, index_ring in enumerate(index_rings):
            e0, e1 = [list(t) for t in zip(*index_ring)]
            if path.contains_point(vertices[e0[0], :]):
                potential_interiors.append(i)
        # filter out nested rings
        real_interiors = list()
        for i, p_interior in reversed(
                list(enumerate(potential_interiors))):
            _p_interior = index_rings[p_interior]
            check = [index_rings[k]
                     for j, k in
                     reversed(list(enumerate(potential_interiors)))
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
            _index_rings[_id]['interiors'].append(
                np.asarray(index_rings.pop(i)))
            areas.pop(i)
        # if no internal rings found, initialize next polygon
        if len(index_rings) > 0:
            idx = areas.index(np.max(areas))
            exterior = index_rings.pop(idx)
            areas.pop(idx)
            _id += 1
            _index_rings[_id] = {
                'exterior': np.asarray(exterior),
                'interiors': []
            }
            e0, e1 = [list(t) for t in zip(*exterior)]
            path = Path(vertices[e0 + [e0[0]], :], closed=True)
    return _index_rings


def signed_polygon_area(vertices):
    # https://code.activestate.com/recipes/578047-area-of-polygon-using-shoelace-formula/
    n = len(vertices)  # of vertices
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
        return area / 2.0


def _mesh_interpolate_worker(coords, coords_crs, raster_path, chunk_size):
    coords = np.array(coords)
    raster = Raster(raster_path)
    idxs = []
    values = []
    for window in raster.iter_windows(chunk_size=chunk_size, overlap=2):

        if not raster.crs.equals(coords_crs):
            transformer = Transformer.from_crs(
                    coords_crs, raster.crs, always_xy=True)
            coords[:, 0], coords[:, 1] = transformer.transform(
                coords[:, 0], coords[:, 1])
        xi = raster.get_x(window)
        yi = raster.get_y(window)
        zi = raster.get_values(window=window)
        f = RectBivariateSpline(
            xi,
            np.flip(yi),
            np.flipud(zi).T,
            kx=3, ky=3, s=0,
            # bbox=[min(x), max(x), min(y), max(y)]  # ??
        )
        _idxs = np.where(
            np.logical_and(
                np.logical_and(
                    np.min(xi) <= coords[:, 0],
                    np.max(xi) >= coords[:, 0]),
                np.logical_and(
                    np.min(yi) <= coords[:, 1],
                    np.max(yi) >= coords[:, 1])))[0]
        _values = f.ev(coords[_idxs, 0], coords[_idxs, 1])
        idxs.append(_idxs)
        values.append(_values)

    return (np.hstack(idxs), np.hstack(values))
