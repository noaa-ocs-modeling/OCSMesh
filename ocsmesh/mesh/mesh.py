import logging
import os
import pathlib
import warnings
from collections import defaultdict
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from typing import List, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from jigsawpy import jigsaw_msh_t, loadmsh, savemsh, savevtk
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from matplotlib.tri import Triangulation
from pyproj import CRS, Transformer
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import linemerge, polygonize

from ocsmesh import utils
from ocsmesh.mesh.base import BaseMesh
from ocsmesh.mesh.parsers import grd, sms2dm
from ocsmesh.raster import Raster

_logger = logging.getLogger(__name__)

class Rings:

    def __init__(self, mesh: 'EuclideanMesh'):
        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self):

        polys = utils.get_mesh_polygons(self.mesh.msh_t)

        data = []
        bnd_id = 0
        for poly in polys:
            data.append({
                    "geometry": poly.exterior,
                    "bnd_id": bnd_id,
                    "type": 'exterior'
                })
            for interior in poly.interiors:
                data.append({
                    "geometry": interior,
                    "bnd_id": bnd_id,
                    "type": 'interior'
                })
            bnd_id = bnd_id + 1
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
        triangles = self.mesh.msh_t.tria3['index'].tolist()
        for quad in self.mesh.msh_t.quad4['index']:
            triangles.extend([
                [quad[0], quad[1], quad[3]],
                [quad[1], quad[2], quad[3]]
            ])
        return Triangulation(self.mesh.coord[:, 0], self.mesh.coord[:, 1], triangles)



class Nodes:

    def __init__(self, mesh: "EuclideanMesh"):
        self.mesh = mesh
        self._id_to_index = None
        self._index_to_id = None

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

    def get_index_by_id(self, node_id):
        return self.id_to_index[node_id]

    def get_id_by_index(self, index: int):
        return self.index_to_id[index]

    @property
    def id_to_index(self):
        if self._id_to_index is None:
            self._id_to_index = {node_id: index for index, node_id
                                 in enumerate(self().keys())}
        return self._id_to_index

    @property
    def index_to_id(self):
        if self._index_to_id is None:
            self._index_to_id = dict(enumerate(self().keys()))
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
        for elem_id, element in self().items():
            data.append({
                'geometry': Polygon(
                    self.mesh.coord[list(
                        map(self.mesh.nodes.get_index_by_id, element))]),
                'id': elem_id})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)


class Boundaries:

    def __init__(self, mesh: "Mesh"):
        # TODO: Add a way to manually initialize
        self.mesh = mesh
        self._ocean = gpd.GeoDataFrame()
        self._land = gpd.GeoDataFrame()
        self._interior = gpd.GeoDataFrame()
        self._data = defaultdict(defaultdict)

    @lru_cache(maxsize=1)
    def _init_dataframes(self):
        boundaries = self._data
        ocean_boundaries = []
        land_boundaries = []
        interior_boundaries = []
        if boundaries is not None:
            for ibtype, bnds in boundaries.items():
                if ibtype is None:
                    for bnd_id, data in bnds.items():
                        indexes = list(map(self.mesh.nodes.get_index_by_id,
                                       data['indexes']))
                        ocean_boundaries.append({
                            'id': bnd_id,
                            "index_id": data['indexes'],
                            "indexes": indexes,
                            'geometry': LineString(self.mesh.coord[indexes])
                            })

                elif str(ibtype).endswith('1'):
                    for bnd_id, data in bnds.items():
                        indexes = list(map(self.mesh.nodes.get_index_by_id,
                                       data['indexes']))
                        interior_boundaries.append({
                            'id': bnd_id,
                            'ibtype': ibtype,
                            "index_id": data['indexes'],
                            "indexes": indexes,
                            'geometry': LineString(self.mesh.coord[indexes])
                            })
                else:
                    for bnd_id, data in bnds.items():
                        _indexes = np.array(data['indexes'])
                        if _indexes.ndim > 1:
                            # ndim > 1 implies we're dealing with an ADCIRC
                            # mesh that includes boundary pairs, such as weir
                            new_indexes = []
                            for i, line in enumerate(_indexes.T):
                                if i % 2 != 0:
                                    new_indexes.extend(np.flip(line))
                                else:
                                    new_indexes.extend(line)
                            _indexes = np.array(new_indexes).flatten()
                        else:
                            _indexes = _indexes.flatten()
                        indexes = list(map(self.mesh.nodes.get_index_by_id,
                                       _indexes))

                        land_boundaries.append({
                            'id': bnd_id,
                            'ibtype': ibtype,
                            "index_id": data['indexes'],
                            "indexes": indexes,
                            'geometry': LineString(self.mesh.coord[indexes])
                            })

        self._ocean = gpd.GeoDataFrame(ocean_boundaries)
        self._land = gpd.GeoDataFrame(land_boundaries)
        self._interior = gpd.GeoDataFrame(interior_boundaries)

    def ocean(self):
        self._init_dataframes()
        return self._ocean

    def land(self):
        self._init_dataframes()
        return self._land

    def interior(self):
        self._init_dataframes()
        return self._interior

    @property
    def data(self):
        return self._data

    @lru_cache(maxsize=1)
    def __call__(self):
        self._init_dataframes()
        data = []
        for bnd in self.ocean().itertuples():
            data.append({
                'id': bnd.id,
                'ibtype': None,
                "index_id": bnd.index_id,
                "indexes": bnd.indexes,
                'geometry': bnd.geometry})

        for bnd in self.land().itertuples():
            data.append({
                'id': bnd.id,
                'ibtype': bnd.ibtype,
                "index_id": bnd.index_id,
                "indexes": bnd.indexes,
                'geometry': bnd.geometry})

        for bnd in self.interior().itertuples():
            data.append({
                'id': bnd.id,
                'ibtype': bnd.ibtype,
                "index_id": bnd.index_id,
                "indexes": bnd.indexes,
                'geometry': bnd.geometry})

        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def __len__(self):
        return len(self())

    def auto_generate(
            self,
            threshold=0.,
            land_ibtype=0,
            interior_ibtype=1,
            ):

        values = self.mesh.value
        if np.any(np.isnan(values)):
            raise Exception(
                "Mesh contains invalid values. Raster values must"
                "be interpolated to the mesh before generating "
                "boundaries.")


        coords = self.mesh.msh_t.vert2['coord']
        coo_to_idx = {
            tuple(coo): idx
            for idx, coo in enumerate(coords)}

        polys = utils.get_mesh_polygons(self.mesh.msh_t)

        # TODO: Split using shapely to get bdry segments

        boundaries = defaultdict(defaultdict)
        bdry_type = dict

        get_id = self.mesh.nodes.get_id_by_index
        # generate exterior boundaries
        for poly in polys:
            ext_ring_coo = poly.exterior.coords
            ext_ring = np.array([
                    (coo_to_idx[ext_ring_coo[e]],
                     coo_to_idx[ext_ring_coo[e + 1]])
                    for e, coo in enumerate(ext_ring_coo[:-1])])

            # find boundary edges
            edge_tag = np.full(ext_ring.shape, 0)
            edge_tag[
                np.where(values[ext_ring[:, 0]] < threshold)[0], 0] = -1
            edge_tag[
                np.where(values[ext_ring[:, 1]] < threshold)[0], 1] = -1
            edge_tag[
                np.where(values[ext_ring[:, 0]] >= threshold)[0], 0] = 1
            edge_tag[
                np.where(values[ext_ring[:, 1]] >= threshold)[0], 1] = 1
            # sort boundary edges
            ocean_boundary = []
            land_boundary = []
            for i, (e0, e1) in enumerate(edge_tag):
                if np.any(np.asarray((e0, e1)) == 1):
                    land_boundary.append(tuple(ext_ring[i, :]))
                elif np.any(np.asarray((e0, e1)) == -1):
                    ocean_boundary.append(tuple(ext_ring[i, :]))
#            ocean_boundaries = utils.sort_edges(ocean_boundary)
#            land_boundaries = utils.sort_edges(land_boundary)
            ocean_boundaries = []
            if len(ocean_boundary) != 0:
                #pylint: disable=not-an-iterable
                ocean_segs = linemerge(coords[np.array(ocean_boundary)])
                ocean_segs = [ocean_segs] if isinstance(ocean_segs, LineString) else ocean_segs
                ocean_boundaries = [
                        [(coo_to_idx[seg.coords[e]], coo_to_idx[seg.coords[e + 1]])
                         for e, coo in enumerate(seg.coords[:-1])]
                        for seg in ocean_segs]
            land_boundaries = []
            if len(land_boundary) != 0:
                #pylint: disable=not-an-iterable
                land_segs = linemerge(coords[np.array(land_boundary)])
                land_segs = [land_segs] if isinstance(land_segs, LineString) else land_segs
                land_boundaries = [
                        [(coo_to_idx[seg.coords[e]], coo_to_idx[seg.coords[e + 1]])
                         for e, coo in enumerate(seg.coords[:-1])]
                        for seg in land_segs]

            _bnd_id = len(boundaries[None])
            for bnd in ocean_boundaries:
                e0, e1 = [list(t) for t in zip(*bnd)]
                e0 = [get_id(vert) for vert in e0]
                data = e0 + [get_id(e1[-1])]
                boundaries[None][_bnd_id] = bdry_type(
                        indexes=data, properties={})
                _bnd_id += 1

            # add land boundaries
            _bnd_id = len(boundaries[land_ibtype])
            for bnd in land_boundaries:
                e0, e1 = [list(t) for t in zip(*bnd)]
                e0 = [get_id(vert) for vert in e0]
                data = e0 + [get_id(e1[-1])]
                boundaries[land_ibtype][_bnd_id] = bdry_type(
                        indexes=data, properties={})

                _bnd_id += 1

        # generate interior boundaries
        _bnd_id = 0
        interior_boundaries = defaultdict()
        for poly in polys:
            interiors = poly.interiors
            for interior in interiors:
                int_ring_coo = interior.coords
                int_ring = [
                        (coo_to_idx[int_ring_coo[e]],
                         coo_to_idx[int_ring_coo[e + 1]])
                        for e, coo in enumerate(int_ring_coo[:-1])]

                # TODO: Do we still need these?
                e0, e1 = [list(t) for t in zip(*int_ring)]
                if utils.signed_polygon_area(self.mesh.coord[e0, :]) < 0:
                    e0 = e0[::-1]
                    e1 = e1[::-1]
                e0 = [get_id(vert) for vert in e0]
                e0.append(e0[0])
                interior_boundaries[_bnd_id] = e0
                _bnd_id += 1

        for bnd_id, data in interior_boundaries.items():
            boundaries[interior_ibtype][bnd_id] = bdry_type(
                        indexes=data, properties={})

        self._data = boundaries
        self._init_dataframes.cache_clear()
        self.__call__.cache_clear()
        self._init_dataframes()


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

        self._hull = None
        self._nodes = None
        self._elements = None
        self._msh_t = mesh

    def write(self,
              path: Union[str, os.PathLike],
              overwrite: bool = False,
              format='grd', # pylint: disable=W0622
              ):
        path = pathlib.Path(path)
        if path.exists() and overwrite is not True:
            raise IOError(
                f'File {str(path)} exists and overwrite is not True.')
        if format == 'grd':
            grd_dict = utils.msh_t_to_grd(self.msh_t)
            if self._boundaries and self._boundaries.data:
                grd_dict.update(boundaries=self._boundaries.data)
            grd.write(grd_dict, path, overwrite)

        elif format == '2dm':
            sms2dm.writer(utils.msh_t_to_2dm(self.msh_t), path, overwrite)

        elif format == 'msh':
            savemsh(str(path), self.msh_t)

        elif format == 'vtk':
            savevtk(str(path), self.msh_t)

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
        if self._hull is None:
            self._hull = Hull(self)
        return self._hull

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = Nodes(self)
        return self._nodes

    @property
    def elements(self):
        if self._elements is None:
            self._elements = Elements(self)
        return self._elements


class EuclideanMesh2D(EuclideanMesh):

    def __init__(self, mesh: jigsaw_msh_t):
        super().__init__(mesh)
        self._boundaries = None

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
                # pylint: disable=E0633
                (xmin, xmax), (ymin, ymax) = transformer.transform(
                    (xmin, xmax), (ymin, ymax))
        if output_type == 'polygon': # pylint: disable=R1705
            return box(xmin, ymin, xmax, ymax)
        elif output_type == 'bbox':
            return Bbox([[xmin, ymin], [xmax, ymax]])

        raise TypeError(
            'Argument output_type must a string literal \'polygon\' or '
            '\'bbox\'')

    @property
    def boundaries(self):
        if self._boundaries is None:
            self._boundaries = Boundaries(self)
        return self._boundaries

    def tricontourf(self, **kwargs):
        return utils.tricontourf(self.msh_t, **kwargs)

    def interpolate(self, raster: Union[Raster, List[Raster]],
                    method='spline', nprocs=None):

        if isinstance(raster, Raster):
            raster = [raster]

        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs

        # Fix an issue on Jupyter notebook where having pool execute
        # interpolation even in case of nprocs == 1 would results in
        # application getting stuck
        if nprocs > 1:
            with Pool(processes=nprocs) as pool:
                res = pool.starmap(
                    _mesh_interpolate_worker,
                    [(self.vert2['coord'], self.crs,
                        _raster.tmpfile, _raster.chunk_size, method)
                     for _raster in raster]
                    )
            pool.join()
        else:
            res = [_mesh_interpolate_worker(
                        self.vert2['coord'], self.crs,
                        _raster.tmpfile, _raster.chunk_size, method)
                   for _raster in raster]

        values = self.msh_t.value.flatten()

        for idxs, _values in res:
            values[idxs] = _values

        self.msh_t.value = np.array(values.reshape((values.shape[0], 1)),
                                    dtype=jigsaw_msh_t.REALS_t)


    def get_contour(self, level: float):

        # ONLY SUPPORTS TRIANGLES
        for attr in ['quad4', 'hexa8']:
            if len(getattr(self.msh_t, attr)) > 0:
                warnings.warn(
                    'Mesh contour extraction only supports triangles')

        coords = self.msh_t.vert2['coord']
        values = self.msh_t.value
        trias = self.msh_t.tria3['index']
        if np.any(np.isnan(values)):
            raise Exception(
                "Mesh contains invalid values. Raster values must"
                "be interpolated to the mesh before generating "
                "boundaries.")

        x, y = coords[:, 0], coords[:, 1]
        features = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            _logger.debug('Computing contours...')
            fig, ax = plt.subplots()
            ax.tricontour(
                x, y, trias, values.ravel(), levels=[level])
            plt.close(fig)
        for path_collection in ax.collections:
            for path in path_collection.get_paths():
                try:
                    features.append(LineString(path.vertices))
                except ValueError:
                    # LineStrings must have at least 2 coordinate tuples
                    pass
        return linemerge(features)


    def get_multipolygon(self, zmin=None, zmax=None):

        values = self.msh_t.value
        mask = np.ones(values.shape)
        if zmin is not None:
            mask = np.logical_and(mask, values > zmin)
        if zmax is not None:
            mask = np.logical_and(mask, values < zmax)

        # Assuming value is of shape (N, 1)
        # ravel to make sure it's 1D
        verts_in = np.argwhere(mask).ravel()

        clipped_mesh = utils.clip_mesh_by_vertex(
            self.msh_t, verts_in,
            can_use_other_verts=True)

        boundary_edges = utils.get_boundary_edges(clipped_mesh)
        coords = clipped_mesh.vert2['coord']
        coo_to_idx = {
            tuple(coo): idx
            for idx, coo in enumerate(coords)}
        poly_gen = polygonize(coords[boundary_edges])
        polys = list(poly_gen)
        polys = sorted(polys, key=lambda p: p.area, reverse=True)

        rings = [p.exterior for p in polys]
        n_parents = np.zeros((len(rings),))
        represent = np.array([r.coords[0] for r in rings])
        for e, ring in enumerate(rings[:-1]):
            path = Path(ring, closed=True)
            n_parents = n_parents + np.pad(
                np.array([
                    path.contains_point(pt) for pt in represent[e+1:]]),
                (e+1, 0), 'constant', constant_values=0)

        # Get actual polygons based on logic described above
        polys = [p for e, p in enumerate(polys) if not n_parents[e] % 2]

        return MultiPolygon(polys)

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

    def __new__(cls, mesh: jigsaw_msh_t):

        if not isinstance(mesh, jigsaw_msh_t):
            raise TypeError(f'Argument mesh must be of type {jigsaw_msh_t}, '
                            f'not type {type(mesh)}.')

        if mesh.mshID == 'euclidean-mesh':
            if mesh.ndims == 2:
                return EuclideanMesh2D(mesh)

            raise NotImplementedError(
                f'mshID={mesh.mshID} + mesh.ndims={mesh.ndims} not '
                'handled.')

        raise NotImplementedError(f'mshID={mesh.mshID} not handled.')

    @staticmethod
    def open(path, crs=None):
        try:
            msh_t = utils.grd_to_msh_t(grd.read(path, crs=crs))
            msh_t.value = np.negative(msh_t.value)
            return Mesh(msh_t)
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




def sort_rings(index_rings, vertices):
    """Sorts a list of index-rings.

    Takes a list of unsorted index rings and sorts them into an "exterior" and
    "interior" components. Any doubly-nested rings are considered exterior
    rings.

    TODO: Refactor and optimize. Calls that use :class:matplotlib.path.Path can
    probably be optimized using shapely.
    """

    # sort index_rings into corresponding "polygons"
    areas = []
    for index_ring in index_rings:
        e0, e1 = [list(t) for t in zip(*index_ring)]
        areas.append(float(Polygon(vertices[e0, :]).area))

    # maximum area must be main mesh
    idx = areas.index(np.max(areas))
    exterior = index_rings.pop(idx)
    areas.pop(idx)
    _id = 0
    _index_rings = {}
    _index_rings[_id] = {
        'exterior': np.asarray(exterior),
        'interiors': []
    }
    e0, e1 = [list(t) for t in zip(*exterior)]
    path = Path(vertices[e0 + [e0[0]], :], closed=True)
    while len(index_rings) > 0:
        # find all internal rings
        potential_interiors = []
        for i, index_ring in enumerate(index_rings):
            e0, e1 = [list(t) for t in zip(*index_ring)]
            if path.contains_point(vertices[e0[0], :]):
                potential_interiors.append(i)
        # filter out nested rings
        real_interiors = []
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



def _mesh_interpolate_worker(
        coords,
        coords_crs,
        raster_path,
        chunk_size,
        method):
    coords = np.array(coords)
    raster = Raster(raster_path)
    idxs = []
    values = []
    for window in raster.iter_windows(chunk_size=chunk_size, overlap=2):

        if not raster.crs.equals(coords_crs):
            transformer = Transformer.from_crs(
                    coords_crs, raster.crs, always_xy=True)
            # pylint: disable=E0633
            coords[:, 0], coords[:, 1] = transformer.transform(
                coords[:, 0], coords[:, 1])
        xi = raster.get_x(window)
        yi = raster.get_y(window)
        # Use masked array to ignore missing values from DEM
        zi = raster.get_values(window=window, masked=True)

        _idxs = np.logical_and(
            np.logical_and(
                np.min(xi) <= coords[:, 0],
                np.max(xi) >= coords[:, 0]),
            np.logical_and(
                np.min(yi) <= coords[:, 1],
                np.max(yi) >= coords[:, 1]))

        # Inspired by StackOverflow 35807321
        interp_mask = None
        if np.any(zi.mask):
            m_interp = RegularGridInterpolator(
                (xi, np.flip(yi)),
                np.flipud(zi.mask).T.astype(bool),
                method=method
            )
            # Pick nodes NOT "contaminated" by masked values
            interp_mask = m_interp(coords[_idxs]) > 0

        if method == 'spline':
            f = RectBivariateSpline(
                xi,
                np.flip(yi),
                np.flipud(zi).T,
                kx=3, ky=3, s=0,
                # bbox=[min(x), max(x), min(y), max(y)]  # ??
            )
            _values = f.ev(coords[_idxs, 0], coords[_idxs, 1])

        elif method in ['nearest', 'linear']:
            f = RegularGridInterpolator(
                (xi, np.flip(yi)),
                np.flipud(zi).T,
                method=method
            )
            _values = f(coords[_idxs])

        else:
            raise ValueError(
                    f"Invalid value method specified <{method}>!")

        if interp_mask is not None:
            # pylint: disable=invalid-unary-operand-type

            helper = np.ones_like(_values).astype(bool)
            helper[interp_mask] = False
            # _idxs is inverse mask
            _idxs[_idxs] = helper
            _values = _values[~interp_mask]
        idxs.append(_idxs)
        values.append(_values)

    return (np.hstack(idxs), np.hstack(values))
