import logging
from copy import deepcopy
from collections import defaultdict

from jigsawpy import jigsaw_msh_t
import numpy as np
from pyproj import CRS, Transformer
import utm

from geomesh.hfun.base import BaseHfun
from geomesh.crs import CRS as CRSDescriptor


_logger = logging.getLogger(__name__)

class HfunMesh(BaseHfun):

    _crs = CRSDescriptor()

    def __init__(self, mesh):
        self._mesh = mesh
        self._crs = mesh.crs

    def msh_t(self) -> jigsaw_msh_t:
        if self.crs.is_geographic:
            x0, y0, x1, y1 = self.mesh.get_bbox().bounds
            _, _, number, letter = utm.from_latlon(
                    (y0 + y1)/2, (x0 + x1)/2)
            utm_crs = CRS(
                    proj='utm',
                    zone=f'{number}{letter}',
                    ellps={
                        'GRS 1980': 'GRS80',
                        'WGS 84': 'WGS84'
                        }[self.crs.ellipsoid.name]
                )
            transformer = Transformer.from_crs(
                self.crs, utm_crs, always_xy=True)
            # TODO: This modifies the underlying mesh, is this
            # intended?
            self.mesh.msh_t.vert2['coord'] = np.vstack(
                transformer.transform(
                    self.mesh.msh_t.vert2['coord'][:, 0],
                    self.mesh.msh_t.vert2['coord'][:, 1]
                    )).T
            self.mesh.msh_t.crs = utm_crs
            self._crs = utm_crs
            return self.mesh.msh_t
        else:
            return self.mesh.msh_t

    def size_from_mesh(self):

        ''' 
        Get size function values based on the mesh underlying
        this size function. This method overwrites the values
        in underlying msh_t.
        Also note that for calculation coordinates are projected
        to utm, but the projected coordinates are discarded
        '''

        # Make sure it's in utm so that sizes are in meters
        hfun_msh = self.mesh.msh_t
        coord = hfun_msh.vert2['coord']

        if self.crs.is_geographic:

            _logger.info('Projecting to utm...')

            x0, y0, x1, y1 = self.mesh.get_bbox().bounds
            _, _, number, letter = utm.from_latlon(
                    (y0 + y1)/2, (x0 + x1)/2)
            utm_crs = CRS(
                    proj='utm',
                    zone=f'{number}{letter}',
                    ellps={
                        'GRS 1980': 'GRS80',
                        'WGS 84': 'WGS84'
                        }[self.crs.ellipsoid.name]
                )
            transformer = Transformer.from_crs(
                self.crs, utm_crs, always_xy=True)
            # Note self.mesh.msh_t is NOT overwritten as coord is 
            # being reassigned, not modified
            coord = np.vstack(
                transformer.transform(coord[:, 0], coord[:, 1])).T

        # NOTE: For msh_t type vertex id and index are the same
        trias = hfun_msh.tria3['index']
        quads = hfun_msh.quad4['index']
        hexas = hfun_msh.hexa8['index']

        _logger.info('Getting edges...')
        # Get unique set of edges by rolling connectivity
        # and joining connectivities in 3rd dimension, then sorting
        # to get all edges with lower index first
        all_edges = np.empty(shape=(0, 2), dtype=trias.dtype)
        if trias.shape[0]:
            _logger.info('Getting tria edges...')
            edges = np.sort(
                    np.stack(
                        (trias, np.roll(trias, shift=1, axis=1)),
                        axis=2),
                    axis=2)
            edges = np.unique(
                    edges.reshape(np.product(edges.shape[0:2]), 2), axis=0)
            all_edges = np.vstack((all_edges, edges))
        if quads.shape[0]:
            _logger.info('Getting quad edges...')
            edges = np.sort(
                    np.stack(
                        (quads, np.roll(quads, shift=1, axis=1)),
                        axis=2),
                    axis=2)
            edges = np.unique(
                    edges.reshape(np.product(edges.shape[0:2]), 2), axis=0)
            all_edges = np.vstack((all_edges, edges))
        if hexas.shape[0]:
            _logger.info('Getting quad edges...')
            edges = np.sort(
                    np.stack(
                        (hexas, np.roll(hexas, shift=1, axis=1)),
                        axis=2),
                    axis=2)
            edges = np.unique(
                    edges.reshape(np.product(edges.shape[0:2]), 2), axis=0)
            all_edges = np.vstack((all_edges, edges))

        all_edges = np.unique(all_edges, axis=0)

        # ONLY TESTED FOR TRIA FOR NOW

        # This part of the function is generic for tria and quad
        
        # Get coordinates for all edge vertices
        _logger.info('Getting coordinate of edges...')
        edge_coords = coord[all_edges, :]

        # Calculate length of all edges based on acquired coords
        _logger.info('Getting length of edges...')
        edge_lens = np.sqrt(
                np.sum(
                    np.power(
                        np.abs(np.diff(edge_coords, axis=1)), 2)
                    ,axis=2)).squeeze()

        # Calculate the mesh size by getting average of lengths
        # associated with each vertex (note there's not id vs index
        # distinction here). This is the most time consuming section
        # as of 04/21
        _logger.info('Creating vertex to edge map...')
        vert_to_edge = defaultdict(list)
        for e, i in enumerate(all_edges.ravel()):
            vert_to_edge[i].append(e // 2)

        _logger.info('Creating size value array for vertices...')
        vert_value = np.array(
                [np.average(edge_lens[vert_to_edge[i]])
                    if i in vert_to_edge else 0
                        for i in range(coord.shape[0])])

        # NOTE: Modifying values of underlying mesh
        hfun_msh.value = vert_value.reshape(len(vert_value), 1)

    @property
    def hmin(self):
        return np.min(self.mesh.msh_t.value)

    @property
    def hmax(self):
        return np.max(self.mesh.msh_t.value)

    @property
    def mesh(self):
        return self._mesh

    @property
    def crs(self):
        return self._crs

    def get_bbox(self, **kwargs):
        return self.mesh.get_bbox(**kwargs)
