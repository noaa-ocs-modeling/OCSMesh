from abc import ABC, abstractmethod
from typing import List, Tuple

from jigsawpy import jigsaw_msh_t  # type: ignore[import]
import numpy as np  # type: ignore[import]
from shapely.geometry import MultiPolygon  # type: ignore[import]

from geomesh.logger import Logger


class BaseGeom(ABC):
    '''Abstract base class used to construct geomesh "geom" objects.

    More concretely, a "geom" object can be visualized as a collection of
    polygons. In terms of data structures, a collection of polygons can be
    represented as a :class:shapely.geometry.MultiPolygon object, or as a
    :class:jigsawpy.jigsaw_msh_t object.

    A 'geom' object can be visualized as the "hull" of the input object, but
    this should not be confused with the convex hull (the geom object does not
    have to be convex).

    Derived classes from :class:`geomesh.geom.BaseGeom` expose the concrete
    implementation of how to compute this hull based on inputs provided by the
    users.
    '''

    logger = Logger()

    @property
    def geom(self) -> jigsaw_msh_t:
        '''Returns a :class:jigsawpy.jigsaw_msh_t object representing the
        currently configured geometry.'''
        return self.get_jigsaw_msh_t()

    @property
    def multipolygon(self) -> MultiPolygon:
        '''Returns a :class:shapely.geometry.MultiPolygon object representing
        the configured geometry.'''
        return self.get_multipolygon()

    def get_jigsaw_msh_t(self, **kwargs) -> jigsaw_msh_t:
        '''Returns a :class:jigsawpy.jigsaw_msh_t object representing the
        geometry constrained by the arguments.'''
        return multipolygon_to_jigsaw_msh_t(self.get_multipolygon(**kwargs))

    @abstractmethod
    def get_multipolygon(self, **kwargs) -> MultiPolygon:
        '''Returns a :class:shapely.geometry.MultiPolygon object representing
        the geometry constrained by the arguments.'''
        raise NotImplementedError


def multipolygon_to_jigsaw_msh_t(multipolygon: MultiPolygon) -> jigsaw_msh_t:
    '''Casts shapely.geometry.MultiPolygon to jigsawpy.jigsaw_msh_t'''
    vert2: List[Tuple[Tuple[float, float], int]] = list()
    for polygon in multipolygon:
        if np.all(
                np.asarray(
                    polygon.exterior.coords).flatten() == float('inf')):
            raise NotImplementedError("ellispoidal-mesh")
        for x, y in polygon.exterior.coords[:-1]:
            vert2.append(((x, y), 0))
        for interior in polygon.interiors:
            for x, y in interior.coords[:-1]:
                vert2.append(((x, y), 0))

    # edge2
    edge2: List[Tuple[int, int]] = list()
    for polygon in multipolygon:
        polygon = [polygon.exterior, *polygon.interiors]
        for linear_ring in polygon:
            _edge2 = list()
            for i in range(len(linear_ring.coords)-2):
                _edge2.append((i, i+1))
            _edge2.append((_edge2[-1][1], _edge2[0][0]))
            edge2.extend(
                [(e0+len(edge2), e1+len(edge2))
                    for e0, e1 in _edge2])
    # geom
    geom = jigsaw_msh_t()
    geom.ndims = +2
    geom.mshID = 'euclidean-mesh'
    # TODO: Consider ellipsoidal case.
    # geom.mshID = 'euclidean-mesh' if self._ellipsoid is None \
    #     else 'ellipsoidal-mesh'
    geom.vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)
    geom.edge2 = np.asarray(
        [((e0, e1), 0) for e0, e1 in edge2],
        dtype=jigsaw_msh_t.EDGE2_t)
    return geom
