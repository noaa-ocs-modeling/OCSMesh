from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from jigsawpy import jigsaw_msh_t
from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import MultiPolygon

from ocsmesh import utils
from ocsmesh.crs import CRS as CRSDescriptor


class BaseGeom(ABC):
    """Abstract base class used to construct ocsmesh "geom" objects.

    More concretely, a "geom" object can be visualized as a collection of
    polygons. In terms of data structures, a collection of polygons can be
    represented as a :class:shapely.geometry.MultiPolygon object, or as a
    :class:jigsawpy.jigsaw_msh_t object.

    A 'geom' object can be visualized as the "hull" of the input object, but
    this should not be confused with the convex hull (the geom object does not
    have to be convex).

    Derived classes from :class:`ocsmesh.geom.BaseGeom` expose the concrete
    implementation of how to compute this hull based on inputs provided by the
    users.
    """

    _crs = CRSDescriptor()

    def __init__(self, crs):
        self._crs = crs

    @property
    def multipolygon(self) -> MultiPolygon:
        """Returns a :class:shapely.geometry.MultiPolygon object representing
        the configured geometry."""
        return self.get_multipolygon()

    def msh_t(self, **kwargs) -> jigsaw_msh_t:
        """Returns a :class:jigsawpy.jigsaw_msh_t object representing the
        geometry constrained by the arguments."""
        return multipolygon_to_jigsaw_msh_t(self.get_multipolygon(**kwargs), self.crs)

    @abstractmethod
    def get_multipolygon(self, **kwargs) -> MultiPolygon:
        """Returns a :class:shapely.geometry.MultiPolygon object representing
        the geometry constrained by the arguments."""
        raise NotImplementedError

    @property
    def crs(self):
        return self._crs


def multipolygon_to_jigsaw_msh_t(multipolygon: MultiPolygon, crs: CRS) -> jigsaw_msh_t:
    """Casts shapely.geometry.MultiPolygon to jigsawpy.jigsaw_msh_t"""
    utm_crs = utils.estimate_bounds_utm(multipolygon.bounds, crs)
    if utm_crs is not None:
        transformer = Transformer.from_crs(crs, utm_crs, always_xy=True)
        multipolygon = ops.transform(transformer.transform, multipolygon)

    vert2: List[Tuple[Tuple[float, float], int]] = []
    for polygon in multipolygon:
        if np.all(np.asarray(polygon.exterior.coords).flatten() == float("inf")):
            raise NotImplementedError("ellispoidal-mesh")
        for x, y in polygon.exterior.coords[:-1]:
            vert2.append(((x, y), 0))
        for interior in polygon.interiors:
            for x, y in interior.coords[:-1]:
                vert2.append(((x, y), 0))

    # edge2
    edge2: List[Tuple[int, int]] = []
    for polygon in multipolygon:
        polygon = [polygon.exterior, *polygon.interiors]
        for linear_ring in polygon:
            _edge2 = []
            for i in range(len(linear_ring.coords) - 2):
                _edge2.append((i, i + 1))
            _edge2.append((_edge2[-1][1], _edge2[0][0]))
            edge2.extend([(e0 + len(edge2), e1 + len(edge2)) for e0, e1 in _edge2])
    # geom
    geom = jigsaw_msh_t()
    geom.ndims = +2
    geom.mshID = "euclidean-mesh"
    # TODO: Consider ellipsoidal case.
    # geom.mshID = 'euclidean-mesh' if self._ellipsoid is None \
    #     else 'ellipsoidal-mesh'
    geom.vert2 = np.asarray(vert2, dtype=jigsaw_msh_t.VERT2_t)
    geom.edge2 = np.asarray(
        [((e0, e1), 0) for e0, e1 in edge2], dtype=jigsaw_msh_t.EDGE2_t
    )
    geom.crs = crs
    if utm_crs is not None:
        geom.crs = utm_crs
    return geom
