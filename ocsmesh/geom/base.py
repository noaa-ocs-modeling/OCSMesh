"""This module defines the base class for all geometry (domain) types
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Union

from jigsawpy import jigsaw_msh_t
import numpy as np
from pyproj import CRS, Transformer
from shapely import ops
from shapely.geometry import MultiPolygon

from ocsmesh.crs import CRS as CRSDescriptor
from ocsmesh import utils


class BaseGeom(ABC):
    """Abstract base class used to construct OCSMesh "geom" objects.

    This abstract class defines the interface that all geometry
    objects are expected to have in OCSMesh. All geometry type
    classes need to inherit this base class.

    Attributes
    ----------
    crs : CRSDescriptor
        Representing the CRS of the geometry's underlying object.
    multipolygon : MultiPolygon
        Lazily calculated `shapely` (multi)polygon of the geometry

    Methods
    -------
    msh_t(**kwargs)
        Returns the `jigsawpy` vertex-edge representation of the geometry
    get_multipolygon(**kwargs)
        Returns `shapely` object representation of the geometry

    Notes
    -----
    A "geom" object represents the domain of meshing (i.e.
    simulation). This domain can be represented either as a `shapely`
    `MultiPolygon` object, or as a `jigsawpy` `jigsaw_msh_t` object.
    """

    _crs = CRSDescriptor()

    def __init__(self, crs: Union[CRS, str, int]) -> None:
        self._crs = crs

    @property
    def multipolygon(self) -> MultiPolygon:
        """Read-only attribute for `shapely` representation of the geometry

        This read-only attribute is calculated lazily and has the same
        value as calling `get_multipolygon` without any arguments.
        """

        return self.get_multipolygon()

    def msh_t(self, **kwargs: Any) -> jigsaw_msh_t:
        """Returns the `jigsawpy` representation of the geometry.

        This method calculates the vertex-edge representation of
        the geometry in the form of `jigsaw_msh_t`. The return value
        is in a projected CRS. If the geometry CRS is geographic, then
        a local UTM CRS is calculated and used for this representation.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments passed to `get_multipolygon` method

        Returns
        -------
        jigsaw_msh_t
            Calculated vertex-edge representation of the geometry
            if a projected or local UTM CRS.

        Notes
        -----
        The output of this method needs to have length unit for
        distances (i.e. not degrees) since mesh size is specified
        in length units and the domain and size function are the
        passed to the mesh engine for cartesian meshing.
        """

        return multipolygon_to_jigsaw_msh_t(
            self.get_multipolygon(**kwargs),
            self.crs
        )

    @abstractmethod
    def get_multipolygon(self, **kwargs: Any) -> MultiPolygon:
        """Calculate the `shapely` representation of the geometry

        This abstract method defines the expected API for a geometry
        object.

        Raises
        ------
        NotImplementedError
            If method is not redefined in the subclass
        """

        raise NotImplementedError

    @property
    def crs(self) -> CRS:
        """Read-only attribute for CRS of the input geometry"""
        return self._crs


def multipolygon_to_jigsaw_msh_t(
        multipolygon: MultiPolygon,
        crs: CRS
    ) -> jigsaw_msh_t:
    """Calculate vertex-edge representation of multipolygon

    Calculate `jigsawpy` vertex-edge representation of the input
    `shapely` multipolygon. The resulting object is in a projected or
    local UTM CRS

    Parameters
    ----------
    multipolygon : MultiPolygon
        Input polygon for which the vertex-edge representation is to
        be calculated
    crs : CRS
        CRS of the input polygon

    Returns
    -------
    jigsaw_msh_t
        Vertex-edge representation of the input multipolygon

    Raises
    ------
    NotImplementedError
    """

    utm_crs = utils.estimate_bounds_utm(
            multipolygon.bounds, crs)
    if utm_crs is not None:
        transformer = Transformer.from_crs(crs, utm_crs, always_xy=True)
        multipolygon = ops.transform(transformer.transform, multipolygon)

    vert2: List[Tuple[Tuple[float, float], int]] = []
    for polygon in multipolygon.geoms:
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
    edge2: List[Tuple[int, int]] = []
    for polygon in multipolygon.geoms:
        polygon = [polygon.exterior, *polygon.interiors]
        for linear_ring in polygon:
            _edge2 = []
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
    geom.crs = crs
    if utm_crs is not None:
        geom.crs = utm_crs
    return geom
