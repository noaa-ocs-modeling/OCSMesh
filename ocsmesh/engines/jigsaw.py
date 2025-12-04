from __future__ import annotations
from typing import Any, Optional, Union
import logging

import geopandas as gpd
import numpy as np
from shapely import Polygon, MultiPolygon
try:
    import jigsawpy
    _HAS_JIGSAW = True
except ImportError:
    _HAS_JIGSAW = False

from ocsmesh import utils
from ocsmesh.internal import MeshData
from ocsmesh.engines.base import BaseMeshEngine, BaseMeshOptions


_logger = logging.getLogger(__name__)


class JigsawOptions(BaseMeshOptions):
    """
    Wraps jigsaw_opts_t options.
    """

    def __init__(
            self,
            hfun_marche=False,
            remesh_tiny_elements=False,
            quality_metric=1.05,
            **other_options
    ):
        if not _HAS_JIGSAW:
            raise ImportError("Jigsawpy not installed.")

        self._hfun_marche = hfun_marche
        self._remesh_tiny = remesh_tiny_elements

        # internal storage for options
        self._opts = jigsawpy.jigsaw_jig_t()

        # Set defaults
        self._opts.hfun_scal = "absolute"
        self._opts.mesh_dims = +2
        self._opts.hfun_hmax = float("inf")
        self._opts.hfun_hmin = 0.0
        self._opts.mesh_top1 = False
        self._opts.geom_feat = False

        self._opts.mesh_rad2 = float(quality_metric)

        # Apply user overrides
        for key, value in other_options.items():
            if hasattr(self._opts, key):
                setattr(self._opts, key, value)
            else:
                _logger.warning(
                    f"Unknown Jigsaw option: {key}"
                )

    def get_config(self) -> Any:
        return {'opts': self._opts, 'marche': self._hfun_marche, 'tiny_elem': self._remesh_tiny}


class JigsawEngine(BaseMeshEngine):
    """
    Concrete implementation using JIGSAW.
    """

    def generate(
        self,
        shape: gpd.GeoSeries,
        sizing: Optional[MeshData | int | float] = None,
        seed: Optional[MeshData] = None,
    ) -> MeshData:

        if not _HAS_JIGSAW:
            raise ImportError("Jigsawpy not installed.")

        # Prepare Geometry
        geom = shape_to_msh_t(shape)

        seed_msh = None
        if seed is not None:
            seed_msh = meshdata_to_msh_t(seed)
            seed_msh.vert2['IDtag'][:] = -1
            seed_edges = utils.get_mesh_edges(seed, unique=True)
            seed_msh.edge2 = np.array(
                [(e, -1) for e in seed_edges],
                dtype=jigsawpy.jigsaw_msh_t.EDGE2_t
            )

        # Convert back to MeshData
        return msh_t_to_mesdata(self._jigsaw_mesh(geom, sizing, seed_msh))


    def remesh(
        self,
        mesh: MeshData,
        remesh_region: Optional[gpd.GeoSeries] = None,
        sizing: Optional[MeshData | int | float] = None,
        seed: Optional[MeshData] = None,
    ) -> MeshData:

        if not _HAS_JIGSAW:
            raise ImportError("Jigsawpy not installed.")

        seed_msh = None
        if seed is not None:
            # NOTE: User should make sure seed mesh is compatible with the
            # input mesh and refinement shape!
            seed_msh = meshdata_to_msh_t(seed)
            seed_msh.vert2['IDtag'][:] = -1
            seed_edges = utils.get_mesh_edges(seed, unique=True)
            seed_msh.edge2 = np.array(
                [(e, -1) for e in seed_edges],
                dtype=jigsawpy.jigsaw_msh_t.EDGE2_t
            )

        init_msh = meshdata_to_msh_t(mesh)
        if remesh_region is not None:
            if seed is not None:
                seed_in_roi = utils.clip_mesh_by_shape(
                    seed, remesh_region.union_all(), fit_inside=True, inverse=False)
                seed_msh = meshdata_to_msh_t(seed_in_roi)
                seed_msh.vert2['IDtag'][:] = -1
                # note: add edge2 from elements
                seed_edges = utils.get_mesh_edges(seed_in_roi, unique=True)
                seed_msh.edge2 = np.array(
                    [(e, -1) for e in seed_edges],
                    dtype=jigsawpy.jigsaw_msh_t.EDGE2_t
                )

            mesh_w_hole = utils.clip_mesh_by_shape(
                mesh, remesh_region.union_all(), fit_inside=True, inverse=True)

            if mesh_w_hole.num_nodes == 0:
                err = 'ERROR: remesh shape covers the whole input mesh!'
                _logger.error(err)
                raise RuntimeError(err)

            # Convert MeshData to Jigsaw format (Initial Mesh)
            init_msh = meshdata_to_msh_t(mesh_w_hole)
            init_msh.vert2['IDtag'][:] = -1
            # note: add edge2 from elements
            init_edges = utils.get_mesh_edges(mesh_w_hole, unique=True)
            init_msh.edge2 = np.array(
                [(e, -1) for e in init_edges],
                dtype=jigsawpy.jigsaw_msh_t.EDGE2_t
            )

        if seed_msh is not None:
            seed_idx_offset = len(init_msh.vert2)
            init_msh.vert2 = np.concatenate((init_msh.vert2, seed_msh.vert2))
            init_msh.edge2 = np.concatenate(
                (init_msh.edge2, seed_msh.edge2 + seed_idx_offset)
            )

        # Prepare Geometry from the input mesh
        bdry_edges = utils.get_boundary_edges(mesh)
        bdry_vert_idx = np.unique(bdry_edges.ravel())
        mapping = np.full(mesh.num_nodes, -1)
        mapping[bdry_vert_idx] = np.arange(len(bdry_vert_idx))
        geom_coords = mesh.coords[bdry_vert_idx, :]
        geom_edges = mapping[bdry_edges]

        geom = jigsawpy.jigsaw_msh_t()
        geom.mshID = 'euclidean-mesh'
        geom.ndims = +2
        geom.vert2 = np.array(
            [(c, 0) for c in geom_coords],
            dtype=jigsawpy.jigsaw_msh_t.VERT2_t
        )
        geom.edge2 = np.array(
            [(e, 0) for e in geom_edges],
            dtype=jigsawpy.jigsaw_msh_t.EDGE2_t
        )

        return msh_t_to_mesdata(self._jigsaw_mesh(geom, sizing, init_msh))


    def _jigsaw_mesh(self, geom, sizing, init_mesh=None):

        # Prepare config
        opt_dict = self._options.get_config()

        opts = opt_dict['opts']
        marche = opt_dict['marche']
        tiny_elem = opt_dict['tiny_elem']

        # Prepare Sizing
        hfun = None
        if sizing is not None:
            # TODO: Should sizing override options or options the sizing?
            if isinstance(sizing, MeshData):
                hfun = meshdata_to_msh_t(sizing)

                opts.hfun_hmin = np.min(sizing.values)
                opts.hfun_hmax = np.max(sizing.values)

                if marche is True:
                    jigsawpy.lib.marche(opts, hfun)
            elif isinstance(sizing, (int, float)):
                opts.hfun_hmin = float(sizing)
                opts.hfun_hmax = float(sizing)

        # Prepare Output Container
        mesh = jigsawpy.jigsaw_msh_t()
        mesh.mshID = 'euclidean-mesh'
        mesh.ndims = +2

        # Run Engine
        jigsawpy.lib.jigsaw(
            opts,
            geom,
            mesh,
            init=init_mesh,
            hfun=hfun
        )

        # Post process
        if mesh.tria3['index'].shape[0] == 0:
            err = 'ERROR: Jigsaw returned empty mesh.'
            _logger.error(err)
            raise RuntimeError(err)


        _logger.info('Cleanup jigsaw mesh...')
        if opts.hfun_hmin > 0 and tiny_elem:
            mesh = remesh_small_elements(opts, geom, mesh, hfun)

        _logger.info('done!')

        return mesh



def meshdata_to_msh_t(mesh: MeshData) -> 'jigsawpy.jigsaw_msh_t':
    if not _HAS_JIGSAW:
        raise ImportError("Jigsawpy not installed.")

    msh = jigsawpy.jigsaw_msh_t()
    msh.ndims = +2
    msh.mshID = 'euclidean-mesh'

    # Vertices
    if mesh.coords is not None:
        msh.vert2 = np.array(
            [(c, 0) for c in mesh.coords],
            dtype=jigsawpy.jigsaw_msh_t.VERT2_t
        )

    # Triangles
    if mesh.tria is not None:
        msh.tria3 = np.array(
            [(t, 0) for t in mesh.tria],
            dtype=jigsawpy.jigsaw_msh_t.TRIA3_t
        )

    # Quads
    if mesh.quad is not None:
        msh.quad4 = np.array(
            [(q, 0) for q in mesh.quad],
            dtype=jigsawpy.jigsaw_msh_t.QUAD4_t
        )

    # Values (Scalars)
    if mesh.values is not None:
        vals = mesh.values
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        msh.value = np.array(
            vals, dtype=jigsawpy.jigsaw_msh_t.REALS_t
        )

    return msh


def msh_t_to_mesdata(msh: 'jigsawpy.jigsaw_msh_t') -> MeshData:
    coords = msh.vert2['coord']

    tria = None
    if msh.tria3.size > 0:
        tria = msh.tria3['index']

    quad = None
    if msh.quad4.size > 0:
        quad = msh.quad4['index']

    values = None
    if msh.value.size > 0:
        values = msh.value

    return MeshData(
        coords=coords,
        tria=tria,
        quad=quad,
        values=values,
    )


def shape_to_msh_t(
    shape: Union[Polygon, MultiPolygon]
) -> 'jigsawpy.jigsaw_msh_t':

    if not _HAS_JIGSAW:
        raise ImportError("Jigsawpy not installed.")

    if isinstance(shape, Polygon):
        shape = MultiPolygon([shape])

    vert2_list = []
    edge2_list = []

    # Helper to process a linear ring
    def process_ring(ring, start_idx):
        coords = ring.coords[:-1] # Drop duplicate end point
        n_pts = len(coords)
        if n_pts < 3:
            return 0

        for xy in coords:
            vert2_list.append((xy, 0))

        # Create edges: i -> i+1, and close last -> first
        for i in range(n_pts):
            u = start_idx + i
            v = start_idx + ((i + 1) % n_pts)
            edge2_list.append(((u, v), 0))

        return n_pts

    current_idx = 0
    for idx, poly in shape.explode().geometry.items():
        # Exterior
        n = process_ring(poly.exterior, current_idx)
        current_idx += n

        # Interiors
        for interior in poly.interiors:
            n = process_ring(interior, current_idx)
            current_idx += n

    msh = jigsawpy.jigsaw_msh_t()
    msh.ndims = +2
    msh.mshID = 'euclidean-mesh'

    if vert2_list:
        msh.vert2 = np.array(
            vert2_list,
            dtype=jigsawpy.jigsaw_msh_t.VERT2_t
        )
    if edge2_list:
        msh.edge2 = np.array(
            edge2_list,
            dtype=jigsawpy.jigsaw_msh_t.EDGE2_t
        )

    return msh


def remesh_small_elements(opts, geom, mesh, hfun):


    """
    This function uses all the inputs for a given jigsaw meshing
    process and based on that finds and fixes tiny elements that
    might occur during initial meshing by iteratively remeshing
    """


    # TODO: Implement for quad
    if not _HAS_JIGSAW:
        raise ImportError("Jigsawpy not installed.")


    hmin = np.min(hfun.value)
    equilat_area = np.sqrt(3)/4 * hmin**2
    # List of arbitrary coef of equilateral triangle area for a given
    # minimum mesh size to come up with a decent cut off.
    coeffs = [0.5, 0.2, 0.1, 0.05]


    fixed_mesh = mesh
    for coef in coeffs:
        tria_areas = utils.calculate_tria_areas(fixed_mesh)
        tiny_sz = coef * equilat_area
        tiny_verts = np.unique(fixed_mesh.tria3['index'][tria_areas<tiny_sz,:].ravel())
        if len(tiny_verts) == 0:
            break
        mesh_clip = utils.clip_mesh_by_vertex(fixed_mesh, tiny_verts, inverse=True)

        fixed_mesh = jigsawpy.jigsaw_msh_t()
        fixed_mesh.mshID = 'euclidean-mesh'
        fixed_mesh.ndims = +2

        jigsawpy.lib.jigsaw(
            opts, geom, fixed_mesh, init=mesh_clip, hfun=hfun)

    return fixed_mesh
