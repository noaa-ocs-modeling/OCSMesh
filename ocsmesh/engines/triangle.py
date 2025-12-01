from typing import Any, Optional, Union, Dict, List
import logging

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

import geopandas as gpd
try:
    import triangle as tr
    _HAS_TRIANGLE = True
except ImportError:
    _HAS_TRIANGLE = False

from ocsmesh.internal import MeshData
from ocsmesh.engines.base import BaseMeshEngine, BaseMeshOptions

_logger = logging.getLogger(__name__)


def _shape_to_triangle_dict(
    shape: Union[Polygon, MultiPolygon]
) -> Dict[str, Any]:
    """
    Convert a Shapely polygon to Triangle library input dict.
    """
    if isinstance(shape, Polygon):
        shape = MultiPolygon([shape])

    if not isinstance(shape, MultiPolygon):
        raise ValueError("Input must be Polygon or MultiPolygon")

    vertices = []
    segments = []
    holes = []

    current_idx = 0

    def process_ring(ring, is_hole=False):
        nonlocal current_idx
        coords = np.array(ring.coords)
        if np.all(coords[0] == coords[-1]):
            coords = coords[:-1]  # Remove duplicate end point

        n_pts = len(coords)
        if n_pts < 3:
            return

        # Add vertices
        for pt in coords:
            vertices.append(pt)

        # Add segments (edges)
        for i in range(n_pts):
            u = current_idx + i
            v = current_idx + ((i + 1) % n_pts)
            segments.append([u, v])

        current_idx += n_pts
        # TODO: Use is_hole value!

    for poly in shape.geoms:
        # Process Exterior
        process_ring(poly.exterior)

        # Process Interiors (Holes topology)
        for interior in poly.interiors:
            process_ring(interior, True)
        
        # Identify hole points (geometric holes)
        # We need a point strictly inside the hole to mark it
        # Triangle uses representative points to identify holes
        world = poly.buffer(np.sqrt(poly.area) * 0.01) # Small buffer
        # A hole in the polygon is 'land' in the negative shape
        # This logic mimics utils.py logic simplified:
        # We assume the user provides a shape where holes are actual
        # empty spaces. Shapely interiors are holes.
        for interior in poly.interiors:
            # Create a small polygon for the hole to find a point
            hole_poly = Polygon(interior)
            # Find a point inside this hole
            rep_pt = hole_poly.representative_point()
            holes.append([rep_pt.x, rep_pt.y])

    data = {
        'vertices': np.array(vertices, dtype=float),
        'segments': np.array(segments, dtype=int)
    }
    if holes:
        data['holes'] = np.array(holes, dtype=float)

    return data


def _meshdata_to_triangle_dict(mesh: MeshData) -> Dict[str, Any]:
    """
    Convert MeshData to Triangle library input dict.
    """
    data = {}
    if mesh.coords is not None:
        data['vertices'] = mesh.coords

    if mesh.tria is not None and mesh.tria.size > 0:
        data['triangles'] = mesh.tria

    # If simple segments are needed (e.g. boundary), they must be
    # calculated. For simple remeshing of an existing mesh,
    # 'vertices' and 'triangles' are often enough for 'r' mode.
    # If using 'p' mode on existing nodes, segments are needed.
    
    # We can infer boundary segments if not provided, but
    # Triangle library usually expects PSLG for 'p'.
    
    return data


def _triangle_dict_to_meshdata(
    data: Dict[str, Any],
) -> MeshData:
    """
    Convert Triangle library output dict to MeshData.
    """
    coords = data.get('vertices')
    tria = data.get('triangles')
    
    # Triangle outputs segments/edges too, usually we ignore
    # unless we want to store them.
    
    return MeshData(
        coords=coords,
        tria=tria,
    )


class TriangleOptions(BaseMeshOptions):
    """
    Wraps options for the Triangle library.
    """

    def __init__(self, opts: str='', **kwargs):
        if not _HAS_TRIANGLE:
            raise ImportError("Triangle library not installed.")
        
        self._opts = opts
        
        # Allow appending/modifying opts via kwargs if needed
        # e.g. min_angle=30 -> append 'q30'
        if 'min_angle' in kwargs:
            self._opts += f"q{kwargs['min_angle']}"
        
        if 'max_area' in kwargs:
            self._opts += f"a{kwargs['max_area']}"

    def get_config(self) -> str:
        return self._opts


class TriangleEngine(BaseMeshEngine):
    """
    Concrete implementation using Triangle.
    """

    def generate(
        self,
        shape: gpd.GeoSeries,
        sizing: Optional[MeshData | int | float] = None,
        seed: Optional[MeshData] = None,
    ) -> MeshData:
        
        if not _HAS_TRIANGLE:
            raise ImportError("Triangle library not installed.")

        # 1. Prepare Input
        shape_dict = _shape_to_triangle_dict(shape)

        seed_dict = None
        if seed is not None:
            seed.tria = []
            seed.quad = []
            seed_dict = _meshdata_to_triangle_dict(seed)

        input_dict = {}
        input_dict['vertices'] = shape_dict['vertices'].copy()
        if seed_dict is not None:
            input_dict['vertices'] = np.concatenate(
                (input_dict['vertices'], seed_dict['vertices'])
            )
        input_dict['segments'] = shape_dict['segments']

        
        # 2. Handle Sizing
        # Triangle uses 'a' switch for global area, or a separate
        # refinement function.
        # If sizing is a scalar float, we can append 'a{value}'
        opts = self._options.get_config()
        
        if isinstance(sizing, (float, int)):
            # If 'a' is not already in opts with a value
            if 'a' not in opts or 'a' == opts[-1]:
                 opts = opts.replace('a', '') + f"a{sizing}"
        
        # 3. Run Engine
        # 'p' is usually required for PSLG input (vertices + segments)
        if 'p' not in opts and 'segments' in input_dict:
            opts += 'p'
            
        out_dict = tr.triangulate(input_dict, opts)
        
        # 4. Convert Output
        # Preserve CRS from shape if available?
        # Shape usually doesn't carry CRS in raw form,
        # usually managed externally.
        return _triangle_dict_to_meshdata(out_dict)


    def remesh(
        self,
        mesh: MeshData,
        shape: Optional[gpd.GeoSeries] = None,
        sizing: Optional[MeshData | int | float] = None,
        seed: Optional[MeshData] = None,
    ) -> MeshData:
        
        if not _HAS_TRIANGLE:
            raise ImportError("Triangle library not installed.")

        if mesh.quad is not None:
            raise NotImplementedError("Triangle does not support quads!")

        # Prepare Input
        bdry_edges = utils.get_boundary_edges(mesh)
        bdry_vert_idx = np.unique(mesh_edges.ravel())
        mapping = np.full(mesh.num_nodes, -1)
        mapping[bdry_vert_idx] = np.range(len(bdry_vert_idx))
        geom_coords = mesh.coords[bdry_vert_idx, :]
        geom_edges = mapping[bdry_edges]

        shape_dict = {
            'vertices': np.array(geom_coords, dtype=float),
            'segments': np.array(geom_edges, dtype=int)
        }

        seed_dict = None
        if seed is not None:
            seed.tria = []
            seed.quad = []
            seed_dict = _meshdata_to_triangle_dict(seed)

        init_dict = _meshdata_to_triangle_dict(mesh)
        is shape is not None:
            if seed is not None:
                seed_in_roi = utils.clip_mesh_by_shape(
                    seed, shape.union_all(), fit_inside=True, inverse=False)
                seed_dict = _meshdata_to_triangle_dict(seed_in_roi)

            mesh_w_hole = utils.clip_mesh_by_shape(
                mesh, shape.union_all(), fit_inside=True, inverse=True)

            if mesh_w_hole.num_nodes == 0:
                err = 'ERROR: refinement shape covers the whole input mesh!'
                _logger.error(err)
                raise RuntimeError(err)

            # Convert MeshData to Jigsaw format (Initial Mesh)
            init_dict = _meshdata_to_triangle_dict(mesh_w_hole)


        input_dict = {}
        input_dict['vertices'] = np.concatenate(
            (shape_dict['vertices'], init_dict['vertices'])
        )
        if seed_dict is not None:
            input_dict['vertices'] = np.concatenate(
                (input_dict['vertices'], seed_dict['vertices'])
            )
        input_dict['segments'] = shape_dict['segments']
        input_dict['triangles'] = init_dict['triangles'] + len(shape_dict['vertices'])

        
        # Handle Options
        # TODO: For remeshing/refining, 'r' switch is often used
#        opts = self._options.get_config()
#        if 'r' not in opts:
#            opts += 'r'

        # Handle Sizing
        if isinstance(sizing, (float, int)):
             if 'a' not in opts or 'a' == opts[-1]:
                 opts = opts.replace('a', '') + f"a{sizing}"

        # Run Engine
        out_dict = tr.triangulate(input_dict, opts)
        
        return _triangle_dict_to_meshdata(out_dict)
