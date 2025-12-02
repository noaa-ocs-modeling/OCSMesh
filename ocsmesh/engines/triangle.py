from typing import Any, Optional, Union, Dict, List
import logging

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box

import pandas as pd
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

    gs_shape = gpd.GeoSeries(shape)

    if np.sum(gs_shape.area) == 0:
        raise ValueError("The shape must have an area, such as Polygon!")

    if not np.all(gs_shape.is_valid):
        raise ValueError("Input contains invalid (multi)polygons!")

    df_lonlat = (
        gs_shape
        .boundary
        .explode(ignore_index=True)
        .map(lambda i: i.coords)
        .explode()
        .apply(lambda s: pd.Series({'lon': s[0], 'lat': s[1]}))
        .reset_index() # put polygon index in the dataframe
        .drop_duplicates() # drop duplicates within polygons
    )

    df_seg = (
        df_lonlat.join(
            df_lonlat.groupby('index').transform(np.roll, 1, axis=0),
            lsuffix='_1',
            rsuffix='_2'
        ).dropna()
        .set_index('index')
    )

    df_nodes = (
        df_lonlat.drop(columns='index') # drop to allow cross poly dupl
        .drop_duplicates() # drop duplicates across multiple polygons
        .reset_index(drop=True) # renumber nodes
        .reset_index() # add node idx as df data column
        .set_index(['lon','lat'])
    )

    # CRD Table
    df_coo = df_nodes.reset_index().drop(columns='index')

    # CNN Table
    df_edg = (
        df_nodes.loc[
            pd.MultiIndex.from_frame(df_seg[['lon_1', 'lat_1']])
        ].reset_index(drop=True)
        .join(
            df_nodes.loc[
                pd.MultiIndex.from_frame(df_seg[['lon_2', 'lat_2']])
            ]
            .reset_index(drop=True),
            lsuffix='_1',
            rsuffix='_2'
        )
    )


    ar_edg = np.sort(df_edg.values, axis=1)
    df_cnn = (
        pd.DataFrame.from_records(
            ar_edg,
            columns=['index_1', 'index_2'],
        )
        .drop_duplicates() # Remove duplicate edges
        .reset_index(drop=True)
    )

    vertices = df_coo[['lon', 'lat']].values
    segments = df_cnn.values

        
    # To make sure islands formed by two polygons touching at multiple
    # points are also considered as holes, instead of getting interiors,
    # we get the negative of the domain and calculate points in all
    # negative polygons!

    # buffer by 1/100 of shape length scale = sqrt(area)/100
    holes = []
    world = box(*shape.buffer(np.sqrt(shape.area) / 100).bounds)
    neg_shape = world.difference(shape)
    if isinstance(neg_shape, Polygon):
        neg_shape = MultiPolygon([neg_shape])

    for poly in neg_shape.geoms:
        holes.append(np.array(poly.representative_point().xy).ravel())

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
        shape_dict = _shape_to_triangle_dict(shape.union_all())

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
        remesh_region: Optional[gpd.GeoSeries] = None,
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
        if remesh_region is not None:
            if seed is not None:
                seed_in_roi = utils.clip_mesh_by_shape(
                    seed, remesh_region.union_all(), fit_inside=True, inverse=False)
                seed_dict = _meshdata_to_triangle_dict(seed_in_roi)

            mesh_w_hole = utils.clip_mesh_by_shape(
                mesh, remesh_region.union_all(), fit_inside=True, inverse=True)

            if mesh_w_hole.num_nodes == 0:
                err = 'ERROR: remesh shape covers the whole input mesh!'
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
