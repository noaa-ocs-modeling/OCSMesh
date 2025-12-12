import logging
import copy
from typing import Union, Optional

import numpy as np
from pyproj import CRS
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from scipy.spatial import cKDTree

from ocsmesh import utils
from ocsmesh.internal import MeshData
from ocsmesh.engines.factory import get_mesh_engine
from ocsmesh.hfun.base import BaseHfun
from ocsmesh.geom.base import BaseGeom
from ocsmesh.mesh import Mesh


_logger = logging.getLogger(__name__)


class MeshDriver:
    """
    High-level API for running meshing jobs.
    """

    def __init__(
        self,
        geom: BaseGeom,
        hfun: Optional[BaseHfun] = None,
        init_mesh: Optional[Union[Mesh, MeshData]] = None,
        crs: Union[str, CRS] = None,
        engine_name: str = 'gmsh',
        bnd_representation: str = 'fixed',
        **engine_kwargs
    ):
        """
        Initialize the driver with a specific engine.

        Parameters
        ----------
        geom: BaseGeom
            The geometry definition for the domain.
        hfun: BaseHfun, optional
            The size function definition. Must be instance of BaseHfun or None.
        init_mesh: Mesh or MeshData, optional
            Initial mesh or points to be used as a seed.
        crs: str, CRS, optional
            Coordinate reference system for the output.
        engine_name : str
            Name of the engine ('jigsaw', 'triangle', 'gmsh').
            Defaults to 'gmsh'.
        bnd_representation : str, default='fixed'
            - 'exact': Mesh boundary matches input segments exactly (locked).
            - 'fixed': Mesh boundary matches input vertices, but engine can refine edges.
            - 'adapt': Input boundary is RESAMPLED in Python to match the hfun resolution 
                       (coarsening or refining) before meshing.
        **engine_kwargs : dict
            Options to pass to the engine's Option class.
        """

        self._geom = geom
        self._hfun = hfun
        self._crs = CRS.from_user_input(crs) if crs is not None else None

        # Validate bnd_representation
        valid_reps = ['exact', 'fixed', 'adapt']
        if bnd_representation not in valid_reps:
            raise ValueError(f"bnd_representation must be {valid_reps}, got '{bnd_representation}'")
        self._bnd_representation = bnd_representation

        # Validate hfun
        if hfun is not None and not isinstance(hfun, BaseHfun):
            raise TypeError(f"hfun must be BaseHfun, got {type(hfun).__name__}")

        # Validate init_mesh
        self._seed_data = None
        if init_mesh is not None:
            if isinstance(init_mesh, MeshData):
                self._seed_data = init_mesh
            elif hasattr(init_mesh, 'meshdata'):
                self._seed_data = init_mesh.meshdata
            else:
                raise TypeError("init_mesh must be MeshData or have 'meshdata' prop")

        # Initialize Engine 
        # Note: If 'adapt' is chosen, we handle the adaptation in the driver, 
        # and tell the engine to treat the result as 'fixed' (respect the new nodes).
        engine_opt_bnd = 'fixed' if bnd_representation == 'adapt' else bnd_representation
        engine_kwargs['bnd_representation'] = engine_opt_bnd
        
        self._engine = get_mesh_engine(engine_name, **engine_kwargs)


    def run(self, sieve=None) -> Mesh:
        """
        Run a mesh generation job.
        """
        shape = self._geom.geoseries()
        calc_crs = shape.crs

        # 1. Project to Metric (UTM) if needed
        # Essential for 'adapt' so that Hfun meters match geometry meters
        if calc_crs is not None and calc_crs.is_geographic:
             _logger.info("Input is Geographic. Projecting to UTM for processing...")
             calc_crs = utils.estimate_bounds_utm(shape.total_bounds, shape.crs)
             shape = shape.to_crs(calc_crs)
             _logger.info(f"Calculations will be performed in: {calc_crs}")

        # 2. Prepare Sizing (Hfun) early (needed for adaptation)
        sizing: Optional[MeshData] = None
        if self._hfun is not None:
            sizing = copy.deepcopy(self._hfun.meshdata())
            # Align Hfun CRS with Calculation CRS
            if calc_crs is None:
                if sizing.crs is not None:
                    calc_crs = sizing.crs
            elif sizing.crs is None:
                sizing.crs = calc_crs
            elif not sizing.crs.equals(calc_crs):
                _logger.info("Reprojecting sizing field to match calculation CRS...")
                utils.reproject(sizing, calc_crs)

        # 3. Handle 'adapt' mode (Resample Geometry based on Hfun)
        if self._bnd_representation == 'adapt':
            if sizing is None:
                _logger.warning("'adapt' boundary requested but no Hfun provided. Skipping adaptation.")
            else:
                _logger.info("Adapting boundary geometry to match Hfun resolution...")
                shape = self._resample_geom_by_hfun(shape, sizing)

        # 4. Handle Seed CRS
        generation_seed = self._seed_data
        if generation_seed is not None:
            generation_seed = copy.deepcopy(generation_seed)
            seed_crs = generation_seed.crs
            if seed_crs is not None and not seed_crs.equals(calc_crs):
                 _logger.info("Reprojecting seed data to match calculation CRS...")
                 utils.reproject(generation_seed, calc_crs)

        # 5. Generate Mesh
        output_mesh: MeshData = self._engine.generate(
            shape,
            sizing,
            seed=generation_seed
        )
        output_mesh.crs = calc_crs

        # 6. Finalize
        utils.finalize_mesh(output_mesh, sieve)

        # 7. Final Projection
        target_crs = self._crs
        if target_crs is None:
             target_crs = CRS.from_user_input("EPSG:4326")

        if output_mesh.crs is None:
            output_mesh.crs = target_crs
        elif not output_mesh.crs.equals(target_crs):
            _logger.info(f"Reprojecting final mesh to {target_crs}...")
            utils.reproject(output_mesh, target_crs)

        return Mesh(output_mesh)

    def _resample_geom_by_hfun(self, shape_series, hfun_data):
        """
        Resamples polygons in the GeoSeries by 'walking' the boundary
        with step sizes determined by the local Hfun value.
        
        Tiny islands (holes) that collapse below 3 vertices are removed.
        """
        # Build KDTree for fast spatial lookup of Hfun values
        tree = cKDTree(hfun_data.coords)
        values = hfun_data.values.ravel()

        def get_local_size(x, y):
            # Find nearest Hfun point
            # optimization: query 1 point
            dist, idx = tree.query([x, y], k=1)
            return values[idx]

        def resample_ring(ring):
            if ring.is_empty: return None  # Empty rings are invalid
            
            new_coords = []
            # Start at the beginning
            curr_dist = 0.0
            total_length = ring.length
            
            # Add first point
            start_pt = ring.interpolate(0.0)
            new_coords.append((start_pt.x, start_pt.y))
            
            while curr_dist < total_length:
                # 1. Get current position
                curr_pt = new_coords[-1]
                
                # 2. Look up target size at this location
                h = get_local_size(curr_pt[0], curr_pt[1])
                
                # 3. Advance
                next_dist = curr_dist + h
                
                if next_dist >= total_length:
                    break
                
                # 4. Interpolate new point along the original geometry
                # This ensures we stick to the shape shape
                next_pt = ring.interpolate(next_dist)
                new_coords.append((next_pt.x, next_pt.y))
                
                curr_dist = next_dist
            
            # Close the loop
            if len(new_coords) < 3:
                # Island is smaller than local resolution (collapsed to line/point)
                # Return None to signal removal
                return None
            
            new_coords.append(new_coords[0]) # Close ring
            return LinearRing(new_coords)

        # Apply to all polygons in the series
        new_geoms = []
        for poly in shape_series:
            if isinstance(poly, Polygon):
                # 1. Resample Exterior
                new_ext = resample_ring(poly.exterior)
                
                # If the main continent/boundary collapses, we have a bigger problem.
                # Usually we want to keep it even if small, or user might want it gone.
                # For safety, let's keep exterior unless it's truly broken.
                if new_ext is None: 
                    # Fallback: keep original if it collapsed? 
                    # Or discard? Let's assume we discard to be consistent.
                    continue 

                # 2. Resample Islands (Interiors)
                new_ints = []
                for inter in poly.interiors:
                    resampled_island = resample_ring(inter)
                    if resampled_island is not None:
                        new_ints.append(resampled_island)
                
                new_geoms.append(Polygon(new_ext, new_ints))

            elif isinstance(poly, MultiPolygon):
                parts = []
                for p in poly.geoms:
                    # Resample Exterior
                    new_ext = resample_ring(p.exterior)
                    if new_ext is None:
                        continue # Skip this part of the MultiPolygon

                    # Resample Islands
                    new_ints = []
                    for inter in p.interiors:
                        resampled_island = resample_ring(inter)
                        if resampled_island is not None:
                            new_ints.append(resampled_island)
                    
                    parts.append(Polygon(new_ext, new_ints))
                
                if parts:
                    new_geoms.append(MultiPolygon(parts))
            else:
                new_geoms.append(poly) # Pass through unknown types

        # Return new GeoSeries with same CRS
        import geopandas as gpd
        return gpd.GeoSeries(new_geoms, crs=shape_series.crs)

    def run_remeshing(self) -> Mesh:
        raise NotImplementedError()
