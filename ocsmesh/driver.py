import logging
import copy
from typing import Union, Optional

from pyproj import CRS

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
            If None, defaults to EPSG:4326.
        engine_name : str
            Name of the engine ('jigsaw', 'triangle', 'gmsh').
        **engine_kwargs : dict
            Options to pass to the engine's Option class.
            Common options:
            - bnd_representation (str): 'exact', 'fixed', or 'adapt'.
        """

        self._geom = geom
        self._hfun = hfun

        # Set default CRS immediately if not provided
        self._crs = CRS.from_user_input(crs) if crs is not None else None

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

        self._engine = get_mesh_engine(engine_name, **engine_kwargs)


    def run(self, sieve=None) -> Mesh:
        """
        Run a mesh generation job.
        """
        shape = self._geom.geoseries()
        calc_crs = shape.crs

        # Track if we auto-projected from Geographic so we can revert later
        is_input_geographic = False
        if calc_crs is not None and calc_crs.is_geographic:
            is_input_geographic = True

        # 1. Project to Metric (UTM) if needed
        # We need metric units for 'adapt' (Hfun is in meters) and for Gmsh
        # to correctly interpret Hfun values vs Edge lengths.
        if is_input_geographic:
            _logger.info("Input is Geographic. Projecting to UTM for processing...")
            calc_crs = utils.estimate_bounds_utm(shape.total_bounds, shape.crs)
            shape = shape.to_crs(calc_crs)
            _logger.info(f"Calculations will be performed in: {calc_crs}")
        elif calc_crs is None:
            # FIXED: Explicitly warn that we assume meters if no CRS is found.
            _logger.warning(
"Input geometry has no CRS. Assuming coordinates are Cartesian (Meters). "
"If coordinates are Degrees, mesh sizing and boundary adaptation will be incorrect."
            )

        # 2. Prepare Sizing (Hfun) early
        # (Needed for adaptation step and engine generation)
        sizing: Optional[MeshData] = None
        if self._hfun is not None:
            sizing = copy.deepcopy(self._hfun.meshdata())

            # Align Hfun CRS with Calculation CRS
            if calc_crs is None:
                if sizing.crs is not None:
                    # If geom has no CRS but hfun does, adopt hfun's CRS for calculation
                    calc_crs = sizing.crs
                    # Update shape to have this CRS so subsequent steps are consistent
                    shape.crs = calc_crs
            elif sizing.crs is None:
                sizing.crs = calc_crs
            elif not sizing.crs.equals(calc_crs):
                _logger.info("Reprojecting sizing field to match calculation CRS...")
                utils.reproject(sizing, calc_crs)

        # 3. Handle Seed CRS
        generation_seed = self._seed_data
        if generation_seed is not None:
            generation_seed = copy.deepcopy(generation_seed)
            seed_crs = generation_seed.crs

            if calc_crs is None:
                # If we don't know the calculation CRS yet, trust the seed
                if seed_crs is not None:
                    calc_crs = seed_crs
                    shape.crs = calc_crs
                    if sizing is not None:
                        sizing.crs = calc_crs
            elif seed_crs is None:
                # If seed has no CRS, assume it matches the calculation frame
                generation_seed.crs = calc_crs
            elif not seed_crs.equals(calc_crs):
                # If both exist and differ, reproject
                _logger.info("Reprojecting seed data to match calculation CRS...")
                utils.reproject(generation_seed, calc_crs)

        # 4. Generate Mesh
        output_mesh: MeshData = self._engine.generate(
            shape,
            sizing,
            seed=generation_seed
        )
        output_mesh.crs = calc_crs

        # 5. Finalize
        utils.finalize_mesh(output_mesh, sieve)

        # 6. Final Projection
        target_crs = self._crs
        if target_crs is None and is_input_geographic:
            target_crs = CRS.from_epsg(4326)

        if target_crs is not None:
            # Only reproject if the current CRS differs from target
            if output_mesh.crs is None:
                output_mesh.crs = target_crs
            elif not output_mesh.crs.equals(target_crs):
                _logger.info(f"Reprojecting final mesh to {target_crs}...")
                utils.reproject(output_mesh, target_crs)

        return Mesh(output_mesh)

    def run_remeshing(self) -> Mesh:
        raise NotImplementedError()
