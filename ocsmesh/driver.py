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
        engine_name: str = 'jigsaw',
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
            Accepts high-level Mesh objects (BaseMesh subclasses) or low-level MeshData.
        crs: str, CRS, optional
            Coordinate reference system for the output.
        engine_name : str
            Name of the engine ('jigsaw', 'triangle', etc).
        **engine_kwargs : dict
            Options to pass to the engine's Option class.
        """

        self._geom = geom

        # 1. Validate hfun
        if hfun is not None and not isinstance(hfun, BaseHfun):
            raise TypeError(
                f"hfun must be an instance of BaseHfun or None, "
                f"got {type(hfun).__name__}"
            )
        self._hfun = hfun

        # 2. Validate and Normalize init_mesh (used as seed)
        self._seed_data = None
        if init_mesh is not None:
            if isinstance(init_mesh, MeshData):
                self._seed_data = init_mesh
            elif hasattr(init_mesh, 'meshdata'):
                # Handle BaseMesh and its subclasses (EuclideanMesh, EuclideanMesh2D)
                # The property defined in base.py is 'meshdata'
                self._seed_data = init_mesh.meshdata
            else:
                raise TypeError(
                    f"init_mesh must be of type MeshData or have a 'meshdata' property "
                    f"(like ocsmesh.mesh.Mesh), got {type(init_mesh).__name__}"
                )

        self._crs = CRS.from_user_input(crs) if crs is not None else None

        # Initialize Engine
        self._engine = get_mesh_engine(engine_name, **engine_kwargs)


    def run(self, sieve=None) -> Mesh:
        """
        Run a mesh generation job.
        """
        shape = self._geom.geoseries()
        calc_crs = shape.crs

        # Handle optional sizing
        sizing: Optional[MeshData] = None
        if self._hfun is not None:
            # TODO: Not the best memory management practice!
            # deepcopy in case we use HfunMesh to avoid mutation
            sizing = copy.deepcopy(self._hfun.meshdata())

            # When the center of geom and hfun are NOT the same, utm
            # zones would be different for resulting meshdata.
            if calc_crs is None:
                if sizing.crs is not None:
                    calc_crs = sizing.crs
            elif sizing.crs is None:
                sizing.crs = calc_crs
            elif not sizing.crs.equals(calc_crs):
                utils.reproject(sizing, calc_crs)

        # Handle Seed CRS mismatch
        # We assume shape has the "correct" projection (likely UTM from BaseGeom)
        generation_seed = self._seed_data

        if generation_seed is not None:
            # Determine the CRS of the seed data
            # If explicit CRS is missing, assume it matches the input geometry's CRS
            # Create a deepcopy to avoid side-effects on the user's original object
            generation_seed = copy.deepcopy(generation_seed)
            seed_crs = generation_seed.crs
            if calc_crs is None:
                if seed_crs is not None:
                    calc_crs = seed_crs
                    shape = shape.set_crs(calc_crs)
                    sizing.crs = calc_crs
            elif seed_crs is None:
                generation_seed.crs = calc_crs
            elif not seed_crs.equals(calc_crs):
                # If reprojection is needed

                _logger.info("Reprojecting seed data to match geometry CRS...")
                utils.reproject(generation_seed, calc_crs)

        # Pass the prepared seed to generate
        output_mesh: MeshData = self._engine.generate(
            shape,
            sizing,
            seed=generation_seed
        )
        output_mesh.crs = calc_crs

        utils.finalize_mesh(output_mesh, sieve)

        if self._crs is not None:
            if output_mesh.crs is None:
                output_mesh.crs = self._crs
            elif not output_mesh.crs.equals(self._crs):
                utils.reproject(output_mesh, self._crs)

        return Mesh(output_mesh)


    def run_remeshing(self) -> Mesh:
        """
        Run a mesh refinement/optimization job.
        """
        # Note: If remeshing logic is implemented later,
        # self._seed_data can be used here as well if needed.
        raise NotImplementedError()
