import logging
from typing import Union

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
        hfun: BaseHfun,
        init_mesh: Mesh = None,
        crs: Union[str, CRS] = None,
        engine_name: str = 'jigsaw',
        **engine_kwargs
    ):
        """
        Initialize the driver with a specific engine.

        Parameters
        ----------
        goem: BaseGeom
        hfun: BaseHfun
        crs: str, CRS, optional
        engine_name : str
            Name of the engine ('jigsaw', 'triangle', etc).
        **engine_kwargs : dict
            Options to pass to the engine's Option class.
        """


        self._geom = geom
        self._hfun = hfun
        self._init_mesh = init_mesh
        self._crs = CRS.from_user_input(crs) if crs is not None else None
        self._engine = get_mesh_engine(engine_name, **engine_kwargs)


    def run(self, sieve=None) -> Mesh:
        """
        Run a mesh generation job.
        """
        shape = self._geom.geoseries()
        sizing : MeshData = self._hfun.meshdata()

        # When the center of geom and hfun are NOT the same, utm
        # zones would be different for resulting meshdata.
        if shape.crs != sizing.crs:
            utils.reproject(sizing, shape.crs)


        output_mesh : MeshData = self._engine.generate(shape, sizing)
        output_mesh.crs = sizing.crs

        utils.finalize_mesh(output_mesh, sieve)

        if self._crs is not None:
            utils.reproject(output_mesh, self._crs)

        return Mesh(output_mesh)


    def run_remeshing(self) -> Mesh:
        """
        Run a mesh refinement/optimization job.
        """
        raise NotImplementedError()
#        return self._engine.remesh(mesh, shape, sizing)
