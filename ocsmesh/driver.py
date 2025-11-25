from typing import Any, Optional, Dict

import shapely

from ocsmesh import utils
from ocsmesh.internal import MeshData
from ocsmesh.engines.base import BaseMeshEngine
from ocsmesh.engines.jigsaw import (
    JigsawEngine, JigsawOptions
)
# from ocsmesh.engines.triangle_wrapper import ...


_logger = logging.getLogger(__name__)


class MeshDriver:
    """
    High-level API for running meshing jobs.
    """

    _ENGINES = {
        'jigsaw': (JigsawEngine, JigsawOptions),
        # 'triangle': (TriangleEngine, TriangleOptions),
    }

    def __init__(
        self,
        engine_name: str,
        **engine_kwargs
    ):
        """
        Initialize the driver with a specific engine.

        Parameters
        ----------
        engine_name : str
            Name of the engine ('jigsaw', 'triangle', etc).
        **engine_kwargs : dict
            Options to pass to the engine's Option class.
        """
        if engine_name not in self._ENGINES:
            raise ValueError(
                f"Engine '{engine_name}' not supported. "
                f"Available: {list(self._ENGINES.keys())}"
            )

        engine_cls, opts_cls = self._ENGINES[engine_name]

        # 1. Create Options Object
        self._options = opts_cls(**engine_kwargs)

        # 2. Instantiate Engine
        self._engine: BaseMeshEngine = engine_cls(self._options)


    def run_generation(
        self,
        shape: shapely.Geometry,
        sizing: Optional[Any] = None
    ) -> MeshData:
        """
        Run a mesh generation job.
        """
        return self._engine.generate(shape, sizing)

    def run_remeshing(
        self,
        mesh: MeshData,
        shape: Optional[shapely.Geometry] = None,
        sizing: Optional[Any] = None
    ) -> MeshData:
        """
        Run a mesh refinement/optimization job.

        Parameters
        ----------
        mesh : MeshData
            Input mesh.
        shape : Any, optional
            Constraint shape or region for remeshing.
        sizing : Any, optional
            Sizing field.
        """
        return self._engine.remesh(mesh, shape, sizing)


# --- Helper Function for Quick Usage ---

def mesh(
    shape: Any,
    engine: str = 'jigsaw',
    sizing: Optional[Any] = None,
    pre: bool = True,
    **kwargs
) -> MeshData:
    """
    One-shot function to generate a mesh.
    """
    driver = MeshDriver(engine, **kwargs)
    meshdata = driver.run_generation(shape, sizing)
    if not pre:
        utils.finalize_mesh(meshdata, sieve)
        mesh = Mesh(meshdata)
        return mesh

    return meshdata
