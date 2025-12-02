from typing import Any, Optional, Dict
import logging

import shapely

from ocsmesh.internal import MeshData
from ocsmesh.engines.base import BaseMeshEngine
from ocsmesh.engines.jigsaw import (
    JigsawEngine, JigsawOptions
)
from ocsmesh.engines.triangle import (
    TriangleEngine, TriangleOptions
)
# from ocsmesh.engines.triangle_wrapper import ...

_ENGINES = {
    'jigsaw': (JigsawEngine, JigsawOptions),
    'triangle': (TriangleEngine, TriangleOptions),
}

_logger = logging.getLogger(__name__)

def get_mesh_engine(engine_name: str, **engine_kwargs: Any) -> BaseMeshEngine:
    """
    Initialize the driver with a specific engine.

    Parameters
    ----------
    engine_name : str
        Name of the engine ('jigsaw', 'triangle', etc).
    **engine_kwargs : dict
        Options to pass to the engine's Option class.
    """
    if engine_name not in _ENGINES:
        raise ValueError(
            f"Engine '{engine_name}' not supported. "
            f"Available: {list(_ENGINES.keys())}"
        )

    engine_cls, opts_cls = _ENGINES[engine_name]

    # 1. Create Options Object
    options = opts_cls(**engine_kwargs)

    # 2. Instantiate Engine
    engine: BaseMeshEngine = engine_cls(options)

    return engine
