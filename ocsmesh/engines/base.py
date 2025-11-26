from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import geopandas as gpd

from ocsmesh.internal import MeshData


class BaseMeshOptions(ABC):
    """
    Abstract base class for mesh engine options.
    """

    @abstractmethod
    def get_config(self) -> Any:
        """
        Returns the configuration object or structure
        required by the specific backend engine.
        """
        pass


class BaseMeshEngine(ABC):
    """
    Abstract base class for a mesh generation engine.
    """

    def __init__(self, options: BaseMeshOptions):
        """
        Initialize the engine with specific options.
        """
        self._options = options

    @abstractmethod
    def generate(
        self,
        shape: gpd.GeoSeries,
        sizing: Optional[MeshData] = None
    ) -> MeshData:
        """
        Generate a new mesh from a geometric shape.

        Parameters
        ----------
        shape : Any
            The input geometry (e.g., Polygon, MultiPolygon).
        sizing : Any, optional
            The sizing function or field.

        Returns
        -------
        MeshData
            The generated mesh.
        """
        pass

    @abstractmethod
    def remesh(
        self,
        mesh: MeshData,
        shape: Optional[gpd.GeoSeries] = None,
        sizing: Optional[MeshData] = None
    ) -> MeshData:
        """
        Refine or optimize an existing mesh.

        Parameters
        ----------
        mesh : MeshData
            The input mesh to be modified.
        shape : Any, optional
            The region to remesh. If None, remesh entire mesh.
        sizing : Any, optional
            The sizing function or field.

        Returns
        -------
        MeshData
            The resulting mesh.
        """
        pass
