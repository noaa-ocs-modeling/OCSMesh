from typing import List, Optional, Union

import numpy as np


class MeshData:
    """
    A simple container for mesh data to replace jigsawpy_msh_t in OCSMesh.

    Parameters
    ----------
    coords : array_like
        Nodal coordinates with shape (K, 2).
    tria : array_like, optional
        Connectivity array for triangles with shape (N, 3). If None, an empty
        array is initialized.
    quad : array_like, optional
        Connectivity array for quads with shape (M, 4). If None, an empty
        array is initialized.
    values : array_like, optional
        Nodal values/attributes with shape (K,) for scalars or (K, D) for
        vectors. If None, initialized to (K,) zeros (scalar default).

    Attributes
    ----------
    coords : ndarray
        Nodal coordinates with shape (K, 2).
    tria : ndarray
        Triangle connectivity array with shape (N, 3).
    quad : ndarray
        Quad connectivity array with shape (M, 4).
    values : ndarray
        Nodal values/attributes with shape (K,) or (K, D).
    num_nodes : int
        Number of nodes in the mesh.
    """

    def __init__(
        self,
        coords: Union[np.ndarray, List],
        tria: Optional[Union[np.ndarray, List]] = None,
        quad: Optional[Union[np.ndarray, List]] = None,
        values: Optional[Union[np.ndarray, List]] = None
    ):
        # Initialize internal storage variables
        self._coords = None
        self._tria = None
        self._quad = None
        self._values = None

        # Set Coordinates (Required)
        # Uses the setter defined below to handle validation immediately
        # This will also initialize self._values to zeros of the correct size
        self.coords = coords

        # Initialize Triangles (Optional, defaults to empty)
        if tria is not None:
            self.tria = tria
        else:
            self._tria = np.empty((0, 3), dtype=int)

        # Initialize Quads (Optional, defaults to empty)
        if quad is not None:
            self.quad = quad
        else:
            self._quad = np.empty((0, 4), dtype=int)

        # Initialize Values (Optional)
        # If provided, overwrite the default zeros created by coords setter
        if values is not None:
            self.values = values

    def __repr__(self):
        # Improve repr to show vector dimension if applicable
        val_shape = self.values.shape if self.values is not None else (0,)
        return (
            f"<MeshData: "
            f"{self.num_nodes} Nodes, "
            f"{self.tria.shape[0]} Triangles, "
            f"{self.quad.shape[0]} Quads, "
            f"Values Shape: {val_shape}>"
        )

    @property
    def num_nodes(self) -> int:
        """
        The number of nodes in the mesh.

        Returns
        -------
        int
            The count of rows in the coordinates array.
        """
        if self._coords is not None:
            return self._coords.shape[0]
        return 0

    # ==========================
    # Coordinates Property
    # ==========================
    @property
    def coords(self) -> np.ndarray:
        """
        Nodal coordinates array.

        Returns
        -------
        ndarray
            Array of shape (K, 2).

        Raises
        ------
        ValueError
            If assigned array is not 2-dimensional or does not have exactly
            2 columns.
        """
        return self._coords

    @coords.setter
    def coords(self, new_coords: Union[np.ndarray, List]):
        arr = np.array(new_coords, dtype=float)

        # Validation: Must be 2D array
        if arr.ndim != 2:
            raise ValueError("Coords array must be 2-dimensional.")

        # Validation: Must have exactly 2 columns (x, y)
        if arr.shape[1] != 2:
            raise ValueError(
                f"Coords must have 2 columns, got {arr.shape[1]}."
            )

        self._coords = arr

        # Enforce consistency: Ensure values array matches new node count
        if self._values is None or self._values.shape[0] != self.num_nodes:
            # If values exist and are 2D (vectors), preserve width
            if self._values is not None and self._values.ndim == 2:
                width = self._values.shape[1]
                self._values = np.zeros((self.num_nodes, width), dtype=float)
            else:
                # Default scalar case (K,)
                self._values = np.zeros((self.num_nodes,), dtype=float)

    # ==========================
    # Triangles Property
    # ==========================
    @property
    def tria(self) -> np.ndarray:
        """
        Triangle connectivity array.

        Returns
        -------
        ndarray
            Array of shape (N, 3).

        Raises
        ------
        ValueError
            If assigned array does not have 3 columns.
        """
        return self._tria

    @tria.setter
    def tria(self, new_tria: Union[np.ndarray, List]):
        arr = np.array(new_tria, dtype=int)

        # Allow empty assignment
        if arr.size == 0:
            self._tria = np.empty((0, 3), dtype=int)
            return

        # Validation: Must have 3 columns
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"Triangles must be shape (N, 3), got shape {arr.shape}."
            )

        self._tria = arr

    # ==========================
    # Quads Property
    # ==========================
    @property
    def quad(self) -> np.ndarray:
        """
        Quad connectivity array.

        Returns
        -------
        ndarray
            Array of shape (M, 4).

        Raises
        ------
        ValueError
            If assigned array does not have 4 columns.
        """
        return self._quad

    @quad.setter
    def quad(self, new_quad: Union[np.ndarray, List]):
        arr = np.array(new_quad, dtype=int)

        # Allow empty assignment
        if arr.size == 0:
            self._quad = np.empty((0, 4), dtype=int)
            return

        # Validation: Must have 4 columns
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(
                f"Quads must be shape (N, 4), got shape {arr.shape}."
            )

        self._quad = arr

    # ==========================
    # Values Property
    # ==========================
    @property
    def values(self) -> np.ndarray:
        """
        Nodal values array.

        Returns
        -------
        ndarray
            Array of shape (K,) for scalars or (K, D) for vectors.

        Raises
        ------
        ValueError
            If the length (first dimension) of the values array does not
            match `num_nodes`.
        """
        return self._values

    @values.setter
    def values(self, new_values: Union[np.ndarray, List]):
        arr = np.array(new_values, dtype=float)

        # Validation: Check length against number of nodes
        if self._coords is not None and arr.shape[0] != self.num_nodes:
            raise ValueError(
                f"Values length {arr.shape[0]} does not match number of "
                f"nodes {self.num_nodes}."
            )

        self._values = arr
