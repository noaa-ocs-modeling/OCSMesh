import unittest

import numpy as np

from ocsmesh import MeshData


class TestMeshData(unittest.TestCase):

    def setUp(self):
        """Set up some basic data for testing."""
        self.coords_2d = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        self.tria = np.array([[0, 1, 2]])
        self.quad = np.array([[0, 1, 3, 2]])
        # Scalar values for basic setup
        self.values = np.array([10.0, 20.0, 30.0, 40.0])

    def test_initialization_full(self):
        """Test initializing with all arguments provided."""
        mesh = MeshData(
            coords=self.coords_2d,
            tria=self.tria,
            quad=self.quad,
            values=self.values
        )

        self.assertEqual(mesh.num_nodes, 4)
        np.testing.assert_array_equal(mesh.coords, self.coords_2d)
        np.testing.assert_array_equal(mesh.tria, self.tria)
        np.testing.assert_array_equal(mesh.quad, self.quad)
        np.testing.assert_array_equal(mesh.values, self.values)

    def test_initialization_minimal(self):
        """Test initializing with only coordinates (required)."""
        mesh = MeshData(coords=self.coords_2d)

        self.assertEqual(mesh.num_nodes, 4)
        # Should default to empty arrays with correct column counts
        self.assertEqual(mesh.tria.shape, (0, 3))
        self.assertEqual(mesh.quad.shape, (0, 4))
        # Should default to zeros (K,)
        self.assertEqual(mesh.values.shape, (4,))
        self.assertTrue(np.all(mesh.values == 0))

    def test_vector_values(self):
        """Test assigning vector values (K, D) where D > 1."""
        # 2D vector per node (e.g., velocity u, v)
        vector_vals = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0]
        ])
        mesh = MeshData(coords=self.coords_2d, values=vector_vals)
        self.assertEqual(mesh.values.shape, (4, 2))
        np.testing.assert_array_equal(mesh.values, vector_vals)

    def test_explicit_column_vector(self):
        """
        Test assigning (K, 1) values.
        Ensure they remain (K, 1) and are not flattened to (K,),
        allowing explicit distinction between scalar and 1-comp vector.
        """
        col_vec = np.array([[1], [2], [3], [4]])
        mesh = MeshData(coords=self.coords_2d, values=col_vec)
        self.assertEqual(mesh.values.shape, (4, 1))

    def test_vector_preservation_on_resize(self):
        """
        Test that if we have vector values and resize the mesh,
        the new zero-values array maintains the vector dimension.
        """
        # Initialize with vectors (4 nodes, 3 components)
        vector_vals = np.zeros((4, 3))
        mesh = MeshData(coords=self.coords_2d, values=vector_vals)

        # Resize mesh to 2 nodes
        mesh.coords = [[0, 0], [1, 1]]

        self.assertEqual(mesh.num_nodes, 2)
        # Should be (2, 3) - width 3 preserved
        self.assertEqual(mesh.values.shape, (2, 3))
        self.assertTrue(np.all(mesh.values == 0))

    def test_scalar_preservation_on_resize(self):
        """
        Test that if we have scalar values (K,) and resize,
        it remains (NewK,).
        """
        mesh = MeshData(coords=self.coords_2d)  # Default (4,)
        self.assertEqual(mesh.values.ndim, 1)

        # Resize
        mesh.coords = [[0, 0], [1, 1]]
        self.assertEqual(mesh.values.shape, (2,))

    def test_coords_3d_rejected(self):
        """Test that 3D coordinates (x, y, z) are NOT accepted."""
        coords_3d = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0]
        ])
        # Should raise ValueError now
        with self.assertRaisesRegex(ValueError, "Coords must have 2 columns"):
            MeshData(coords=coords_3d)

    def test_values_length_mismatch(self):
        """Test setting values with mismatching length raises ValueError."""
        mesh = MeshData(coords=self.coords_2d)  # 4 nodes

        # Try to set 3 vectors for 4 nodes
        vector_mismatch = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            mesh.values = vector_mismatch

        # Try to set 5 scalars for 4 nodes
        with self.assertRaises(ValueError):
            mesh.values = [1, 2, 3, 4, 5]

    def test_coords_update_logic(self):
        """
        Test that updating coordinates handles values array correctly:
        1. Resizes (resets) values if node count changes.
        2. Preserves values if node count stays same.
        """
        mesh = MeshData(coords=self.coords_2d, values=[1, 2, 3, 4])

        # 1. Update coords with DIFFERENT size (2 nodes instead of 4)
        mesh.coords = [[0, 0], [1, 1]]
        self.assertEqual(mesh.num_nodes, 2)
        # Values should be reset to zeros of length 2
        self.assertEqual(mesh.values.shape, (2,))
        self.assertTrue(np.all(mesh.values == 0))

        # 2. Update coords with SAME size (2 nodes)
        # Set specific values first
        mesh.values = [99, 100]
        # Update coords
        mesh.coords = [[5, 5], [6, 6]]
        # Values should remain [99, 100]
        np.testing.assert_array_equal(mesh.values, [99, 100])

    def test_list_input_compatibility(self):
        """Test that Python lists are correctly converted to numpy arrays."""
        mesh = MeshData(
            coords=[[0, 0], [1, 1]],
            tria=[[0, 1, 0]]
        )
        self.assertIsInstance(mesh.coords, np.ndarray)
        self.assertIsInstance(mesh.tria, np.ndarray)


if __name__ == '__main__':
    unittest.main()
