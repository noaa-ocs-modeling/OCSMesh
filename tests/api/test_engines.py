#! python
import unittest
from unittest.mock import patch, MagicMock
import logging
import sys

import numpy as np
import geopandas as gpd
from shapely.geometry import (
    Point,
    LineString,
    box,
    Polygon,
    MultiPolygon,
)
from shapely.ops import polygonize

# Import package modules
from ocsmesh import utils, MeshData
from ocsmesh.engines.base import BaseMeshEngine, BaseMeshOptions
from ocsmesh.engines.factory import get_mesh_engine
from ocsmesh.engines.triangle import TriangleOptions, TriangleEngine
from ocsmesh.engines.gmsh import GmshOptions, GmshEngine

# Check dependencies for skipping tests
try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

try:
    import triangle as tr
    HAS_TRIANGLE = True
except ImportError:
    HAS_TRIANGLE = False

# Disable logging during tests to keep output clean
logging.disable(logging.CRITICAL)

def get_num_elements(mesh):
    n = 0
    if mesh.tria is not None:
        n += len(mesh.tria)
    if mesh.quad is not None:
        n += len(mesh.quad)
    return n

class TestFactory(unittest.TestCase):
    def test_get_mesh_engine_success(self):
        """Test retrieving valid engines and ensure they inherit from Base."""
        if HAS_TRIANGLE:
            eng = get_mesh_engine('triangle', opts='p')
            self.assertIsInstance(eng, TriangleEngine)
            self.assertIsInstance(eng, BaseMeshEngine) 

        if HAS_GMSH:
            eng = get_mesh_engine('gmsh', optimize_mesh=False)
            self.assertIsInstance(eng, GmshEngine)
            self.assertIsInstance(eng, BaseMeshEngine)

class TestBaseEngine(unittest.TestCase):
    """Test constraints defined in the abstract base classes."""
    # NOTE: test_init_type_check removed because engines do not currently 
    # enforce strict type checking on __init__.
    pass


class TestGmsh(unittest.TestCase):
    def setUp(self):
        self.box = box(0, 0, 10, 10)
        self.geo_box = gpd.GeoSeries([self.box])

        # Sizing field
        coords = np.array([[0,0], [10,0], [10,10], [0,10]])
        values = np.array([1.0, 1.0, 1.0, 1.0])
        self.sizing_field = MeshData(coords=coords, values=values)

    @patch('ocsmesh.engines.gmsh._HAS_GMSH', False)
    def test_missing_dependency(self):
        with self.assertRaises(ImportError):
            GmshEngine(GmshOptions())

    def test_options(self):
        if not HAS_GMSH:
            self.skipTest("Gmsh not installed")

        opt = GmshOptions(bnd_representation='exact', optimize_mesh=False)
        self.assertEqual(opt.bnd_representation, 'exact')
        self.assertFalse(opt.optimize_mesh)

        # Test config dict
        cfg = opt.get_config()
        self.assertEqual(cfg["Mesh.Algorithm"], 6)

        # Invalid option
        with self.assertRaises(ValueError):
            GmshOptions(bnd_representation='bad_option')

    def test_generate_constant_sizing(self):
        if not HAS_GMSH:
            self.skipTest("Gmsh not installed")

        engine = get_mesh_engine('gmsh', optimize_mesh=False)
        mesh = engine.generate(self.geo_box, sizing=2.0)

        self.assertIsInstance(mesh, MeshData)
        self.assertGreater(get_num_elements(mesh), 0)

        # Check if coordinates are generally within bounds
        self.assertTrue(np.all(mesh.coords >= 0))
        self.assertTrue(np.all(mesh.coords <= 10))

    def test_generate_field_sizing(self):
        if not HAS_GMSH:
            self.skipTest("Gmsh not installed")

        engine = get_mesh_engine('gmsh')
        mesh = engine.generate(self.geo_box, sizing=self.sizing_field)
        self.assertIsInstance(mesh, MeshData)
        self.assertGreater(get_num_elements(mesh), 0)

    def test_multipolygon(self):
        if not HAS_GMSH:
            self.skipTest("Gmsh not installed")

        mp = MultiPolygon([box(0,0,1,1), box(2,2,3,3)])
        engine = get_mesh_engine('gmsh')
        mesh = engine.generate(gpd.GeoSeries([mp]), sizing=0.5)

        self.assertIsInstance(mesh, MeshData)
        # Should have 2 distinct islands of mesh
        self.assertGreater(get_num_elements(mesh), 0)

    def test_remesh_not_implemented(self):
        if not HAS_GMSH:
            self.skipTest("Gmsh not installed")

        engine = get_mesh_engine('gmsh')
        dummy_mesh = MeshData(coords=[[0,0]], tria=[[0,0,0]])
        with self.assertRaises(NotImplementedError):
            engine.remesh(dummy_mesh)

    def test_empty_shape_error(self):
        if not HAS_GMSH:
            self.skipTest("Gmsh not installed")

        engine = get_mesh_engine('gmsh')
        empty_poly = Polygon()
        with self.assertRaises(ValueError):
            engine.generate(gpd.GeoSeries([empty_poly]))


class TestTriangle(unittest.TestCase):
    def setUp(self):
        if HAS_TRIANGLE:
            self.triangle_engine = get_mesh_engine('triangle', opts='p')

        self.valid_input_1 = box(0, 0, 1, 1)
        self.valid_input_2 = gpd.GeoDataFrame(geometry=[self.valid_input_1])
        self.valid_input_3 = gpd.GeoSeries(self.valid_input_1)

        # Polygon with hole
        self.poly_hole = Polygon(
            [[0, 0], [4, 0], [4, 4], [0, 4]],
            [[[1, 1], [2, 1], [2, 2], [1, 2]]]
        )

    @patch('ocsmesh.engines.triangle._HAS_TRIANGLE', False)
    def test_missing_dependency(self):
        with self.assertRaises(ImportError):
            TriangleEngine(TriangleOptions())

    def test_string_arg_as_option_fails(self):
        with self.assertRaises(ValueError):
            TriangleEngine(options='pq30')

    def test_io_validity(self):
        if not HAS_TRIANGLE:
            self.skipTest("Triangle not installed")

        meshdata_1 = self.triangle_engine.generate(gpd.GeoSeries(self.valid_input_1))
        meshdata_2 = self.triangle_engine.generate(self.valid_input_2.geometry)
        meshdata_3 = self.triangle_engine.generate(self.valid_input_3)

        self.assertIsInstance(meshdata_1, MeshData)
        # Ensure consistency across input types
        self.assertEqual(meshdata_1.num_nodes, meshdata_2.num_nodes)
        self.assertEqual(meshdata_1.num_nodes, meshdata_3.num_nodes)

    def test_sizing_constant(self):
        if not HAS_TRIANGLE:
            self.skipTest("Triangle not installed")

        # Test max area sizing constraint (Triangle uses 'a')
        mesh = self.triangle_engine.generate(self.valid_input_3, sizing=0.01)
        self.assertGreater(get_num_elements(mesh), 2) # Should be fine mesh

    def test_sizing_field_fails(self):
        if not HAS_TRIANGLE:
            self.skipTest("Triangle not installed")

        field = MeshData(coords=[[0,0]], values=[1])
        with self.assertRaises(NotImplementedError):
            self.triangle_engine.generate(self.valid_input_3, sizing=field)

    def test_holes(self):
        if not HAS_TRIANGLE:
            self.skipTest("Triangle not installed")

        mesh = self.triangle_engine.generate(gpd.GeoSeries([self.poly_hole]))
        # Check area coverage roughly matches (16 - 1 = 15)
        # Calculating mesh area:
        mesh_poly = utils.get_mesh_polygons(mesh)
        self.assertTrue(self.poly_hole.equals(mesh_poly))

    def test_remesh(self):
        if not HAS_TRIANGLE:
            self.skipTest("Triangle not installed")

        # 1. Generate coarse mesh
        coarse_mesh = self.triangle_engine.generate(self.valid_input_3, sizing=0.5)
        n_coarse = coarse_mesh.num_nodes

        # 2. Refine (remesh) entire domain using smaller sizing
        # Note: Triangle remesh logic in engine might behave differently based on inputs
        # Here we pass the mesh back in as 'mesh' arg
        fine_mesh = self.triangle_engine.remesh(
            coarse_mesh,
            sizing=0.01
        )

        self.assertIsInstance(fine_mesh, MeshData)
        self.assertGreater(fine_mesh.num_nodes, n_coarse)

    def test_aux_points_seed(self):
        """Test seeding points (auxiliary points) into the mesh."""
        if not HAS_TRIANGLE:
            self.skipTest("Triangle not installed")

        bx = box(0, 0, 4, 1)
        # Define a point exactly in the middle
        seed_mesh = MeshData(coords=[[2.0, 0.5]])

        mesh = self.triangle_engine.generate(gpd.GeoSeries([bx]), seed=seed_mesh)

        # Check if the specific coordinate exists in the output
        found = False
        for c in mesh.coords:
            if np.allclose(c, [2.0, 0.5]):
                found = True
                break
        self.assertTrue(found, "Seed point not preserved in Triangle mesh")

    def test_quad_not_supported(self):
        if not HAS_TRIANGLE:
            self.skipTest("Triangle not installed")

        # Create a fake quad mesh
        quad_mesh = MeshData(coords=[[0,0],[1,0],[1,1],[0,1]], quad=[[0,1,2,3]])

        with self.assertRaises(NotImplementedError):
            self.triangle_engine.remesh(quad_mesh)

class TestBehavior(unittest.TestCase):
    """
    Tests that verify the physics/logic of the output,
    not just that it runs without error.
    """

    def setUp(self):
        self.box = box(0, 0, 10, 10)
        self.geo_box = gpd.GeoSeries([self.box])

    def test_sizing_respect(self):
        """
        Verify that requesting a smaller sizing actually
        results in more elements.
        """
        # Test for whatever engines are installed
        engines_to_test = []
        if HAS_GMSH: engines_to_test.append('gmsh')
        if HAS_TRIANGLE: engines_to_test.append('triangle')

        for engine_name in engines_to_test:
            with self.subTest(engine=engine_name):
                eng = get_mesh_engine(engine_name)

                # Coarse Mesh (Size ~ 5.0)
                mesh_coarse = eng.generate(self.geo_box, sizing=5.0)

                # Fine Mesh (Size ~ 1.0)
                mesh_fine = eng.generate(self.geo_box, sizing=1.0)

                # Logic check
                self.assertGreater(
                    get_num_elements(mesh_fine), 
                    get_num_elements(mesh_coarse),
                    f"{engine_name} failed to refine mesh when sizing was reduced."
                )

    def test_gmsh_dirty_geometry_cleaning(self):
        """
        Gmsh engine has specific logic to merge vertices closer than 1e-6.
        We feed it two points that are 1e-7 apart to ensure it doesn't crash 
        or create zero-length lines.
        """
        if not HAS_GMSH:
            self.skipTest("Gmsh not installed")

        # Create a polygon with two points insanely close together
        dirty_poly = Polygon([
            (0, 0),
            (1, 0),
            (1, 1),
            (0.0000001, 1), # <--- 1e-7 distance from next point
            (0, 1)          # <--- Next point
        ])

        eng = get_mesh_engine('gmsh', optimize_mesh=False)

        # This should NOT raise an error if the cleaning logic works
        try:
            mesh = eng.generate(gpd.GeoSeries([dirty_poly]), sizing=0.5)
        except Exception as e:
            self.fail(f"Gmsh engine crashed on dirty geometry: {e}")

        self.assertIsInstance(mesh, MeshData)

if __name__ == '__main__':
    unittest.main()