import unittest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, box
from ocsmesh import MeshData, utils
from ocsmesh.ops import river_mesh


class TestRiverMesh(unittest.TestCase):

    def test_quadrangulate_rivermapper_arcs_simple(self):
        """
        Synthetic Test:
        Create a mock GeoDataFrame representing one river segment with two parallel banks.
        Expected result: A strip of Quad elements connecting them.
        """
        # 1. Create Synthetic Arcs (2 Parallel lines)
        # Bank 1: (0,0) -> (10,0)
        # Bank 2: (0,2) -> (10,2)
        # 11 points each = 10 segments
        x = np.linspace(0, 10, 11)
        line1 = LineString(np.column_stack((x, np.zeros_like(x))))
        line2 = LineString(np.column_stack((x, np.ones_like(x) * 2)))

        # 2. Build the specific DataFrame structure expected by the function
        # Required cols: 'river_idx', 'local_arc_', 'geometry'
        df = pd.DataFrame({
            'river_idx': [1, 1],
            'local_arc_': [1, 2], # Order matters (Bank 1 then Bank 2)
            'geometry': [line1, line2]
        })
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

        # 3. Run the function
        mesh = river_mesh.quadrangulate_rivermapper_arcs(gdf, crs="EPSG:4326")

        # 4. Assertions
        self.assertIsInstance(mesh, MeshData)

        # Calculate element count manually (MeshData fix)
        n_quads = len(mesh.quad) if mesh.quad is not None else 0
        n_trias = len(mesh.tria) if mesh.tria is not None else 0

        # We had 11 points per line. We expect 10 quads connecting them.
        self.assertEqual(n_quads, 10)
        self.assertEqual(n_trias, 0)

        # Check area
        # 10m long * 2m wide = 20m^2
        poly = utils.get_mesh_polygons(mesh)
        self.assertAlmostEqual(poly.area, 20.0, delta=1e-5)

    def test_quadrangulate_sorting_logic(self):
        """
        Verify that the function correctly sorts the input dataframe.
        If we feed arcs in reverse order (Arc 2 before Arc 1), 
        it should still work if 'local_arc_' is set correctly.
        """
        # 5 points = 4 segments
        x = np.linspace(0, 10, 5)
        line1 = LineString(np.column_stack((x, np.zeros_like(x))))
        line2 = LineString(np.column_stack((x, np.ones_like(x))))

        # INVERTED ORDER in dataframe (Arc 2 appears first in the list)
        df = pd.DataFrame({
            'river_idx': [1, 1],
            'local_arc_': [2, 1], # <--- Notice 2 comes before 1 in list
            'geometry': [line2, line1]
        })
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

        mesh = river_mesh.quadrangulate_rivermapper_arcs(gdf)

        n_quads = len(mesh.quad) if mesh.quad is not None else 0
        self.assertEqual(n_quads, 4)

    def test_triangulate_poly_integration(self):
        """
        Test the triangulation of polygons (Delaunay within) used for 
        irregular river patches.
        """
        # Create a simple C-shape or square
        poly = Polygon([(0,0), (10,0), (10,10), (0,10), (0,0)])
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        
        # Run triangulation
        mesh = river_mesh.triangulate_poly(gdf)

        n_elements = 0
        if mesh.tria is not None:
            n_elements += len(mesh.tria)
        if mesh.quad is not None:
            n_elements += len(mesh.quad)

        self.assertGreater(n_elements, 0)
        # Should be triangles only
        self.assertEqual(len(mesh.quad), 0)

if __name__ == '__main__':
    unittest.main()
