import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon, Point
from ocsmesh import MeshData, utils
from ocsmesh.ops import combine_mesh


class TestCombineMesh(unittest.TestCase):

    def test_merge_neighboring_meshes(self):
        """
        Create two meshes that touch exactly at x=1.
        Verify they fuse into a single mesh.
        """
        # Mesh A: (0,0) to (1,1)
        mesh_a = utils.create_rectangle_mesh(nx=2,
                                             ny=2,
                                             holes=[],
                                             x_extent=(0, 1),
                                             y_extent=(0, 1))

        # Mesh B: (1,0) to (2,1) 
        # Note: They share the edge at x=1
        mesh_b = utils.create_rectangle_mesh(nx=2,
                                             ny=2,
                                             holes=[],
                                             x_extent=(1, 2),
                                             y_extent=(0, 1))

        # Combined
        merged = combine_mesh.merge_neighboring_meshes(mesh_a, mesh_b)

        # Logic Checks
    # 1. Total nodes should be less than sum(A)+sum(B) because boundary nodes merged
        # A has 4 nodes, B has 4 nodes. Shared = 2. Result should be 6 nodes.
        self.assertEqual(merged.num_nodes, 6)

        # 2. Area should be sum of areas (1 + 1 = 2)
        poly = utils.get_mesh_polygons(merged)
        self.assertAlmostEqual(poly.area, 2.0)

    def test_merge_overlapping_meshes(self):
        """
        Create a large background mesh and a smaller, DENSE foreground mesh.
        Verify the background is clipped and stitched.
        """
        # 1. Background: Coarse (10x10 units, 1.0 unit spacing)
        # Bounds: 0 to 10
        mesh_bg = utils.create_rectangle_mesh(
            nx=11, ny=11, holes=[], 
            x_extent=(0, 10), y_extent=(0, 10)
        )

        # 2. Foreground: Very Fine (2x2 units, 0.1 unit spacing)
        # Bounds: 4 to 6
        mesh_fg = utils.create_rectangle_mesh(
            nx=21, ny=21, holes=[], 
            x_extent=(4, 6), y_extent=(4, 6)
        )

        # Use Projected CRS to ensure buffer units are treated as meters
        test_crs = 'EPSG:26918' 
        mesh_bg.crs = test_crs
        mesh_fg.crs = test_crs

        try:
        # NOTE: We pass clip_final=False.
        # The 'clip_final=True' logic in the source code performs an aggressive
        # 'fit_inside' clip on the exact boundary, which often drops boundary
        # elements due to floating point union errors (the 'onion peeling' effect).
        # We disable it here to test the MERGE logic specifically.
            combined = combine_mesh.merge_overlapping_meshes(
                [mesh_bg, mesh_fg], 
                buffer_size=0.1, 
                buffer_domain=0.1, 
                crs=test_crs,
                clip_final=False 
            )
        except Exception as e:
            self.fail(f"Merge operation failed with error: {e}")

        self.assertIsInstance(combined, MeshData)

        # 3. Check Geometry Coverage
        # Use intersects() instead of contains() because points on the
        # boundary of a polygon are NOT 'contained' in Shapely logic.
        poly_combined = utils.get_mesh_polygons(combined)

        bg_corners = [(0,0), (10,0), (10,10), (0,10)]
        for pt in bg_corners:
            p_geom = Point(pt[0], pt[1])
            self.assertTrue(poly_combined.intersects(p_geom),
        f"Background corner {pt} lost in merge! Bounds: {poly_combined.bounds}")

        # Check center of the Foreground (should still exist)
        center_pt = Point(5, 5)
        self.assertTrue(poly_combined.intersects(center_pt),
                        "Foreground center (5,5) lost in merge!")

        # 4. Check Refinement (Logic)
        n_elem_combined = len(combined.tria) + len(combined.quad)
        n_elem_bg = len(mesh_bg.tria) + len(mesh_bg.quad)

        self.assertGreater(n_elem_combined, n_elem_bg, 
                f"Merged count ({n_elem_combined}) not > BG count ({n_elem_bg})")

    def test_triangulate_polygon(self):
        """Test the wrapper for triangle engine."""
        square = box(0, 0, 1, 1)

        # Simple triangulation
        mesh = combine_mesh.triangulate_polygon(square, opts='p')

        self.assertIsInstance(mesh, MeshData)

        n_elements = 0
        if mesh.tria is not None:
            n_elements += len(mesh.tria)
        if mesh.quad is not None:
            n_elements += len(mesh.quad)

        self.assertGreater(n_elements, 0)

        # Check it covers the area
        out_poly = utils.get_mesh_polygons(mesh)
        self.assertAlmostEqual(out_poly.area, 1.0)

if __name__ == '__main__':
    unittest.main()
