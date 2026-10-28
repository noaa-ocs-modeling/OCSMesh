"""Sanity checks of the shared multi-tile fixture and its helpers.

These assert properties of the synthetic data and of the comparison
helpers only; behaviour of OCSMesh on this stack is tested next to each
fix.
"""

import unittest

import numpy as np
import rasterio
from shapely.geometry import box

from tests.api import multitile


class TestMultiTileFixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0)

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def test_priority_order_and_count(self):
        names = [s.name for s in self.stack.specs]
        self.assertEqual(names[0], 'background')
        self.assertEqual(len(self.stack.paths), 6)
        self.assertNotIn('empty', [s.name for s in multitile.default_specs(include_empty=False)])

    def test_every_tile_intersects_the_aoi(self):
        for path in self.stack.paths:
            with rasterio.open(path) as src:
                self.assertTrue(box(*src.bounds).intersects(self.stack.aoi), path)

    def test_nodata_layout(self):
        with rasterio.open(self.stack.by_name['empty']) as src:
            self.assertTrue(np.all(src.read(1) == multitile.NODATA))
        with rasterio.open(self.stack.by_name['fine_a']) as src:
            z = src.read(1)
            self.assertEqual(src.nodata, multitile.NODATA)
            self.assertTrue(np.all(z[0, :] == multitile.NODATA))
            self.assertTrue(np.all(z[z.shape[0] // 2, 10:-10] != multitile.NODATA))
        with rasterio.open(self.stack.by_name['background']) as src:
            self.assertFalse(np.any(src.read(1) == multitile.NODATA))

    def test_background_without_declared_nodata(self):
        stack = multitile.build_stack(scale=8.0, include_empty=False,
                                      background_declares_nodata=False)
        try:
            with rasterio.open(stack.by_name['background']) as src:
                self.assertIsNone(src.nodata)
        finally:
            multitile.cleanup(stack)

    def test_same_bathymetry_where_tiles_overlap(self):
        # both medium tiles report the same depth at a common interior point
        x, y = 0.41, 0.30
        values = []
        for name in ('medium_a', 'medium_b'):
            with rasterio.open(self.stack.by_name[name]) as src:
                values.append(next(src.sample([(x, y)]))[0])
        self.assertAlmostEqual(float(values[0]), float(values[1]), places=1)

    def test_compare_rasters(self):
        a = [np.zeros((2, 2)), np.ones((3, 3))]
        self.assertEqual(multitile.compare_rasters(a, [x.copy() for x in a]), '')
        b = [np.zeros((2, 2)), np.full((3, 3), 2.0)]
        self.assertIn('9 of 9 cells differ', multitile.compare_rasters(a, b))
        self.assertIn('1 vs 2', multitile.compare_rasters(a[:1], a))

    def test_modes_agree_without_clipping(self):
        """Helper smoke test: unclipped tiles give identical refinements in every mode."""
        paths = self.stack.without('empty')
        reference = multitile.refined_rasters(
            paths, 'serial', configure=multitile.add_depth_bands)
        for mode in multitile.available_modes()[1:]:
            other = multitile.refined_rasters(
                paths, mode, configure=multitile.add_depth_bands)
            if multitile.is_manager():
                self.assertEqual(multitile.compare_rasters(reference, other), '', mode)


if __name__ == '__main__':
    unittest.main()
