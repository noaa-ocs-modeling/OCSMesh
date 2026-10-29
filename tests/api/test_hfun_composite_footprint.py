"""Exact-method compositing by valid-data footprint (tile-edge seams).

Overlapping size functions used to be resolved by bounding box: a
higher-priority tile removed every lower-priority node inside its bbox,
including where its own pixels are nodata (margins, clipped corners).
Its nodata pixels carried no refinement, so strips of the wrong size
appeared along tile edges in the NJ exact size function.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ocsmesh import Hfun, utils
from pyproj import CRS
from tests.api import multitile

# 1000 m band of multitile.DEPTH_BANDS: elevation above -15 m, i.e.
# x > 0.2889 for multitile.bathymetry. Stay a pixel clear of the edge.
X_1000 = 0.30


class TestFootprintCompositing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pylint: disable=consider-using-with
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.low = multitile.write_tile(
            tmp / 'low.tif',
            multitile.TileSpec('low', 0.004, (0.0, 0.0, 0.6, 0.3)))
        cls.high = multitile.write_tile(
            tmp / 'high.tif',
            multitile.TileSpec('high', 0.002, (0.2, 0.05, 0.5, 0.25),
                               nodata_border_px=8))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _meshdata(self, mode):
        hfun = multitile.make_hfun([str(self.low), str(self.high)], mode,
                                   configure=multitile.add_depth_bands)
        md = hfun.meshdata()
        if md is None:
            return None
        utils.reproject(md, CRS.from_epsg(4326))
        return md

    def test_margin_does_not_hide_lower_tile(self):
        for mode in multitile.available_modes():
            md = self._meshdata(mode)
            if not multitile.is_manager():
                continue
            x = np.asarray(md.coords)[:, 0]
            values = np.asarray(md.values).ravel()
            band = x > X_1000
            with self.subTest(mode=mode):
                self.assertTrue(band.any())
                off = int(np.sum(values[band] > 1000.0 + 1e-3))
                self.assertEqual(off, 0, f'{off} nodes off the 1000 m band')


if __name__ == '__main__':
    unittest.main()
