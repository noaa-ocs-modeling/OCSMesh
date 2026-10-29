"""Topography-based refinements must ignore nodata pixels.

``add_constant_value``, ``add_subtidal_flow_limiter`` and the topography
constraints read the DEM without its nodata mask, so a nodata value such
as -99999 was used as an elevation (inside any "deeper than" band), and
the flow limiter saw a cliff next to every nodata pixel, forcing the
valid pixels there towards hmin. ``Raster.clip`` filled the outside of
the clip shape with 0 (valid sea level) when the DEM declares no nodata.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box

from ocsmesh import Raster
from ocsmesh.hfun.raster import HfunRaster
from tests.api import multitile

BORDER = 6
BOUNDS = (0.0, 0.0, 0.6, 0.3)


class TestNodataIgnored(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pylint: disable=consider-using-with
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.plain = multitile.write_tile(
            tmp / 'plain.tif', multitile.TileSpec('plain', 0.004, BOUNDS))
        cls.bordered = multitile.write_tile(
            tmp / 'bordered.tif',
            multitile.TileSpec('bordered', 0.004, BOUNDS,
                               nodata_border_px=BORDER))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _refine(self, path, configure):
        hfun = HfunRaster(Raster(str(path)),
                          hmin=multitile.HMIN, hmax=multitile.HMAX)
        before = hfun.get_values().copy()
        configure(hfun)
        return before, hfun.get_values().copy()

    def _check(self, configure, rim=0):
        before, refined = self._refine(self.bordered, configure)
        _, reference = self._refine(self.plain, configure)
        border = np.zeros(refined.shape, dtype=bool)
        border[:BORDER] = border[-BORDER:] = True
        border[:, :BORDER] = border[:, -BORDER:] = True
        # untouched, apart from the global clamp to hmax some methods apply
        np.testing.assert_array_equal(
            np.minimum(refined[border], multitile.HMAX),
            np.minimum(before[border], multitile.HMAX))
        # away from the border the refinement is the same as without it
        inner = slice(BORDER + rim, -(BORDER + rim))
        np.testing.assert_allclose(refined[inner, inner],
                                   reference[inner, inner], rtol=1e-6)

    def test_constant_value(self):
        self._check(lambda h: h.add_constant_value(700.0, upper_bound=-30.0))

    def test_flow_limiter(self):
        # one-sided differences on the first valid pixels: compare inside
        self._check(lambda h: h.add_subtidal_flow_limiter(
            hmin=100.0, hmax=3000.0, upper_bound=0.0), rim=1)

    def test_flow_limiter_no_hmin_rim(self):
        _, refined = self._refine(self.bordered, lambda h: h.add_subtidal_flow_limiter(
            hmin=100.0, hmax=3000.0, upper_bound=0.0))
        rim = refined[BORDER, BORDER:-BORDER]
        self.assertGreater(float(rim.min()), 100.0)

    def test_topo_bound_constraint(self):
        def configure(hfun):
            hfun.add_constant_value(500.0)
            hfun.add_topo_bound_constraint(
                value=1500.0, upper_bound=-30.0, value_type='min', rate=None)
        self._check(configure)


class TestClipUndeclaredNodata(unittest.TestCase):

    def test_outside_is_nodata_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = multitile.write_tile(
                Path(tmp) / 'undeclared.tif',
                multitile.TileSpec('undeclared', 0.004, BOUNDS,
                                   declare_nodata=False))
            raster = Raster(str(path))
            raster.clip(box(0.0, 0.0, 0.3, 0.3).buffer(0.0).union(
                box(0.0, 0.0, 0.6, 0.1)))
            self.assertIsNotNone(raster.src.nodata)
            masked = raster.get_values(masked=True)
            self.assertTrue(np.ma.getmaskarray(masked).any())
            self.assertFalse(np.any(masked.compressed() == 0.0))


if __name__ == '__main__':
    unittest.main()
