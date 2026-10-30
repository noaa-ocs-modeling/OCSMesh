"""HfunRaster(initial_value=path) wraps an existing size function file.

Workers rebuild every size function this way, and the coordinator does
it again to take their results. It used to write a blank
full-resolution raster first and then overwrite it with the file, one
full read and write per tile and stage for nothing.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ocsmesh import Raster
from ocsmesh.hfun.raster import HfunRaster
from tests.api import multitile


class TestInitialValue(unittest.TestCase):

    def setUp(self):
        # pylint: disable=consider-using-with
        self._tmp = tempfile.TemporaryDirectory()
        self.dem = multitile.write_tile(
            Path(self._tmp.name) / 'dem.tif',
            multitile.TileSpec('dem', 0.004, (0.0, 0.0, 0.6, 0.3)))
        self.source = HfunRaster(Raster(str(self.dem)), hmin=50.0, hmax=3000.0)
        self.source.add_constant_value(700.0, upper_bound=-30.0)

    def tearDown(self):
        self._tmp.cleanup()

    def test_wraps_without_writing_a_blank_raster(self):
        with mock.patch.object(HfunRaster, 'modifying_raster',
                               autospec=True) as writes:
            wrapped = HfunRaster(Raster(str(self.dem)), hmin=50.0,
                                 hmax=3000.0,
                                 initial_value=self.source.tmpfile)
        writes.assert_not_called()
        np.testing.assert_array_equal(wrapped.get_values(),
                                      self.source.get_values())
        self.assertNotEqual(wrapped.tmpfile, self.source.tmpfile)

    def test_wrapped_raster_can_be_refined(self):
        wrapped = HfunRaster(Raster(str(self.dem)), hmin=50.0, hmax=3000.0,
                             initial_value=self.source.tmpfile)
        before = self.source.get_values().copy()
        wrapped.add_constant_value(400.0, upper_bound=-30.0)
        self.assertLess(wrapped.get_values().min(), before.min())
        # the source file is untouched
        np.testing.assert_array_equal(self.source.get_values(), before)

    def test_new_size_function_starts_at_float32_max(self):
        fresh = HfunRaster(Raster(str(self.dem)))
        self.assertTrue(np.all(
            fresh.get_values() == np.finfo(np.float32).max))


if __name__ == '__main__':
    unittest.main()
