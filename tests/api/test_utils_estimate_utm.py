"""estimate_bounds_utm says when it cannot estimate a UTM CRS.

For geographic bounds outside the valid range it returned None silently
and the callers computed sizes in degrees.
"""

import unittest
import warnings

from ocsmesh import utils


class TestEstimateBoundsUtm(unittest.TestCase):

    def test_valid_bounds(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            crs = utils.estimate_bounds_utm((-74.5, 39.5, -73.5, 40.5))
        self.assertEqual(crs.to_dict()['zone'], 18)

    def test_out_of_range_warns(self):
        with self.assertWarnsRegex(UserWarning, 'computed in degrees'):
            self.assertIsNone(
                utils.estimate_bounds_utm((190, 55, 200, 60)))

    def test_projected_input_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertIsNone(utils.estimate_bounds_utm(
                (500000, 4400000, 600000, 4500000), 'EPSG:32618'))


if __name__ == '__main__':
    unittest.main()
