"""estimate_bounds_utm gives a UTM CRS for any valid geographic bounds,
0-360 longitudes included, and raises instead of returning None (which
made the callers compute sizes in degrees).
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

    def test_0_360_bounds(self):
        # 190..200 E is -170..-160: zone 3, as the -180..180 bounds give
        crs = utils.estimate_bounds_utm((190, 55, 200, 60))
        same = utils.estimate_bounds_utm((-170, 55, -160, 60))
        self.assertEqual(crs.to_dict()['zone'], 3)
        self.assertTrue(crs.equals(same))

    def test_bounds_across_180(self):
        # centre at 180: zone 1 (as -180)
        crs = utils.estimate_bounds_utm((175, 50, 185, 55))
        self.assertEqual(crs.to_dict()['zone'], 1)

    def test_south_up_bounds_are_sorted(self):
        crs = utils.estimate_bounds_utm((-74.5, 40.5, -73.5, 39.5))
        self.assertEqual(crs.to_dict()['zone'], 18)
        self.assertNotIn('south', crs.to_dict())

    def test_southern_hemisphere(self):
        crs = utils.estimate_bounds_utm((150, -35, 151, -33))
        self.assertEqual(crs.to_dict()['zone'], 56)
        self.assertIn('south', crs.to_dict())

    def test_wider_than_180_raises(self):
        with self.assertRaisesRegex(ValueError, 'metric_crs'):
            utils.estimate_bounds_utm((-180, 50, 180, 60))

    def test_not_geographic_coordinates_raise(self):
        # projected coordinates labelled EPSG:4326
        with self.assertRaisesRegex(ValueError, 'not longitudes'):
            utils.estimate_bounds_utm((500000, 4400000, 600000, 4500000))

    def test_projected_input_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertIsNone(utils.estimate_bounds_utm(
                (500000, 4400000, 600000, 4500000), 'EPSG:32618'))


if __name__ == '__main__':
    unittest.main()
