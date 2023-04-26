#! python
import unittest
import tempfile
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from shapely import geometry

from ocsmesh.features.linefeature import LineFeature

class LineFeatureCapabilities(unittest.TestCase):

    def setUp(self):
        self._crs_1 = CRS.from_epsg(4326)
        self._crs_2 = CRS.from_string('nad83')
        self._lines_1 = geometry.LineString(
            [ 
                [0, 0], [1, 1], [2, 2]
            ]
        )
        self._lines_2 = geometry.LineString(
            [
                [1, 0], [2, 1]
            ]
        )
        self._mult_line_1 = geometry.MultiLineString(
            [self._lines_1, self._lines_2]
        )


    def test_ctor_args_1(self):
        lf = LineFeature(self._lines_1)
        shape, crs = lf.get_multiline()
        self.assertIsInstance(shape, geometry.MultiLineString)
        self.assertEqual(crs, self._crs_1)


    def test_ctor_args_2(self):
        lf = LineFeature(self._mult_line_1, self._crs_2)
        shape, crs = lf.get_multiline()
        self.assertIsInstance(shape, geometry.MultiLineString)
        self.assertEqual(crs, self._crs_2)


    def test_ctor_args_3(self):
        lf = LineFeature(shape=self._lines_1, shape_crs=self._crs_2)
        shape, crs = lf.get_multiline()
        self.assertIsInstance(shape, geometry.MultiLineString)
        self.assertEqual(crs, self._crs_2)


    def test_ctor_args_4(self):

        with tempfile.TemporaryDirectory() as tdir:

            shape_path = tdir

            gdf = gpd.GeoDataFrame(geometry=[self._mult_line_1], crs=self._crs_2)
            gdf.to_file(shape_path)

            lf = LineFeature(shapefile=shape_path)
            shape, crs = lf.get_multiline()
            self.assertIsInstance(shape, geometry.MultiLineString)
            self.assertEqual(crs, self._crs_2)


    def test_ctor_args_5(self):

        with tempfile.TemporaryDirectory() as tdir:

            # NOTE: Path is file this time
            shape_path = Path(tdir) / 'test_shape.shp'

            gdf = gpd.GeoDataFrame(geometry=[self._mult_line_1], crs=self._crs_2)
            gdf.to_file(shape_path)

            lf = LineFeature(shapefile=shape_path)
            shape, crs = lf.get_multiline()
            self.assertIsInstance(shape, geometry.MultiLineString)
            self.assertEqual(crs, self._crs_2)


    def test_ctor_args_6(self):

        with tempfile.TemporaryDirectory() as tdir:

            shape_path = tdir

            # NOTE: This time shapefile doesn't have CRS
            gdf = gpd.GeoDataFrame(geometry=[self._mult_line_1])
            gdf.to_file(shape_path)

            lf = LineFeature(shapefile=shape_path)
            with self.assertRaises(ValueError):
                lf.get_multiline()


class CourantNumConstraint(unittest.TestCase):

    def test_calculate_element_size_from_courant_num(self):
        self.assertEqual(
            ocsmesh.utils.get_element_size_courant(
                target_courant=1,
                characteristic_velocity_magnitude=1,
                timestep=1
            ),
            1
        )

        self.assertEqual(
            ocsmesh.utils.get_element_size_courant(
                target_courant=0.5,
                characteristic_velocity_magnitude=2,
                timestep=3.5
            ),
            14
        )

        self.assertTrue(
            np.array_equal(
                ocsmesh.utils.get_element_size_courant(
                    target_courant=1,
                    characteristic_velocity_magnitude=np.array([1, 1]),
                    timestep=1
                ),
                np.array([1, 1])
            )
        )

        # TODO: Add non-trivial cases


    def test_approx_courant_number_for_depth(self):
        elemsize = np.ones((10, 20)) * 1
        depths = -np.arange(1, 201).reshape(elemsize.shape)
        timestep = 1
        wave_amplitude = 2

        C_apprx = ocsmesh.utils.approximate_courant_number_for_depth(
            depths, timestep, elemsize, wave_amplitude
        )

        self.assertTrue(np.all(C_apprx > 0))

        # TODO: Add non-trivial cases


    def test_can_velocity_be_approx_line_wave_theory(self):
        self.assertFalse(
            ocsmesh.utils.can_velocity_be_approximated_by_linear_wave_theory(1, 2)
        )

        self.assertFalse(
            ocsmesh.utils.can_velocity_be_approximated_by_linear_wave_theory(-1, 2)
        )

        self.assertTrue(
            ocsmesh.utils.can_velocity_be_approximated_by_linear_wave_theory(-3, 2)
        )

        self.assertTrue(
            ocsmesh.utils.can_velocity_be_approximated_by_linear_wave_theory(-2)
        )

        self.assertFalse(
            ocsmesh.utils.can_velocity_be_approximated_by_linear_wave_theory(-1.9)
        )

        self.assertTrue(
            np.array_equal(
                ocsmesh.utils.can_velocity_be_approximated_by_linear_wave_theory(
                    np.array([-1, -1, 0, 1, -2], dtype=float),
                    1
                ),
                np.array([True, True, False, False, True])
            )
        )


    def test_estimate_velocity_magnitude_for_depth(self):
        # Velocity approx for depths shallower than wave amp is the same
        self.assertEqual(
            ocsmesh.utils.estimate_particle_velocity_from_depth(-1),
            ocsmesh.utils.estimate_particle_velocity_from_depth(0)
        )
        self.assertEqual(
            ocsmesh.utils.estimate_particle_velocity_from_depth(-1, 3),
            ocsmesh.utils.estimate_particle_velocity_from_depth(0, 3)
        )

        self.assertNotEqual(
            ocsmesh.utils.estimate_particle_velocity_from_depth(-2, 3),
            ocsmesh.utils.estimate_particle_velocity_from_depth(-4, 3)
        )

        # TODO: Should tests for exact approx value matches be added here?


    def test_cfl_constraint_object_api_type(self):
        sizes_before_constraint = np.array([[500, 600]])
        depths = np.array([[-10, -11]])
        constraint = ocsmesh.features.constraint.CourantNumConstraint(
            value=0.5
        )
        sizes_after_constraint = constraint.apply(depths, sizes_before_constraint)

        # Assertions
        self.assertIsInstance(
            sizes_after_constraint, type(sizes_before_constraint)
        )

    def test_cfl_constraint_object_api_io(self):
        sizes_before_constraint = np.array([[500, 600]])
        depths_1 = np.array([[-10, -11]])
        depths_2 = np.array([[-10]])
        constraint = ocsmesh.features.constraint.CourantNumConstraint(
            value=0.5
        )

        # Assertions
        self.assertEqual(
            constraint.apply(depths_1, sizes_before_constraint).shape,
            sizes_before_constraint.shape
        )
        self.assertRaises(
            ValueError,
            constraint.apply, depths_2, sizes_before_constraint,
        )


    def test_cfl_constraint_max(self):
        sizes_before_constraint = np.ones((10, 20)) * np.finfo(float).resolution
        depths = np.ones((10, 20)) * -20

        constraint = ocsmesh.features.constraint.CourantNumConstraint(
            value=0.5,
            value_type='max',
            wave_amplitude=2,
            timestep=150,
        )

        sizes_after_constraint = constraint.apply(depths, sizes_before_constraint)

        # Assertions
        # NOTE: Max is on Courant # NOT the element size
        self.assertTrue(
            np.all(sizes_after_constraint > sizes_before_constraint)
        )


    def test_cfl_constraint_min(self):
        sizes_before_constraint = np.ones((10, 20)) * np.inf
        depths = np.ones((10, 20)) * -20

        constraint = ocsmesh.features.constraint.CourantNumConstraint(
            value=0.1,
            value_type='min',
            wave_amplitude=2,
            timestep=150,
        )

        sizes_after_constraint = constraint.apply(depths, sizes_before_constraint)

        # Assertions
        # NOTE: Max is on Courant # NOT the element size
        self.assertTrue(
            np.all(sizes_after_constraint < sizes_before_constraint)
        )


    def test_cfl_constraint_unaffected(self):
        sizes_before_constraint = np.vstack(
            (
                np.ones((10, 20)) * np.finfo(float).resolution,
                np.ones((10, 20)) * np.inf
            )
        )
        depths = -np.arange(1, 401).reshape(sizes_before_constraint.shape)

        constraint = ocsmesh.features.constraint.CourantNumConstraint(
            value=0.1,
            value_type='min',
            wave_amplitude=2,
            timestep=150,
        )

        sizes_after_constraint = constraint.apply(depths, sizes_before_constraint)
        self.assertTrue(
            np.all(
                np.equal(
                    sizes_after_constraint.ravel()[:200],
                    sizes_before_constraint.ravel()[:200]
                )
            )
        )


    def test_cfl_constraint_result(self):
        sizes_before_constraint = np.ones((10, 20)) * np.inf
        depths = -np.arange(1, 201).reshape(sizes_before_constraint.shape)

        constraint = ocsmesh.features.constraint.CourantNumConstraint(
            value=0.1,
            value_type='min',
            wave_amplitude=2,
            timestep=150,
        )

        sizes_after_constraint = constraint.apply(depths, sizes_before_constraint)

        C_apprx = ocsmesh.utils.approximate_courant_number_for_depth(
            depths, 150, sizes_after_constraint, 2
        )

        # Assertions
        self.assertTrue(np.all(np.isclose(C_apprx, 0.1)))




if __name__ == '__main__':
    unittest.main()
