#! python
import unittest
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import warnings

from jigsawpy import jigsaw_msh_t
import geopandas as gpd
import numpy as np
from pyproj import CRS
import rasterio as rio
import requests
from shapely import geometry

import ocsmesh

from tests.api.common import raster_from_numpy, msht_from_numpy, create_rectangle_mesh


class SizeFunctionType(unittest.TestCase):
    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())

        self.rast = self.tdir / 'rast_1.tif'
        self.mesh = self.tdir / 'mesh_1.gr3'

        rast_xy = np.mgrid[-1:0.1:0.1, -0.7:0.1:0.1]
        rast_xy = np.mgrid[0:1.1:0.1, -0.7:0.1:0.1]
        rast_z = np.ones_like(rast_xy[0])

        raster_from_numpy(
            self.rast, rast_z, rast_xy, 4326
        )

        msh_t = create_rectangle_mesh(
            nx=17, ny=7, holes=[40, 41], x_extent=(-1, 1), y_extent=(0, 1))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, message='Input mesh has no CRS information'
            )
            mesh = ocsmesh.Mesh(msh_t)
            mesh.write(str(self.mesh), format='grd', overwrite=False)


    def test_create_raster_hfun(self):
        hfun = ocsmesh.Hfun(
            ocsmesh.Raster(self.rast),
            hmin=500,
            hmax=10000
        )
        self.assertTrue(isinstance(hfun, ocsmesh.hfun.raster.HfunRaster))

    def test_mesh_raster_hfun(self):
        hfun = ocsmesh.Hfun(
            ocsmesh.Mesh.open(self.mesh, crs=4326),
        )
        self.assertTrue(isinstance(hfun, ocsmesh.hfun.mesh.HfunMesh))

    def test_collector_raster_hfun(self):
        hfun = ocsmesh.Hfun(
            [self.rast],
            hmin=500,
            hmax=10000
        )
        self.assertTrue(isinstance(hfun, ocsmesh.hfun.collector.HfunCollector))


class SizeFunctionCollector(unittest.TestCase):

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())

        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.gr3'

        rast_xy_1 = np.mgrid[-1:0.1:0.1, -0.7:0.1:0.1]
        rast_xy_2 = np.mgrid[0:1.1:0.1, -0.7:0.1:0.1]
        rast_z_1 = np.ones_like(rast_xy_1[0])

        raster_from_numpy(
            self.rast1, rast_z_1, rast_xy_1, 4326
        )
        raster_from_numpy(
            self.rast2, rast_z_1, rast_xy_2, 4326
        )

        msh_t = create_rectangle_mesh(
            nx=17, ny=7, holes=[40, 41], x_extent=(-1, 1), y_extent=(0, 1))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, message='Input mesh has no CRS information'
            )
            mesh = ocsmesh.Mesh(msh_t)
            mesh.write(str(self.mesh1), format='grd', overwrite=False)


    def test_multi_path_input(self):
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=10000
        )
        hfun_msht = hfun_coll.msh_t()
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))

    def test_multi_str_input(self):
        hfun_coll = ocsmesh.Hfun(
            [str(i) for i in [self.rast1, self.rast2, self.mesh1]],
            hmin=500,
            hmax=10000
        )
        hfun_msht = hfun_coll.msh_t()
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))

    def test_multi_raster_input(self):

        rast1 = ocsmesh.Raster(self.rast1)
        rast2 = ocsmesh.Raster(self.rast2)
        hfun_coll = ocsmesh.Hfun(
            [rast1, rast2],
            hmin=500,
            hmax=10000
        )
        hfun_msht = hfun_coll.msh_t()
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))

    def test_multi_mix_input(self):
        rast1 = ocsmesh.Raster(self.rast1)
        mesh1 = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        hfun_coll = ocsmesh.Hfun(
            [rast1, self.rast2, mesh1],
            hmin=500,
            hmax=10000
        )
        hfun_msht = hfun_coll.msh_t()
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))

    def test_mesh_input(self):
        mesh1 = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        hfun_coll = ocsmesh.Hfun(
            [mesh1],
            hmin=500,
            hmax=10000
        )
        hfun_msht = hfun_coll.msh_t()
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))



class SizeFromMesh(unittest.TestCase):

    def setUp(self):
        rast = ocsmesh.raster.Raster('/tmp/test_dem.tif')

        hfun_orig = ocsmesh.hfun.hfun.Hfun(rast, hmin=100, hmax=1500)
        hfun_orig.add_contour(level=0, expansion_rate=0.001, target_size=100)
        hfun_orig_jig = hfun_orig.msh_t()

        self.hfun_orig_val = hfun_orig_jig.value

        hfun_calc_jig = deepcopy(hfun_orig_jig)
        mesh_calc = ocsmesh.mesh.mesh.Mesh(hfun_calc_jig)
        self.hfun_calc = ocsmesh.hfun.hfun.Hfun(mesh_calc)
        self.hfun_calc.size_from_mesh()

    def test_calculated_size(self):
        hfun_calc_jig = self.hfun_calc.msh_t()

        hfun_calc_val = hfun_calc_jig.value
        hfun_val_diff = self.hfun_orig_val - hfun_calc_val

        # TODO: Come up with a more robust criteria
        threshold = 0.2
        err_value = np.max(np.abs(hfun_val_diff))/np.max(self.hfun_orig_val)
        self.assertTrue(err_value < threshold)



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


class SizeFunctionWithCourantNumConstraint(unittest.TestCase):

    # NOTE: Since mesh size function doesn't have built-in depth
    # information, just like other constraints Courant number constraint
    # is not implemented for it either!

    def test_hfun_raster_cfl_constraint_support(self):
        dt = 100
        nu = 2
        courant_hi = 0.8
        courant_lo = 0.2

        rast = ocsmesh.raster.Raster('/tmp/test_dem.tif')

        hfun_raster = ocsmesh.hfun.hfun.Hfun(rast, hmin=100, hmax=5000)
        hfun_raster.add_courant_num_constraint(
            upper_bound=courant_hi,
            lower_bound=courant_lo,
            timestep=dt,
            wave_amplitude=nu
        )

        C_apprx = ocsmesh.utils.approximate_courant_number_for_depth(
            rast.values, dt, hfun_raster.values, nu
        )

        self.assertTrue(
            np.all(
                np.logical_and(
                    np.logical_or(
                        C_apprx > courant_lo,
                        np.isclose(C_apprx, courant_lo)
                    ),
                    np.logical_or(
                        C_apprx < courant_hi,
                        np.isclose(C_apprx, courant_hi)
                    )
                )
            )
        )

        hfun_jig = hfun_raster.msh_t()
        mesh_jig = deepcopy(hfun_jig)
        mesh = ocsmesh.mesh.mesh.Mesh(mesh_jig)
        mesh.interpolate(rast, nprocs=1)

        C_apprx_mesh = ocsmesh.utils.approximate_courant_number_for_depth(
            mesh.msh_t.value, dt, hfun_jig.value, nu
        )

        # Note using higher tolerance for closeness since sizes and
        # depths are interpolated. Is this ideal? No!
        self.assertTrue(
            np.all(
                np.logical_and(
                    np.logical_or(
                        C_apprx_mesh > courant_lo,
                        np.isclose(C_apprx_mesh, courant_lo, atol=0.03)
                    ),
                    np.logical_or(
                        C_apprx_mesh < courant_hi,
                        np.isclose(C_apprx_mesh, courant_hi, atol=0.03)
                    )
                )
            )
        )

    def test_hfun_raster_cfl_constraint_io(self):
        rast = ocsmesh.raster.Raster('/tmp/test_dem.tif')
        hfun_raster = ocsmesh.hfun.hfun.Hfun(rast, hmin=100, hmax=5000)
        self.assertRaises(
            ValueError,
            hfun_raster.add_courant_num_constraint,
            upper_bound=None,
            lower_bound=None,
            timestep=100,
            wave_amplitude=2
        )


    def test_hfun_coll_cfl_constraint(self):
        dt = 100
        nu = 2
        courant_hi = 0.90
        courant_lo = 0.2

        # Fast method is much less accurate!
        method_tolerance = {
            'exact': 0.03,
            'fast': 0.2
        }

        for method, tol in method_tolerance.items():
            # TODO: Add subTest

            # Creating adjacent rasters from the test raster
            rast1 = ocsmesh.raster.Raster('/tmp/test_dem.tif')
            rast2 = ocsmesh.raster.Raster('/tmp/test_dem.tif')
            bounds = rast1.bbox.bounds
            bbox1 = geometry.box(
                bounds[0], bounds[1], (bounds[0] + bounds[2]) / 2, bounds[3]
            )
            bbox2 = geometry.box(
                (bounds[0] + bounds[2]) / 2, bounds[1], bounds[2], bounds[3]
            )
            rast1.clip(bbox1)
            rast2.clip(bbox2)

            hfun_coll = ocsmesh.hfun.hfun.Hfun(
                [rast1, rast2], hmin=100, hmax=5000, method=method
            )
            hfun_coll.add_courant_num_constraint(
                upper_bound=courant_hi,
                lower_bound=courant_lo,
                timestep=dt,
                wave_amplitude=nu
            )

            hfun_jig = hfun_coll.msh_t()
            mesh_jig = deepcopy(hfun_jig)
            mesh = ocsmesh.mesh.mesh.Mesh(mesh_jig)
            mesh.interpolate([rast1, rast2], method='nearest', nprocs=1)

            C_apprx_mesh = ocsmesh.utils.approximate_courant_number_for_depth(
                mesh.msh_t.value, dt, hfun_jig.value, nu
            )

            valid_courant = np.logical_and(
                np.logical_or(
                    C_apprx_mesh > courant_lo,
                    np.isclose(C_apprx_mesh, courant_lo, atol=tol)
                ),
                np.logical_or(
                    C_apprx_mesh < courant_hi,
                    np.isclose(C_apprx_mesh, courant_hi, atol=tol)
                )
            )
            # Note using higher tolerance for closeness since sizes and
            # depths are interpolated. Is this ideal? No!
            self.assertTrue(
                np.all(valid_courant),
                msg=f"Courant constraint failed for '{method}' method!"
                    + f"\n{C_apprx_mesh[~valid_courant]}"
            )


class SizeFunctionCollectorAddFeature(unittest.TestCase):

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())

        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.gr3'
        self.feat1 = self.tdir / 'feature_1'
        self.feat2 = self.tdir / 'feature_2'

        rast_xy_1 = np.mgrid[-1:0.1:0.1, -0.7:0.1:0.1]
        rast_xy_2 = np.mgrid[0:1.1:0.1, -0.7:0.1:0.1]
        rast_z_1 = np.ones_like(rast_xy_1[0])

        raster_from_numpy(
            self.rast1, rast_z_1, rast_xy_1, 4326
        )
        raster_from_numpy(
            self.rast2, rast_z_1, rast_xy_2, 4326
        )

        crd = np.array([
            [-1, 0],
            [-0.1, 0],
            [-0.5, 0.2],
            [0.1, 0],
            [1, 0],
            [0.5, 0.2],
            [-1, 0.4],
            [1, 0.4],
        ])
        tria = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [2, 1, 3],
            [2, 3, 5],
            [2, 6, 7],
            [2, 7, 5],
            [0, 2, 6],
            [5, 4, 7],
        ])
        msh_t = msht_from_numpy(crd, tria, 4326)
        mesh = ocsmesh.Mesh(msh_t)
        mesh.write(str(self.mesh1), format='grd', overwrite=False)

        self.shape1 = geometry.LineString([
            [-1, 0], [1, 0]
        ])

        self.shape2 = geometry.LineString([
            [0, -1], [0, 1]
        ])

        gdf_feature = gpd.GeoDataFrame(geometry=[self.shape1], crs=4326)
        gdf_feature.to_file(self.feat1)

        gdf_feature = gpd.GeoDataFrame(geometry=[self.shape2], crs=4326)
        gdf_feature.to_file(self.feat2)

    def tearDown(self):
        shutil.rmtree(self.tdir)

    def _create_hfun(self, hmin, hmax):

        rast1 = ocsmesh.Raster(self.rast1)
        rast2 = ocsmesh.Raster(self.rast2)

        hfun_mesh_1 = ocsmesh.Hfun(ocsmesh.Mesh.open(self.mesh1, crs=4326))
        hfun_mesh_1.size_from_mesh()

        hfun = ocsmesh.Hfun(
            [rast1, rast2, hfun_mesh_1],
            hmin=hmin,
            hmax=hmax,
            method='exact')

        return hfun

    def _check_applied_refinement(self, msh_t, refine_gdf, target_size):

        refine_msh_t = ocsmesh.Hfun(ocsmesh.Mesh(ocsmesh.utils.clip_mesh_by_shape(
            mesh=msh_t,
            shape=refine_gdf.unary_union,
            fit_inside=True,
            inverse=False
        )))
        refine_msh_t.size_from_mesh()
        refine_avg = np.mean(refine_msh_t.msh_t().value)
        rest_msh_t = ocsmesh.Hfun(ocsmesh.Mesh(ocsmesh.utils.clip_mesh_by_shape(
            mesh=msh_t,
            shape=refine_gdf.unary_union,
            fit_inside=False,
            inverse=True
        )))
        rest_msh_t.size_from_mesh()
        rest_avg = np.mean(rest_msh_t.msh_t().value)

        self.assertTrue(np.isclose(refine_avg, target_size, rtol=3e-1))
        self.assertTrue(rest_avg > target_size * 10)

    def _is_refined_by_shape1(self, hfun, target_size):

        hfun_msh_t = hfun.msh_t()

        # Nodes close to the feature line must be small
        gdf_feature = gpd.GeoDataFrame(
            geometry=[self.shape1], crs=4326
        )
        gdf_clip = gdf_feature.to_crs(hfun_msh_t.crs).buffer(target_size*1.1)
        self._check_applied_refinement(hfun_msh_t, gdf_clip, target_size)


    def _is_refined_by_feat1(self, hfun, target_size):

        hfun_msh_t = hfun.msh_t()

        # Nodes close to the feature line must be small
        gdf_feature = gpd.read_file(self.feat1)
        gdf_clip = gdf_feature.to_crs(hfun_msh_t.crs).buffer(target_size*1.1)
        self._check_applied_refinement(hfun_msh_t, gdf_clip, target_size)

    def _is_refined_by_feat2(self, hfun, target_size):

        hfun_msh_t = hfun.msh_t()

        # Nodes close to the feature line must be small
        gdf_feature = gpd.read_file(self.feat2)
        gdf_clip = gdf_feature.to_crs(hfun_msh_t.crs).buffer(target_size)
        self._check_applied_refinement(hfun_msh_t, gdf_clip, target_size)



    def test_by_shape(self):

        hmin = 500
        hmax = 10000

        hfun = self._create_hfun(hmin, hmax)

        hfun.add_feature(
            shape=self.shape1,
            expansion_rate=0.002,
            target_size=hmin,
            crs=4326
        )
        self._is_refined_by_shape1(hfun, hmin)


    def test_by_shapefile(self):

        hmin = 500
        hmax = 10000

        hfun = self._create_hfun(hmin, hmax)

        hfun.add_feature(
            shapefile=self.feat1,
            expansion_rate=0.002,
            target_size=hmin,
        )

        self._is_refined_by_feat1(hfun, hmin)


    def test_by_linedefn(self):

        hmin = 500
        hmax = 10000

        hfun = self._create_hfun(hmin, hmax)

        hfun.add_feature(
            line_defn=ocsmesh.features.linefeature.LineFeature(shapefile=self.feat1),
            expansion_rate=0.002,
            target_size=hmin,
        )

        self._is_refined_by_feat1(hfun, hmin)


    def test_optional_arg_priority(self):

        hmin = 500
        hmax = 10000

        # First priority is `line_defn`
        hfun1 = self._create_hfun(hmin, hmax)
        hfun1.add_feature(
            line_defn=ocsmesh.features.linefeature.LineFeature(shapefile=self.feat1),
            shape=self.shape2,
            crs=4326,
            shapefile=self.feat2,
            expansion_rate=0.002,
            target_size=hmin,
        )
        self._is_refined_by_feat1(hfun1, hmin)


        # Second priority is `shape` + `crs`
        hfun2 = self._create_hfun(hmin, hmax)
        hfun2.add_feature(
            shape=self.shape1,
            crs=4326,
            shapefile=self.feat2,
            expansion_rate=0.002,
            target_size=hmin,
        )
        self._is_refined_by_feat1(hfun2, hmin)


if __name__ == '__main__':
    unittest.main()
