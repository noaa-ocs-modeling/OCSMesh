#! python
import unittest
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import warnings
from unittest.mock import MagicMock

from jigsawpy import jigsaw_msh_t
import geopandas as gpd
import numpy as np
from shapely import geometry
from pyproj import CRS

import ocsmesh

from tests.api import TEST_FILE
from tests.api.common import (
    topo_2rast_1mesh,
)


class SizeFunctionType(unittest.TestCase):
    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())

        self.rast = self.tdir / 'rast_1.tif'
        self.mesh = self.tdir / 'mesh_1.gr3'

        rast_xy = np.mgrid[-1:0.1:0.1, -0.7:0.1:0.1]
        rast_xy = np.mgrid[0:1.1:0.1, -0.7:0.1:0.1]
        rast_z = np.ones_like(rast_xy[0])

        ocsmesh.utils.raster_from_numpy(
            self.rast, rast_z, rast_xy, 4326
        )

        msh_t = ocsmesh.utils.create_rectangle_mesh(
            nx=17, ny=7, holes=[40, 41], x_extent=(-1, 1), y_extent=(0, 1))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, message='Input mesh has no CRS information'
            )
            mesh = ocsmesh.Mesh(msh_t)
            mesh.write(str(self.mesh), format='grd', overwrite=False)

    def tearDown(self):
        shutil.rmtree(self.tdir)


    def test_create_raster_hfun(self):
        hfun = ocsmesh.Hfun(
            ocsmesh.Raster(self.rast),
            hmin=500,
            hmax=10000
        )
        self.assertTrue(isinstance(hfun, ocsmesh.hfun.raster.HfunRaster))

    def test_create_mesh_hfun(self):
        hfun = ocsmesh.Hfun(
            ocsmesh.Mesh.open(self.mesh, crs=4326),
        )
        self.assertTrue(isinstance(hfun, ocsmesh.hfun.mesh.HfunMesh))

    def test_create_collector_hfun(self):
        hfun = ocsmesh.Hfun(
            [self.rast],
            hmin=500,
            hmax=10000
        )
        self.assertTrue(isinstance(hfun, ocsmesh.hfun.collector.HfunCollector))


class SizeFunctionCollector(unittest.TestCase):
    # NOTE: Testing a mixed collector size function indirectly tests
    # all the other types as it is currently calling all the underlying
    # size functions to apply each feature or constraint in "exact" mode

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.grd'
        topo_2rast_1mesh(self.rast1, self.rast2, self.mesh1)

    def tearDown(self):
        shutil.rmtree(self.tdir)

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


    def test_add_topo_bound_constraint_exact(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=10000,
            method='exact'
        )
        hfun_coll.add_topo_bound_constraint(
            value=1000,
            upper_bound=-10,
            value_type='max',
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))

    def test_add_topo_func_constraint_exact(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=10,
            hmax=200,
            method='exact'
        )
        hfun_coll.add_topo_func_constraint(
            func=lambda i: abs(i) / 2.0,
            upper_bound=-10,
            value_type='min',
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))

    def test_add_contour_exact(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='exact'
        )
        hfun_coll.add_contour(
            level=5,
            target_size=1000,
            )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_contour_single_vertex(self):
        rast3 = self.tdir / 'rast_3.tif'
        rast_xy_3 = np.mgrid[-1:0.1:0.1, -0.7:0.1:0.1]
        rast_z_3 = np.ones_like(rast_xy_3[0]) * 2.0
        rast_z_3[0, 0] = 1
        rast_z_3[-1, -1] = 3

        ocsmesh.utils.raster_from_numpy(
            rast3, rast_z_3, rast_xy_3, 4326
        )

        # regression test for single point contours
        hfun_coll = ocsmesh.Hfun(
            [rast3],
            hmin=500,
            hmax=5000,
            method='exact'
        )
        hfun_coll.add_contour(
            level=1,
            target_size=1000,
            )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_channel_exact(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='exact'
        )
        hfun_coll.add_channel(
            level=0,
            width=200,
            target_size=600,
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_channel_exact(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='exact'
        )
        hfun_coll.add_subtidal_flow_limiter(
            hmin=700,
            hmax=2000,
            upper_bound=0,
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_constant_value_exact(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='exact'
        )
        hfun_coll.add_constant_value(
            value=700,
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_patch_exact(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='exact'
        )
        bx = geometry.box(-0.4, -0.3, 0.4, 0.6)
        hfun_coll.add_patch(
            shape=bx,
            target_size=1000
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_feature_exact(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='exact'
        )
        ln = geometry.LineString([[-1, 0], [1, 0]])
        hfun_coll.add_feature(
            shape=ln,
            expansion_rate=0.01,
            target_size=1000,
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_topo_bound_constraint_fast(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=10000,
            method='fast'
        )
        hfun_coll.add_topo_bound_constraint(
            value=1000,
            upper_bound=-10,
            value_type='max',
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))

    def test_add_topo_func_constraint_fast(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=10,
            hmax=200,
            method='fast'
        )
        hfun_coll.add_topo_func_constraint(
            func=lambda i: abs(i) / 2.0,
            upper_bound=-10,
            value_type='min',
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))

    def test_add_contor_fast(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='fast'
        )
        hfun_coll.add_contour(
            level=5,
            target_size=1000,
            )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_channel_fast(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='fast'
        )
        hfun_coll.add_channel(
            level=0,
            width=200,
            target_size=600,
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_channel_fast(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='fast'
        )
        hfun_coll.add_subtidal_flow_limiter(
            hmin=700,
            hmax=2000,
            upper_bound=0,
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_constant_value_fast(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='fast'
        )
        hfun_coll.add_constant_value(
            value=700,
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_patch_fast(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='fast'
        )
        bx = geometry.box(-0.4, -0.3, 0.4, 0.6)
        hfun_coll.add_patch(
            shape=bx,
            target_size=1000
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_add_feature_fast(self):
        # TODO: Improve this test (added for upgrade to shapely2)
        hfun_coll = ocsmesh.Hfun(
            [self.rast1, self.rast2, self.mesh1],
            hmin=500,
            hmax=5000,
            method='fast'
        )
        ln = geometry.LineString([[-1, 0], [1, 0]])
        hfun_coll.add_feature(
            shape=ln,
            expansion_rate=0.01,
            target_size=1000,
        )

        hfun_msht = hfun_coll.msh_t()
        
        self.assertTrue(isinstance(hfun_msht, jigsaw_msh_t))


    def test_hfun_fast_extent(self):
        r_path = self.tdir / 'rast_large.tif'
        rast_xy = np.mgrid[-80:80:0.2, 20:70:0.2]
        rast_z = np.ones_like(rast_xy[0]) * 100

        ocsmesh.utils.raster_from_numpy(
            r_path, rast_z, rast_xy, 4326
        )

        rast = ocsmesh.Raster(r_path)
        hfun_coll = ocsmesh.Hfun(
            [rast], hmin=100000, hmax=200000, method='fast'
        )
        hfun_msht = hfun_coll.msh_t()

        ocsmesh.utils.reproject(hfun_msht, rast.crs)
        rast_box = rast.get_bbox()
        hfun_box = ocsmesh.utils.get_mesh_polygons(hfun_msht)

        # NOTE: It's good enough if it covers most of it (?)
        self.assertTrue(
            rast_box.difference(hfun_box).area / rast_box.area < 5e-4
        )
        # This fails due to coarse elements!
#        self.assertTrue(hfun_box.covers(rast_box)) 



class SizeFromMesh(unittest.TestCase):

    def setUp(self):
        rast = ocsmesh.raster.Raster(TEST_FILE)

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

        # TODO: Come up with a more robust criteria!
        threshold = 0.06
        err_value = np.mean(np.abs(hfun_val_diff))/np.mean(self.hfun_orig_val)
        self.assertTrue(err_value < threshold)



class SizeFunctionWithCourantNumConstraint(unittest.TestCase):

    # NOTE: Since mesh size function doesn't have built-in depth
    # information, just like other constraints Courant number constraint
    # is not implemented for it either!

    def test_hfun_raster_cfl_constraint_support(self):
        dt = 100
        nu = 2
        courant_hi = 0.8
        courant_lo = 0.2

        rast = ocsmesh.raster.Raster(TEST_FILE)

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
        rast = ocsmesh.raster.Raster(TEST_FILE)
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
            rast1 = ocsmesh.raster.Raster(TEST_FILE)
            rast2 = ocsmesh.raster.Raster(TEST_FILE)
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

        ocsmesh.utils.raster_from_numpy(
            self.rast1, rast_z_1, rast_xy_1, 4326
        )
        ocsmesh.utils.raster_from_numpy(
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
        msh_t = ocsmesh.utils.msht_from_numpy(
            crd, triangles=tria, crs=4326
        )
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


class SizeFunctionWithRegionConstraint(unittest.TestCase):

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())

        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.gr3'

        rast_xy_1 = np.mgrid[-1:0.1:0.1, -0.7:0.1:0.1]
        rast_xy_2 = np.mgrid[0:1.1:0.1, -0.7:0.1:0.1]
        rast_z_1 = np.ones_like(rast_xy_1[0])

        ocsmesh.utils.raster_from_numpy(
            self.rast1, rast_z_1, rast_xy_1, 4326
        )
        ocsmesh.utils.raster_from_numpy(
            self.rast2, rast_z_1, rast_xy_2, 4326
        )

        msh_t = ocsmesh.utils.create_rectangle_mesh(
            nx=17, ny=7, holes=[40, 41], x_extent=(-1, 1), y_extent=(0, 1))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, message='Input mesh has no CRS information'
            )
            mesh = ocsmesh.Mesh(msh_t)
            mesh.write(str(self.mesh1), format='grd', overwrite=False)


    def tearDown(self):
        shutil.rmtree(self.tdir)


    def test_hfun_raster(self):
        rast = ocsmesh.raster.Raster(self.rast1)
        bx = geometry.box(-0.75, -0.6, -0.25, -0.4)

        hfun_raster = ocsmesh.hfun.hfun.Hfun(rast, hmin=100, hmax=5000)
        hfun_raster.add_constant_value(value=500)
        hfun_raster.add_region_constraint(
            value=1000,
            value_type='min',
            shape=bx,
            crs='4326',
            rate=None,
            )
        hfun_msht = hfun_raster.msh_t()
        ocsmesh.utils.reproject(hfun_msht, rast.crs)
        clipped_hfun = ocsmesh.utils.clip_mesh_by_shape(hfun_msht, bx)
        inv_clipped_hfun = ocsmesh.utils.clip_mesh_by_shape(
            hfun_msht, bx, fit_inside=False, inverse=True
        )

        # Due to hfun msh_t zigzag, some nodes of size 1000 might fall
        # outside the box and viceversa

        n_in_is1000 = np.sum(clipped_hfun.value == 1000)
        n_in_is500 = np.sum(clipped_hfun.value == 500)
        n_out_is1000 = np.sum(inv_clipped_hfun.value == 1000)
        n_out_is500 = np.sum(inv_clipped_hfun.value == 500)

        self.assertTrue(n_in_is500 / n_in_is1000 < 0.05)
        self.assertTrue(n_out_is1000 / n_out_is500 < 0.05)

        self.assertTrue(np.isclose(np.mean(clipped_hfun.value), 1000, rtol=0.025))
        self.assertTrue(np.isclose(np.mean(inv_clipped_hfun.value), 500, rtol=0.025))


    def test_hfun_mesh(self):
        mesh = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        bx = geometry.box(-0.75, 0.21, 0.75, 0.79)

        hfun_mesh = ocsmesh.hfun.hfun.Hfun(mesh)
        hfun_mesh.mesh.msh_t.value[:] = 200
        hfun_mesh.add_region_constraint(
            value=1000,
            value_type='min',
            shape=bx,
            crs='4326',
            rate=None,
            )
        hfun_msht = hfun_mesh.msh_t()
        ocsmesh.utils.reproject(hfun_msht, 4326)
        clipped_hfun = ocsmesh.utils.clip_mesh_by_shape(hfun_msht, bx)
        inv_clipped_hfun = ocsmesh.utils.clip_mesh_by_shape(
            hfun_msht, bx, fit_inside=False, inverse=True
        )

        # Due to hfun msh_t zigzag, some nodes of size 1000 might fall
        # outside the box and viceversa

        n_in_is1000 = np.sum(clipped_hfun.value == 1000)
        n_in_is200 = np.sum(clipped_hfun.value == 200)
        n_out_is1000 = np.sum(inv_clipped_hfun.value == 1000)
        n_out_is200 = np.sum(inv_clipped_hfun.value == 200)

        self.assertTrue(np.all(clipped_hfun.value == 1000))
        self.assertTrue(np.all(inv_clipped_hfun.value == 200))


    def test_hfun_collector_exact(self):
        rast1 = ocsmesh.Raster(self.rast1)
        mesh1 = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        mesh1.msh_t.value[:] = 500

        bx = geometry.box(-0.75, -0.4, 0.75, 0.6)

        hfun_coll = ocsmesh.Hfun(
            [rast1, self.rast2, mesh1],
        )
        hfun_coll.add_constant_value(value=500)
        hfun_coll.add_region_constraint(
            value=1000,
            value_type='min',
            shape=bx,
            crs='4326',
            rate=None,
            )
        hfun_msht = hfun_coll.msh_t()
        ocsmesh.utils.reproject(hfun_msht, 4326)
        clipped_hfun = ocsmesh.utils.clip_mesh_by_shape(hfun_msht, bx)
        inv_clipped_hfun = ocsmesh.utils.clip_mesh_by_shape(
            hfun_msht, bx, fit_inside=False, inverse=True
        )

        # Due to hfun msh_t zigzag, some nodes of size 1000 might fall
        # outside the box and viceversa

        n_in_is1000 = np.sum(clipped_hfun.value == 1000)
        n_in_is500 = np.sum(clipped_hfun.value == 500)
        n_out_is1000 = np.sum(inv_clipped_hfun.value == 1000)
        n_out_is500 = np.sum(inv_clipped_hfun.value == 500)

        self.assertTrue(n_in_is500 / n_in_is1000 < 0.1)
        self.assertTrue(n_out_is1000 / n_out_is500 < 0.05)

        self.assertTrue(np.isclose(np.mean(clipped_hfun.value), 1000, rtol=0.050))
        self.assertTrue(np.isclose(np.mean(inv_clipped_hfun.value), 500, rtol=0.025))


    def test_hfun_collector_fast(self):
        rast1 = ocsmesh.Raster(self.rast1)
        mesh1 = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        mesh1.msh_t.value[:] = 500

        bx = geometry.box(-0.75, -0.4, 0.75, -0.1)

        # TODO: Constraint application with Mesh input is NOT tested
        # with "fast" method
        hfun_coll = ocsmesh.Hfun(
            [rast1, self.rast2],
            hmin=100,
            hmax=10000,
            method='fast'
        )
        hfun_coll.add_constant_value(value=500)
        hfun_coll.add_region_constraint(
            value=1000,
            value_type='min',
            shape=bx,
            crs='4326',
            rate=None,
            )
        hfun_msht = hfun_coll.msh_t()
        ocsmesh.utils.reproject(hfun_msht, 4326)
        clipped_hfun = ocsmesh.utils.clip_mesh_by_shape(hfun_msht, bx)
        inv_clipped_hfun = ocsmesh.utils.clip_mesh_by_shape(
            hfun_msht, bx, fit_inside=False, inverse=True
        )

        # Due to hfun msh_t zigzag, some nodes of size 1000 might fall
        # outside the box and viceversa

        n_in_is1000 = np.sum(clipped_hfun.value == 1000)
        n_in_is500 = np.sum(clipped_hfun.value == 500)
        n_out_is1000 = np.sum(inv_clipped_hfun.value == 1000)
        n_out_is500 = np.sum(inv_clipped_hfun.value == 500)

        self.assertTrue(n_in_is500 / n_in_is1000 < 0.1)
        self.assertTrue(n_out_is1000 / n_out_is500 < 0.05)

        self.assertTrue(np.isclose(np.mean(clipped_hfun.value), 1000, rtol=0.25))
        self.assertTrue(np.isclose(np.mean(inv_clipped_hfun.value), 500, rtol=0.1))


class HfunConstraintApplyHold(unittest.TestCase):
    def test_hold_rasterhfun_constraint(self):
        hfun_rast = ocsmesh.Hfun(
            ocsmesh.Raster(TEST_FILE),
            hmin=500,
            hmax=10000
        )
        hfun_rast.apply_constraints = MagicMock()

        with hfun_rast.hold_applying_added_constratins():
            hfun_rast.add_topo_bound_constraint(
                value=2700, lower_bound=0, value_type='min')
            hfun_rast.add_constant_value(value=150, lower_bound=-10)
            hfun_rast.add_constant_value(value=150, lower_bound=-20)
            hfun_rast.add_constant_value(value=150, lower_bound=-30)

        hfun_rast.apply_constraints.assert_called_once()


    def test_hold_meshhfun_constraint(self):
        msh_t = ocsmesh.utils.create_rectangle_mesh(
            nx=29, ny=17, holes=[40, 41],
            x_extent=(-76, -75), y_extent=(37, 38)
        )
        msh_t.crs = CRS.from_epsg(4326)

        hfun_mesh = ocsmesh.Hfun(ocsmesh.Mesh(msh_t))

        hfun_mesh.apply_constraints = MagicMock()

        with hfun_mesh.hold_applying_added_constratins():
            hfun_mesh.add_region_constraint(
                value=150,
                shape=geometry.box(-75.75, 37.25, -75.5, 37.5),
                value_type='min',
                crs=CRS.from_epsg(4326)
            )
            hfun_mesh.add_feature(
                feature=geometry.LineString(
                    [[-75.5, 37.5], [-75.25, 37.75]]
                ),
                expansion_rate=0.02,
                target_size=200,
            )
            hfun_mesh.add_feature(
                feature=geometry.LineString(
                    [[-75.5, 37.25], [-75.25, 37.75]]
                ),
                expansion_rate=0.03,
                target_size=300,
            )

        hfun_mesh.apply_constraints.assert_called_once()


if __name__ == '__main__':
    unittest.main()
