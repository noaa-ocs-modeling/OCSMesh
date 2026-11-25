"""``metric_crs``: where OCSMesh measures metres.

* ``ocsmesh.geodesy`` picks UTM for small domains (as before) and the
  best fitted conformal projection for wide ones, on either side of 180°
  and in either longitude convention.
* An explicit CRS reaches every step, in the workers too.
* Rasters east of 180 (0-360 longitudes) give the same channels and
  feature distances as the same raster in -180..180 longitudes; they
  used to be computed in degrees.
"""

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import LineString, box
import geopandas as gpd

from ocsmesh import Raster, Mesh, utils, geodesy
from ocsmesh.driver import MeshDriver
from ocsmesh.geom.base import geoseries_metric_crs
from ocsmesh.hfun.collector import HfunCollector
from ocsmesh.hfun.mesh import HfunMesh
from ocsmesh.hfun.raster import HfunRaster


def grid(lon0, lon1, lat0, lat1, n=40):
    lon, lat = np.meshgrid(np.linspace(lon0, lon1, n),
                           np.linspace(lat0, lat1, n))
    return lon.ravel(), lat.ravel()


def proj_name(crs):
    return CRS.from_user_input(crs).to_dict().get('proj')


class TestLonHelpers(unittest.TestCase):

    def test_lon_arc(self):
        self.assertEqual(geodesy.lon_arc([-75, -72]), (-75, -72))
        self.assertEqual(geodesy.lon_arc([170, 200, 185]), (170, 200))
        # split at 180 in -180..180: the same arc
        self.assertEqual(geodesy.lon_arc([170, 179, -179, -170]), (170, 190))
        # 0-360 around 0
        self.assertEqual(geodesy.lon_arc([350, 5]), (-10, 5))

    def test_lon_center(self):
        self.assertAlmostEqual(geodesy.lon_center([170, -170]), -180)
        self.assertAlmostEqual(geodesy.lon_center([190, 200]), -165)


class TestPickMetricCrs(unittest.TestCase):

    def test_small_domain_keeps_utm(self):
        lon, lat = grid(-76, -72, 38, 42)
        utm_crs = utils.estimate_bounds_utm((-76, 38, -72, 42))
        crs = geodesy.pick_metric_crs(lon, lat, utm_crs=utm_crs)
        self.assertTrue(crs.equals(utm_crs))

    def test_hawaii_0_360_keeps_utm(self):
        lon, lat = grid(199, 206, 18, 23)
        crs = geodesy.pick_metric_crs(lon, lat)
        self.assertEqual(proj_name(crs), 'utm')
        self.assertEqual(crs.to_dict()['zone'], 4)

    def test_wide_domains_get_conformal_fits(self):
        for (lon0, lon1, lat0, lat1), expected in [
                ((172, 230, 51, 72), 'lcc'),       # Alaska
                ((160, 210, -30, -5), 'lcc'),      # South Pacific
                ((-98, -60, 8, 46), 'stere')]:     # US Atlantic coast
            lon, lat = grid(lon0, lon1, lat0, lat1)
            crs = geodesy.pick_metric_crs(lon, lat)
            self.assertEqual(proj_name(crs), expected)
            err, _ = geodesy.scale_error(crs, lon, lat)
            utm_err, _ = geodesy.scale_error(
                geodesy.pick_metric_crs(lon, lat, 'utm'), lon, lat)
            self.assertLess(err, utm_err)

    def test_alaska_fit_numbers(self):
        lon, lat = grid(172, 230, 51, 72, n=61)
        crs = geodesy.pick_metric_crs(lon, lat)
        err, _ = geodesy.scale_error(crs, lon, lat)
        self.assertAlmostEqual(100 * err, 0.92, delta=0.05)
        self.assertAlmostEqual(crs.to_dict()['lon_0'], -159)

    def test_split_at_180_same_as_0_360(self):
        lon, lat = grid(172, 230, 51, 72)
        crs_360 = geodesy.pick_metric_crs(lon, lat)
        crs_split = geodesy.pick_metric_crs(geodesy.wrap_lon(lon), lat)
        self.assertTrue(crs_360.equals(crs_split))

    def test_utm_mode_and_explicit(self):
        lon, lat = grid(172, 230, 51, 72)
        self.assertEqual(
            proj_name(geodesy.pick_metric_crs(lon, lat, 'utm')), 'utm')
        crs = geodesy.pick_metric_crs(lon, lat, 'EPSG:3338')
        self.assertTrue(crs.equals(CRS.from_epsg(3338)))

    def test_check_metric_crs(self):
        self.assertEqual(geodesy.check_metric_crs(None), 'auto')
        self.assertEqual(geodesy.check_metric_crs('UTM'), 'utm')
        self.assertTrue(geodesy.check_metric_crs(3338).equals(
            CRS.from_epsg(3338)))
        with self.assertRaisesRegex(ValueError, 'projected'):
            geodesy.check_metric_crs('EPSG:4326')
        with self.assertRaisesRegex(ValueError, 'metre'):
            geodesy.check_metric_crs('EPSG:2263')   # US survey feet
        with self.assertRaisesRegex(ValueError, "'auto'"):
            geodesy.check_metric_crs('nonsense')


class TestMeshMetricCrs(unittest.TestCase):

    def _mesh(self, lon0, lon1, lat0, lat1):
        msh = utils.create_rectangle_mesh(
            nx=20, ny=20, holes=[], x_extent=(lon0, lon1),
            y_extent=(lat0, lat1))
        msh.crs = CRS.from_epsg(4326)
        return msh

    def test_small_mesh_unchanged(self):
        msh = self._mesh(-74.5, -73.5, 39.5, 40.5)
        self.assertTrue(utils.mesh_metric_crs(msh).equals(
            utils.estimate_mesh_utm(msh)))

    def test_project_to_metric_wide_mesh(self):
        msh = self._mesh(172, 230, 51, 72)
        utils.project_to_metric(msh)
        self.assertEqual(proj_name(msh.crs), 'lcc')
        self.assertTrue(np.all(np.isfinite(msh.coords)))

    def test_mesh_split_at_180_utm_zone(self):
        msh = self._mesh(176, 184, 50, 52)
        msh.coords[:, 0] = geodesy.wrap_lon(msh.coords[:, 0])
        # nodes on both sides of 180: zone 1 (centre 180), not zone 31
        self.assertEqual(utils.estimate_mesh_utm(msh).to_dict()['zone'], 1)

    def test_metric_mesh_left_alone(self):
        msh = self._mesh(-74.5, -73.5, 39.5, 40.5)
        utils.project_to_utm(msh)
        crs = msh.crs
        utils.project_to_metric(msh)
        self.assertTrue(msh.crs.equals(crs))
        utils.project_to_metric(msh, 'EPSG:3338')
        self.assertTrue(msh.crs.equals(CRS.from_epsg(3338)))


def channel_dem(path, lon0, lat0=55.0, res=0.001, size=0.2):
    """Sea in the west, land in the east, and a narrow (about 330 m)
    inlet running east into the land."""
    n = int(round(size / res))
    rows, cols = np.mgrid[0:n, 0:n]
    x = (cols + 0.5) * res
    y = size - (rows + 0.5) * res
    z = np.where(x < 0.08, -10.0, 10.0)
    inlet = (np.abs(y - size / 2) < 0.0015) & (x < 0.17)
    z[inlet] = -10.0
    with rasterio.open(
            path, 'w', driver='GTiff', width=n, height=n, count=1,
            dtype='float32', crs='EPSG:4326', nodata=-99999.0,
            transform=from_origin(lon0, lat0 + size, res, res)) as dst:
        dst.write(z.astype('float32'), 1)
    return path


class TestEastOf180(unittest.TestCase):
    """The same DEM at 190 E (0-360) and at -170 (-180..180)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.east = str(channel_dem(d / 'east_360.tif', 190.0))
        cls.west = str(channel_dem(d / 'west_180.tif', -170.0))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_channels_east_of_180(self):
        ch_east = Raster(self.east).get_channels(level=0, width=1000)
        ch_west = Raster(self.west).get_channels(level=0, width=1000)
        self.assertIsNotNone(ch_east)
        self.assertIsNotNone(ch_west)
        # same channel, shifted by 360 (in the raster's convention)
        self.assertAlmostEqual(ch_east.area, ch_west.area, places=8)
        self.assertAlmostEqual(ch_east.bounds[0] - 360, ch_west.bounds[0],
                               places=6)
        # the inlet, not the whole sea
        self.assertLess(ch_east.area, 0.01 * 0.2 * 0.2 * 10)

    def test_add_feature_distances_east_of_180(self):
        values = []
        for path, lon0 in ((self.east, 190.0), (self.west, -170.0)):
            hfun = HfunRaster(Raster(path), hmin=100, hmax=5000)
            line = LineString([(lon0 + 0.1, 55.02), (lon0 + 0.1, 55.18)])
            hfun.add_feature(line, expansion_rate=0.002, target_size=100)
            values.append(hfun.get_values(band=1))
        np.testing.assert_allclose(values[0], values[1], rtol=1e-5)
        # sizes grow away from the line (not hmin everywhere)
        self.assertGreater(values[0].max(), 1000)


class TestExplicitMetricCrs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.east = str(channel_dem(Path(cls.tmp.name) / 'east.tif', 190.0))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_hfun_raster(self):
        hfun = HfunRaster(Raster(self.east), hmin=100, hmax=5000,
                          metric_crs='EPSG:3338')
        hfun.add_constant_value(1000)
        self.assertTrue(hfun.meshdata().crs.equals(CRS.from_epsg(3338)))
        self.assertTrue(hfun._local_metric_crs(  # pylint: disable=W0212
            (190, 55, 190.1, 55.1)).equals(CRS.from_epsg(3338)))

    def test_hfun_raster_auto_small_is_utm(self):
        hfun = HfunRaster(Raster(self.east), hmin=100, hmax=5000)
        hfun.add_constant_value(1000)
        crs = hfun.meshdata().crs
        self.assertEqual(proj_name(crs), 'utm')
        # 190..190.2 E is -170..-169.8: zone 2
        self.assertEqual(crs.to_dict()['zone'], 2)

    def test_collector_serial_and_parallel(self):
        results = {}
        for mode in ('serial', 'parallel'):
            hfun = HfunCollector(
                [self.east], hmin=100, hmax=5000, nprocs=2,
                execution_mode=mode, metric_crs='EPSG:3338')
            hfun.add_channel(level=0, width=1000, target_size=200,
                             expansion_rate=0.002)
            md = hfun.meshdata()
            self.assertTrue(md.crs.equals(CRS.from_epsg(3338)))
            results[mode] = md
        np.testing.assert_allclose(
            np.sort(results['serial'].values.ravel()),
            np.sort(results['parallel'].values.ravel()), rtol=1e-6)
        # the channel refinement was applied (sizes below hmax)
        self.assertLess(results['serial'].values.min(), 1000)

    def test_collector_auto_wide_domain(self):
        # two small tiles 40 degrees apart, across 180 in 0-360
        d = Path(self.tmp.name)
        paths = [str(channel_dem(d / f'tile_{lon0:.0f}.tif', lon0))
                 for lon0 in (175.0, 215.0)]
        for mode in ('serial', 'parallel'):
            hfun = HfunCollector(paths, hmin=100, hmax=5000, nprocs=2,
                                 execution_mode=mode)
            hfun.add_constant_value(1000)
            md = hfun.meshdata()
            self.assertEqual(proj_name(md.crs), 'lcc')
            # nodes of both tiles are kept (none dropped east of 180)
            lon, _ = utils.Transformer.from_crs(
                md.crs, 'EPSG:4326', always_xy=True).transform(
                    md.coords[:, 0], md.coords[:, 1])
            lon = geodesy.wrap_lon(lon, 0)
            self.assertTrue(np.any(np.abs(lon - 175.1) < 0.1))
            self.assertTrue(np.any(np.abs(lon - 215.1) < 0.1))

    def test_collector_rejects_geographic(self):
        with self.assertRaisesRegex(ValueError, 'projected'):
            HfunCollector([self.east], hmin=100, hmax=5000,
                          metric_crs='EPSG:4326')

    def test_hfun_mesh(self):
        msh = utils.create_rectangle_mesh(
            nx=10, ny=10, holes=[], x_extent=(190, 191), y_extent=(55, 56))
        msh.crs = CRS.from_epsg(4326)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hfun = HfunMesh(Mesh(msh), metric_crs='EPSG:3338')
        self.assertTrue(hfun.meshdata().crs.equals(CRS.from_epsg(3338)))


class TestGeomAndDriver(unittest.TestCase):

    def test_geoseries_metric_crs(self):
        small = gpd.GeoSeries([box(-74.5, 39.5, -73.5, 40.5)], crs=4326)
        self.assertEqual(proj_name(geoseries_metric_crs(small)), 'utm')
        alaska = gpd.GeoSeries([box(172, 51, 230, 72)], crs=4326)
        self.assertEqual(proj_name(geoseries_metric_crs(alaska)), 'lcc')
        self.assertTrue(geoseries_metric_crs(alaska, 'EPSG:3338').equals(
            CRS.from_epsg(3338)))
        metric = small.to_crs(32618)
        self.assertIsNone(geoseries_metric_crs(metric))

    def test_driver_takes_hfun_metric_crs(self):
        class _Hfun(HfunMesh):  # pylint: disable=abstract-method
            def __init__(self):  # pylint: disable=super-init-not-called
                self._metric_crs = CRS.from_epsg(3338)

        from ocsmesh.geom.shapely import PolygonGeom
        geom = PolygonGeom(box(190, 55, 191, 56), 'EPSG:4326')
        driver = MeshDriver(geom, _Hfun())
        self.assertTrue(driver._metric_crs.equals(  # pylint: disable=W0212
            CRS.from_epsg(3338)))
        driver = MeshDriver(geom, _Hfun(), metric_crs='utm')
        self.assertEqual(driver._metric_crs, 'utm')  # pylint: disable=W0212
        self.assertEqual(MeshDriver(geom)._metric_crs, 'auto')  # pylint: disable=W0212


if __name__ == '__main__':
    unittest.main()
