"""Inputs in either longitude convention meet each other.

The same DEM is written at 190 E (0-360 longitudes) and at -170
(-180..180 longitudes). Refinements, constraints, clipping and
interpolation give identical results on both, whichever convention the
shapes and meshes are given in, and for shapes split at 180.
"""

import gc
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, box

from ocsmesh import Raster, Mesh, utils, geodesy
from ocsmesh.hfun.collector import HfunCollector
from ocsmesh.hfun.raster import HfunRaster

LAT0 = 55.0
SIZE = 0.2
RES = 0.002


def write_dem(path, lon0):
    """Sea in the west half, land in the east half."""
    n = int(round(SIZE / RES))
    cols = np.arange(n)[None, :].repeat(n, axis=0)
    z = np.where(cols < n // 2, -10.0, 10.0).astype('float32')
    with rasterio.open(
            path, 'w', driver='GTiff', width=n, height=n, count=1,
            dtype='float32', crs='EPSG:4326', nodata=-99999.0,
            transform=from_origin(lon0, LAT0 + SIZE, RES, RES)) as dst:
        dst.write(z, 1)
    return str(path)


def shifted(geom, dx):
    return affinity.translate(geom, xoff=dx)


class TestAlignHelpers(unittest.TestCase):

    def test_wrap_lon_near_is_exact(self):
        x = np.array([0.75, -0.75, 179.0])
        np.testing.assert_array_equal(geodesy.wrap_lon_near(x, 0.0), x)
        np.testing.assert_array_equal(
            geodesy.wrap_lon_near([-170.0, 190.0], 185.0), [190.0, 190.0])

    def test_align_geometry(self):
        split = MultiPolygon([box(179, 50, 180, 51), box(-180, 50, -179, 51)])
        self.assertEqual(geodesy.align_geometry(split, 180).bounds,
                         (179, 50, 181, 51))
        self.assertEqual(geodesy.align_geometry(box(170, 50, 200, 51), -175)
                         .bounds, (-190, 50, -160, 51))
        b = box(-75, 40, -74, 41)
        self.assertIs(geodesy.align_geometry(b, -74.5), b)


class TestTwinDems(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.east = write_dem(d / 'east.tif', 190.0)    # 0-360
        cls.west = write_dem(d / 'west.tif', -170.0)   # -180..180

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        cls.tmp.cleanup()

    def _hfun(self, path):
        return HfunRaster(Raster(path), hmin=100, hmax=5000)

    def _values(self, path, apply):
        hfun = self._hfun(path)
        apply(hfun)
        return hfun.get_values(band=1)

    def _check_all_equal(self, make_apply, geom_west):
        """``make_apply(geom)`` refines a size function with ``geom``;
        the west geometry, and its copy shifted by 360, on both DEMs."""
        reference = self._values(self.west, make_apply(geom_west))
        for path in (self.west, self.east):
            for dx in (0.0, 360.0):
                got = self._values(path, make_apply(shifted(geom_west, dx)))
                np.testing.assert_allclose(got, reference, rtol=1e-6)
        return reference

    def test_add_patch(self):
        patch = box(-169.95, 55.05, -169.9, 55.1)
        ref = self._check_all_equal(
            lambda g: lambda h: h.add_patch(g, target_size=200,
                                            expansion_rate=0.002), patch)
        self.assertEqual(ref.min(), 200)

    def test_add_patch_without_rate(self):
        patch = box(-169.95, 55.05, -169.9, 55.1)
        ref = self._check_all_equal(
            lambda g: lambda h: h.add_patch(g, target_size=200), patch)
        self.assertEqual(ref.min(), 200)

    def test_add_feature(self):
        line = LineString([(-169.9, 55.02), (-169.9, 55.18)])
        ref = self._check_all_equal(
            lambda g: lambda h: h.add_feature(g, expansion_rate=0.002,
                                              target_size=100), line)
        self.assertLess(ref.min(), 150)
        self.assertGreater(ref.max(), 1000)

    def test_region_constraint(self):
        region = box(-169.95, 55.0, -169.85, 55.2)

        def make(g):
            def apply(h):
                h.add_constant_value(3000)
                h.add_region_constraint(500, g, crs='EPSG:4326',
                                        value_type='max', rate=None)
            return apply
        ref = self._check_all_equal(make, region)
        self.assertEqual(ref.min(), 500)

    def test_clip(self):
        shape = box(-169.95, 55.02, -169.85, 55.15)
        counts = []
        for path in (self.west, self.east):
            for dx in (0.0, 360.0):
                raster = Raster(path)
                raster.clip(shifted(shape, dx))
                counts.append(int(np.sum(~raster.get_values(
                    band=1, masked=True).mask)))
        self.assertGreater(counts[0], 0)
        self.assertEqual(len(set(counts)), 1, counts)

    def test_collector_base_shape_other_convention(self):
        sizes = []
        for path, dx in ((self.west, 0.0), (self.east, 0.0),
                         (self.east, -360.0)):
            base = shifted(box(-170.0 + 360.0, 55.0, -169.85 + 360.0, 55.2),
                           dx - 360.0) if path == self.west else shifted(
                box(190.0, 55.0, 190.15, 55.2), dx)
            hfun = HfunCollector([path], hmin=100, hmax=5000,
                                 base_shape=base, execution_mode='serial')
            hfun.add_constant_value(1000)
            md = hfun.meshdata()
            sizes.append(len(md.values))
        self.assertGreater(sizes[0], 0)
        self.assertEqual(sizes[0], sizes[1])
        self.assertEqual(sizes[0], sizes[2])

    def test_interpolate_mesh_other_convention(self):
        values = []
        for path in (self.west, self.east):
            msh = utils.create_rectangle_mesh(
                nx=8, ny=8, holes=[], x_extent=(-169.98, -169.82),
                y_extent=(55.02, 55.18))
            msh.crs = CRS.from_epsg(4326)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mesh = Mesh(msh)
                mesh.interpolate(Raster(path), method='nearest', nprocs=1)
            values.append(mesh.value.ravel().copy())
        self.assertFalse(np.any(np.isnan(values[0])))
        np.testing.assert_array_equal(values[0], values[1])

    def test_raster_bbox_in_other_geographic_crs(self):
        # NAD83 is another geographic CRS: PROJ wraps 190 to -170
        bbox = Raster(self.east).get_bbox(crs='EPSG:4269')
        x0, _, x1, _ = bbox.bounds
        self.assertAlmostEqual(x1 - x0, SIZE, places=3)
        self.assertGreater(x0, 180 - 1e-6 - 360)

    def test_mesh_bbox_split_at_180(self):
        msh = utils.create_rectangle_mesh(
            nx=5, ny=5, holes=[], x_extent=(179.0, 181.0), y_extent=(50, 51))
        msh.crs = CRS.from_epsg(4326)
        msh.coords[:, 0] = geodesy.wrap_lon(msh.coords[:, 0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x0, _, x1, _ = Mesh(msh).get_bbox().bounds
        self.assertAlmostEqual(x0, 179.0)
        self.assertAlmostEqual(x1, 181.0)


class TestShapesSplitAt180(unittest.TestCase):
    """A DEM across 180 (179.9..180.1 in 0-360) and a patch given as two
    pieces either side of 180 in -180..180 longitudes."""

    @classmethod
    def setUpClass(cls):
        # Removed in tearDownClass, after the size functions holding the
        # file open are gone (Windows cannot delete an open file).
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = write_dem(Path(cls.tmp.name) / 'across.tif', 179.9)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        cls.tmp.cleanup()

    def test_split_patch(self):
        whole = box(179.95, 55.05, 180.05, 55.1)
        split = MultiPolygon([box(179.95, 55.05, 180.0, 55.1),
                              box(-180.0, 55.05, -179.95, 55.1)])
        out = []
        for patch in (whole, split):
            hfun = HfunRaster(Raster(self.path), hmin=100, hmax=5000)
            hfun.add_patch(patch, target_size=200, expansion_rate=0.002)
            out.append(hfun.get_values(band=1))
            del hfun
        # Same patch area; outside it, add_feature samples the two
        # pieces' edges at other points than the whole box's, so the
        # sizes agree to its sampling resolution (about 1%). The seam
        # inside the patch is never the nearest edge.
        np.testing.assert_allclose(out[0], out[1], rtol=2e-2)
        np.testing.assert_array_equal(out[0] == 200, out[1] == 200)
        self.assertEqual(out[1].min(), 200)


if __name__ == '__main__':
    unittest.main()
