"""Splitting large rasters into windows (HfunCollector split_size).

A raster larger than ``split_size`` pixels is cut into windows that are
computed with a halo and keep only their core. Without the halo the
window edges look like channels; with it the result must match the
unsplit raster, up to the one-pixel flips at the edges of detected
polygons (they lie on pixel edges, and ``add_patch`` marks the pixels
they touch).
"""

import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString, MultiPolygon, box

from ocsmesh import Raster, utils
from ocsmesh.hfun import split as hsplit
from ocsmesh.hfun.collector import HfunCollector
from tests.api import multitile

TARGET = 100.0


def coast_with_creeks(x, y):
    """Water west of a diagonal coast, land east of it, with creeks
    (water strips about 450 m wide) running into the land."""
    d = (x + 0.5 * y) - 0.06
    z = np.where(d < 0, -10.0, 3.0)
    for yc in (0.015, 0.035, 0.055, 0.075, 0.09):
        z = np.where((np.abs(y - yc - 0.00007) < 0.00113) & (d < 0.04),
                     -2.0, z)
    return z.astype('float32')


def _core_values(collector):
    """The size function on the input grid, assembled from the cores of
    the windows (or the unsplit raster)."""
    if collector._input_slots is None:  # pylint: disable=protected-access
        return collector._hfun_list[0].get_values().copy()  # pylint: disable=protected-access
    out = None
    for hfun in collector._hfun_list:  # pylint: disable=protected-access
        core = collector._core_of(hfun)  # pylint: disable=protected-access
        if out is None:
            out = np.full(core.shape, np.nan, dtype='float32')
        values = hfun.get_values()
        with rasterio.open(hfun.raster.tmpfile) as src:
            transform = src.transform
        rows, cols = np.mgrid[0:values.shape[0], 0:values.shape[1]]
        x, y = transform * (cols + 0.5, rows + 0.5)
        pcol, prow = ~core.affine * (x, y)
        pcol = np.floor(pcol).astype(int)
        prow = np.floor(prow).astype(int)
        keep = ((pcol >= core.cols[0]) & (pcol < core.cols[1])
                & (prow >= core.rows[0]) & (prow < core.rows[1]))
        out[prow[keep], pcol[keep]] = values[keep]
    return out


class _TileCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.path = cls.tmp / 'tile.tif'
        multitile.write_tile(
            cls.path, multitile.TileSpec('t', 0.0002, (0.0, 0.0, 0.1, 0.1)),
            coast_with_creeks)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _collector(self, inputs=None, **kwargs):
        mode = kwargs.pop('mode', 'serial')
        return HfunCollector(
            inputs or [str(self.path)], hmin=50.0, hmax=3000.0,
            nprocs=2 if mode == 'parallel' else 1, execution_mode=mode,
            **kwargs)


class TestCores(_TileCase):

    def test_cores_tile_the_grid(self):
        raster = Raster(str(self.path))
        cores = hsplit.plan_cores(raster, raster, 170)
        self.assertEqual(len(cores), 9)
        height, width = raster.src.height, raster.src.width
        rows, cols = np.mgrid[0:height, 0:width]
        x, y = raster.src.transform * (cols + 0.5, rows + 0.5)
        count = sum(core.contains(x, y).astype(int) for core in cores)
        np.testing.assert_array_equal(count, 1)
        # Points on the outer edge, and a little outside it, still count
        edge = np.array([[0.0, 0.05], [0.1, 0.05], [0.05, -0.00005]])
        count = sum(core.contains(edge[:, 0], edge[:, 1]).astype(int)
                    for core in cores)
        np.testing.assert_array_equal(count, 1)

    def test_small_raster_not_split(self):
        raster = Raster(str(self.path))
        self.assertEqual(hsplit.plan_cores(raster, raster, 500), [])

    def test_clip_to_core(self):
        raster = Raster(str(self.path))
        core = hsplit.plan_cores(raster, raster, 250)[0]
        west, south, east, north = core.bounds()
        poly = box(west - 0.01, south + 0.001, west + 0.01, south + 0.002)
        clipped = hsplit.clip_to_core(poly, raster.crs, core)
        self.assertIsInstance(clipped, MultiPolygon)
        self.assertAlmostEqual(clipped.bounds[0], west)
        line = LineString([(west - 0.01, south + 0.001),
                           (east + 0.01, south + 0.001)])
        clipped = hsplit.clip_to_core(line, raster.crs, core)
        self.assertAlmostEqual(clipped.length, east - west)
        self.assertIsNone(hsplit.clip_to_core(
            box(5.0, 5.0, 6.0, 6.0), raster.crs, core))


class TestSplitChannels(_TileCase):

    def _channels(self, **kwargs):
        collector = self._collector(**kwargs)
        collector.add_channel(level=0.0, width=1000.0, target_size=TARGET)
        collector._apply_features()  # pylint: disable=protected-access
        values = _core_values(collector)
        self.assertFalse(np.isnan(values).any())
        return values <= TARGET, collector

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.reference = None

    def _reference(self):
        if self.reference is None:
            type(self).reference, _ = self._channels()
        return self.reference

    @staticmethod
    def _beyond_one_pixel(a, b):
        """Pixels of ``a`` more than one pixel away from ``b``."""
        return int((a & ~binary_dilation(b, np.ones((3, 3), bool))).sum())

    def test_split_matches_unsplit(self):
        ref = self._reference()
        split, collector = self._channels(split_size=170)
        self.assertEqual(len(collector._hfun_list), 9)  # pylint: disable=protected-access
        self.assertEqual(self._beyond_one_pixel(ref, split), 0)
        self.assertEqual(self._beyond_one_pixel(split, ref), 0)

    def test_without_halo_seams_look_like_channels(self):
        ref = self._reference()
        split, _ = self._channels(split_size=170, split_halo=0.0)
        self.assertGreater(self._beyond_one_pixel(split, ref), 1000)

    def test_parallel_matches_serial(self):
        serial, _ = self._channels(split_size=170)
        parallel, _ = self._channels(split_size=170, mode='parallel')
        np.testing.assert_array_equal(serial, parallel)

    def test_auto_halo(self):
        collector = self._collector(split_size=170)
        collector.add_channel(level=0.0, width=1000.0, target_size=TARGET,
                              tolerance=30.0)
        detect, reach, _ = collector._split_halo_parts()  # pylint: disable=protected-access
        self.assertEqual(detect, 1030.0)
        self.assertEqual(reach, 0.0)
        collector.add_topo_bound_constraint(value=100.0, upper_bound=0.0,
                                            rate=0.01)
        _, reach, _ = collector._split_halo_parts()  # pylint: disable=protected-access
        self.assertAlmostEqual(reach, (3000.0 / 50.0 - 1.0) / 0.01)
        # 5.9 km reach > 170 px * 22 m windows: limited, with a warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            collector._apply_features()  # pylint: disable=protected-access
        self.assertTrue(any('split_halo' in str(w.message) for w in caught))


class TestSplitInputIndices(_TileCase):
    """``source_index`` counts input rasters: windows share theirs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.small = cls.tmp / 'small.tif'
        multitile.write_tile(
            cls.small, multitile.TileSpec('s', 0.0002, (0.1, 0.0, 0.12, 0.02)),
            lambda x, y: np.full_like(x, -5.0, dtype='float32'))

    def _refined(self, source_index):
        collector = self._collector([str(self.path), str(self.small)],
                                    split_size=170)
        collector.add_constant_value(700.0, source_index=source_index)
        collector._apply_features()  # pylint: disable=protected-access
        hfuns = collector._hfun_list  # pylint: disable=protected-access
        self.assertEqual(len(hfuns), 10)
        return [bool((h.get_values() == 700.0).any()) for h in hfuns]

    def test_constant_value(self):
        self.assertEqual(self._refined([0]), [True] * 9 + [False])
        self.assertEqual(self._refined([1]), [False] * 9 + [True])

    def test_constraint(self):
        collector = self._collector([str(self.path), str(self.small)],
                                    split_size=170)
        collector.add_topo_bound_constraint(
            value=600.0, upper_bound=100.0, value_type='max', rate=None,
            source_index=[1])
        collector._apply_features()  # pylint: disable=protected-access
        hfuns = collector._hfun_list  # pylint: disable=protected-access
        self.assertEqual([bool((h.get_values() == 600.0).any())
                          for h in hfuns], [False] * 9 + [True])


class TestSplitComposite(_TileCase):

    def test_nodes_from_one_core_each(self):
        collector = self._collector(split_size=170)
        collector.add_constant_value(500.0, upper_bound=0.0)
        meshdata = collector.meshdata()
        utils.reproject(meshdata, 'EPSG:4326')
        coords = np.asarray(meshdata.coords)
        cores = list(collector._cores.values())  # pylint: disable=protected-access
        count = sum(core.contains_lonlat(coords).astype(int)
                    for core in cores)
        # Every node comes from the core of one window, none from a halo
        # (nodes on a seam may be claimed by the cores on both sides
        # after the round trip through UTM).
        self.assertTrue(np.all(count >= 1))
        self.assertLess(np.mean(count > 1), 0.05)
        values = np.asarray(meshdata.values).ravel()
        self.assertTrue(np.all((values >= 50.0) & (values <= 3000.0)))


if __name__ == '__main__':
    unittest.main()
