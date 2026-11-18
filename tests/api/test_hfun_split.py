"""Channel and contour detection on windows of large rasters
(HfunCollector split_size).

A raster larger than ``split_size`` pixels is detected on windows: each
is the raster's detection grid cut to a core plus a halo, and keeps the
shapes in its core. The size function itself is computed on the whole
raster. Without the halo the window edges look like channels; with it
the result must match the unsplit detection, up to the one-pixel flips
at the edges of detected polygons (they lie on pixel edges, and
``add_patch`` marks the pixels they touch).
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString, MultiPolygon, Polygon, box

from ocsmesh import Raster
from ocsmesh.hfun import resampling
from ocsmesh.hfun import split as hsplit
from ocsmesh.hfun.collector import HfunCollector
from tests.api import multitile

TARGET = 100.0


def coast_with_creeks(x, y):
    """Water west of a diagonal coast, land east of it, with creeks
    (water strips about 450 m wide) running into the land, and a thin
    lake (70 m by 4.4 km, 0.3 km2) across the middle of the tile.

    The lake is a channel only as a whole: the pieces a window edge cuts
    it into are smaller than the channel size cutoff (0.21 km2 for a
    1000 m width), so a window that only sees its own piece drops it.
    """
    d = (x + 0.5 * y) - 0.06
    z = np.where(d < 0, -10.0, 3.0)
    for yc in (0.015, 0.035, 0.055, 0.075, 0.09):
        z = np.where((np.abs(y - yc - 0.00007) < 0.00113) & (d < 0.04),
                     -2.0, z)
    lake = (np.abs(y - 0.0645) < 0.0003) & (x > 0.035) & (x < 0.075)
    z = np.where(lake, -2.0, z)
    # Some relief, so that resampling has values to combine
    z = z + (0.3 * np.sin(x * 900.0) * np.cos(y * 700.0)).astype(z.dtype)
    return z.astype('float32')


def _beyond_one_pixel(a, b):
    """Pixels of ``a`` more than one pixel away from ``b``."""
    return int((a & ~binary_dilation(b, np.ones((3, 3), bool))).sum())


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

    def _grid(self):
        return hsplit.Grid.of(Raster(str(self.path)))


class TestCoresAndWindows(_TileCase):

    def test_cores_tile_the_grid(self):
        raster = Raster(str(self.path))
        cores = hsplit.plan_cores(self._grid(), raster, 170)
        self.assertEqual(len(cores), 9)
        count = np.zeros((raster.src.height, raster.src.width), int)
        for core in cores:
            count[slice(*core.rows), slice(*core.cols)] += 1
        np.testing.assert_array_equal(count, 1)

    def test_small_raster_not_split(self):
        raster = Raster(str(self.path))
        self.assertEqual(hsplit.plan_cores(self._grid(), raster, 500), [])

    def test_window_copies_the_grid(self):
        raster = Raster(str(self.path))
        core = hsplit.plan_cores(self._grid(), raster, 250)[3]
        window = hsplit.Window(str(self.path), core, (7, 5))
        out = hsplit.write_window(window, str(self.tmp / 'w_native.tif'))
        (c0, c1), (r0, r1) = window.pixels()
        self.assertEqual((c0, c1, r0, r1), (243, 500, 245, 500))
        with rasterio.open(out) as w, rasterio.open(self.path) as src:
            np.testing.assert_array_equal(
                w.read(1), src.read(1)[r0:r1, c0:c1])
            self.assertEqual(w.transform,
                             src.transform * rasterio.Affine.translation(
                                 c0, r0))

    def test_resampled_window_is_part_of_the_whole_resample(self):
        # The pixels of a window of a resampled grid are those of the
        # whole raster resampled: the windows of a raster share one grid
        # (resampling each window on its own would shift its pixels).
        raster = Raster(str(self.path))
        scale = resampling.scaling_for(raster, 31.0)
        whole = resampling.resampled_copy(raster, 31.0, 'average')
        grid = hsplit.Grid.resampled(raster, scale)
        self.assertEqual(grid, hsplit.Grid.of(whole))
        for core in hsplit.plan_cores(grid, raster, 250):
            window = hsplit.Window(str(self.path), core, (6, 6), 'average')
            out = hsplit.write_window(window, str(self.tmp / 'w_rs.tif'))
            (c0, c1), (r0, r1) = window.pixels()
            with rasterio.open(out) as w:
                np.testing.assert_allclose(
                    w.read(1), whole.src.read(1)[r0:r1, c0:c1],
                    rtol=0, atol=1e-6)

    def test_clip_to_core(self):
        raster = Raster(str(self.path))
        core = hsplit.plan_cores(self._grid(), raster, 250)[0]
        west, south, east, north = core.bounds()
        poly = box(west - 0.01, south + 0.001, west + 0.01, south + 0.002)
        clipped = hsplit.clip_to_core(poly, raster.crs, core)
        self.assertIsInstance(clipped, MultiPolygon)
        self.assertAlmostEqual(clipped.bounds[0], west)
        line = LineString([(west - 0.01, south + 0.001),
                           (east + 0.01, south + 0.001)])
        clipped = hsplit.clip_to_core(line, raster.crs, core)
        self.assertAlmostEqual(clipped.length, east - west)
        # A line that only touches the core leaves nothing
        touching = LineString([(west - 0.01, north + 0.01),
                               (west, north), (west - 0.01, north)])
        self.assertIsNone(hsplit.clip_to_core(touching, raster.crs, core))
        self.assertIsNone(hsplit.clip_to_core(
            box(5.0, 5.0, 6.0, 6.0), raster.crs, core))
        # An invalid polygon (with a spike) that make_valid turns into a
        # GeometryCollection keeps its polygon part
        spiked = Polygon([(west - 0.01, south + 0.001),
                          (west + 0.01, south + 0.001),
                          (west + 0.01, south + 0.002),
                          (west - 0.01, south + 0.002),
                          (west - 0.01, south + 0.001),
                          (west - 0.015, south + 0.0005),
                          (west - 0.01, south + 0.001)])
        self.assertFalse(spiked.is_valid)
        clipped = hsplit.clip_to_core(spiked, raster.crs, core)
        self.assertIsInstance(clipped, MultiPolygon)
        self.assertAlmostEqual(clipped.area, 0.01 * 0.001)


class TestDropSmallChannels(_TileCase):
    """Pieces of one channel cut by core edges are joined before the
    channel size cutoff."""

    def _cores(self):
        return hsplit.plan_cores(self._grid(), Raster(str(self.path)), 250)

    def test_pieces_joined_across_cores(self):
        raster = Raster(str(self.path))
        left, right = self._cores()[:2]   # 2x2, seam x=0.05
        width = 1000.0   # cutoff 0.21 km2
        # A 70 m x 4.4 km strip across the seam (0.3 km2), in two
        # pieces of 0.17 and 0.14 km2; a 0.1 km2 strip on its own.
        strip = box(0.035, 0.0642, 0.075, 0.0648)
        alone = box(0.01, 0.0642, 0.025, 0.0648)
        items = [
            (hsplit.clip_to_core(strip, raster.crs, left), raster.crs, left,
             None),
            (hsplit.clip_to_core(strip, raster.crs, right), raster.crs,
             right, None),
            (hsplit.clip_to_core(alone, raster.crs, left), raster.crs, left,
             None),
        ]
        out = hsplit.drop_small_channels(items, width)
        self.assertIsNotNone(out[0])
        self.assertIsNotNone(out[1])
        self.assertAlmostEqual(out[0].union(out[1]).area, strip.area)
        self.assertIsNone(out[2])
        # Each piece on its own is below the cutoff
        self.assertIsNone(hsplit.drop_small_channels(items[:1], width)[0])
        # Shapes of unsplit rasters are left alone
        self.assertIs(hsplit.drop_small_channels(
            [(alone, raster.crs, None, None)], width)[0], alone)

    def test_corner_contact_not_joined(self):
        raster = Raster(str(self.path))
        left, right = self._cores()[:2]
        # Squares of 0.2 km2 (below the 0.21 km2 cutoff), two together
        # above it: joined when they share part of the seam x=0.05,
        # not when they only meet at a corner.
        sq_left = box(0.046, 0.060, 0.05, 0.064)
        beside = box(0.05, 0.060, 0.054, 0.064)
        corner = box(0.05, 0.064, 0.054, 0.068)
        # Sharing only 1 m of the seam (a twentieth of a pixel), like a
        # channel with a narrow neck there: still one channel, as unsplit
        neck = box(0.05, 0.064 - 0.000009, 0.054, 0.068)
        for other, joined in ((beside, True), (corner, False),
                              (neck, True)):
            with self.subTest(joined=joined):
                items = [(MultiPolygon([sq_left]), raster.crs, left, None),
                         (MultiPolygon([other]), raster.crs, right, None)]
                out = hsplit.drop_small_channels(items, 1000.0)
                self.assertEqual(out[0] is not None, joined)
                self.assertEqual(out[1] is not None, joined)

    def test_pieces_of_one_detected_polygon_kept_together(self):
        raster = Raster(str(self.path))
        left = self._cores()[0]
        # Two 0.2 km2 squares meeting at a corner: as two detected
        # polygons they are judged apart (and dropped); as one detected
        # polygon whose ring touches itself there (invalid, repaired
        # into the same two squares) they are one channel (kept).
        separate = MultiPolygon([box(0.03, 0.06, 0.034, 0.064),
                                 box(0.034, 0.064, 0.038, 0.068)])
        pinched = Polygon([(0.03, 0.06), (0.034, 0.06), (0.034, 0.064),
                           (0.038, 0.064), (0.038, 0.068), (0.034, 0.068),
                           (0.034, 0.064), (0.03, 0.064)])
        self.assertFalse(pinched.is_valid)
        for shape, kept in ((separate, False), (pinched, True)):
            with self.subTest(kept=kept):
                pieces, labels = hsplit.clip_channels_to_core(
                    shape, raster.crs, left)
                self.assertEqual(len(pieces.geoms), 2)
                out = hsplit.drop_small_channels(
                    [(pieces, raster.crs, left, labels)], 1000.0)
                self.assertEqual(out[0] is not None, kept)


class TestSplitDetection(_TileCase):
    """Sizes with detection on windows against detection on the whole
    raster."""

    def _sizes(self, refine, **kwargs):
        collector = self._collector(**kwargs)
        refine(collector)
        collector._apply_features()  # pylint: disable=protected-access
        # The size function is never split
        self.assertEqual(len(collector._hfun_list), 1)  # pylint: disable=protected-access
        values = collector._hfun_list[0].get_values()  # pylint: disable=protected-access
        self.assertFalse(np.isnan(values).any())
        return values

    @staticmethod
    def _channels(factor=None):
        def refine(collector):
            collector.add_channel(level=0.0, width=1000.0,
                                  target_size=TARGET,
                                  resampling_factor=factor)
        return refine

    @staticmethod
    def _contour(collector):
        collector.add_contour(level=0.0, expansion_rate=0.02,
                              target_size=60.0)

    def _assert_same_channels(self, ref, split):
        # Pixels along channel edges may flip either way (the polygons
        # lie on pixel edges and add_patch marks the pixels they touch),
        # but no channel is lost or added beyond that.
        ref, split = ref <= TARGET, split <= TARGET
        self.assertEqual(_beyond_one_pixel(ref, split), 0)
        self.assertEqual(_beyond_one_pixel(split, ref), 0)

    def test_channels_match_unsplit(self):
        ref = self._sizes(self._channels())
        # 3x3 and 2x2: with 2x2 the lake is cut in two pieces that are
        # each below the channel size cutoff.
        for split_size in (170, 250):
            with self.subTest(split_size=split_size):
                self._assert_same_channels(
                    ref, self._sizes(self._channels(),
                                     split_size=split_size))

    def test_resampled_channels_match_unsplit(self):
        # Detection on 25 m pixels (from 22 m): the windows are cut from
        # the grid of the whole raster resampled.
        ref = self._sizes(self._channels(0.5))
        self._assert_same_channels(
            ref, self._sizes(self._channels(0.5), split_size=170))

    def test_contours_match_unsplit(self):
        # The lines are those of the whole raster (window positions
        # mapped to it, pieces merged back); add_feature samples them
        # from their start, so the few lines that start elsewhere once
        # merged (a ring cut open at another point) give other sizes
        # next to them.
        ref = self._sizes(self._contour)
        split = self._sizes(self._contour, split_size=170)
        rel = np.abs(split - ref) / ref
        self.assertEqual(np.median(rel), 0.0)
        self.assertLess(np.percentile(rel, 99), 0.01)

    def test_without_halo_seams_look_like_channels(self):
        ref = self._sizes(self._channels()) <= TARGET
        split = self._sizes(self._channels(), split_size=170,
                            split_halo=0.0) <= TARGET
        self.assertGreater(_beyond_one_pixel(split, ref), 1000)

    def test_parallel_matches_serial(self):
        serial = self._sizes(self._channels(), split_size=250)
        parallel = self._sizes(self._channels(), split_size=250,
                               mode='parallel')
        np.testing.assert_array_equal(serial, parallel)

    def test_halo(self):
        collector = self._collector(split_size=170)
        collector.add_channel(level=0.0, width=1000.0, target_size=TARGET,
                              tolerance=30.0)
        defn = next(iter(collector._ch_info_coll))[0]  # pylint: disable=protected-access
        grid = self._grid()
        four_px = 4 * max(grid.pixel_size_m())
        self.assertAlmostEqual(
            collector._split_halo_m(defn, grid), 1030.0 + four_px)  # pylint: disable=protected-access
        given = self._collector(split_size=170, split_halo=500.0)
        self.assertEqual(given._split_halo_m(defn, grid), 500.0)  # pylint: disable=protected-access

    def test_optimum(self):
        with self.assertWarns(UserWarning) as caught:
            collector = self._collector(split_size='optimum')
        self.assertIn('New Jersey', str(caught.warning))
        self.assertEqual(collector._split_size, 4100)  # pylint: disable=protected-access

    def test_arguments(self):
        for bad in (1, 2.5, True, 'auto', 'best'):
            with self.assertRaises(ValueError):
                self._collector(split_size=bad)
        with self.assertRaises(ValueError):
            self._collector(split_size=100, split_halo=-1.0)
        with self.assertWarns(UserWarning):
            collector = self._collector(split_size=100, method='fast')
        self.assertIsNone(collector._split_size)  # pylint: disable=protected-access


if __name__ == '__main__':
    unittest.main()
