"""resampling_factor='optimum': the New Jersey factors, with a warning."""

import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from ocsmesh.hfun import resampling
from ocsmesh.hfun.collector import HfunCollector
from tests.api import multitile


def _sloped(x, y):
    return (x - 0.05 + 0.2 * (y - 0.05)).astype('float32') * 200.0


class TestOptimum(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.path = cls.tmp / 'tile.tif'
        multitile.write_tile(
            cls.path, multitile.TileSpec('t', 0.0002, (0.0, 0.0, 0.1, 0.1)),
            _sloped)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _collector(self, factor):
        return HfunCollector([str(self.path)], hmin=50.0, hmax=3000.0,
                             nprocs=1, resampling_factor=factor)

    @staticmethod
    def _added(collector, coll_attr, add):
        """The resampling arguments ``add`` (an ``add_*`` call) stores."""
        coll = getattr(collector, coll_attr)
        with mock.patch.object(coll, 'add', wraps=coll.add) as spy:
            add(collector)
        kwargs = spy.call_args.kwargs
        return kwargs['resampling_factor'], kwargs['resampling_method']

    _REFINEMENTS = {
        'contour': ('_contour_info_coll',
                    lambda c, **kw: c.add_contour(
                        level=0.0, expansion_rate=0.01, target_size=100.0,
                        **kw)),
        'channel': ('_ch_info_coll',
                    lambda c, **kw: c.add_channel(
                        level=0.0, width=1000.0, target_size=100.0, **kw)),
        'flow_limiter': ('_flow_lim_coll',
                         lambda c, **kw: c.add_subtidal_flow_limiter(
                             hmin=100.0, hmax=3000.0, upper_bound=0.0,
                             lower_bound=-10.0, **kw)),
        'constant_value': ('_const_val_contour_coll',
                           lambda c, **kw: c.add_constant_value(
                               500.0, upper_bound=0.0, **kw)),
    }

    def test_warns_and_sets_the_grid(self):
        with self.assertWarns(UserWarning) as caught:
            collector = self._collector('optimum')
        self.assertIn('New Jersey', str(caught.warning))
        self.assertIn('Flow limiters are not covered', str(caught.warning))
        # pylint: disable=protected-access
        self.assertEqual(collector._resampling_factor, 0.5)
        self.assertEqual(collector._resampling_method, 'average')

    def test_refinement_defaults(self):
        expected = {
            'contour': (0.5, 'min'),
            'channel': (0.125, None),
            'flow_limiter': ('native', None),
            'constant_value': (None, None),   # the grid's
        }
        self.assertEqual(expected, resampling.OPTIMUM_REFINEMENTS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            collector = self._collector('optimum')
        for kind, (attr, add) in self._REFINEMENTS.items():
            with self.subTest(kind=kind):
                self.assertEqual(self._added(collector, attr, add),
                                 expected[kind])

    def test_own_factor_wins(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            collector = self._collector('optimum')
        for kind, (attr, add) in self._REFINEMENTS.items():
            with self.subTest(kind=kind):
                # Coarser than the optimum is allowed when asked for
                self.assertEqual(
                    self._added(collector, attr,
                                lambda c, a=add: a(c, resampling_factor=1.0)),
                    (1.0, None))
        # A method alone keeps the optimum factor
        attr, add = self._REFINEMENTS['contour']
        self.assertEqual(
            self._added(collector, attr,
                        lambda c: add(c, resampling_method='average')),
            (0.5, 'average'))

    def test_off_by_default(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            collector = self._collector(None)
            coarse = self._collector(1.0)
        # pylint: disable=protected-access
        self.assertIsNone(collector._resampling_factor)
        self.assertEqual(coarse._resampling_factor, 1.0)
        for kind, (attr, add) in self._REFINEMENTS.items():
            with self.subTest(kind=kind):
                self.assertEqual(self._added(collector, attr, add),
                                 (None, None))

    def test_flow_limiter_on_a_resampled_grid_warns(self):
        add = self._REFINEMENTS['flow_limiter'][1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            coarse = self._collector(1.0)
        with self.assertWarns(UserWarning) as caught:
            add(coarse)
        self.assertIn('Flow limiters', str(caught.warning))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            add(self._collector(None))

    def test_other_strings_rejected(self):
        for bad in ('native', 'best', 'auto'):
            with self.assertRaises(ValueError):
                self._collector(bad)

    def test_sizes_computed(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            collector = self._collector('optimum')
        collector.add_constant_value(500.0, upper_bound=0.0)
        collector._apply_features()  # pylint: disable=protected-access
        hfun = collector._hfun_list[0]  # pylint: disable=protected-access
        # 25 m pixels (0.5 * hmin) instead of the DEM's 22 m
        self.assertAlmostEqual(hfun.raster.pixel_size_m()[1], 25.0, delta=1.0)
        values = hfun.get_values()
        self.assertTrue(np.any(values == 500.0))


if __name__ == '__main__':
    unittest.main()
