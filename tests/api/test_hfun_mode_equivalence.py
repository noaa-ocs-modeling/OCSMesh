"""Serial, parallel and MPI modes give the same size function.

Refinement types not covered by the other multi-tile tests: line
features, region constraints, the Courant number constraint, and all
refinement types together. Each is compared on the refined rasters and
on the final meshdata, with the AOI clipping the tiles.

Run serially (serial, parallel) and under ``mpiexec -n 2`` or more (mpi).
"""

import unittest

import numpy as np
from shapely.geometry import LineString, Point

from tests.api import multitile


def _line_feature(hfun):
    hfun.add_constant_value(1500.0)
    hfun.add_feature(shape=LineString([(0.2, 0.1), (0.6, 0.45)]),
                     expansion_rate=0.01, target_size=300.0, crs=4326)


def _region_constraints(hfun):
    hfun.add_constant_value(1000.0)
    hfun.add_region_constraint(value=2500.0, value_type='min',
                               shape=Point(0.5, 0.3).buffer(0.08),
                               crs='EPSG:4326', rate=0.0005)
    hfun.add_region_constraint(value=400.0, value_type='max',
                               shape=Point(0.3, 0.2).buffer(0.05),
                               crs='EPSG:4326', rate=None)


def _courant(hfun):
    multitile.add_depth_bands(hfun)
    hfun.add_courant_num_constraint(upper_bound=0.9, lower_bound=0.1,
                                    timestep=150, wave_amplitude=2)


def _everything(hfun):
    multitile.add_depth_bands(hfun)
    hfun.add_subtidal_flow_limiter(hmin=100.0, hmax=3000.0,
                                   lower_bound=-30.0, upper_bound=0.0)
    hfun.add_contour(level=0.0, expansion_rate=0.01, target_size=500.0)
    hfun.add_channel(level=0.0, width=2000.0, target_size=500.0,
                     expansion_rate=0.01, tolerance=50)
    hfun.add_patch(shape=Point(0.45, 0.3).buffer(0.05),
                   target_size=700.0, expansion_rate=None)
    hfun.add_feature(shape=LineString([(0.2, 0.1), (0.6, 0.45)]),
                     expansion_rate=0.01, target_size=300.0, crs=4326)
    hfun.add_topo_bound_constraint(value=1500.0, lower_bound=-30.0,
                                   upper_bound=-20.0, value_type='min',
                                   rate=None)
    hfun.add_region_constraint(value=2500.0, value_type='min',
                               shape=Point(0.5, 0.3).buffer(0.08),
                               crs='EPSG:4326', rate=None)


REFINEMENTS = {
    'line_feature': _line_feature,
    'region_constraints': _region_constraints,
    'courant': _courant,
    'everything': _everything,
}


class TestModeEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0, include_empty=True)

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def _check(self, name):
        configure = REFINEMENTS[name]
        kwargs = {'aoi': self.stack.aoi, 'configure': configure}
        reference = multitile.refined_rasters(
            self.stack.paths, 'serial', **kwargs)
        ref_stats = multitile.meshdata_stats(
            self.stack.paths, 'serial', **kwargs)
        for mode in multitile.available_modes()[1:]:
            rasters = multitile.refined_rasters(
                self.stack.paths, mode, **kwargs)
            stats = multitile.meshdata_stats(self.stack.paths, mode, **kwargs)
            if not multitile.is_manager():
                continue
            with self.subTest(refinement=name, mode=mode):
                self.assertEqual(
                    multitile.compare_rasters(reference, rasters), '')
                self.assertEqual(stats['nodes'], ref_stats['nodes'])
                np.testing.assert_allclose(stats['mean'], ref_stats['mean'],
                                           rtol=1e-9)

    def test_line_feature(self):
        self._check('line_feature')

    def test_region_constraints(self):
        self._check('region_constraints')

    def test_courant(self):
        self._check('courant')

    def test_everything(self):
        self._check('everything')


if __name__ == '__main__':
    unittest.main()
