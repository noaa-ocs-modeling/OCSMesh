"""Serial vs distributed equivalence when rasters are clipped by ``base_shape``.

Regression tests for distributed workers reading the topography from the
original, unclipped DEM while the size function they refine is the clipped
one: every depth-based refinement then lands on the wrong pixels (fine
sizes outside the domain, shifted bands along tile edges).

Run serially (serial vs parallel) and under ``mpiexec -n 2`` or more
(serial vs parallel vs mpi).
"""

import unittest

import numpy as np

from tests.api import multitile


def _depth_bands(hfun):
    multitile.add_depth_bands(hfun)


def _flow_limiter(hfun):
    hfun.add_subtidal_flow_limiter(hmin=100.0, hmax=3000.0,
                                   lower_bound=-30.0, upper_bound=0.0)


def _topo_constraint(hfun):
    hfun.add_constant_value(500.0)
    hfun.add_topo_bound_constraint(value=1500.0, lower_bound=-30.0,
                                   upper_bound=-20.0, value_type='min',
                                   rate=None)


REFINEMENTS = {
    'depth_bands': _depth_bands,
    'flow_limiter': _flow_limiter,
    'topo_constraint': _topo_constraint,
}


class TestClippedRasterEquivalence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        stack = multitile.build_stack(scale=2.0, include_empty=False)
        cls.stack = stack
        cls.paths = stack.paths
        cls.aoi = stack.aoi

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def _check(self, name):
        configure = REFINEMENTS[name]
        reference = multitile.refined_rasters(
            self.paths, 'serial', aoi=self.aoi, configure=configure)
        for mode in multitile.available_modes()[1:]:
            other = multitile.refined_rasters(
                self.paths, mode, aoi=self.aoi, configure=configure)
            if multitile.is_manager():
                with self.subTest(refinement=name, mode=mode):
                    self.assertEqual(
                        multitile.compare_rasters(reference, other), '')

    def test_depth_bands(self):
        self._check('depth_bands')

    def test_flow_limiter(self):
        self._check('flow_limiter')

    def test_topo_constraint(self):
        self._check('topo_constraint')

    def test_no_sizes_outside_aoi(self):
        """Outside the AOI the topography is nodata: no refinement applies
        there, and the composite has no nodes there at all."""
        for mode in multitile.available_modes():
            stats = multitile.meshdata_stats(
                self.paths, mode, aoi=self.aoi, configure=_depth_bands)
            if not multitile.is_manager():
                continue
            from pyproj import CRS
            from shapely import contains_xy
            from ocsmesh import utils
            md = stats['meshdata']
            utils.reproject(md, CRS.from_epsg(4326))
            values = np.asarray(md.values).ravel()
            outside = ~contains_xy(self.aoi.buffer(0.01),
                                   md.coords[:, 0], md.coords[:, 1])
            with self.subTest(mode=mode):
                self.assertEqual(int(np.sum(values[outside] < multitile.HMAX)), 0)
                self.assertEqual(int(outside.sum()), 0)


if __name__ == '__main__':
    unittest.main()
