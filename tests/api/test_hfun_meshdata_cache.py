"""HfunCollector.meshdata() caching.

The cached result used to be returned forever: refinements added after a
first meshdata() call were ignored, so were different kwargs, and callers
modifying the returned object (utils.reproject works in place) changed
the cache.
"""

import unittest
from unittest import mock

import numpy as np

from tests.api import multitile


# pylint: disable=protected-access
class TestMeshdataCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0, include_empty=False)
        cls.paths = cls.stack.paths[:2]

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def _hfun(self):
        # Creating a collector is collective under MPI (work dir broadcast),
        # so every rank creates it; the serial checks then run on rank 0.
        hfun = multitile.make_hfun(self.paths, 'serial')
        if not multitile.is_manager():
            self.skipTest('serial checks run on rank 0 only')
        return hfun

    def test_refinement_added_after_meshdata_is_used(self):
        hfun = self._hfun()
        before = np.asarray(hfun.meshdata().values).max()
        hfun.add_constant_value(500.0)
        after = np.asarray(hfun.meshdata().values).max()
        self.assertAlmostEqual(before, multitile.HMAX, delta=1.0)
        self.assertLessEqual(after, 500.0 + 1e-6)

    def test_region_constraint_invalidates(self):
        hfun = self._hfun()
        hfun.meshdata()
        version = hfun._state_version
        hfun.add_region_constraint(
            value=600.0, value_type='min',
            shape=multitile.default_aoi(), crs='EPSG:4326')
        self.assertGreater(hfun._state_version, version)
        self.assertFalse(hfun._applied)

    def test_same_inputs_reuse_and_kwargs_recompute(self):
        hfun = self._hfun()
        with mock.patch.object(hfun, '_meshdata_pipeline',
                               wraps=hfun._meshdata_pipeline) as pipeline:
            hfun.meshdata()
            hfun.meshdata()
            self.assertEqual(pipeline.call_count, 1)
            hfun.meshdata(stride=2)
            self.assertEqual(pipeline.call_count, 2)
            # MeshDriver reuses whatever is current, kwargs aside
            hfun._current_meshdata()
            self.assertEqual(pipeline.call_count, 2)
            hfun.add_constant_value(500.0)
            hfun._current_meshdata()
            self.assertEqual(pipeline.call_count, 3)

    def test_returned_object_does_not_alias_cache(self):
        hfun = self._hfun()
        first = hfun.meshdata()
        first.values[:] = -1.0
        second = hfun.meshdata()
        self.assertTrue(np.all(np.asarray(second.values) > 0))

    def test_repeated_calls_stay_collective_under_mpi(self):
        """Every rank takes the same cached/recompute decision (no hang)."""
        if 'mpi' not in multitile.available_modes():
            self.skipTest('needs mpiexec -n 2 or more')
        hfun = multitile.make_hfun(self.paths, 'mpi')
        first = hfun.meshdata()
        hfun.meshdata()                # cached on every rank
        hfun.meshdata(stride=2)        # recomputed on every rank
        current = hfun._current_meshdata()   # what MeshDriver uses
        hfun.add_constant_value(500.0)
        refined = hfun._current_meshdata()   # recomputed on every rank
        if multitile.is_manager():
            self.assertIsNotNone(first)
            self.assertIsNotNone(current)
            self.assertLessEqual(np.asarray(refined.values).max(), 500.0 + 1e-6)
        else:
            self.assertIsNone(refined)


if __name__ == '__main__':
    unittest.main()
