"""Parallel (Pool) paths must stop when a worker fails.

The flow limiter, constant value and meshdata Pool paths used to log a
worker error and carry on, returning a size function without that
refinement (or without that tile). Each test makes every worker fail by
pointing the tasks at a topography file that does not exist and expects
a ``RuntimeError`` naming the failed stage. Each stage is called
directly so an earlier stage cannot fail first.
"""

import tempfile
import unittest
from unittest import mock

from tests.api import multitile


def _missing_topo(_hfun):
    return '/nonexistent/ocsmesh_fail_fast_topo.tif'


# pylint: disable=protected-access
class TestPoolFailFast(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0, include_empty=False)
        cls.paths = cls.stack.paths[:2]

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def _assert_raises(self, configure, stage, run):
        # Construction is collective under MPI; the Pool runs on rank 0.
        hfun = multitile.make_hfun(self.paths, 'parallel', configure=configure)
        if not multitile.is_manager():
            self.skipTest('Pool paths run on rank 0')
        with mock.patch('ocsmesh.hfun.collector._task_topo_path', _missing_topo):
            with self.assertRaisesRegex(RuntimeError, stage):
                run(hfun)

    def test_constant_value(self):
        self._assert_raises(
            lambda h: h.add_constant_value(500.0),
            'constant value worker', lambda h: h._apply_const_val())

    def test_flow_limiter(self):
        self._assert_raises(
            lambda h: h.add_subtidal_flow_limiter(
                hmin=100.0, hmax=3000.0, lower_bound=-30.0, upper_bound=0.0),
            'flow limiter worker', lambda h: h._apply_flow_limiters())

    def test_constraint(self):
        self._assert_raises(
            lambda h: h.add_topo_bound_constraint(
                value=1500.0, lower_bound=-30.0, upper_bound=-20.0,
                value_type='min', rate=None),
            'constraint worker', lambda h: h._apply_constraints())

    def test_meshdata(self):
        def run(hfun):
            with tempfile.TemporaryDirectory() as out_dir:
                hfun._calculate_and_write_hfun_to_disk(out_dir)
        self._assert_raises(None, 'meshdata worker', run)


if __name__ == '__main__':
    unittest.main()
