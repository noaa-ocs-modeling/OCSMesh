"""Topography function constraints given as lambdas.

``TopoFuncConstraint`` rejected lambdas so that every constraint could
be pickled for parallel and MPI runs, which broke serial scripts that
upstream OCSMesh accepted. Lambdas (and nested functions) are accepted
again; when one cannot be pickled, the constraints are applied on the
coordinator with a warning.

Run serially (serial, parallel) and under ``mpiexec -n 2`` or more (mpi).
"""

import unittest
import warnings

from tests.api import multitile


def _half_depth(depth):
    return abs(depth) / 2.0


def _constraint(func):
    def configure(hfun):
        multitile.add_depth_bands(hfun)
        hfun.add_topo_func_constraint(
            func=func, upper_bound=-10.0, value_type='min', rate=None)
    return configure


class TestLambdaConstraint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0, include_empty=False)
        cls.paths = cls.stack.paths[:2]
        cls.reference = multitile.refined_rasters(
            cls.paths, 'serial', configure=_constraint(_half_depth))

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def test_lambda_matches_named_function(self):
        for mode in multitile.available_modes():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = multitile.refined_rasters(
                    self.paths, mode,
                    configure=_constraint(lambda d: abs(d) / 2.0))
            fallback = [w for w in caught if 'cannot be pickled' in str(w.message)]
            with self.subTest(mode=mode):
                # the fallback is only needed when constraints are distributed
                self.assertEqual(bool(fallback), mode != 'serial')
                if multitile.is_manager():
                    self.assertEqual(
                        multitile.compare_rasters(self.reference, result), '')

    def test_named_function_is_distributed(self):
        for mode in multitile.available_modes():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = multitile.refined_rasters(
                    self.paths, mode, configure=_constraint(_half_depth))
            with self.subTest(mode=mode):
                self.assertFalse(
                    any('cannot be pickled' in str(w.message) for w in caught))
                if multitile.is_manager():
                    self.assertEqual(
                        multitile.compare_rasters(self.reference, result), '')


if __name__ == '__main__':
    unittest.main()
